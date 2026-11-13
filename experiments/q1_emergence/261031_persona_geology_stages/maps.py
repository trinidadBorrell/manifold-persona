#!/usr/bin/env python3
"""CPU numerical apparatus; this module never trains or downloads a model.

Manifest paths are relative to the manifest. Schema 1::

  {"schema_version": 1, "panel": "panel.json", "arms": {
    "real": {"trait-slug": "real/trait-slug/adapter.pt", ...},
    "scrambled": {...}, "seed1": {...}, "permuted": {...}},
    "permuted_data_order": [1, 2, ..., 0],
    "text_contrasts": "contrasts.npy", "activation_shifts": {"real": "shifts.npy"}}

permuted_data_order[i] is the original DATA trait index at wrong-name row i.
Panel order defines every array's trait order. text_contrasts is [traits, embedding].
Activation shifts are [traits, inputs, hidden], already adapter-minus-reference.
Use activation.py's NPZ files to verify trait order, layer and fixed-input hash.
Bare NPY arrays use panel order but have no embedded order or layer evidence.
Missing arms or text predictions produce an incomplete, never passing, pilot.
All numeric arrays, masks, null draws and factor fits are saved separately in NPZ.
Only observed cross-fold text residual blocks have a reported singular spectrum.

PAF iterates reduced-correlation eigenvectors with SMC initialization; it is not
PCA or minres. Oblimin uses gamma=0 with unit-column oblique transformations.
The rotation criterion agrees with GPArotation/factor_analyzer's gamma=0
criterion, independently optimized here with scipy BFGS; no Kaiser normalization.
References: https://personality-project.org/r/psych/help/fa.html and
https://github.com/EducationalTestingService/factor_analyzer/blob/main/factor_analyzer/rotator.py
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import spearmanr

PARALLEL_SEED = 3102026
LABEL_SEED = 3102027
ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000.)


class FactorFitError(ValueError):
    """A nonconvergent or inadmissible fit must not imply a pilot verdict."""


def _array(value, name, ndim=None):
    if hasattr(value, "detach"):
        if value.is_complex():
            raise ValueError(f"{name}: complex values")
        value = value.detach().cpu().double().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or np.iscomplexobj(value):
        raise ValueError(f"{name}: expected real numeric values")
    value = value.astype(np.float64, copy=False)
    if (ndim is not None and value.ndim != ndim) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: wrong dimensions or nonfinite values")
    if value.size == 0 and ndim != 2:
        raise ValueError(f"{name}: empty values")
    return value


def _square(value, name="map", correlation=False):
    value = _array(value, name, 2)
    if len(value) < 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name}: expected square matrix of at least two traits")
    if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: not symmetric")
    if correlation:
        if not np.allclose(np.diag(value), 1., atol=1e-9, rtol=1e-9):
            raise ValueError(f"{name}: correlation diagonal differs from one")
        if np.linalg.eigvalsh(value)[0] < -1e-8:
            raise ValueError(f"{name}: not positive semidefinite")
    return value


def _modules(modules):
    if not isinstance(modules, dict) or not modules:
        raise ValueError("adapter: no named modules")
    result = {}
    for name, module in modules.items():
        if not isinstance(name, str) or not isinstance(module, dict) or set(module) != {"A", "B", "scale"}:
            raise ValueError("adapter: expected named {A, B, scale} modules")
        a, b = _array(module["A"], name + ".A", 2), _array(module["B"], name + ".B", 2)
        scale = _array(module["scale"], name + ".scale")
        if scale.ndim != 0 or min(*a.shape, *b.shape) <= 0 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{name}: incompatible low-rank shapes or scale")
        result[name] = {"A": a, "B": b, "scale": float(scale)}
    return result


def load_adapter(path):
    """Read the training apparatus's tensor-only adapter.pt on CPU."""
    import torch
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"modules", "metadata"} or not isinstance(saved["metadata"], dict):
        raise ValueError("adapter.pt must contain modules and metadata only")
    return {"modules": _modules(saved["modules"]), "metadata": saved["metadata"]}


def validate_adapter_identities(metadata):
    """Reject mixed stage/checkpoint/tokenizer maps when identity is recorded."""
    identities = [{key: info.get(key) for key in ("stage", "model", "tokenizer")}
                  for arm in metadata.values() for info in arm.values()]
    if not identities or any(value != identities[0] for value in identities[1:]):
        raise ValueError("raw weight map mixes checkpoint/stage/tokenizer identities")
    return {"checkpoint_verified": identities[0]["stage"] is not None and identities[0]["model"] is not None,
            "tokenizer_verified": identities[0]["tokenizer"] is not None, "identity": identities[0]}


def low_rank_gram(adapters, trait_names):
    """Exact <s_i B_i A_i, s_j B_j A_j> without dense B@A updates."""
    if len(trait_names) < 2 or len(set(trait_names)) != len(trait_names) or set(adapters) != set(trait_names):
        raise ValueError("adapters: trait keys differ, repeat, or are insufficient")
    clean = [_modules(adapters[name]) for name in trait_names]
    names = sorted(clean[0])
    shapes = {name: (clean[0][name]["B"].shape[0], clean[0][name]["A"].shape[1]) for name in names}
    for trait, modules in zip(trait_names, clean):
        if set(modules) != set(names):
            raise ValueError(f"{trait}: mismatched module keys")
        for name, (outdim, indim) in shapes.items():
            if (modules[name]["B"].shape[0], modules[name]["A"].shape[1]) != (outdim, indim):
                raise ValueError(f"{trait}/{name}: mismatched update shape")
    n = len(clean)
    gram = np.zeros((n, n))
    module_squared_norms = np.zeros((n, len(names)))
    for i in range(n):
        for j in range(i, n):
            products = []
            for m, name in enumerate(names):
                left, right = clean[i][name], clean[j][name]
                value = left["scale"] * right["scale"] * np.sum(
                    (left["B"].T @ right["B"]) * (left["A"] @ right["A"].T))
                products.append(value)
                if i == j:
                    module_squared_norms[i, m] = value
            gram[i, j] = gram[j, i] = sum(products)
    if not np.all(np.isfinite(gram)) or np.any(np.diag(gram) <= 0):
        failed = [name for i, name in enumerate(trait_names) if not np.isfinite(gram[i, i]) or gram[i, i] <= 0]
        raise ValueError(f"nonfinite Gram or zero overall update norms: {failed}")
    norms = np.sqrt(np.diag(gram))
    cosine = gram / np.outer(norms, norms)
    _square(cosine, "adapter cosine", correlation=True)
    return {"gram": gram, "cosine": cosine, "norms": norms, "modules": names,
            "module_squared_norms": module_squared_norms,
            "zero_modules": {trait_names[i]: [name for m, name in enumerate(names) if module_squared_norms[i, m] == 0]
                             for i in range(n)}, "update_shapes": shapes,
            "scales": {trait: {name: clean[i][name]["scale"] for name in names} for i, trait in enumerate(trait_names)}}


@lru_cache(maxsize=12)
def parallel_null(p, n_observations=150, simulations=2000, seed=PARALLEL_SEED):
    """Ordered Gaussian-correlation eigenvalues; identical frozen draws per arm."""
    if p < 2 or n_observations < 3 or simulations < 2:
        raise ValueError("invalid parallel-analysis dimensions")
    rng = np.random.default_rng(seed)
    eigenvalues = np.empty((simulations, p))
    for start in range(0, simulations, 100):
        size = min(100, simulations - start)
        rows = rng.standard_normal((size, n_observations, p))
        rows -= rows.mean(axis=1, keepdims=True)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        eigenvalues[start:start + size] = np.linalg.eigvalsh(rows.transpose(0, 2, 1) @ rows)[:, ::-1]
    thresholds = np.quantile(eigenvalues, .95, axis=0, method="linear")
    eigenvalues.setflags(write=False)
    thresholds.setflags(write=False)
    return {"eigenvalues": eigenvalues, "thresholds": thresholds, "observations": n_observations,
            "simulations": simulations, "seed": seed, "percentile": .95}


def parallel_analysis(correlation, null=None):
    correlation = _square(correlation, correlation=True)
    null = parallel_null(len(correlation)) if null is None else null
    thresholds = _array(null["thresholds"], "null thresholds", 1)
    if thresholds.shape != (len(correlation),):
        raise ValueError("null trait dimension mismatch")
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    above = eigenvalues > thresholds
    count = next((i for i, retained in enumerate(above) if not retained), len(above))
    return {"n_factors": count, "eigenvalues": eigenvalues, "thresholds": thresholds, "above_null": above}


def parallel_sensitivities(correlation):
    return {n: parallel_analysis(correlation, parallel_null(len(correlation), n)) for n in (75, 150, 300)}


def oblimin(unrotated, tol=1e-7, max_iter=2000):
    """Gamma-zero oblimin, optimized over normalized oblique transformations."""
    unrotated = _array(unrotated, "unrotated loadings", 2)
    k = unrotated.shape[1]
    if k <= 1:
        return {"loadings": unrotated.copy(), "phi": np.eye(k), "rotation": np.eye(k),
                "converged": True, "iterations": 0, "gradient_norm": 0.}

    def objective(flat):
        raw = flat.reshape(k, k)
        norms = np.linalg.norm(raw, axis=0)
        if np.any(norms < 1e-12):
            return 1e100, np.zeros_like(flat)
        transform = raw / norms
        if abs(np.linalg.det(transform)) < 1e-12:
            return 1e100, np.zeros_like(flat)
        inverse = np.linalg.inv(transform)
        pattern = unrotated @ inverse.T
        squared = pattern ** 2
        others = squared.sum(axis=1, keepdims=True) - squared
        value = .25 * np.sum(squared * others)
        loading_gradient = pattern * others
        transform_gradient = -inverse.T @ loading_gradient.T @ pattern
        raw_gradient = (transform_gradient - transform * np.sum(transform * transform_gradient, axis=0)) / norms
        return float(value), raw_gradient.ravel()

    result = minimize(objective, np.eye(k).ravel(), method="BFGS", jac=True,
                      options={"gtol": tol, "maxiter": max_iter})
    gradient_norm = float(np.linalg.norm(objective(result.x)[1]))
    if not np.isfinite(result.fun) or gradient_norm > tol * 10:
        raise FactorFitError(f"oblimin did not converge: {result.message}; gradient={gradient_norm}")
    transform = result.x.reshape(k, k)
    transform /= np.linalg.norm(transform, axis=0)
    pattern = unrotated @ np.linalg.inv(transform).T
    phi = transform.T @ transform
    if np.linalg.eigvalsh(phi)[0] <= 0 or not np.all(np.isfinite(pattern)):
        raise FactorFitError("undefined oblimin rotation")
    return {"loadings": pattern, "phi": phi, "rotation": transform, "converged": True,
            "iterations": int(result.nit), "gradient_norm": gradient_norm}


def principal_axis_factoring(correlation, n_factors, tol=1e-9, max_iter=2000):
    """Iterative principal-axis factoring, SMC start, no smoothing or clipping."""
    correlation = _square(correlation, correlation=True)
    p = len(correlation)
    if not isinstance(n_factors, (int, np.integer)) or not 0 <= n_factors < p:
        raise ValueError("invalid factor count")
    if n_factors == 0:
        return {"unrotated": np.empty((p, 0)), "loadings": np.empty((p, 0)), "phi": np.empty((0, 0)),
                "rotation": np.empty((0, 0)), "uniquenesses": np.ones(p), "converged": True,
                "iterations": 0, "rotation_converged": True, "rotation_iterations": 0,
                "rotation_gradient_norm": 0., "communality_history": np.zeros((1, p)),
                "method": "principal_axis", "rotation_method": "oblimin_gamma0"}
    if np.linalg.eigvalsh(correlation)[0] <= 1e-12:
        raise FactorFitError("SMC initialization requires positive definite correlation; no smoothing authorized")
    communalities = 1 - 1 / np.diag(np.linalg.inv(correlation))
    history = [communalities.copy()]
    for iteration in range(1, max_iter + 1):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communalities)
        values, vectors = np.linalg.eigh(reduced)
        values, vectors = values[::-1][:n_factors], vectors[:, ::-1][:, :n_factors]
        if np.any(values <= 0):
            raise FactorFitError("nonpositive principal-axis factor eigenvalue")
        unrotated = vectors * np.sqrt(values)
        updated = np.sum(unrotated ** 2, axis=1)
        history.append(updated.copy())
        change = np.max(np.abs(updated - communalities))
        communalities = updated
        if not np.all(np.isfinite(updated)) or np.max(updated) > 1e6:
            raise FactorFitError("divergent principal-axis communalities")
        if change < tol:
            break
    else:
        raise FactorFitError("principal-axis factoring did not converge")
    uniquenesses = 1 - communalities
    if np.any(uniquenesses <= 0):
        raise FactorFitError("improper principal-axis solution: nonpositive uniqueness")
    rotated = oblimin(unrotated)
    return {"unrotated": unrotated, "loadings": rotated["loadings"], "phi": rotated["phi"],
            "rotation": rotated["rotation"], "uniquenesses": uniquenesses, "converged": True,
            "iterations": iteration, "rotation_converged": rotated["converged"],
            "rotation_iterations": rotated["iterations"], "rotation_gradient_norm": rotated["gradient_norm"],
            "communality_history": np.stack(history), "method": "principal_axis", "rotation_method": "oblimin_gamma0"}


def factor_map(correlation):
    parallel = parallel_sensitivities(correlation)
    result = {"parallel": parallel, "n_factors": parallel[150]["n_factors"]}
    try:
        result["fit"] = principal_axis_factoring(correlation, result["n_factors"])
        result["admissible"] = True
    except FactorFitError as error:
        result.update(admissible=False, error=str(error))
    return result


def match_factors(reference, candidate, threshold=.85):
    reference, candidate = _array(reference, "reference loadings", 2), _array(candidate, "candidate loadings", 2)
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("factor trait dimension mismatch")
    if reference.shape[1] != candidate.shape[1] or reference.shape[1] == 0:
        return {"pass": False, "count_match": reference.shape[1] == candidate.shape[1], "congruences": np.empty(0)}
    norms = np.outer(np.linalg.norm(reference, axis=0), np.linalg.norm(candidate, axis=0))
    if np.any(norms <= 0):
        raise ValueError("zero loading column")
    congruence = reference.T @ candidate / norms
    rows, columns = linear_sum_assignment(-np.abs(congruence))
    signs = np.where(congruence[rows, columns] < 0, -1., 1.)
    values = np.abs(congruence[rows, columns])
    return {"pass": bool(np.all(values >= threshold)), "count_match": True, "congruences": values,
            "permutation": columns, "signs": signs, "aligned": candidate[:, columns] * signs,
            "all_congruences": congruence}


def _labels(labels):
    if not isinstance(labels, list) or not labels:
        raise ValueError("trait metadata missing")
    seen = set()
    result = []
    for item in labels:
        if not isinstance(item, dict) or not {"trait", "slug", "factor", "keyed"} <= set(item):
            raise ValueError("expected trait, slug, factor, keyed metadata")
        if any(not isinstance(item[key], str) or not item[key] for key in ("trait", "slug", "factor")) or item["slug"] in seen:
            raise ValueError("invalid or duplicate trait metadata")
        seen.add(item["slug"])
        key = item["keyed"]
        if isinstance(key, bool) or key not in ("+", "-", 1, -1):
            raise ValueError("keyed must be '+'/'-' or numeric +1/-1")
        result.append({**item, "key": 1 if key in ("+", 1) else -1})
    return result


def label_permutation_test(loadings, labels, permutations=9999, seed=LABEL_SEED):
    labels = _labels(labels)
    loadings = _array(loadings, "label-test loadings", 2)
    if loadings.shape[0] != len(labels) or permutations < 1:
        raise ValueError("label test dimensions or permutation count")
    groups = sorted({item["factor"] for item in labels})
    keys = np.array([[item["key"] if item["factor"] == group else 0 for group in groups] for item in labels])
    key_norms = np.linalg.norm(keys, axis=0)
    loading_norms = np.linalg.norm(loadings, axis=0)
    if loadings.shape[1] == 0:
        return {"score": 0., "pvalue": 1., "null_scores": np.zeros(permutations),
                "permutations": permutations, "seed": seed, "groups": groups}
    if np.any(loading_norms <= 0):
        raise ValueError("undefined zero loading in label test")
    normalized = loadings / loading_norms
    def score(order):
        association = (keys[order].T @ normalized) / key_norms[:, None]
        return float(np.max(np.abs(association), axis=1).sum())
    observed = score(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    null = np.array([score(rng.permutation(len(labels))) for _ in range(permutations)])
    return {"score": observed, "pvalue": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
            "null_scores": null, "permutations": permutations, "seed": seed, "groups": groups}


def recover_permuted_gram(wrong_named, data_order, real):
    wrong_named, real = _square(wrong_named), _square(real)
    order = np.asarray(data_order)
    if wrong_named.shape != real.shape or order.dtype.kind not in "iu" or sorted(order.tolist()) != list(range(len(real))):
        raise ValueError("permuted_data_order must be a complete index permutation")
    if np.any(order == np.arange(len(real))):
        raise ValueError("permutation control must be a derangement")
    inverse = np.argsort(order)
    recovered = wrong_named[np.ix_(inverse, inverse)]
    return {"pass": bool(np.allclose(recovered, real, rtol=1e-5, atol=1e-6)), "recovered": recovered,
            "max_absolute_error": float(np.max(np.abs(recovered - real))), "data_order": order}


def balanced_trait_folds(labels):
    labels = _labels(labels)
    groups = {}
    for i, item in enumerate(labels):
        groups.setdefault((item["factor"], item["key"]), []).append(i)
    if any(len(indices) != 2 for indices in groups.values()) or len(groups) != 2 * len({i["factor"] for i in labels}):
        raise ValueError("two markers per factor/pole cell required for balanced holdout")
    folds = [[], []]
    for cell in sorted(groups):
        indices = sorted(groups[cell], key=lambda i: labels[i]["trait"].casefold())
        for fold, index in enumerate(indices):
            folds[fold].append(index)
    return [np.array(sorted(indices), dtype=int) for indices in folds]


def mean_text_contrasts(chosen, rejected):
    chosen, rejected = _array(chosen, "chosen embeddings", 3), _array(rejected, "rejected embeddings", 3)
    if chosen.shape != rejected.shape or min(chosen.shape) < 1:
        raise ValueError("chosen/rejected embedding shapes differ or are empty")
    return (chosen - rejected).mean(axis=1)


def _ridge_predict(x, y, fit_indices, predict_indices, alpha):
    train_x, train_y = x[fit_indices], y[fit_indices]
    center, intercept = train_x.mean(axis=0), train_y.mean()
    centered = train_x - center
    # Dual form keeps the solve small for 384-dimensional MiniLM features.
    coefficients = np.linalg.solve(centered @ centered.T + alpha * np.eye(len(fit_indices)), train_y - intercept)
    return (x[predict_indices] - center) @ centered.T @ coefficients + intercept


def _spearman(left, right):
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        raise ValueError("map correlation undefined for constant or insufficient edges")
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("nonfinite map correlation")
    return value


def text_holdout_prediction(gram, contrasts, labels, alphas=ALPHAS):
    """Nested trait holdout, target-specific self exclusion, no test-label tuning."""
    gram = _square(gram)
    contrasts = _array(contrasts, "text contrasts", 2)
    if contrasts.shape[0] != len(gram) or len(labels) != len(gram) or contrasts.shape[1] == 0:
        raise ValueError("text prediction trait dimensions differ")
    alphas = _array(alphas, "ridge alphas", 1)
    if np.any(alphas <= 0):
        raise ValueError("ridge alphas must be positive")
    halves = balanced_trait_folds(labels)
    if min(map(len, halves)) < 3:
        raise ValueError("inner trait holdout requires at least three outer training traits")
    predictions = np.zeros_like(gram)
    mask = np.zeros_like(gram, dtype=bool)
    folds = []
    for fold in range(2):
        train, test = halves[fold], halves[1 - fold]
        losses = []
        for alpha in alphas:
            errors = []
            for omitted in train:
                for target in train:
                    if omitted == target:
                        continue
                    fit_indices = train[(train != omitted) & (train != target)]
                    prediction = _ridge_predict(contrasts, gram[:, target], fit_indices, [omitted], alpha)[0]
                    errors.append((prediction - gram[omitted, target]) ** 2)
            losses.append(float(np.mean(errors)))
        alpha = float(alphas[np.argmin(losses)])
        for target in train:
            fit_indices = train[train != target]
            predictions[test, target] = _ridge_predict(contrasts, gram[:, target], fit_indices, test, alpha)
            mask[test, target] = True
        residual_block = gram[np.ix_(test, train)] - predictions[np.ix_(test, train)]
        folds.append({"train": train, "test": test, "alpha": alpha, "inner_mse": np.array(losses),
                      "residual_singular_values": np.linalg.svd(residual_block, compute_uv=False)})
    residuals = np.where(mask, gram - predictions, 0.)
    actual, predicted = gram[mask], predictions[mask]
    denominator = np.sum((actual - actual.mean()) ** 2)
    if denominator <= 0:
        raise ValueError("heldout R2 undefined for constant observed similarities")
    averaged_predictions = (predictions + predictions.T) / 2
    averaged_residuals = (residuals + residuals.T) / 2
    upper = np.triu(mask | mask.T, 1)
    # A rectangular cross-half block is fully observed; no within-half values are invented.
    cross_block = averaged_residuals[np.ix_(halves[0], halves[1])]
    return {"predictions": predictions, "prediction_mask": mask, "residuals": residuals,
            "averaged_predictions": averaged_predictions, "averaged_residuals": averaged_residuals,
            "heldout_r2": float(1 - np.sum((actual - predicted) ** 2) / denominator),
            "map_correlation": _spearman(gram[upper], averaged_predictions[upper]), "alphas": alphas,
            "folds": folds, "residual_singular_values": np.linalg.svd(cross_block, compute_uv=False),
            "spectrum_scope": "averaged heldout cross-half residual block; within-half edges unobserved"}


def check_no_adapter(shifts, atol=0.):
    shifts = _array(shifts, "no-adapter shift")
    if not np.all(np.abs(shifts) <= atol):
        raise ValueError("no-adapter activation shift is not zero")
    return True


def load_activation_shifts(path, trait_names):
    """Read activation.py NPZ output, enforcing order and any no-adapter check."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        values = _array(loaded, "activation shifts", 3)
        if trait_names and len(values) != len(trait_names):
            raise ValueError("activation NPY trait count mismatch")
        if not trait_names:
            check_no_adapter(values)
        return {"shifts": values, "metadata": {"format": "npy", "trait_order_verified": False}}
    with loaded as archive:
        expected = {"shifts", "traits", "layer_label", "hidden_state_index", "input_sha256"}
        if set(archive.files) != expected:
            raise ValueError("unknown or missing activation archive fields")
        values = _array(archive["shifts"], "activation shifts", 3)
        traits = archive["traits"]
        if traits.ndim != 1 or traits.dtype.kind not in "US" or len(traits) != len(values):
            raise ValueError("activation archive trait metadata mismatch")
        traits = traits.tolist()
        if len(set(traits)) != len(traits):
            raise ValueError("duplicate activation trait names")
        control = [i for i, name in enumerate(traits) if name == "no_adapter"]
        indices = [i for i, name in enumerate(traits) if name != "no_adapter"]
        if [traits[i] for i in indices] != list(trait_names):
            raise ValueError("activation trait order differs from panel order")
        metadata = {"format": "npz", "trait_order_verified": True}
        for key in ("layer_label", "hidden_state_index"):
            value = archive[key]
            if value.ndim != 0 or value.dtype.kind not in "iu" or int(value) < 0:
                raise ValueError("invalid activation layer metadata")
            metadata[key] = int(value)
        bank_hash = archive["input_sha256"]
        if bank_hash.ndim != 0 or bank_hash.dtype.kind not in "US" or not str(bank_hash):
            raise ValueError("invalid activation input hash")
        metadata["input_sha256"] = str(bank_hash)
        result = {"shifts": values[indices], "metadata": metadata}
        if control:
            check_no_adapter(values[control])
            result["no_adapter"] = values[control]
        return result


def activation_similarity(shifts):
    shifts = _array(shifts, "activation shifts", 3)
    if min(shifts.shape) < 1 or len(shifts) < 2:
        raise ValueError("empty or insufficient activation shifts")
    flat = shifts.reshape(len(shifts), -1)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms <= 0):
        raise ValueError("zero overall activation shift")
    normalized = flat / norms[:, None]
    return normalized @ normalized.T


def compare_maps(left, right):
    left, right = _square(left), _square(right)
    if left.shape != right.shape:
        raise ValueError("map dimensions differ")
    upper = np.triu_indices(len(left), 1)
    return _spearman(left[upper], right[upper])


def analyze_pilot(arms, labels, permuted_data_order=None, text_contrasts=None, activation_shifts=None):
    """Analyze available arm dicts {arm: {trait_slug: named_module_dict}}."""
    if not arms or not set(arms) <= {"real", "scrambled", "seed1", "permuted"} or "real" not in arms:
        raise ValueError("unknown arm or missing real arm")
    labels = _labels(labels)
    traits = [item["slug"] for item in labels]
    maps = {arm: low_rank_gram(adapters, traits) for arm, adapters in arms.items()}
    factors = {arm: factor_map(value["cosine"]) for arm, value in maps.items()}
    result = {"maps": maps, "factors": factors, "traits": traits, "conditions": {}}
    conditions = result["conditions"]
    conditions["real_structure"] = factors["real"]["n_factors"] >= 1
    conditions["scrambled_zero"] = "scrambled" in factors and factors["scrambled"]["n_factors"] == 0
    conditions["factor_fit"] = all(value["admissible"] for value in factors.values())
    conditions["seed_stability"] = False
    if "seed1" in factors and factors["real"]["admissible"] and factors["seed1"]["admissible"]:
        result["seed_match"] = match_factors(factors["real"]["fit"]["loadings"], factors["seed1"]["fit"]["loadings"])
        conditions["seed_stability"] = result["seed_match"]["pass"]
    conditions["permutation_control"] = False
    if "permuted" in maps:
        if permuted_data_order is None:
            raise ValueError("permuted arm requires data-order derangement")
        result["permutation_recovery"] = recover_permuted_gram(maps["permuted"]["gram"], permuted_data_order, maps["real"]["gram"])
        result["permutation_cosine_recovery"] = recover_permuted_gram(maps["permuted"]["cosine"], permuted_data_order, maps["real"]["cosine"])
        if factors["permuted"]["admissible"]:
            result["wrong_label_test"] = label_permutation_test(factors["permuted"]["fit"]["loadings"], labels)
            conditions["permutation_control"] = result["permutation_recovery"]["pass"] and result["wrong_label_test"]["pvalue"] > .05
    conditions["data_baseline"] = text_contrasts is not None
    if text_contrasts is not None:
        result["text_contrasts"] = _array(text_contrasts, "text contrasts", 2)
        result["text_prediction"] = text_holdout_prediction(maps["real"]["cosine"], text_contrasts, labels)
    if activation_shifts is not None:
        if not isinstance(activation_shifts, dict) or not set(activation_shifts) <= set(arms) | {"no_adapter"}:
            raise ValueError("unknown activation shift arm")
        activation = {}
        for arm, shifts in activation_shifts.items():
            if arm == "no_adapter":
                result["no_adapter_zero"] = check_no_adapter(shifts)
            else:
                if len(shifts) != len(traits):
                    raise ValueError("activation trait count mismatch")
                activation[arm] = activation_similarity(shifts)
        result["activation_maps"] = activation
        result["activation_shifts"] = activation_shifts
        result["activation_map_agreement"] = {f"{a}:{b}": compare_maps(activation[a], activation[b])
            for i, a in enumerate(sorted(activation)) for b in sorted(activation)[i + 1:]}
    complete = {"real", "scrambled", "seed1", "permuted"} <= set(arms) and text_contrasts is not None
    result["verdict"] = "pass" if complete and all(conditions.values()) else "inconclusive" if not complete or not conditions["factor_fit"] else "fail"
    return result


def _json_arrays(value, arrays, prefix="analysis"):
    """Keep strict JSON small; every array is saved under an explicit NPZ key."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fci" and not np.all(np.isfinite(value)):
            raise ValueError(f"nonfinite output at {prefix}")
        arrays[prefix] = value
        return {"npz_key": prefix, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _json_arrays(item, arrays, f"{prefix}/{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_arrays(item, arrays, f"{prefix}/{i}") for i, item in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"nonfinite output at {prefix}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "arrays.npz").exists():
        raise ValueError("output already contains an analysis; use a separate run folder")
    manifest = json.loads(manifest_path.read_text())
    allowed = {"schema_version", "panel", "arms", "permuted_data_order", "text_contrasts", "activation_shifts"}
    if not isinstance(manifest, dict) or set(manifest) - allowed:
        raise ValueError("unknown manifest fields")
    if manifest.get("schema_version") != 1 or not {"panel", "arms"} <= set(manifest):
        raise ValueError("expected manifest schema_version 1, panel and arms")
    base = manifest_path.parent
    inputs = {str(manifest_path): _sha256(manifest_path)}
    def path(relative):
        item = (base / relative).resolve()
        inputs[str(item)] = _sha256(item)
        return item
    panel = json.loads(path(manifest["panel"]).read_text())
    if panel.get("schema_version") != 1 or "traits" not in panel:
        raise ValueError("expected panel schema_version 1 and traits")
    labels = _labels(panel["traits"])
    if len(labels) != 20 or len({item["factor"] for item in labels}) != 5:
        raise ValueError("pilot CLI requires twenty markers in five groups")
    balanced_trait_folds(labels)
    metadata, arms = {}, {}
    if not isinstance(manifest["arms"], dict):
        raise ValueError("arms must be an object")
    for arm, files in manifest["arms"].items():
        if arm not in {"real", "scrambled", "seed1", "permuted"} or not isinstance(files, dict):
            raise ValueError("unknown arm or invalid adapter paths")
        arms[arm], metadata[arm] = {}, {}
        for trait, relative in files.items():
            adapter = load_adapter(path(relative))
            arms[arm][trait], metadata[arm][trait] = adapter["modules"], adapter["metadata"]
    adapter_identity = validate_adapter_identities(metadata)
    contrasts = np.load(path(manifest["text_contrasts"]), allow_pickle=False) if "text_contrasts" in manifest else None
    shifts, activation_metadata = None, {}
    if "activation_shifts" in manifest:
        shifts = {}
        for name, relative in manifest["activation_shifts"].items():
            if name not in set(arms) | {"no_adapter"}:
                raise ValueError("unknown activation arm")
            loaded = load_activation_shifts(path(relative), [] if name == "no_adapter" else [item["slug"] for item in labels])
            activation_metadata[name] = loaded["metadata"]
            if name == "no_adapter":
                shifts[name] = loaded.get("no_adapter", loaded["shifts"])
            else:
                shifts[name] = loaded["shifts"]
                if "no_adapter" in loaded:
                    shifts["no_adapter"] = loaded["no_adapter"]
        archives = [value for name, value in activation_metadata.items() if name != "no_adapter"]
        if len(archives) > 1:
            if len({value["format"] for value in archives}) > 1:
                raise ValueError("cannot verify shared activation inputs across mixed NPY/NPZ files")
            if archives[0]["format"] == "npz":
                identities = {(value["layer_label"], value["hidden_state_index"], value["input_sha256"]) for value in archives}
                if len(identities) != 1:
                    raise ValueError("activation maps use different layers or fixed inputs")
    result = analyze_pilot(arms, labels, manifest.get("permuted_data_order"), contrasts, shifts)
    result.update(schema_version=1, manifest=manifest, panel=panel, adapter_metadata=metadata,
                  adapter_identity=adapter_identity,
                  activation_input_metadata=activation_metadata, maps_code_sha256=_sha256(Path(__file__)),
                  input_sha256=inputs, invocation=sys.argv, software={"python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__}, scope="within-stage apparatus; provenance/training-integrity gates require separate checks")
    result["null_draws"] = {n: parallel_null(len(labels), n) for n in (75, 150, 300)}
    arrays = {}
    summary = _json_arrays(result, arrays)
    np.savez_compressed(output / "arrays.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_manifest(args.manifest, args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "error.json").write_text(json.dumps({"error_type": type(error).__name__, "error": str(error)}, indent=2, allow_nan=False) + "\n")
        print(f"analysis refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
