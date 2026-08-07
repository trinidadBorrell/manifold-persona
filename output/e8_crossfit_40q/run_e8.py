#!/usr/bin/env python3
"""E8: disjoint-question cross-fitting of role geometry (40q response cloud).

Implements the adversarial review's highest-value missing control #1
(output/codex_review_2026-07-30.md): split the 40 shared questions into
disjoint DEV/VERIFY panels before any analysis, and cross that with
non-overlapping role folds, so that role geometry is estimated on one
question panel and tested on the other with held-out roles.

Review findings addressed:
- #1 (via design): model-free reliability endpoint P1 needs no Gaussian null
  of any covariance; its null is a role-label permutation.
- #2/#11: the E1b GPLVM machinery is reused (its analytic training gradients
  were finite-difference-verified by the review) with seeded multi-start,
  L-BFGS-B analytic-gradient held-out latent inference (no Nelder-Mead),
  explicit convergence checks, and per-fit convergence records.
- #5: each role contributes exactly one out-of-fold paired loss from a
  pre-specified non-overlapping 4-fold partition; inference is role-level.
- #9: DEV/VERIFY question panels are disjoint; centroids are per panel.
- #10: everything is repeated on view='prompt_last' (response-last token).
- #15: one pre-specified primary latent dimension (d=5); other dims are
  labeled exploratory.

Run from the repository root:

    .venv/bin/python output/e8_crossfit_40q/run_e8.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Thread limits must be set before NumPy/SciPy import.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["MP_ROLE_DIR"] = str(REPO_ROOT / "data" / "embeddings_roles_resp_40q")

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import sklearn
from sklearn.decomposition import PCA

sys.path.insert(0, str(REPO_ROOT / "src"))
from manifold_persona.common import load_points  # noqa: E402


# ----------------------------------------------------------------------------
# Pre-specified design constants (frozen before any result is computed).
# ----------------------------------------------------------------------------
N_ROLES = 276
N_VARIANTS = 5
N_QUESTIONS = 40
PANEL_SIZE = 20
N_FOLDS = 4
R_DENOISE = 50
PRIMARY_D = 5
SECONDARY_DIMS = (2, 3, 8, 12)
ALL_DIMS = (2, 3, 5, 8, 12)
N_NULL_DRAWS = 200
N_P1_BOOT = 1000
N_P2_BOOT = 1000
N_SIGN_PERM = 10000
N_RESTARTS = 3
VIEWS = ("prompt_avg", "prompt_last")
VIEW_IDX = {"prompt_avg": 0, "prompt_last": 1}
DEFAULT_LAYER = 19

P1_CI_LOWER_THRESHOLD = 0.5   # bootstrap CI lower bound for panel-stability
P2_GAP_FRACTION = 0.25        # GPLVM advantage must be >= 25% of panel-shift gap
HELDOUT_CONVERGENCE_GATE = 0.80

ROOT_SEED = 0
CHILD_NAMES = (
    "question_panel",   # 0: DEV/VERIFY split of the 40 question ids
    "role_folds",       # 1: 4-fold partition of the 276 roles
    "p1_null",          # 2: role-label shuffles of VERIFY centroids
    "p1_bootstrap",     # 3: role bootstrap for distance correlation
    "p2_bootstrap",     # 4: role bootstrap for paired loss differences
    "p2_signperm",      # 5: role-level sign permutations
    "gplvm_init",       # 6: jittered-PCA GPLVM initializations
)
_ROOT_SS = np.random.SeedSequence(ROOT_SEED)
_CHILDREN = dict(zip(CHILD_NAMES, _ROOT_SS.spawn(len(CHILD_NAMES))))


def rng_for(name: str, *key: int) -> np.random.Generator:
    """Named, order-independent child stream of SeedSequence(0)."""
    ss = _CHILDREN[name]
    if key:
        ss = np.random.SeedSequence(
            entropy=ss.entropy, spawn_key=tuple(ss.spawn_key) + tuple(key)
        )
    return np.random.default_rng(ss)


def seed_registry() -> dict[str, Any]:
    return {
        "root_entropy": ROOT_SEED,
        "scheme": (
            "numpy SeedSequence(0).spawn per named child; per-use streams "
            "extend the child's spawn_key with (view_idx, ...) integers, "
            "so every stream is deterministic and order-independent"
        ),
        "children": {
            name: {"spawn_key": list(_CHILDREN[name].spawn_key)}
            for name in CHILD_NAMES
        },
    }


@dataclass
class FitConfig:
    """GPLVM numerical settings (E1b machinery + review fixes)."""

    alternations: int = 8
    round_rel_ftol: float = 1e-9
    hyper_maxiter: int = 200
    latent_maxiter: int = 500
    heldout_maxiter: int = 400
    jitter: float = 1e-8
    train_grad_tol: float = 1e-3
    heldout_grad_rel_tol: float = 1e-6
    init_jitter_scale: float = 0.3


# ----------------------------------------------------------------------------
# JSON helpers (from run_e1b.py).
# ----------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), indent=1, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def squared_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if b is None:
        b = a
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


# ----------------------------------------------------------------------------
# GPLVM.  Kernel, marginal likelihood, and analytic training gradients are
# copied unchanged from output/e1b_gplvm_repro/run_e1b.py (finite-difference
# verified by the adversarial review).  Changes, per review findings #2/#11:
# seeded initializations supplied by the caller, convergence recording,
# early-stopped alternation, and L-BFGS-B analytic-gradient held-out latent
# inference in place of Nelder-Mead.  The new held-out gradient is
# finite-difference verified at runtime (recorded in results.json).
# ----------------------------------------------------------------------------
class GPLVM:
    """Small, shared-kernel GPLVM for independent output dimensions."""

    def __init__(self, latent_dim: int, kernel: str, config: FitConfig) -> None:
        if kernel not in {"rbf", "linear"}:
            raise ValueError(f"unsupported kernel: {kernel}")
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.config = config
        self.z_: np.ndarray | None = None
        self.z_pca_base_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] = {}

    # --- verbatim from E1b -------------------------------------------------
    def _kernel_train(
        self, z: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        signal_var = float(np.exp(theta[0]))
        if self.kernel == "rbf":
            lengthscale = float(np.exp(theta[1]))
            noise_var = float(np.exp(theta[2]))
            d2 = squared_distances(z)
            core = signal_var * np.exp(-0.5 * d2 / (lengthscale**2))
            params = {
                "signal_variance": signal_var,
                "lengthscale": lengthscale,
                "noise_variance": noise_var,
            }
        else:
            noise_var = float(np.exp(theta[1]))
            core = signal_var * (z @ z.T)
            params = {
                "signal_variance": signal_var,
                "noise_variance": noise_var,
            }
        kernel = core.copy()
        kernel.flat[:: len(z) + 1] += noise_var + self.config.jitter
        return kernel, core, params

    def _objective_and_gradients(
        self, z: np.ndarray, theta: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Negative log marginal likelihood and analytic derivatives."""
        n, n_outputs = y.shape
        kernel, core, params = self._kernel_train(z, theta)
        try:
            chol = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve(chol, y, check_finite=False)
            kernel_inv = cho_solve(
                chol, np.eye(n, dtype=np.float64), check_finite=False
            )
        except (np.linalg.LinAlgError, ValueError):
            return 1e100, np.zeros_like(z), np.zeros_like(theta)

        logdet = 2.0 * np.log(np.diag(chol[0])).sum()
        objective = 0.5 * (
            n_outputs * logdet
            + np.sum(y * alpha)
            + n * n_outputs * np.log(2.0 * np.pi)
        )

        g_kernel = 0.5 * (n_outputs * kernel_inv - alpha @ alpha.T)
        g_kernel = 0.5 * (g_kernel + g_kernel.T)

        if self.kernel == "rbf":
            length2 = params["lengthscale"] ** 2
            d2 = squared_distances(z)
            weighted = g_kernel * core
            grad_z = (
                2.0
                * (weighted @ z - weighted.sum(axis=1, keepdims=True) * z)
                / length2
            )
            grad_theta = np.array(
                [
                    np.sum(g_kernel * core),
                    np.sum(g_kernel * core * d2 / length2),
                    params["noise_variance"] * np.trace(g_kernel),
                ],
                dtype=np.float64,
            )
        else:
            grad_z = 2.0 * params["signal_variance"] * (g_kernel @ z)
            grad_theta = np.array(
                [
                    np.sum(g_kernel * core),
                    params["noise_variance"] * np.trace(g_kernel),
                ],
                dtype=np.float64,
            )
        return float(objective), grad_z, grad_theta

    def _fix_scale_gauge(
        self, z: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scale = float(np.sqrt(np.mean(z * z)))
        if not np.isfinite(scale) or scale < 1e-10:
            return z, theta
        z = z / scale
        theta = theta.copy()
        if self.kernel == "rbf":
            theta[1] -= np.log(scale)
        else:
            theta[0] += 2.0 * np.log(scale)
        return z, theta

    # --- adapted: caller-supplied init, convergence recording ---------------
    def fit(self, y: np.ndarray, z_init: np.ndarray | None = None) -> "GPLVM":
        y = np.asarray(y, dtype=np.float64)
        if np.max(np.abs(y.mean(axis=0))) > 1e-8:
            raise ValueError("GPLVM expects training outputs centered by train PCA")
        pca_init = PCA(
            n_components=self.latent_dim, svd_solver="full", random_state=0
        )
        z_raw = pca_init.fit_transform(y)
        z_base = z_raw / max(float(np.sqrt(np.mean(z_raw * z_raw))), 1e-12)
        self._pca_init = pca_init
        self.z_pca_base_ = z_base.copy()
        z = z_base.copy() if z_init is None else np.asarray(
            z_init, dtype=np.float64
        ).copy()

        data_var = max(float(np.mean(y * y)), 1e-8)
        if self.kernel == "rbf":
            nonzero = squared_distances(z)
            median_distance = float(np.sqrt(np.median(nonzero[nonzero > 1e-14])))
            theta = np.log(
                [
                    data_var,
                    max(median_distance, 0.2),
                    max(0.05 * data_var, 1e-7),
                ]
            )
            bounds = [
                (np.log(data_var * 1e-4), np.log(data_var * 1e4)),
                (np.log(0.03), np.log(100.0)),
                (np.log(data_var * 1e-7), np.log(data_var * 10.0)),
            ]
        else:
            theta = np.log([data_var / self.latent_dim, 0.05 * data_var])
            bounds = [
                (np.log(data_var * 1e-5), np.log(data_var * 1e5)),
                (np.log(data_var * 1e-7), np.log(data_var * 10.0)),
            ]

        start_obj = self._objective_and_gradients(z, theta, y)[0]
        rounds: list[dict[str, Any]] = []
        prev_obj = start_obj
        early_stop = False
        for round_index in range(self.config.alternations):
            def hyper_fg(th: np.ndarray) -> tuple[float, np.ndarray]:
                objective, _, gradient = self._objective_and_gradients(z, th, y)
                return objective, gradient

            hyper_result = minimize(
                hyper_fg,
                theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.config.hyper_maxiter,
                    "ftol": 1e-10,
                    "gtol": 1e-6,
                    "maxls": 40,
                },
            )
            theta = hyper_result.x

            z_shape = z.shape

            def latent_fg(flat: np.ndarray) -> tuple[float, np.ndarray]:
                objective, gradient, _ = self._objective_and_gradients(
                    flat.reshape(z_shape), theta, y
                )
                return objective, gradient.ravel()

            latent_result = minimize(
                latent_fg,
                z.ravel(),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self.config.latent_maxiter,
                    "ftol": 1e-10,
                    "gtol": 1e-5,
                    "maxls": 40,
                    "maxcor": 20,
                },
            )
            z = latent_result.x.reshape(z_shape)
            z, theta = self._fix_scale_gauge(z, theta)
            round_obj = self._objective_and_gradients(z, theta, y)[0]
            rounds.append(
                {
                    "round": round_index,
                    "objective": round_obj,
                    "hyper_success": bool(hyper_result.success),
                    "hyper_nit": int(hyper_result.nit),
                    "latent_success": bool(latent_result.success),
                    "latent_nit": int(latent_result.nit),
                }
            )
            if prev_obj - round_obj <= self.config.round_rel_ftol * (
                1.0 + abs(round_obj)
            ):
                early_stop = True
                break
            prev_obj = round_obj

        def final_hyper_fg(th: np.ndarray) -> tuple[float, np.ndarray]:
            objective, _, gradient = self._objective_and_gradients(z, th, y)
            return objective, gradient

        hyper_result = minimize(
            final_hyper_fg,
            theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.config.hyper_maxiter,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 40,
            },
        )
        theta = hyper_result.x
        kernel, _, params = self._kernel_train(z, theta)
        chol = cho_factor(kernel, lower=True, check_finite=False)
        alpha = cho_solve(chol, y, check_finite=False)
        final_obj, grad_z, grad_theta = self._objective_and_gradients(z, theta, y)
        joint_grad_inf = float(
            max(np.max(np.abs(grad_z)), np.max(np.abs(grad_theta)))
        )
        last_round = rounds[-1]
        # Convergence: coordinate-wise stationarity (last latent L-BFGS-B and
        # the final hyperparameter L-BFGS-B both terminated by their own
        # ftol/gtol criteria, i.e. success=True) OR small joint gradient.
        converged = bool(
            (last_round["latent_success"] and hyper_result.success)
            or joint_grad_inf <= self.config.train_grad_tol
        )

        self.z_ = z
        self.theta_ = theta
        self.alpha_ = alpha
        self.y_train_ = y
        self.fit_diagnostics_ = {
            "start_objective": start_obj,
            "final_objective": float(final_obj),
            "rounds": rounds,
            "n_rounds": len(rounds),
            "early_stop": early_stop,
            "final_hyper_success": bool(hyper_result.success),
            "joint_grad_inf": joint_grad_inf,
            "converged": converged,
            "hyperparameters": params,
        }
        return self

    def _cross_kernel(self, z: np.ndarray) -> np.ndarray:
        if self.z_ is None or self.theta_ is None:
            raise RuntimeError("fit must be called first")
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        signal_var = float(np.exp(self.theta_[0]))
        if self.kernel == "rbf":
            lengthscale = float(np.exp(self.theta_[1]))
            return signal_var * np.exp(
                -0.5 * squared_distances(z, self.z_) / (lengthscale**2)
            )
        return signal_var * (z @ self.z_.T)

    def predict(self, z: np.ndarray) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("fit must be called first")
        return self._cross_kernel(z) @ self.alpha_

    def _initial_heldout_latents(self, y_test: np.ndarray) -> np.ndarray:
        """Affine continuation of the (unjittered) PCA initialization."""
        if self.z_pca_base_ is None or self.z_ is None:
            raise RuntimeError("fit must be called first")
        raw_test = self._pca_init.transform(y_test)
        raw_train = self._pca_init.transform(self.y_train_)
        scale = max(float(np.sqrt(np.mean(raw_train * raw_train))), 1e-12)
        initial_test = raw_test / scale
        design = np.column_stack(
            [self.z_pca_base_, np.ones(len(self.z_pca_base_))]
        )
        affine, *_ = np.linalg.lstsq(design, self.z_, rcond=None)
        return np.column_stack(
            [initial_test, np.ones(len(initial_test))]
        ) @ affine

    def _heldout_objective_grad(
        self, z: np.ndarray, observed: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Mean squared residual to the GP posterior mean, with analytic grad."""
        k = self._cross_kernel(z)[0]              # (n_train,)
        pred = k @ self.alpha_                    # (q,)
        residual = pred - observed
        q = observed.shape[0]
        objective = float(np.mean(residual * residual))
        c = self.alpha_ @ residual                # (n_train,)
        if self.kernel == "rbf":
            length2 = float(np.exp(self.theta_[1])) ** 2
            w = c * k
            grad = (-2.0 / (q * length2)) * (w.sum() * z - w @ self.z_)
        else:
            signal_var = float(np.exp(self.theta_[0]))
            grad = (2.0 / q) * signal_var * (self.z_.T @ c)
        return objective, grad

    def reconstruct_heldout(
        self, y_test: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """L-BFGS-B latent inference (analytic gradients) per held-out row."""
        if self.z_ is None or self.y_train_ is None:
            raise RuntimeError("fit must be called first")
        y_test = np.asarray(y_test, dtype=np.float64)
        affine_initial = self._initial_heldout_latents(y_test)
        nearest_indices = np.argmin(
            squared_distances(y_test, self.y_train_), axis=1
        )
        nearest_initial = self.z_[nearest_indices]
        reconstruction = np.empty_like(y_test)
        diagnostics: list[dict[str, Any]] = []

        for row_index, observed in enumerate(y_test):
            def fg(z_flat: np.ndarray) -> tuple[float, np.ndarray]:
                return self._heldout_objective_grad(z_flat, observed)

            best = None
            best_start = -1
            for start_index, start in enumerate(
                (affine_initial[row_index], nearest_initial[row_index])
            ):
                result = minimize(
                    fg,
                    start,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "maxiter": self.config.heldout_maxiter,
                        "ftol": 1e-13,
                        "gtol": 1e-9,
                        "maxls": 40,
                    },
                )
                if best is None or result.fun < best.fun:
                    best = result
                    best_start = start_index
            grad_inf = float(np.max(np.abs(best.jac)))
            converged = bool(
                best.success
                or grad_inf
                <= self.config.heldout_grad_rel_tol * (1.0 + abs(best.fun))
            )
            reconstruction[row_index] = self.predict(best.x)[0]
            diagnostics.append(
                {
                    "success": bool(best.success),
                    "converged": converged,
                    "nit": int(best.nit),
                    "fun": float(best.fun),
                    "grad_inf": grad_inf,
                    "start_used": best_start,
                }
            )
        return reconstruction, diagnostics


def finite_difference_check_heldout(
    model: GPLVM, y_rows: np.ndarray, eps: float = 1e-6
) -> dict[str, Any]:
    """Verify the NEW held-out analytic gradient against central differences."""
    z0 = model._initial_heldout_latents(y_rows)
    max_rel_err = 0.0
    for i in range(len(y_rows)):
        _, grad = model._heldout_objective_grad(z0[i], y_rows[i])
        numeric = np.zeros_like(grad)
        for k in range(len(grad)):
            zp = z0[i].copy()
            zp[k] += eps
            zm = z0[i].copy()
            zm[k] -= eps
            numeric[k] = (
                model._heldout_objective_grad(zp, y_rows[i])[0]
                - model._heldout_objective_grad(zm, y_rows[i])[0]
            ) / (2.0 * eps)
        rel = float(
            np.max(np.abs(numeric - grad)) / (np.max(np.abs(grad)) + 1e-12)
        )
        max_rel_err = max(max_rel_err, rel)
    return {"n_points": int(len(y_rows)), "eps": eps, "max_rel_err": max_rel_err}


# ----------------------------------------------------------------------------
# Data loading, grid assertion, panel centroids.
# ----------------------------------------------------------------------------
def assert_balanced_grid(meta: pd.DataFrame) -> tuple[list[str], list[int]]:
    if len(meta) != N_ROLES * N_VARIANTS * N_QUESTIONS:
        raise ValueError(f"expected 55200 rows, got {len(meta)}")
    roles = sorted(meta["role"].unique())
    questions = sorted(int(q) for q in meta["question_idx"].unique())
    instructions = sorted(int(i) for i in meta["instruction_idx"].unique())
    if len(roles) != N_ROLES:
        raise ValueError(f"expected {N_ROLES} roles, got {len(roles)}")
    if len(questions) != N_QUESTIONS:
        raise ValueError(f"expected {N_QUESTIONS} questions, got {len(questions)}")
    if instructions != list(range(N_VARIANTS)):
        raise ValueError(f"unexpected instruction_idx values: {instructions}")
    cell_sizes = meta.groupby(
        ["role", "instruction_idx", "question_idx"], observed=True
    ).size()
    if len(cell_sizes) != N_ROLES * N_VARIANTS * N_QUESTIONS:
        raise ValueError("grid is not complete: missing cells")
    if not (cell_sizes == 1).all():
        raise ValueError("grid is not balanced: duplicated cells")
    return roles, questions


def panel_centroids(
    X: np.ndarray, role_codes: np.ndarray, panel_mask: np.ndarray
) -> np.ndarray:
    """[276, hidden] float64 raw per-role centroid over one question panel."""
    codes = role_codes[panel_mask]
    counts = np.bincount(codes, minlength=N_ROLES)
    if not (counts == N_VARIANTS * PANEL_SIZE).all():
        raise ValueError("panel is not balanced across roles")
    order = np.argsort(codes, kind="stable")
    rows = X[panel_mask][order].astype(np.float64)
    return rows.reshape(N_ROLES, N_VARIANTS * PANEL_SIZE, -1).mean(axis=1)


def residualization_equivalence_check(
    X: np.ndarray,
    meta: pd.DataFrame,
    role_codes: np.ndarray,
    panel_questions: list[int],
    train_indices: np.ndarray,
    raw_centroids: np.ndarray,
) -> dict[str, float]:
    """On the balanced grid, subtracting per-question TRAIN-role means from
    every row and then averaging per role must equal subtracting the mean of
    TRAIN-role raw centroids from each raw centroid.  Verified numerically."""
    train_role_set = set(int(i) for i in train_indices)
    in_train = np.isin(role_codes, list(train_role_set))
    q_values = meta["question_idx"].to_numpy()
    qmeans = []
    for q in panel_questions:
        mask = (q_values == q) & in_train
        if mask.sum() != len(train_indices) * N_VARIANTS:
            raise ValueError("per-question train-row count mismatch")
        qmeans.append(X[mask].astype(np.float64).mean(axis=0))
    qmean_avg = np.mean(qmeans, axis=0)
    train_centroid_mean = raw_centroids[train_indices].mean(axis=0)
    max_abs_diff = float(np.max(np.abs(qmean_avg - train_centroid_mean)))
    scale = float(np.max(np.abs(train_centroid_mean)))
    if max_abs_diff > 1e-8 * max(scale, 1.0):
        raise AssertionError(
            f"residualization shortcut invalid: max diff {max_abs_diff}"
        )
    return {"max_abs_diff": max_abs_diff, "reference_scale": scale}


# ----------------------------------------------------------------------------
# P1: model-free cross-panel reliability.
# ----------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a_ranked: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a_ranked, rankdata(b))[0, 1])


def run_p1(
    dev_cent: np.ndarray,
    ver_cent: np.ndarray,
    view_index: int,
    n_null: int,
    n_boot: int,
) -> dict[str, Any]:
    # All-roles residualization is used here (no model fitting -> no leakage
    # concern).  On the balanced grid it is exactly a per-panel translation of
    # every centroid, so pairwise distances and the centered statistics below
    # are invariant to it; it is retained as the declared estimand definition.
    dev_c = dev_cent - dev_cent.mean(axis=0)
    ver_c = ver_cent - ver_cent.mean(axis=0)

    d_dev = squareform(pdist(dev_c))
    d_ver = squareform(pdist(ver_c))
    iu = np.triu_indices(N_ROLES, 1)
    ut_dev = d_dev[iu]
    ut_ver = d_ver[iu]
    ut_dev_ranked = rankdata(ut_dev)

    pearson_real = _pearson(ut_dev, ut_ver)
    spearman_real = _spearman(ut_dev_ranked, ut_ver)

    # NULL: shuffle role labels of the VERIFY centroids.
    rng = rng_for("p1_null", view_index)
    null_pearson = np.empty(n_null)
    null_spearman = np.empty(n_null)
    for i in range(n_null):
        perm = rng.permutation(N_ROLES)
        ut_perm = d_ver[np.ix_(perm, perm)][iu]
        null_pearson[i] = _pearson(ut_dev, ut_perm)
        null_spearman[i] = _spearman(ut_dev_ranked, ut_perm)

    # Bootstrap over roles.  Duplicate-role pairs (distance exactly 0 in both
    # panels) are excluded from each draw's correlation.
    rng = rng_for("p1_bootstrap", view_index)
    boot_pearson = np.empty(n_boot)
    boot_spearman = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N_ROLES, N_ROLES)
        sub_dev = d_dev[np.ix_(idx, idx)][iu]
        sub_ver = d_ver[np.ix_(idx, idx)][iu]
        valid = idx[iu[0]] != idx[iu[1]]
        boot_pearson[i] = _pearson(sub_dev[valid], sub_ver[valid])
        boot_spearman[i] = _spearman(rankdata(sub_dev[valid]), sub_ver[valid])

    # Per-role position reliability after per-panel centering.
    a = dev_c - dev_c.mean(axis=1, keepdims=True)
    b = ver_c - ver_c.mean(axis=1, keepdims=True)
    role_r = (a * b).sum(axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )

    # PC-score stability: covariance PCA (centered, no z-scoring) on DEV.
    pca5 = PCA(n_components=5, svd_solver="full", random_state=0)
    scores_dev = pca5.fit_transform(dev_c)
    scores_ver = (ver_c - pca5.mean_) @ pca5.components_.T
    pc_corr = [
        _pearson(scores_dev[:, k], scores_ver[:, k]) for k in range(5)
    ]

    sp_ci = np.percentile(boot_spearman, [2.5, 97.5])
    pe_ci = np.percentile(boot_pearson, [2.5, 97.5])
    exceeds_null = bool(spearman_real > null_spearman.max())
    panel_stable = bool(exceeds_null and sp_ci[0] > P1_CI_LOWER_THRESHOLD)

    return {
        "residualization": (
            "all-roles per-question means (no model fitting here); on the "
            "balanced grid this is a per-panel translation, so all P1 "
            "statistics are invariant to it"
        ),
        "n_pairs": int(len(ut_dev)),
        "distance_corr": {"pearson": pearson_real, "spearman": spearman_real},
        "null": {
            "n_draws": n_null,
            "spearman_draws": null_spearman,
            "pearson_draws": null_pearson,
            "spearman_max": float(null_spearman.max()),
            "spearman_mean": float(null_spearman.mean()),
            "pearson_max": float(null_pearson.max()),
            "n_null_ge_real_spearman": int((null_spearman >= spearman_real).sum()),
            "n_null_ge_real_pearson": int((null_pearson >= pearson_real).sum()),
        },
        "bootstrap": {
            "n_draws": n_boot,
            "spearman_ci95": sp_ci,
            "pearson_ci95": pe_ci,
            "spearman_draws": boot_spearman,
            "pearson_draws": boot_pearson,
            "duplicate_pair_handling": "pairs with identical resampled roles excluded",
        },
        "role_position_reliability": {
            "values": role_r,
            "mean": float(role_r.mean()),
            "median": float(np.median(role_r)),
            "q25": float(np.percentile(role_r, 25)),
            "q75": float(np.percentile(role_r, 75)),
            "min": float(role_r.min()),
            "max": float(role_r.max()),
            "frac_above_0.5": float((role_r > 0.5).mean()),
        },
        "pc_score_stability": {
            "pearson_pc1_to_pc5": pc_corr,
            "dev_explained_variance_ratio": pca5.explained_variance_ratio_,
            "projection": (
                "each panel centered by its own role mean, both projected on "
                "DEV covariance-PCA components"
            ),
        },
        "decision": {
            "rule": (
                "panel-stable iff real Spearman distance correlation exceeds "
                "all null draws AND bootstrap Spearman CI lower bound > "
                f"{P1_CI_LOWER_THRESHOLD}"
            ),
            "exceeds_all_null_draws": exceeds_null,
            "bootstrap_ci_lower_gt_threshold": bool(
                sp_ci[0] > P1_CI_LOWER_THRESHOLD
            ),
            "panel_stable": panel_stable,
        },
    }


# ----------------------------------------------------------------------------
# P2: cross-fitted PCA vs GPLVM reconstruction.
# ----------------------------------------------------------------------------
def run_p2_dim(
    dev_cent: np.ndarray,
    ver_cent: np.ndarray,
    folds: list[np.ndarray],
    d: int,
    view_index: int,
    config: FitConfig,
    n_restarts: int,
    n_boot: int,
    n_perm: int,
    run_fd_check: bool,
) -> dict[str, Any]:
    nmse_pca_ver = np.full(N_ROLES, np.nan)
    nmse_pca_dev = np.full(N_ROLES, np.nan)
    nmse_gplvm_ver = np.full(N_ROLES, np.nan)
    heldout_converged = np.zeros(N_ROLES, dtype=bool)
    fold_records: list[dict[str, Any]] = []
    fd_check_record = None
    train_fit_records: list[dict[str, Any]] = []

    for fold_index, test_idx in enumerate(folds):
        started = time.time()
        train_idx = np.concatenate(
            [folds[j] for j in range(len(folds)) if j != fold_index]
        )
        # Question-effect residualization with TRAIN-role means only (leakage
        # rule).  On the balanced grid this equals subtracting the train-role
        # raw-centroid mean of each panel, which aligns the two panels using
        # train information only (verified in residualization_check).
        mu_dev = dev_cent[train_idx].mean(axis=0)
        mu_ver = ver_cent[train_idx].mean(axis=0)
        dev_r = dev_cent - mu_dev
        ver_r = ver_cent - mu_ver

        denoiser = PCA(
            n_components=R_DENOISE, svd_solver="full", random_state=0
        )
        y_train = denoiser.fit_transform(dev_r[train_idx])
        y_test_ver = denoiser.transform(ver_r[test_idx])
        y_test_dev = denoiser.transform(dev_r[test_idx])
        train_mean = y_train.mean(axis=0)
        base_ver = float(np.mean((y_test_ver - train_mean) ** 2))
        base_dev = float(np.mean((y_test_dev - train_mean) ** 2))

        pca_d = PCA(n_components=d, svd_solver="full", random_state=0)
        pca_d.fit(y_train)
        rec_ver = pca_d.inverse_transform(pca_d.transform(y_test_ver))
        rec_dev = pca_d.inverse_transform(pca_d.transform(y_test_dev))
        mse_pca_ver = np.mean((y_test_ver - rec_ver) ** 2, axis=1)
        mse_pca_dev = np.mean((y_test_dev - rec_dev) ** 2, axis=1)
        nmse_pca_ver[test_idx] = mse_pca_ver / base_ver
        nmse_pca_dev[test_idx] = mse_pca_dev / base_dev

        # GPLVM: 3 seeded initializations (PCA, jittered PCA x2); restart
        # selected by training objective only.
        restart_summaries = []
        best_model = None
        pca_base = PCA(n_components=d, svd_solver="full", random_state=0)
        z_raw = pca_base.fit_transform(y_train)
        z_base = z_raw / max(float(np.sqrt(np.mean(z_raw * z_raw))), 1e-12)
        for restart in range(n_restarts):
            if restart == 0:
                z_init = None
            else:
                jrng = rng_for("gplvm_init", view_index, d, fold_index, restart)
                z_init = z_base + config.init_jitter_scale * jrng.standard_normal(
                    z_base.shape
                )
            model = GPLVM(d, kernel="rbf", config=config)
            model.fit(y_train, z_init=z_init)
            diag = model.fit_diagnostics_
            restart_summaries.append(
                {
                    "restart": restart,
                    "init": "pca" if restart == 0 else "jittered_pca",
                    "final_objective": diag["final_objective"],
                    "converged": diag["converged"],
                    "early_stop": diag["early_stop"],
                    "joint_grad_inf": diag["joint_grad_inf"],
                    "n_rounds": diag["n_rounds"],
                }
            )
            if (
                best_model is None
                or diag["final_objective"]
                < best_model.fit_diagnostics_["final_objective"]
            ):
                best_model = model
                best_restart = restart
        objectives = [r["final_objective"] for r in restart_summaries]
        train_fit_records.append(
            {
                "fold": fold_index,
                "restarts": restart_summaries,
                "selected_restart": best_restart,
                "selected_converged": restart_summaries[best_restart]["converged"],
                "objective_spread": float(max(objectives) - min(objectives)),
                "fit_diagnostics_selected": best_model.fit_diagnostics_,
            }
        )

        if run_fd_check and fold_index == 0:
            fd_check_record = finite_difference_check_heldout(
                best_model, np.asarray(y_test_ver[:3], dtype=np.float64)
            )
            if fd_check_record["max_rel_err"] > 1e-4:
                raise AssertionError(
                    "held-out analytic gradient failed finite-difference "
                    f"check: {fd_check_record['max_rel_err']}"
                )

        rec_gplvm, heldout_diags = best_model.reconstruct_heldout(y_test_ver)
        mse_gplvm = np.mean((y_test_ver - rec_gplvm) ** 2, axis=1)
        nmse_gplvm_ver[test_idx] = mse_gplvm / base_ver
        for local_i, role_i in enumerate(test_idx):
            heldout_converged[role_i] = heldout_diags[local_i]["converged"]

        fold_records.append(
            {
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "baseline_mse_verify": base_ver,
                "baseline_mse_dev": base_dev,
                "denoiser_evr_sum": float(
                    denoiser.explained_variance_ratio_.sum()
                ),
                "pca_nmse_verify": float(np.mean(mse_pca_ver) / base_ver),
                "pca_nmse_dev": float(np.mean(mse_pca_dev) / base_dev),
                "gplvm_nmse_verify": float(np.mean(mse_gplvm) / base_ver),
                "heldout_converged_frac": float(
                    np.mean([h["converged"] for h in heldout_diags])
                ),
                "heldout_success_frac": float(
                    np.mean([h["success"] for h in heldout_diags])
                ),
                "heldout_mean_nit": float(
                    np.mean([h["nit"] for h in heldout_diags])
                ),
                "heldout_per_role": heldout_diags,
                "elapsed_seconds": time.time() - started,
            }
        )
        print(
            f"    fold {fold_index}: PCA(ver)={fold_records[-1]['pca_nmse_verify']:.4f} "
            f"PCA(dev)={fold_records[-1]['pca_nmse_dev']:.4f} "
            f"GPLVM(ver)={fold_records[-1]['gplvm_nmse_verify']:.4f} "
            f"conv={fold_records[-1]['heldout_converged_frac']:.2f} "
            f"({fold_records[-1]['elapsed_seconds']:.0f}s)",
            flush=True,
        )

    if np.isnan(nmse_pca_ver).any() or np.isnan(nmse_gplvm_ver).any():
        raise RuntimeError("some roles were never scored out-of-fold")

    diffs = nmse_pca_ver - nmse_gplvm_ver  # > 0 means GPLVM better
    mean_diff = float(diffs.mean())

    rng = rng_for("p2_bootstrap", view_index, d)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N_ROLES, N_ROLES)
        boot_means[i] = diffs[idx].mean()
    ci = np.percentile(boot_means, [2.5, 97.5])

    rng = rng_for("p2_signperm", view_index, d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, N_ROLES))
    perm_means = signs @ diffs / N_ROLES
    p_one_sided = float((1 + np.sum(perm_means >= mean_diff - 1e-15)) / (n_perm + 1))
    p_two_sided = float(
        (1 + np.sum(np.abs(perm_means) >= abs(mean_diff) - 1e-15)) / (n_perm + 1)
    )

    gap = float(
        np.mean(
            [f["pca_nmse_verify"] - f["pca_nmse_dev"] for f in fold_records]
        )
    )
    conv_frac = float(heldout_converged.mean())
    train_conv_selected = float(
        np.mean([r["selected_converged"] for r in train_fit_records])
    )
    train_conv_all = float(
        np.mean(
            [
                s["converged"]
                for r in train_fit_records
                for s in r["restarts"]
            ]
        )
    )

    return {
        "d": d,
        "n_restarts": n_restarts,
        "per_role": {
            "nmse_pca_verify": nmse_pca_ver,
            "nmse_pca_dev": nmse_pca_dev,
            "nmse_gplvm_verify": nmse_gplvm_ver,
            "paired_diff_pca_minus_gplvm": diffs,
            "heldout_converged": heldout_converged.astype(int),
        },
        "folds": fold_records,
        "training_fits": train_fit_records,
        "fd_check_heldout_gradient": fd_check_record,
        "summary": {
            "pca_nmse_verify_mean": float(nmse_pca_ver.mean()),
            "pca_nmse_dev_mean": float(nmse_pca_dev.mean()),
            "gplvm_nmse_verify_mean": float(nmse_gplvm_ver.mean()),
            "mean_paired_diff_pca_minus_gplvm": mean_diff,
            "bootstrap_ci95": ci,
            "bootstrap_n_draws": n_boot,
            "sign_permutation_p_one_sided": p_one_sided,
            "sign_permutation_p_two_sided": p_two_sided,
            "sign_permutation_n_draws": n_perm,
            "panel_shift_gap_pca": gap,
            "heldout_converged_frac": conv_frac,
            "training_selected_converged_frac": train_conv_selected,
            "training_all_restarts_converged_frac": train_conv_all,
        },
    }


def p2_decision(record: dict[str, Any]) -> dict[str, Any]:
    s = record["summary"]
    ci_excludes_zero = bool(s["bootstrap_ci95"][0] > 0.0)
    gap = s["panel_shift_gap_pca"]
    threshold = P2_GAP_FRACTION * gap
    advantage_large_enough = bool(s["mean_paired_diff_pca_minus_gplvm"] >= threshold)
    gap_degenerate = bool(gap <= 0.0)  # flagged in report; rule applied literally
    gate_ok = bool(s["heldout_converged_frac"] >= HELDOUT_CONVERGENCE_GATE)
    # The convergence gate is part of the verdict, not just an annotation —
    # otherwise a run with unconverged fits still reports survives=True
    # (review finding: gate computed but ignored).
    survives = bool(ci_excludes_zero and advantage_large_enough and gate_ok)
    return {
        "rule": (
            "nonlinearity survives cross-fitting iff role-level bootstrap CI "
            "of mean(PCA - GPLVM) excludes 0 AND the advantage is >= "
            f"{P2_GAP_FRACTION} x the same-fold PCA question-panel transfer "
            "gap AND held-out convergence is >= "
            f"{HELDOUT_CONVERGENCE_GATE:.0%}"
        ),
        "ci_excludes_zero": ci_excludes_zero,
        "advantage_threshold": float(threshold),
        "advantage_ge_threshold": advantage_large_enough,
        "panel_shift_gap_degenerate": gap_degenerate,
        "heldout_convergence_gate_met": gate_ok,
        "survives_crossfit": survives,
    }


# ----------------------------------------------------------------------------
# s3: response-length covariate probe.
# ----------------------------------------------------------------------------
def run_s3(
    X: np.ndarray, meta: pd.DataFrame, questions: list[int]
) -> dict[str, Any]:
    responses = meta["response"]
    if responses.isna().any():
        raise ValueError("missing responses in metadata")
    lengths = responses.str.len().to_numpy(dtype=np.float64)
    q_values = meta["question_idx"].to_numpy()
    global_mean = np.mean(X, axis=0, dtype=np.float64)
    mean_len = np.empty(N_QUESTIONS)
    shift = np.empty(N_QUESTIONS)
    for i, q in enumerate(questions):
        mask = q_values == q
        mean_len[i] = lengths[mask].mean()
        qmean = X[mask].astype(np.float64).mean(axis=0)
        shift[i] = float(np.linalg.norm(qmean - global_mean))
    pear = _pearson(mean_len, shift)
    spear = _spearman(rankdata(mean_len), shift)
    return {
        "note": (
            "exploratory covariate probe over all rows (no fitting); response "
            "length is character count (token counts not stored in metadata)"
        ),
        "question_idx": list(questions),
        "mean_response_char_len": mean_len,
        "question_shift_magnitude": shift,
        "pearson_len_vs_shift": pear,
        "spearman_len_vs_shift": spear,
    }


# ----------------------------------------------------------------------------
# Report generation: every number is read back from results.json.
# ----------------------------------------------------------------------------
def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def build_report(results: dict[str, Any]) -> str:
    cfg = results["config"]
    L: list[str] = []
    add = L.append
    add(
        "# E8: disjoint-question cross-fitting of role geometry "
        f"(layer {cfg['layer']})"
    )
    add("")
    add(
        "Generated programmatically from `results.json` by `run_e8.py` — "
        "no hand-maintained numbers."
    )
    if cfg["layer_extension"]:
        add(
            f"This is a post-hoc layer-{cfg['layer']} sensitivity run using "
            f"the frozen layer-{cfg['protocol_origin_layer']} design, splits, "
            "seeds, dimensions, and fitting settings without retuning."
        )
    add("")
    add("## Pre-specified endpoints and decision rules")
    add("")
    add(
        "Declared before results were inspected; all randomness derives from "
        f"`numpy.random.SeedSequence({cfg['root_seed']})` via named children "
        "(recorded in `results.json: seeds`)."
    )
    add("")
    add(
        "- **Design**: the 40 shared questions are split once into disjoint "
        "DEV (20) / VERIFY (20) panels; the 276 roles are partitioned once "
        "into 4 non-overlapping folds of 69. Per-panel role centroid = mean "
        "of that role's 100 rows (5 prompt variants x 20 panel questions). "
        "Question-effect residualization subtracts per-question means; for "
        "P2 those means are computed from TRAIN-fold roles only."
    )
    add(
        "- **P1 (primary, model-free)**: cross-panel reliability of role "
        "geometry on all 276 roles. Primary statistic: Spearman correlation "
        "between the DEV and VERIFY pairwise role-distance upper triangles "
        "(37,950 pairs); Pearson reported alongside. Null: "
        f"{cfg['n_null_draws']} role-label shuffles of the VERIFY centroids. "
        f"Uncertainty: {cfg['n_p1_bootstrap']}-draw role bootstrap. "
        "**Decision rule**: role geometry is *panel-stable* iff the real "
        "Spearman exceeds all null draws AND the bootstrap CI lower bound > "
        f"{cfg['p1_ci_lower_threshold']}. Supporting metrics (no rule): "
        "per-role position reliability and PC1-PC5 score stability."
    )
    add(
        f"- **P2 (primary, d={cfg['primary_d']})**: per role fold, fit a "
        f"{cfg['r_denoise']}-D PCA denoiser then PCA-{cfg['primary_d']} and "
        f"RBF-GPLVM-{cfg['primary_d']} on DEV-panel centroids of the 207 "
        "train roles; score reconstruction of VERIFY-panel centroids of the "
        "69 held-out roles, normalized by the fold's train-centroid-mean "
        "baseline. GPLVM: E1b machinery with 3 seeded inits (PCA, jittered "
        "PCA x2), L-BFGS-B analytic gradients for training AND held-out "
        "latents, restart selected by training objective, convergence "
        "recorded per fit. Each role contributes exactly one out-of-fold "
        "paired loss difference (PCA - GPLVM). Inference: mean difference, "
        f"{cfg['n_p2_bootstrap']}-draw role bootstrap 95% CI, one-sided "
        f"role-level sign-permutation p ({cfg['n_sign_perm']} draws, "
        "alternative: GPLVM better). **Decision rule**: nonlinearity "
        "survives cross-fitting iff the CI excludes 0 AND the advantage is "
        f">= {cfg['p2_gap_fraction']} x the same-fold PCA question-panel "
        "transfer gap (PCA trained on DEV: nMSE on VERIFY minus nMSE on DEV, "
        "held-out roles). **Gate**: if < "
        f"{cfg['heldout_convergence_gate']:.0%} of held-out GPLVM fits "
        "converge at d=5, interpretation is gated."
    )
    add(
        "- **Secondary (exploratory, no decision rules)**: (s1) d in "
        f"{list(cfg['secondary_dims'])}; (s2) everything on "
        "`view='prompt_last'` (response-last token); (s3) per-question mean "
        "response length vs question centroid-shift magnitude."
    )
    add("")

    add("## Design as executed")
    add("")
    design = results["design"]
    add(f"- Balanced grid asserted: {design['grid_assertion']}")
    add(f"- DEV questions: {design['dev_questions']}")
    add(f"- VERIFY questions: {design['verify_questions']}")
    add(
        f"- Role folds: {design['n_folds']} folds x "
        f"{design['fold_sizes']} roles (full lists in `results.json`)."
    )
    add(
        "- Residualization shortcut check (per-question train-role means vs "
        "train-role centroid mean): max abs diff = "
        f"{design['residualization_check']['max_abs_diff']:.3e} against "
        f"reference scale {design['residualization_check']['reference_scale']:.3f}."
    )
    add(
        "- Note: on this balanced grid, per-question mean subtraction "
        "translates every role centroid of a panel by the same vector. For "
        "P1 (all-roles means) every reported statistic is invariant to it; "
        "for P2 (train-role means) it aligns the DEV and VERIFY panels using "
        "train-fold information only, which is exactly the leakage rule."
    )
    add("")

    for view in results["config"]["views"]:
        v = results["views"][view]
        primary = view == "prompt_avg"
        tag = "primary" if primary else "secondary/exploratory (s2)"
        add(f"## P1 — cross-panel reliability, `{view}` ({tag})")
        add("")
        p1 = v["p1"]
        dc = p1["distance_corr"]
        nu = p1["null"]
        bo = p1["bootstrap"]
        add(
            f"- Distance-profile correlation over {p1['n_pairs']} role pairs: "
            f"Spearman {dc['spearman']:.4f}, Pearson {dc['pearson']:.4f}."
        )
        add(
            f"- Null ({nu['n_draws']} role shuffles): Spearman max "
            f"{nu['spearman_max']:.4f} (mean {nu['spearman_mean']:.4f}); "
            f"{nu['n_null_ge_real_spearman']}/{nu['n_draws']} draws >= real."
        )
        add(
            f"- Bootstrap 95% CI ({bo['n_draws']} draws): Spearman "
            f"{_fmt_ci(bo['spearman_ci95'])}, Pearson {_fmt_ci(bo['pearson_ci95'])}."
        )
        rr = p1["role_position_reliability"]
        add(
            "- Per-role position reliability (DEV vs VERIFY centroid "
            "correlation after per-panel centering): mean "
            f"{rr['mean']:.4f}, median {rr['median']:.4f}, IQR "
            f"[{rr['q25']:.4f}, {rr['q75']:.4f}], min {rr['min']:.4f}, "
            f"{rr['frac_above_0.5']:.1%} of roles > 0.5."
        )
        pc = p1["pc_score_stability"]
        pc_txt = ", ".join(
            f"PC{k + 1} {r:.3f}" for k, r in enumerate(pc["pearson_pc1_to_pc5"])
        )
        evr = ", ".join(f"{e:.3f}" for e in pc["dev_explained_variance_ratio"])
        add(f"- PC-score stability (DEV-fit covariance PCA): {pc_txt}.")
        add(f"- DEV PC1-PC5 explained variance ratios: {evr}.")
        dec = p1["decision"]
        if primary:
            verdict = "PANEL-STABLE" if dec["panel_stable"] else "NOT panel-stable"
            add("")
            add(
                f"**P1 verdict ({view}): {verdict}** — exceeds all null draws: "
                f"{dec['exceeds_all_null_draws']}; bootstrap CI lower bound > "
                f"{results['config']['p1_ci_lower_threshold']}: "
                f"{dec['bootstrap_ci_lower_gt_threshold']}."
            )
        else:
            add(
                f"- Same rule applied (exploratory): panel_stable = "
                f"{dec['panel_stable']}."
            )
        add("")

        add(f"## P2 — cross-fitted PCA vs RBF-GPLVM, `{view}` ({tag})")
        add("")
        d_primary = str(results["config"]["primary_d"])
        rec = v["p2"][d_primary]
        s = rec["summary"]
        add(f"### Primary endpoint: d={d_primary}")
        add("")
        add("| fold | PCA nMSE (VERIFY) | PCA nMSE (DEV) | GPLVM nMSE (VERIFY) | held-out conv. |")
        add("|---:|---:|---:|---:|---:|")
        for f in rec["folds"]:
            add(
                f"| {f['fold']} | {f['pca_nmse_verify']:.4f} | "
                f"{f['pca_nmse_dev']:.4f} | {f['gplvm_nmse_verify']:.4f} | "
                f"{f['heldout_converged_frac']:.1%} |"
            )
        add("")
        add(
            f"- Out-of-fold means over all 276 roles: PCA {s['pca_nmse_verify_mean']:.4f}, "
            f"GPLVM {s['gplvm_nmse_verify_mean']:.4f}."
        )
        add(
            f"- Mean paired difference (PCA - GPLVM): "
            f"{s['mean_paired_diff_pca_minus_gplvm']:+.4f}, bootstrap 95% CI "
            f"{_fmt_ci(s['bootstrap_ci95'])}, one-sided sign-permutation "
            f"p = {s['sign_permutation_p_one_sided']:.4g} (two-sided "
            f"{s['sign_permutation_p_two_sided']:.4g})."
        )
        add(
            f"- Panel-shift reference gap (PCA, VERIFY minus DEV): "
            f"{s['panel_shift_gap_pca']:+.4f}; decision threshold "
            f"{results['config']['p2_gap_fraction']} x gap = "
            f"{v['p2_decision'][d_primary]['advantage_threshold']:+.4f}."
        )
        add(
            f"- Convergence: held-out latent fits {s['heldout_converged_frac']:.1%} "
            f"converged; selected training fits "
            f"{s['training_selected_converged_frac']:.1%}; all training "
            f"restarts {s['training_all_restarts_converged_frac']:.1%}."
        )
        spread = max(
            t["objective_spread"] for t in rec["training_fits"]
        )
        add(
            f"- Restart audit: max across folds of (max-min) training "
            f"objective across the {rec['n_restarts']} restarts = {spread:.4g}."
        )
        if rec["fd_check_heldout_gradient"] is not None:
            fd = rec["fd_check_heldout_gradient"]
            add(
                f"- Held-out analytic gradient finite-difference check: max "
                f"relative error {fd['max_rel_err']:.2e} over "
                f"{fd['n_points']} test points."
            )
        dec = v["p2_decision"][d_primary]
        if dec["panel_shift_gap_degenerate"]:
            add(
                "- Note: the panel-shift gap is <= 0 (in the train-fitted "
                "denoiser subspace, VERIFY-panel centroids reconstruct no "
                "worse than DEV-panel centroids), so the pre-specified 25% "
                "threshold is non-binding and the decision reduces to the CI "
                "criterion."
            )
        if not dec["heldout_convergence_gate_met"]:
            add("")
            add(
                "**WARNING: fewer than "
                f"{results['config']['heldout_convergence_gate']:.0%} of "
                "held-out GPLVM fits converged at d="
                f"{d_primary}. This gates interpretation of the P2 endpoint.**"
            )
        if primary:
            verdict = (
                "SURVIVES cross-fitting"
                if dec["survives_crossfit"]
                else "does NOT survive cross-fitting"
            )
            add("")
            add(
                f"**P2 verdict ({view}): nonlinearity {verdict}** — CI "
                f"excludes 0: {dec['ci_excludes_zero']}; advantage >= "
                f"threshold: {dec['advantage_ge_threshold']}; convergence "
                f"gate met: {dec['heldout_convergence_gate_met']}."
            )
        else:
            add(
                f"- Same rule applied (exploratory): survives = "
                f"{dec['survives_crossfit']}."
            )
        add("")
        add("### Exploratory dimensions (s1)")
        add("")
        add("| d | PCA nMSE (VER) | GPLVM nMSE (VER) | mean diff (PCA-GPLVM) | boot 95% CI | p (1-sided) | held-out conv. | panel gap |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in results["config"]["all_dims"]:
            r = v["p2"][str(d)]["summary"]
            marker = " (primary)" if d == results["config"]["primary_d"] else ""
            add(
                f"| {d}{marker} | {r['pca_nmse_verify_mean']:.4f} | "
                f"{r['gplvm_nmse_verify_mean']:.4f} | "
                f"{r['mean_paired_diff_pca_minus_gplvm']:+.4f} | "
                f"{_fmt_ci(r['bootstrap_ci95'])} | "
                f"{r['sign_permutation_p_one_sided']:.4g} | "
                f"{r['heldout_converged_frac']:.1%} | "
                f"{r['panel_shift_gap_pca']:+.4f} |"
            )
        add("")
        s3 = v["s3"]
        add(f"### s3 — response-length probe, `{view}` (exploratory)")
        add("")
        add(
            f"- Correlation of per-question mean response length (chars) with "
            f"question centroid-shift magnitude over 40 questions: Pearson "
            f"{s3['pearson_len_vs_shift']:.3f}, Spearman "
            f"{s3['spearman_len_vs_shift']:.3f}."
        )
        add("")

    add("## Limitations")
    add("")
    add(
        "- Single model (Qwen2.5-3B-Instruct), evaluated here at layer "
        f"{cfg['layer']}; nothing here speaks to other models."
    )
    add(
        "- Greedy (do_sample=false) generations: one response per cell, so "
        "sampling variability of the response distribution is not measured."
    )
    add(
        "- Questions are shared across roles within a panel, so panel-level "
        "question effects are common to all roles; residualization removes "
        "only the additive per-question component."
    )
    add(
        "- Response-token averaging remains the primary representation; the "
        "prompt_last (response-last token) repeat is a control, not a "
        "resolution, of the token-composition concern (review finding #10)."
    )
    add(
        "- The GPLVM held-out objective is nearest-point projection onto the "
        "learned surface (as in E1b), not probabilistic GPLVM inference "
        "(review finding #11); the PCA comparison uses the same projection "
        "logic, so the contrast is like-for-like."
    )
    add(
        "- The panel split and folds were drawn once from SeedSequence(0); "
        "sensitivity to other panel draws is not explored here."
    )
    add("")
    add("## Environment")
    add("")
    env = results["environment"]
    add(
        f"- Python {env['python']}; NumPy {env['numpy']}; SciPy "
        f"{env['scipy']}; scikit-learn {env['sklearn']}; pandas "
        f"{env['pandas']}."
    )
    add(f"- Elapsed: {results['elapsed_seconds'] / 60.0:.1f} min.")
    add("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast validation run")
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER,
        help=f"activation layer to analyse (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent,
        help="directory for results and report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer = int(args.layer)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    results_path = out_dir / f"results{suffix}.json"
    report_path = out_dir / f"REPORT{suffix}.md"

    if args.smoke:
        n_null, n_p1_boot, n_p2_boot, n_perm = 20, 50, 50, 500
        n_restarts = 1
        dims = (PRIMARY_D,)
        views = ("prompt_avg",)
        config = FitConfig(alternations=2, hyper_maxiter=50, latent_maxiter=80)
    else:
        n_null, n_p1_boot, n_p2_boot, n_perm = (
            N_NULL_DRAWS,
            N_P1_BOOT,
            N_P2_BOOT,
            N_SIGN_PERM,
        )
        n_restarts = N_RESTARTS
        dims = ALL_DIMS
        views = VIEWS
        config = FitConfig()

    started = time.time()
    results: dict[str, Any] = {
        "schema_version": 1,
        "experiment": (
            "E8 disjoint-question cross-fitting of role geometry "
            "(review control #1)"
        ),
        "status": "running",
        "config": {
            "root_seed": ROOT_SEED,
            "layer": layer,
            "protocol_origin_layer": DEFAULT_LAYER,
            "layer_extension": layer != DEFAULT_LAYER,
            "views": list(views),
            "n_roles": N_ROLES,
            "n_variants": N_VARIANTS,
            "n_questions": N_QUESTIONS,
            "panel_size": PANEL_SIZE,
            "n_folds": N_FOLDS,
            "r_denoise": R_DENOISE,
            "primary_d": PRIMARY_D,
            "secondary_dims": list(SECONDARY_DIMS),
            "all_dims": list(dims),
            "n_null_draws": n_null,
            "n_p1_bootstrap": n_p1_boot,
            "n_p2_bootstrap": n_p2_boot,
            "n_sign_perm": n_perm,
            "n_restarts": n_restarts,
            "p1_ci_lower_threshold": P1_CI_LOWER_THRESHOLD,
            "p2_gap_fraction": P2_GAP_FRACTION,
            "heldout_convergence_gate": HELDOUT_CONVERGENCE_GATE,
            "fit": asdict(config),
            "smoke": args.smoke,
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "seeds": seed_registry(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "mp_role_dir": os.environ["MP_ROLE_DIR"],
        },
        "design": {},
        "views": {},
        "elapsed_seconds": 0.0,
    }

    # ---- fixed design draws (before touching any activations) -------------
    meta_probe = pd.read_parquet(
        Path(os.environ["MP_ROLE_DIR"]) / "metadata.parquet"
    )
    roles, questions = assert_balanced_grid(meta_probe)

    q_rng = rng_for("question_panel")
    q_perm = q_rng.permutation(N_QUESTIONS)
    dev_questions = sorted(int(questions[i]) for i in q_perm[:PANEL_SIZE])
    verify_questions = sorted(int(questions[i]) for i in q_perm[PANEL_SIZE:])

    f_rng = rng_for("role_folds")
    role_perm = f_rng.permutation(N_ROLES)
    folds = [
        np.sort(role_perm[i * (N_ROLES // N_FOLDS): (i + 1) * (N_ROLES // N_FOLDS)])
        for i in range(N_FOLDS)
    ]
    assert sorted(np.concatenate(folds).tolist()) == list(range(N_ROLES))

    results["design"] = {
        "grid_assertion": (
            f"{N_ROLES} roles x {N_VARIANTS} prompt variants x "
            f"{N_QUESTIONS} questions, exactly one row per cell (55200 rows)"
        ),
        "dev_questions": dev_questions,
        "verify_questions": verify_questions,
        "n_folds": N_FOLDS,
        "fold_sizes": [int(len(f)) for f in folds],
        "roles": roles,
        "folds_role_indices": [f.tolist() for f in folds],
        "folds_role_names": [[roles[i] for i in f] for f in folds],
    }
    del meta_probe
    atomic_write_json(results_path, results)
    print(f"design fixed: DEV={dev_questions}", flush=True)
    print(f"              VERIFY={verify_questions}", flush=True)

    for view in views:
        view_index = VIEW_IDX[view]
        print(f"[{view}] loading layer {layer} ...", flush=True)
        X, meta, manifest = load_points(view=view, layer=layer, aggregate="none")
        results["config"]["dataset_primary_layer"] = int(
            manifest.get("primary_layer", -1)
        )
        if X.shape != (N_ROLES * N_VARIANTS * N_QUESTIONS, 2048):
            raise ValueError(f"unexpected activation shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("non-finite activations")
        roles_v, questions_v = assert_balanced_grid(meta)
        if roles_v != roles or questions_v != questions:
            raise ValueError("metadata differs across views")
        role_codes = pd.Categorical(
            meta["role"], categories=roles
        ).codes.astype(np.int64)
        q_values = meta["question_idx"].to_numpy()
        dev_mask = np.isin(q_values, dev_questions)
        ver_mask = np.isin(q_values, verify_questions)

        dev_cent = panel_centroids(X, role_codes, dev_mask)
        ver_cent = panel_centroids(X, role_codes, ver_mask)
        check = residualization_equivalence_check(
            X, meta, role_codes, dev_questions, np.concatenate(folds[1:]), dev_cent
        )
        if view_index == 0:
            results["design"]["residualization_check"] = {
                **check,
                "checked_on": "prompt_avg, DEV panel, fold-0 train roles "
                "(folds 1-3)",
            }

        view_record: dict[str, Any] = {}
        print(f"[{view}] P1 ...", flush=True)
        t0 = time.time()
        view_record["p1"] = run_p1(
            dev_cent, ver_cent, view_index, n_null, n_p1_boot
        )
        print(
            f"[{view}] P1 done in {time.time() - t0:.0f}s: spearman="
            f"{view_record['p1']['distance_corr']['spearman']:.4f}",
            flush=True,
        )

        view_record["s3"] = run_s3(X, meta, questions)
        del X
        results["views"][view] = view_record
        atomic_write_json(results_path, results)

        view_record["p2"] = {}
        view_record["p2_decision"] = {}
        ordered_dims = (PRIMARY_D,) + tuple(d for d in dims if d != PRIMARY_D)
        for d in ordered_dims:
            print(f"[{view}] P2 d={d} ...", flush=True)
            t0 = time.time()
            rec = run_p2_dim(
                dev_cent,
                ver_cent,
                folds,
                d,
                view_index,
                config,
                n_restarts,
                n_p2_boot,
                n_perm,
                run_fd_check=(d == PRIMARY_D),
            )
            view_record["p2"][str(d)] = rec
            view_record["p2_decision"][str(d)] = p2_decision(rec)
            results["elapsed_seconds"] = time.time() - started
            atomic_write_json(results_path, results)
            print(
                f"[{view}] P2 d={d} done in {time.time() - t0:.0f}s: "
                f"diff={rec['summary']['mean_paired_diff_pca_minus_gplvm']:+.4f} "
                f"conv={rec['summary']['heldout_converged_frac']:.1%}",
                flush=True,
            )

    results["status"] = "complete"
    results["elapsed_seconds"] = time.time() - started
    atomic_write_json(results_path, results)

    # Report is built strictly from the serialized results file.
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    report_path.write_text(build_report(stored), encoding="utf-8")
    print(
        f"complete in {results['elapsed_seconds'] / 60.0:.1f} min; wrote "
        f"{results_path} and {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
