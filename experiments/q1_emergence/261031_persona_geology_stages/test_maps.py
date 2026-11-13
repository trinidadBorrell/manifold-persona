"""CPU-only synthetic checks. No released or pilot adapter data is read."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("geology_maps", Path(__file__).with_name("maps.py"))
maps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(maps)


def adapters_from_rows(rows):
    return {f"t{i}": {"linear": {"A": np.eye(len(row)), "B": row[None, :], "scale": 1.0}}
            for i, row in enumerate(rows)}


def labels_small():
    return [{"trait": f"t{i}", "slug": f"t{i}", "factor": f"f{i // 4}",
             "keyed": "+" if i % 4 < 2 else "-"} for i in range(8)]


def planted_correlation():
    loading = np.zeros((20, 2))
    loading[:10, 0] = np.linspace(.62, .82, 10)
    loading[10:, 1] = np.linspace(.65, .84, 10)
    phi = np.array([[1., .25], [.25, 1.]])
    common = loading @ phi @ loading.T
    return common + np.diag(1 - np.diag(common)), loading, phi


def test_low_rank_gram_matches_dense_with_scaling_and_different_ranks():
    rng = np.random.default_rng(102)
    adapters = {}
    dense = []
    for i, rank in enumerate([1, 2, 3]):
        modules = {}
        row = []
        for module, outdim, indim in [("a", 4, 5), ("b", 2, 7)]:
            a, b = rng.normal(size=(rank, indim)), rng.normal(size=(outdim, rank))
            scale = (i + 1) * -.3
            modules[module] = {"A": a, "B": torch.tensor(b), "scale": scale}
            row.append((scale * b @ a).ravel())
        adapters[f"t{i}"] = modules
        dense.append(np.concatenate(row))
    dense = np.stack(dense)
    result = maps.low_rank_gram(adapters, list(adapters))
    expected = dense @ dense.T
    np.testing.assert_allclose(result["gram"], expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result["cosine"], expected / np.sqrt(np.outer(np.diag(expected), np.diag(expected))))


@pytest.mark.parametrize("failure", ["zero", "nan", "key", "shape", "complex"])
def test_low_rank_refuses_invalid_updates(failure):
    adapters = adapters_from_rows(np.eye(3))
    if failure == "zero":
        adapters["t0"]["linear"]["B"] *= 0
    elif failure == "nan":
        adapters["t0"]["linear"]["B"][0, 0] = np.nan
    elif failure == "key":
        adapters["t0"]["other"] = adapters["t0"].pop("linear")
    elif failure == "shape":
        adapters["t0"]["linear"]["A"] = np.eye(2)
    else:
        adapters["t0"]["linear"]["B"] = np.ones((1, 3), dtype=complex) * 1j
    with pytest.raises(ValueError):
        maps.low_rank_gram(adapters, list(adapters))


def test_loader_reads_named_tensor_modules_and_preserves_metadata(tmp_path):
    path = tmp_path / "adapter.pt"
    torch.save({"modules": {"q": {"A": torch.ones(1, 2), "B": torch.ones(3, 1), "scale": 2.}},
                "metadata": {"seed": 0}}, path)
    result = maps.load_adapter(path)
    assert result["metadata"] == {"seed": 0}
    np.testing.assert_array_equal(result["modules"]["q"]["A"], [[1, 1]])
    torch.save({"not_modules": {}}, path)
    with pytest.raises(ValueError):
        maps.load_adapter(path)


def test_parallel_analysis_frozen_draws_identity_and_planted_counts():
    correlation, _, _ = planted_correlation()
    null = maps.parallel_null(20)
    assert null["eigenvalues"].shape == (2000, 20)
    np.testing.assert_array_equal(null["eigenvalues"], maps.parallel_null(20)["eigenvalues"])
    assert maps.parallel_analysis(np.eye(20), null)["n_factors"] == 0
    assert maps.parallel_analysis(correlation, null)["n_factors"] == 2
    assert set(maps.parallel_sensitivities(correlation)) == {75, 150, 300}


def test_factor_count_stops_at_first_failed_ordered_eigenvalue():
    # Leading 1.5 clears; second 1.1 misses; the third .9 clearing cannot restart.
    correlation = np.array([[1., .5, 0, 0], [.5, 1., 0, 0], [0, 0, 1., .1], [0, 0, .1, 1.]])
    result = maps.parallel_analysis(correlation, {"thresholds": np.array([1.4, 1.2, .8, .4])})
    assert result["n_factors"] == 1


def test_principal_axis_and_oblimin_reconstruct_known_common_covariance():
    correlation, true_loading, true_phi = planted_correlation()
    fit = maps.principal_axis_factoring(correlation, 2)
    assert fit["converged"] and np.all(fit["uniquenesses"] > 0)
    np.testing.assert_allclose(fit["loadings"] @ fit["phi"] @ fit["loadings"].T,
                               true_loading @ true_phi @ true_loading.T, atol=2e-6)
    # PCA keeps diagonal uniqueness in its eigenvalues, and fails this check.
    assert fit["rotation_converged"]
    np.testing.assert_allclose(fit["unrotated"] @ fit["unrotated"].T,
                               fit["loadings"] @ fit["phi"] @ fit["loadings"].T, atol=1e-9)
    assert np.max(np.abs(fit["phi"] - np.eye(2))) > .1
    assert maps.match_factors(true_loading, fit["loadings"])["pass"]


def test_factor_fit_refuses_nonconvergence_and_improper_input():
    correlation, _, _ = planted_correlation()
    with pytest.raises(maps.FactorFitError):
        maps.principal_axis_factoring(correlation, 2, max_iter=1)
    with pytest.raises(ValueError):
        maps.principal_axis_factoring(np.array([[1., 2.], [2., 1.]]), 1)
    # Positive-definite correlation with an inadmissible one-factor solution.
    with pytest.raises(maps.FactorFitError):
        maps.principal_axis_factoring(np.array([[1., .7, .7], [.7, 1., .1], [.7, .1, 1.]]), 1)
    fit = maps.principal_axis_factoring(np.eye(3), 0)
    assert fit["loadings"].shape == (3, 0)


def test_oblimin_is_oblique_and_preserves_common_covariance():
    pattern = np.array([[.8, 0], [.7, 0], [.6, 0], [0, .8], [0, .7], [0, .6]])
    phi = np.array([[1., .4], [.4, 1.]])
    unrotated = pattern @ np.linalg.cholesky(phi)
    result = maps.oblimin(unrotated)
    np.testing.assert_allclose(result["loadings"] @ result["phi"] @ result["loadings"].T,
                               unrotated @ unrotated.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(result["phi"]), 1., atol=1e-10)
    assert maps.match_factors(pattern, result["loadings"])["pass"]


def test_tucker_matching_handles_signs_orders_and_count_mismatch():
    _, loading, _ = planted_correlation()
    candidate = loading[:, [1, 0]] * [-1, 1]
    result = maps.match_factors(loading, candidate)
    np.testing.assert_allclose(result["congruences"], [1., 1.])
    np.testing.assert_allclose(result["aligned"], loading)
    assert result["pass"]
    assert not maps.match_factors(loading, loading[:, :1])["pass"]


def test_label_omnibus_uses_signed_keys_and_reproducible_permutations():
    labels = labels_small()
    loadings = np.zeros((8, 2))
    for i, item in enumerate(labels):
        loadings[i, i // 4] = 1 if item["keyed"] == "+" else -1
    result = maps.label_permutation_test(loadings, labels)
    assert result["permutations"] == 9999 and result["pvalue"] <= .05
    reverse = maps.label_permutation_test(loadings[:, ::-1] * -1, labels)
    assert result["score"] == reverse["score"] and result["pvalue"] == reverse["pvalue"]
    with pytest.raises(ValueError):
        maps.label_permutation_test(loadings, [{**labels[0], "keyed": "unknown"}, *labels[1:]])


def test_balanced_holdout_contains_one_trait_per_pole_and_factor():
    labels = labels_small()
    folds = maps.balanced_trait_folds(labels)
    for fold in folds:
        assert len(fold) == 4
        assert len({(labels[i]["factor"], labels[i]["keyed"]) for i in fold}) == 4


def test_heldout_ridge_excludes_test_labels_and_self_edges_from_tuning():
    rng = np.random.default_rng(109)
    contrasts = rng.normal(size=(8, 3))
    rows = rng.normal(size=(8, 5))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    gram = rows @ rows.T
    labels = labels_small()
    result = maps.text_holdout_prediction(gram, contrasts, labels)
    train = result["folds"][0]["train"]
    test = result["folds"][0]["test"]
    changed = gram.copy()
    changed[np.ix_(test, train)] += .777
    changed[np.ix_(train, test)] += .777
    np.fill_diagonal(changed, 23.)  # All self edges must be ignored even during tuning.
    second = maps.text_holdout_prediction(changed, contrasts, labels)
    assert result["folds"][0]["alpha"] == second["folds"][0]["alpha"]
    np.testing.assert_allclose(result["predictions"][np.ix_(test, train)],
                               second["predictions"][np.ix_(test, train)])
    assert np.isfinite(result["heldout_r2"]) and np.isfinite(result["map_correlation"])
    assert not np.any(np.diag(result["prediction_mask"]))
    assert np.all(np.isfinite(result["residuals"]))
    assert len(result["residual_singular_values"]) == 4


def test_mean_contrasts_are_chosen_minus_rejected_and_no_adapter_is_zero():
    chosen = np.array([[[2., 1.], [4., 3.]], [[0., 2.], [2., 4.]]])
    rejected = np.ones_like(chosen)
    np.testing.assert_array_equal(maps.mean_text_contrasts(chosen, rejected), [[2., 1.], [0., 2.]])
    assert maps.check_no_adapter(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        maps.check_no_adapter(np.ones((2, 3)))


def test_activation_similarity_and_map_correlation_use_upper_triangle():
    shifts = np.array([[[1., 0]], [[0, 1]], [[1., 1]]])
    cosine = maps.activation_similarity(shifts)
    np.testing.assert_allclose(cosine, [[1, 0, 2**-.5], [0, 1, 2**-.5], [2**-.5, 2**-.5, 1]])
    assert maps.compare_maps(cosine, cosine) == pytest.approx(1.)
    with pytest.raises(ValueError):
        maps.compare_maps(np.eye(3), np.eye(3))


def test_permutation_check_restores_data_labels_and_checks_tolerance():
    gram, _, _ = planted_correlation()
    order = np.roll(np.arange(20), 1)
    wrong_named = gram[np.ix_(order, order)]
    assert maps.recover_permuted_gram(wrong_named, order, gram)["pass"]
    broken = wrong_named.copy()
    broken[0, 1] += .01
    broken[1, 0] += .01
    assert not maps.recover_permuted_gram(broken, order, gram)["pass"]


def test_cli_refuses_unknown_manifest_fields_and_retains_error(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "panel": "never_read.json", "arms": {}, "mystery": True}))
    output = tmp_path / "analysis"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("maps.py")),
                          "--manifest", str(manifest), "--output", str(output)], capture_output=True, text=True)
    assert run.returncode != 0
    assert "unknown" in (output / "error.json").read_text().lower()


def pilot_fixture():
    correlation, _, _ = planted_correlation()
    rows = np.linalg.cholesky(correlation)
    order = np.roll(np.arange(20), 7)
    labels = [{"trait": f"t{i:02}", "slug": f"t{i}", "factor": f"f{i // 4}",
               "keyed": "+" if i % 4 < 2 else "-"} for i in range(20)]
    arms = {"real": adapters_from_rows(rows), "seed1": adapters_from_rows(rows),
            "scrambled": adapters_from_rows(np.eye(20)), "permuted": adapters_from_rows(rows[order])}
    return arms, labels, order


def test_complete_synthetic_pilot_saves_all_conditions_and_incomplete_cannot_pass():
    arms, labels, order = pilot_fixture()
    contrasts = np.random.default_rng(113).normal(size=(20, 6))
    result = maps.analyze_pilot(arms, labels, order, contrasts)
    assert result["verdict"] == "pass"
    assert all(result["conditions"].values())
    incomplete = maps.analyze_pilot({"real": arms["real"]}, labels)
    assert incomplete["verdict"] == "inconclusive"


def test_permutation_rule_detects_raw_update_scale_change_even_when_cosine_matches():
    arms, labels, order = pilot_fixture()
    for module in arms["permuted"].values():
        module["linear"]["scale"] *= 2
    result = maps.analyze_pilot(arms, labels, order, np.random.default_rng(114).normal(size=(20, 5)))
    assert not result["conditions"]["permutation_control"]
    assert result["verdict"] == "fail"


def test_cli_writes_strict_summary_raw_arrays_and_input_hashes(tmp_path):
    arms, labels, order = pilot_fixture()
    (tmp_path / "panel.json").write_text(json.dumps({"schema_version": 1, "traits": labels}))
    paths = {}
    for arm, traits in arms.items():
        paths[arm] = {}
        for trait, modules in traits.items():
            path = tmp_path / f"{arm}-{trait}.pt"
            tensors = {name: {"A": torch.tensor(m["A"]), "B": torch.tensor(m["B"]), "scale": m["scale"]}
                       for name, m in modules.items()}
            torch.save({"modules": tensors, "metadata": {"test_only": True}}, path)
            paths[arm][trait] = path.name
    np.save(tmp_path / "text.npy", np.random.default_rng(115).normal(size=(20, 6)))
    shift_rows = np.random.default_rng(116).normal(size=(20, 3, 4))
    shift_rows = np.concatenate([np.zeros((1, 3, 4)), shift_rows])
    for arm in ("real", "seed1"):
        np.savez(tmp_path / f"{arm}-shifts.npz", shifts=shift_rows,
                 traits=np.array(["no_adapter", *[label["slug"] for label in labels]]),
                 layer_label=16, hidden_state_index=16, input_sha256="fixed-input-hash")
    manifest = {"schema_version": 1, "panel": "panel.json", "arms": paths,
                "permuted_data_order": order.tolist(), "text_contrasts": "text.npy",
                "activation_shifts": {"real": "real-shifts.npz", "seed1": "seed1-shifts.npz"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = maps.run_manifest(tmp_path / "manifest.json", tmp_path / "analysis")
    summary = json.loads((tmp_path / "analysis" / "summary.json").read_text(),
                         parse_constant=lambda value: pytest.fail(f"nonstandard JSON constant {value}"))
    assert result["verdict"] == summary["verdict"] == "pass"
    assert len(summary["input_sha256"]) == 85
    assert summary["activation_input_metadata"]["real"]["trait_order_verified"]
    assert summary["no_adapter_zero"]
    with np.load(tmp_path / "analysis" / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["analysis/maps/real/gram"].shape == (20, 20)
        assert arrays["analysis/null_draws/150/eigenvalues"].shape == (2000, 20)
        np.testing.assert_array_equal(arrays["analysis/text_contrasts"], np.load(tmp_path / "text.npy"))
    with pytest.raises(ValueError):
        maps.run_manifest(tmp_path / "manifest.json", tmp_path / "analysis")
    np.savez(tmp_path / "seed1-shifts.npz", shifts=shift_rows,
             traits=np.array(["no_adapter", *[label["slug"] for label in labels]]),
             layer_label=16, hidden_state_index=16, input_sha256="different-input-hash")
    with pytest.raises(ValueError, match="different layers or fixed inputs"):
        maps.run_manifest(tmp_path / "manifest.json", tmp_path / "wrong-bank-analysis")


def test_actual_cpu_training_output_loads_with_tensor_only_loader(tmp_path):
    import importlib
    fixture = importlib.import_module("experiments.q1_emergence.261031_persona_geology_stages.test_train")
    train = fixture.apparatus()
    train.train_adapter(fixture.tiny_model(), fixture.input_file(tmp_path / "rows.json"),
                        tmp_path / "run", tmp_path / "init.pt", fixture.config())
    adapter = maps.load_adapter(tmp_path / "run" / "adapter.pt")
    assert adapter["metadata"]["completed_steps"] == 2
    gram = maps.low_rank_gram({"a": adapter["modules"], "b": adapter["modules"]}, ["a", "b"])
    np.testing.assert_allclose(gram["cosine"], np.ones((2, 2)))


def test_activation_archive_accepts_production_shape_and_checks_trait_order_and_zero(tmp_path):
    path = tmp_path / "shifts_L16.npz"
    shifts = np.array([[[0., 0]], [[1., 0]], [[0., 1.]]])
    def save(traits, values=shifts):
        np.savez(path, shifts=values, traits=np.array(traits), layer_label=16,
                 hidden_state_index=16, input_sha256="fixed-bank-hash")
    save(["no_adapter", "t0", "t1"])
    result = maps.load_activation_shifts(path, ["t0", "t1"])
    np.testing.assert_array_equal(result["shifts"], shifts[1:])
    assert result["metadata"]["layer_label"] == 16
    assert maps.check_no_adapter(result["no_adapter"])
    save(["no_adapter", "t1", "t0"])
    with pytest.raises(ValueError):
        maps.load_activation_shifts(path, ["t0", "t1"])
    bad = shifts.copy()
    bad[0, 0, 0] = .01
    save(["no_adapter", "t0", "t1"], bad)
    with pytest.raises(ValueError):
        maps.load_activation_shifts(path, ["t0", "t1"])


def test_adapter_identity_rejects_cross_checkpoint_raw_weight_map():
    with pytest.raises(ValueError, match="checkpoint"):
        maps.validate_adapter_identities({"real": {"a": {"stage": "base", "model": {"revision": "one"}},
                                                   "b": {"stage": "instruct", "model": {"revision": "two"}}}})
    assert maps.validate_adapter_identities({"real": {"a": {"stage": "instruct", "model": {"revision": "one"}},
                                                     "b": {"stage": "instruct", "model": {"revision": "one"}}}})["checkpoint_verified"]
