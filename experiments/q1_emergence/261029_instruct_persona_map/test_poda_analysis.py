"""Synthetic checks for the frozen A analysis; no real arrays or models are loaded."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("poda_analysis", HERE / "analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def panel(c=4, q=8, w=5, reps=2, d=7, seed=3):
    rng = np.random.default_rng(seed)
    role_names = [f"r{i}" for i in range(c)] + ["default"]
    vec = rng.normal(size=(c + 1, d))
    rows, X = [], []
    for role, v in zip(role_names, vec):
        for question in range(q):
            for wording in range(w):
                for rep in range(reps):
                    rows.append(dict(role=role, format="chat", question=question, wording=wording,
                                     replicate=rep, valid=True, reply=f"answer {role} question{question}"))
                    X.append(v + 0.05 * rng.normal(size=d))
    return np.array(X), rows, role_names[:-1], list(range(q))


def test_prompt_cells_average_replicates_not_row_counts():
    rows = [dict(role="r", format="chat", question=0, wording=0, replicate=i, valid=True) for i in range(3)]
    rows += [dict(role="r", format="chat", question=0, wording=1, replicate=0, valid=True)]
    S, N = A.cell_means(np.array([[1.], [3.], [5.], [9.]]), rows, ["r"], [0])
    assert S[0, 0, 0, 0] == 3 and S[0, 0, 1, 0] == 9
    assert N[0, 0, :2].tolist() == [1, 1]
    assert S[0, 0, :2, 0].mean() == 6


def test_halves_have_disjoint_questions_and_wordings():
    h = A.make_halves(list(range(8)), n=10)
    assert len(h) == 10
    for (qa, wa), (qb, wb) in h:
        assert not (qa & qb).any() and not (wa & wb).any()
        assert (qa | qb).all() and (wa | wb).all()


def test_axis_fixed_offset_passes_but_does_not_prove_role_structure():
    X, rows, roles, qs = panel(q=40)
    X[:] = 0
    X[:, 0] = 1
    X[np.array([r["role"] == "default" for r in rows]), 1] = 2
    S, N = A.cell_means(X, rows, roles + ["default"], qs)
    rr, W = A.M.make_draws(len(roles), len(qs), n=30)
    result, details = A.axis_statistics(S, N, A.make_halves(qs, n=4), rr, W)
    assert result["cosine"]["point"] == pytest.approx(1)
    assert result["passed"]
    assert result["scope"] == "repeatable default-vs-role contrast"
    assert details["units"].shape == (4, 2, X.shape[1])


def test_axis_projection_uses_training_origin_only():
    X, rows, roles, qs = panel()
    S, N = A.cell_means(X, rows, roles + ["default"], qs)
    half = A.make_halves(qs, n=1)
    rr, W = A.M.make_draws(len(roles), len(qs), n=0)
    _, d = A.axis_statistics(S, N, half, rr, W)
    qa, wa = half[0][0]
    vb = S[:, ~qa][:, :, ~wa].mean((1, 2))
    va = S[:, qa][:, :, wa].mean((1, 2))
    origin = va[:-1].mean(0)
    unit = (va[-1] - origin) / np.linalg.norm(va[-1] - origin)
    np.testing.assert_allclose(d["origins"][0, 0], origin)
    np.testing.assert_allclose(d["test_scores"][0, 0], (vb - origin) @ unit)


def test_zero_axis_is_unavailable_not_a_pass():
    X, rows, roles, qs = panel()
    X[:] = 1
    S, N = A.cell_means(X, rows, roles + ["default"], qs)
    rr, W = A.M.make_draws(len(roles), len(qs), n=10)
    result, _ = A.axis_statistics(S, N, A.make_halves(qs, n=3), rr, W)
    assert result["cosine"]["point"] is None and not result["passed"]


def test_planted_map_and_one_half_null():
    X, rows, roles, qs = panel(c=8, d=12)
    result = A.map_statistics(X, rows, roles, qs, n_boot=50, n_perm=99, n_halves=6)
    assert result["raw_r"]["point"] > .9
    assert result["p"] == pytest.approx(.01)
    assert result["null_method"] == "one half role labels permuted"


def test_undefined_interval_cannot_pass_precision():
    assert not A.precision_ok(dict(point=.9, lo=None, hi=None), .1)
    assert not A.precision_ok(dict(point=.9, lo=.4, hi=.95), .1)


def test_a_intervals_keep_frozen_percentiles_after_c_rule_change(monkeypatch):
    monkeypatch.setattr(A.M, "summary", lambda _: pytest.fail("A must not inherit C's interval amendment"))
    draws = np.array([.9, .1, .2, .3, .4])
    result = A.summary(draws)
    expected = np.percentile(draws[1:], [2.5, 97.5])
    assert result["point"] == .9
    np.testing.assert_allclose([result["lo"], result["hi"]], expected)
    assert "draw_shift" not in result


def test_reduced_counts_cannot_claim_confirmatory_pass():
    assert not A.confirmatory_settings(n_boot=20, n_perm=99, n_halves=3)
    assert A.confirmatory_settings(n_boot=2000, n_perm=9999, n_halves=50)


def test_checkpoint_rejects_changed_input_fingerprint(tmp_path):
    A.phase(tmp_path, "test", "first", lambda: {"value": 1})
    with pytest.raises(A.AnalysisError, match="fingerprint"):
        A.phase(tmp_path, "test", "second", lambda: {"value": 2})


def test_checkpoint_rejects_changed_result_with_same_fingerprint(tmp_path):
    A.phase(tmp_path, "test", "same", lambda: {"value": 1})
    path = tmp_path / "test.json"
    saved = json.loads(path.read_text())
    saved["result"]["value"] = 7
    A.write_json(path, saved)
    with pytest.raises(A.AnalysisError, match="result checksum"):
        A.phase(tmp_path, "test", "same", lambda: {"value": 1})


def test_json_output_is_standard_and_preserves_unavailable(tmp_path):
    A.write_json(tmp_path / "result.json", {"value": float("nan"), "array": np.array([1., np.inf])})
    text = (tmp_path / "result.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"value": None, "array": [1., None]}


def test_classifier_folds_hold_out_questions_and_wordings():
    _, rows, roles, qs = panel()
    meta = A.prompt_index(rows, roles, qs)
    for tr, te in A.decode_splits(meta, n=8):
        assert set(meta["q"][tr]).isdisjoint(meta["q"][te])
        assert set(meta["w"][tr]).isdisjoint(meta["w"][te])


def test_synthetic_gate_runs_without_model_or_data():
    r = A.synthetic_gate()
    assert r["passed"] and r["axis_offset_recorded"] == pytest.approx(1)


def fixture_run(tmp_path):
    X, rows, roles, qs = panel()
    out = tmp_path / "run_full"
    out.mkdir()
    A.write_json(out / "manifest.json", dict(model=A.MODEL, revision=A.REVISION, roles=roles + ["default"],
                                             replicates=2, n_hidden_states=33, limit=None, run_id="run_full"))
    keys = [A.row_key(r) for r in rows]
    A.write_json(out / "grid.json", dict(keys=keys))
    for r in rows:
        r.update(prompt_ids=[0], kept_ids=[1], raw_ids=[1], span=[1, 2])
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(out / "reply_mean_acts.npy", np.repeat(X[:, None], 33, axis=1))
    gates = {f"G{i}": {"pass": True} for i in range(1, 6)}
    gates["G1"].update(n_rows=len(rows), invalid=0)
    gates["G3"].update(n_rows=40, identical_same_batches=40)
    gates["G4"].update(cells={f"L{i}": dict(n=len(rows), **{"pass": True}) for i in (9, 17, 23)})
    A.write_json(out / "gates.json", dict(passed=True, gates=gates))
    files = ["manifest.json", "grid.json", "rows.jsonl", "reply_mean_acts.npy"]
    (out / "SHA256SUMS.txt").write_text("".join(f"{A.sha256(out / f)}  {f}\n" for f in files))
    note = tmp_path / "read.md"
    note.write_text("Raw read: twenty transcripts read before metrics; failures retained.")
    return out, note


def test_loader_refuses_unchecked_or_changed_inputs(tmp_path):
    run, note = fixture_run(tmp_path)
    obj = A.load_run(run, note, engineering=True)
    assert obj.X.shape[1] == 33 and len(obj.roles) == 4
    (run / "rows.jsonl").write_text((run / "rows.jsonl").read_text() + " ")
    with pytest.raises(A.AnalysisError, match="checksum"):
        A.load_run(run, note, engineering=True)


def test_canary_cannot_become_confirmatory_input(tmp_path):
    run, note = fixture_run(tmp_path)
    with pytest.raises(A.AnalysisError, match="full|grid"):
        A.load_run(run, note)


def test_decode_has_chance_on_constant_features():
    _, rows, roles, qs = panel()
    meta = A.prompt_index(rows, roles, qs)
    X = np.zeros((len(meta["y"]), 3))
    result = A.decode_features(meta, lambda tr, te: (X[tr], X[te]), A.decode_splits(meta, n=8))
    assert result["mean_fold_accuracy"] == pytest.approx(1 / len(roles))


def test_no_family_verdict_is_invented():
    result = A.holm_decisions({"D1": .001, "D2": .004}, {"D1": True, "D2": True})
    assert result["status"] == "pending_owner_alpha" and result["decisions"] is None
    assert result["adjusted_p"] == {"D1": .002, "D2": .004}


def test_map_bootstrap_recentres_distinct_sampled_roles():
    from scipy.stats import spearmanr
    X, rows, roles, qs = panel(c=6, q=8)
    S, N = A.cell_means(X, rows, roles, qs)
    halves = A.make_halves(qs, n=3)
    rr = np.array([[0, 0, 1, 2, 2, 4]])
    W = np.array([[1, 2, 1, 0, 1, 1, 1, 1.]])
    actual = A.map_panel({"a": A.M.build_sides(S, N, halves)}, rr, W, [("a", "a")])
    unique, pos = np.unique(rr[0], return_inverse=True)
    i, j = np.triu_indices(rr.shape[1], 1)
    keep = rr[0, i] != rr[0, j]
    expected = []
    for half in halves:
        distances = []
        for q, w in half:
            weight = W[0] * q
            v = np.einsum("cqwd,q->cd", S[unique][:, :, w], weight)
            v /= np.einsum("cqw,q->c", N[unique][:, :, w], weight)[:, None]
            v -= v.mean(0)
            d = 1 - v @ v.T / np.outer(np.linalg.norm(v, axis=1), np.linalg.norm(v, axis=1))
            distances.append(d[pos[i[keep]], pos[j[keep]]])
        expected.append(spearmanr(*distances).statistic)
    assert actual[("a", "a")]["r"][0] == pytest.approx(np.mean(expected))


def test_noise_reports_cover_axis_wording_and_heldout_replicate_decoding():
    from types import SimpleNamespace
    X, rows, roles, qs = panel()
    run = SimpleNamespace(X=np.repeat(X[:, None], 33, axis=1), rows=rows, roles=roles,
                          questions=qs, manifest={"replicates": 2})
    result = A.noise_reports(run, n_halves=2)
    assert set(result["wording_held_out"]) == set(map(str, range(5)))
    assert set(result["replicate_held_out"]) == {"0", "1"}
    for item in result["replicate_held_out"].values():
        assert "axis_cosine" in item and "heldout_decode_accuracy" in item


def test_snapshot_check_refuses_changed_sampling_before_tokenizer_access():
    manifest = dict(sampling={"temperature": .9})
    with pytest.raises(A.AnalysisError, match="sampling"):
        A.verify_snapshot(manifest, [])


def test_generation_identity_is_separate_from_current_analysis_helper():
    approved = {"gen_extract.py": "fd030a35a7abf7900d5dfe3cdc25ba2a298d3a679a77a75b3099405c9bbcfa8b",
                "rows.py": "b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a"}
    A.verify_generation_identity({"code": approved})
    with pytest.raises(A.AnalysisError, match="generation"):
        A.verify_generation_identity({"code": dict(approved, **{"rows.py": "changed"})})


def test_a_null_permutes_one_half_without_averaging_inverse_permutation():
    from scipy.stats import spearmanr
    X, rows, roles, qs = panel(c=6, d=9)
    S, N = A.cell_means(X, rows, roles, qs)
    sides = A.M.build_sides(S, N, A.make_halves(qs, n=3))
    got = A.one_half_null(sides, len(qs), n_perm=7, seed=83)
    rng = np.random.default_rng(83)
    pairs = np.triu_indices(len(roles), 1)
    want = []
    for _ in range(7):
        p = rng.permutation(len(roles))
        values = []
        for left, right in sides:
            a = A.M.dist(left.grams(np.ones((1, len(qs))))[0])
            b = A.M.dist(right.grams(np.ones((1, len(qs))))[0])
            values.append(spearmanr(a[pairs], b[p[:, None], p][pairs]).statistic)
        want.append(np.mean(values))
    np.testing.assert_allclose(got, want)


def test_synthetic_cli_needs_no_model_or_hf(tmp_path, capsys):
    assert A.main(["--synthetic-only"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]


def test_small_engineering_cli_is_resumable_and_not_confirmatory(tmp_path):
    run, note = fixture_run(tmp_path)
    out = tmp_path / "analysis"
    args = ["--run", str(run), "--out", str(out), "--raw-read-note", str(note), "--execute-on-pod",
            "--engineering", "--n-boot", "20", "--n-perm", "19", "--n-halves", "3", "--skip-decoding"]
    assert A.main(args) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["confirmatory_settings"] is False and result["family"]["decisions"] is None
    before = (out / "primary.json").stat().st_mtime_ns
    assert A.main(args) == 0
    assert (out / "primary.json").stat().st_mtime_ns == before


def test_corrupt_phase_sidecar_is_not_reused(tmp_path):
    side = tmp_path / "side.npz"
    side.write_bytes(b"original")
    A.phase(tmp_path, "side", "same", lambda: {"artifact_hashes": {"side.npz": A.sha256(side)}})
    side.write_bytes(b"corrupt")
    with pytest.raises(A.AnalysisError, match="artifact"):
        A.phase(tmp_path, "side", "same", lambda: {})
