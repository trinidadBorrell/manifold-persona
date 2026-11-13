"""CPU tests on a synthetic release in Han Xiao's file layout (no real data).

  .venv/bin/python -m pytest experiments/q1_emergence/261112_han_reanalysis/test_han_reanalysis.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("han_re", HERE / "han_reanalysis.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
pa = H.pa
D, P, Q = 6, 5, 240
ROLES = [f"r{i}" for i in range(10)]


def make_release(root, rng, default_noise=1.0, stage_scale=None):
    """Roles at distinct means; r9 has fewer than 50 score-3 replies at SFT (no SFT vector), like translator."""
    stage_scale = stage_scale or {"SFT": 1.0, "DPO": 1.3, "RLVR": 1.3}
    means = {r: rng.normal(0, 3, D) for r in ROLES}
    means["default"] = rng.normal(0, 3, D)
    keys = [f"pos_p{p}_q{q}" for p in range(P) for q in range(Q)]
    for stage, folder in pa.STAGES.items():
        base = Path(root) / folder
        for sub in ("activations-layer16", f"scores-{pa.JUDGE_TAG}", f"vectors-{pa.JUDGE_TAG}"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        role_vecs = {}
        for role in ROLES + ["default"]:
            noise = default_noise if role == "default" else 1.0
            acts = stage_scale[stage] * means[role] + rng.normal(0, noise, (len(keys), D))
            torch.save({"keys": keys, "acts": torch.tensor(acts).to(torch.bfloat16)},
                       base / "activations-layer16" / f"{role}.pt")
            acts32 = torch.tensor(acts).to(torch.bfloat16).float().numpy()
            if role == "default":
                v = acts32.mean(0)
            else:
                p3 = 0.02 if (role == "r9" and stage == "SFT") else 0.8
                scores = {k: int(3 if rng.random() < p3 else rng.integers(0, 3)) for k in keys}
                (base / f"scores-{pa.JUDGE_TAG}" / f"{role}.json").write_text(json.dumps(scores))
                kept = np.array([scores[k] == 3 for k in keys])
                if kept.sum() < 50:
                    continue
                v = acts32[kept].mean(0)
            role_vecs[role] = v
            vec = np.stack([rng.normal(0, 1, D), v, rng.normal(0, 1, D)])
            torch.save({"vector": torch.tensor(vec), "type": "pos_3", "role": role},
                       base / f"vectors-{pa.JUDGE_TAG}" / f"{role}.pt")
        R = np.stack([role_vecs[r] for r in sorted(role_vecs) if r != "default"])
        axis = np.stack([np.zeros(D), role_vecs["default"] - R.mean(0), np.zeros(D)])
        torch.save(torch.tensor(axis).to(torch.bfloat16), base / f"axis-{pa.JUDGE_TAG}.pt")
    return root


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("rel"), np.random.default_rng(1))


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    lab = {r: dict(label="desirable" if i < 4 else "undesirable" if i < 7 else "neutral", reason="", borderline=(r == "r3"))
           for i, r in enumerate(ROLES)}
    p = tmp_path_factory.mktemp("lab") / "labels.json"
    p.write_text(json.dumps(lab))
    return p


def test_sums_match_han_function(release):
    keep = H.supported(H.passed_ids(release, "DPO"))
    mine = H.sums_by_question(release, "DPO", keep)
    his = pa.sum_activations_by_question(Path(release), "DPO", 16, keep)
    assert mine[0] == his[0]
    assert np.allclose(mine[1], his[1]) and np.allclose(mine[2], his[2])


def test_supported_excludes_low_score_role(release):
    assert "r9" not in H.supported(H.passed_ids(release, "SFT"))
    assert "r9" in H.supported(H.passed_ids(release, "DPO"))


def test_part_a_matches_rebuilt_projections(release, labels):
    a = H.part_a(release, labels, n_perm=99)
    assert a["checks"]["SFT"]["n_roles_with_vectors"] == 9 and a["checks"]["DPO"]["n_roles_with_vectors"] == 10
    assert a["checks"]["SFT"]["axis_file_cosine"] > 0.999
    assert not a["checks"]["SFT"]["roles_with_vector_not_supported"] and not a["checks"]["SFT"]["supported_without_vector"]
    run_s, run_d = H.stage_run(release, "SFT", 4), H.stage_run(release, "DPO", 4)
    common = sorted(set(run_s["roles"]) & set(run_d["roles"]))
    shift = {r: run_d["projection"][run_d["roles"].index(r)] - run_s["projection"][run_s["roles"].index(r)] for r in common}
    good = [r for r in common if r in ("r0", "r1", "r2", "r3")]
    assert abs(a["desirable_shift"] - np.mean([shift[r] for r in good])) < 1e-4
    assert a["n_desirable"] == 4 and a["n_undesirable"] == 3


def test_gate_rule():
    ok = lambda d, p: bool(abs(d - H.FIG_DESIRABLE) <= 0.01 and abs(p - H.FIG_P) <= 0.02)
    assert ok(0.183, 0.071) and ok(0.19, 0.05) and not ok(0.20, 0.066) and not ok(0.183, 0.09)


def test_units_scale_change_vanishes_in_role_sd_units(labels):
    x = np.array([2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -2.0, 0.2, -0.1, 0.3])
    z = np.zeros((3, 10))
    run_a = dict(roles=ROLES, projection=x, half_a=z, half_b=z)
    run_b = dict(roles=ROLES, projection=1.5 * x, half_a=z, half_b=z)
    lab = H.labels_table(labels)
    raw = H.pair_stats(run_a, run_b, lab, "raw", False, n_perm=99)
    zc = H.pair_stats(run_a, run_b, lab, "z_common", False, n_perm=99)
    assert abs(raw["gap"] - 0.5 * (x[:4].mean() - x[4:7].mean())) < 1e-12 and raw["gap"] > 0
    assert abs(zc["gap"]) < 1e-12
    nb = H.pair_stats(run_a, run_b, lab, "raw", True, n_perm=99)
    assert nb["n_desirable"] == 3


def test_split_half_se_uses_han_rule(labels):
    rng = np.random.default_rng(0)
    run_a = dict(roles=ROLES, projection=np.zeros(10), half_a=rng.normal(0, 1, (50, 10)), half_b=rng.normal(0, 1, (50, 10)))
    run_b = dict(roles=ROLES, projection=np.zeros(10), half_a=np.zeros((50, 10)), half_b=np.zeros((50, 10)))
    s = H.pair_stats(run_a, run_b, H.labels_table(labels), "raw", False, n_perm=9)
    per = -(run_a["half_a"] - run_a["half_b"])[:, :4].mean(1)
    assert abs(s["desirable_se"] - per.std(ddof=1) / 2) < 1e-12


def test_boot_weights():
    W = H.boot_weights(20, seed=3)
    assert (W[0] == 1).all() and (W.sum(1) == 240).all()
    assert np.array_equal(W, H.boot_weights(20, seed=3))


def line_sums(values, default_value):
    """Roles on a line along dimension 0; one item per question, so the vector equals the value."""
    n = len(values)
    sums = np.zeros((n + 1, Q, D), np.float32)
    sums[:n, :, 0] = np.array(values)[:, None]
    sums[n, :, 0] = default_value
    return sums, np.ones((n + 1, Q), np.float32)


def test_pc1_position_extreme_middle_outside():
    vals = list(np.linspace(0, 1, 9))
    W = H.boot_weights(5, seed=0)
    for dv, dist, out in ((1.0, 0.0, False), (0.5, 0.5, False), (2.0, 0.0, True)):
        r = H.pc1_position(*line_sums(vals, dv), W)
        assert abs(r["distance"][0] - dist) < 1e-3 and bool(r["outside"][0]) == out
    r = H.pc1_position(*line_sums(vals, 1.0), W)
    assert r["pc1_share"][0] > 0.99 and abs(abs(r["z"][0]) - (0.5 / np.std(np.linspace(0, 1, 9)))) < 1e-2


def test_cloud_variance_matches_direct():
    rng = np.random.default_rng(0)
    keys = [f"pos_p{p}_q{q}" for p in range(P) for q in range(Q)]
    acts = rng.normal(0, 2, (len(keys), D)) + 100.0
    W = H.boot_weights(3, seed=1)
    v = H.cloud_variance(keys, acts, W)
    direct = np.mean([((acts[p * Q:(p + 1) * Q] - acts[p * Q:(p + 1) * Q].mean(0)) ** 2).sum(1).mean() for p in range(P)])
    assert abs(v[0] - direct) < 1e-8 and abs(direct - 4 * D) < 0.6


def test_part_cd_end_to_end_with_a_tight_default(tmp_path, labels):
    rel = make_release(tmp_path / "rel", np.random.default_rng(2), default_noise=0.5)
    r = H.part_cd(rel, labels, n_boot=10, n_perm=99)
    for name in ("C1", "C1_unfiltered", "C2"):
        assert set(r[name]) == {"SFT", "DPO", "RLVR"}
    assert r["C1"]["SFT"]["n_roles"] == 9 and r["C1_unfiltered"]["SFT"]["n_roles"] == 10
    for s in ("SFT", "DPO", "RLVR"):  # default noise 0.5 vs 1.0: variance ratio 0.25
        assert abs(r["C2"][s]["log_var_default_over_median_role"]["observed"] - np.log(0.25)) < 0.1
    assert r["C1"]["RLVR"]["signed_z"]["observed"] >= 0
    assert set(r["C3"]) == {"SFT->DPO", "DPO->RLVR"} and set(r["D_shifts"]["SFT->DPO"]) == {"raw", "z_common"}
    json.dumps(r, default=float)


def test_cloud_variance_with_missing_replies():
    rng = np.random.default_rng(4)
    keys = [f"pos_p{p}_q{q}" for p in range(P) for q in range(Q)]
    acts = rng.normal(0, 2, (len(keys), D)) + 50.0
    drop = {5, 300, 301}
    keys2 = [k for i, k in enumerate(keys) if i not in drop]
    acts2 = np.delete(acts, sorted(drop), axis=0)
    W = H.boot_weights(3, seed=1)
    v = H.cloud_variance(keys2, acts2, W)
    direct = []
    for p in range(P):
        rows = [i for i, k in enumerate(keys2) if k.startswith(f"pos_p{p}_")]
        x = acts2[rows]
        direct.append(((x - x.mean(0)) ** 2).sum(1).mean())
    assert abs(v[0] - np.mean(direct)) < 1e-8 and np.isfinite(v).all()
    with pytest.raises(SystemExit):
        H.cloud_variance(keys2 + keys2[:1], np.vstack([acts2, acts2[:1]]), W)
