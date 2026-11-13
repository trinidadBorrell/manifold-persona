"""Offline tests for maps.py, including the G5 synthetic cases (THRESHOLDS.md): identical, rotated and scaled maps
give r near 1; independent maps give r near 0 and a uniform null p; the cross-half design removes planted
same-row noise; the equal-map, wording-sensitivity case is computed and recorded."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import maps as M  # noqa: E402

C, Q, D = 20, 40, 48


def roles(rng, scale=1.0):
    return scale * rng.standard_normal((C, D))


def sides_of(S, N, halves, support=None):
    return M.build_sides(S, N, halves, support)


@pytest.fixture(scope="module")
def halves():
    return M.halves_from(F.make_halves(range(Q), n=20))


def test_grams_match_direct_role_vectors():
    rng = np.random.default_rng(0)
    S, N = rng.standard_normal((4, 6, 5, 7)), (rng.random((4, 6, 5)) > 0.2).astype(float)
    S = S * N[..., None]
    q, w = np.array([1, 1, 0, 1, 0, 1], bool), np.array([1, 0, 1, 0, 0], bool)
    W = np.array([[1, 2, 5, 0, 1, 1], [1, 1, 1, 1, 1, 1]], float)
    G = M.Side(S, N, q, w).grams(W)
    for b in range(2):
        a = W[b] * q
        num = np.einsum("cqwd,q,w->cd", S, a, w.astype(float))
        den = np.einsum("cqw,q,w->c", N, a, w.astype(float))
        V = num / den[:, None]
        V = V - V.mean(0)
        np.testing.assert_allclose(G[b], V @ V.T, rtol=1e-9, atol=1e-9)


def test_cells_average_replicates_first():
    keys = [json.dumps(["a", "chat", 0, 7, r]) for r in range(3)] + [json.dumps(["b", "chat", 1, 7, 0])]
    X = np.array([[1.0], [3.0], [np.nan], [5.0]])
    S, N = M.cells(X, keys, ["a", "b"], [7])
    assert S[0, 0, 0, 0] == 2.0 and N[0, 0, 0] == 1 and N[0, 0, 1] == 0 and S[1, 0, 1, 0] == 5.0


def stats_for(cks, halves, comps, n_boot=200):
    rr, W = M.make_draws(C, Q, n=n_boot)
    return M.panel({k: sides_of(*v, halves) for k, v in cks.items()}, rr, W, comps)


def test_identical_rotated_scaled_maps(halves):
    rng = np.random.default_rng(1)
    R, qe = roles(rng), 0.5 * rng.standard_normal((Q, D))
    a, b = (M.synthetic_cells(R, qe, 2.0, rng) for _ in range(2))
    O = np.linalg.qr(rng.standard_normal((D, D)))[0]
    cks = dict(a=a, b=b, rot=(b[0] @ O, b[1]), scl=(3 * b[0], b[1]))
    st = stats_for(cks, halves, [("a", "a"), ("b", "b"), ("a", "b"), ("a", "rot"), ("a", "scl"), ("rot", "rot")])
    norm, _ = M.r_norm(st, "a", "b")
    assert norm["available"] and abs(norm["value"]["point"] - 1) < 0.1
    for k in ("rot", "scl"):
        np.testing.assert_allclose(st[("a", k)]["r"], st[("a", "b")]["r"], atol=1e-9)
    np.testing.assert_allclose(st[("rot", "rot")]["r"], st[("b", "b")]["r"], atol=1e-9)
    assert st[("a", "b")]["cka"][0] > 0.8


def test_independent_maps_and_uniform_null(halves):
    rng = np.random.default_rng(2)
    ps, rs = [], []
    for _ in range(30):
        cks = {k: M.synthetic_cells(roles(rng), np.zeros((Q, D)), 2.0, rng) for k in "ab"}
        sides = {k: sides_of(*v, halves[:5]) for k, v in cks.items()}
        rr, W = M.make_draws(C, Q, n=0)
        obs = M.panel(sides, rr, W, [("a", "b")])[("a", "b")]["r"][0]
        ps.append(M.p_value(obs, M.null(sides["a"], sides["b"], Q, n_perm=199, seed=len(ps))))
        rs.append(obs)
    assert abs(np.mean(rs)) < 0.1
    assert 0.3 < np.mean(ps) < 0.7 and np.mean(np.array(ps) < 0.1) < 0.25


def test_undefined_permutations_are_not_evidence():
    assert np.isnan(M.p_value(0.3, np.full(99, np.nan)))
    assert np.isnan(M.p_value(np.nan, np.zeros(99)))
    assert M.p_value(0.3, np.r_[np.full(50, np.nan), np.zeros(49)]) == 1 / 50


def test_planted_map_is_found_by_the_null(halves):
    rng = np.random.default_rng(3)
    R = roles(rng)
    cks = {k: M.synthetic_cells(R, np.zeros((Q, D)), 2.0, rng) for k in "ab"}
    sides = {k: sides_of(*v, halves[:5]) for k, v in cks.items()}
    rr, W = M.make_draws(C, Q, n=0)
    obs = M.panel(sides, rr, W, [("a", "b")])[("a", "b")]["r"][0]
    assert M.p_value(obs, M.null(sides["a"], sides["b"], Q, n_perm=499)) == 1 / 500


def test_cross_half_removes_shared_row_noise(halves):
    rng = np.random.default_rng(4)
    shared = 3.0 * rng.standard_normal((C, Q, 5, D))  # the same text noise in both checkpoints' prompts
    a, b = (M.synthetic_cells(roles(rng, 0.05), np.zeros((Q, D)), 0.5, rng, shared=shared) for _ in range(2))
    cross = stats_for(dict(a=a, b=b), halves, [("a", "b")], n_boot=0)[("a", "b")]["r"][0]
    same = []
    for (qa, wa), _ in halves:
        Ga, Gb = (M.Side(S, N, qa, wa).grams(np.ones((1, Q)))[0] for S, N in (a, b))
        i, j = np.triu_indices(C, 1)
        same.append(np.corrcoef(M.dist(Ga)[i, j], M.dist(Gb)[i, j])[0, 1])
    assert np.mean(same) > 0.5 and abs(cross) < 0.15


def test_wording_sensitivity_case_is_recorded(halves):
    """Equal complete role maps; checkpoint b moves with the wordings (k = -3, -3, 2, 2, 2)."""
    rng = np.random.default_rng(5)
    R, E = roles(rng), roles(rng, 4.0)
    a = M.synthetic_cells(R, np.zeros((Q, D)), 2.0, rng)
    b = M.synthetic_cells(R, np.zeros((Q, D)), 2.0, rng, wording_dirs=E, k=(-3, -3, 2, 2, 2))
    st = stats_for(dict(a=a, b=b), halves, [("a", "a"), ("b", "b"), ("a", "b")])
    norm, _ = M.r_norm(st, "a", "b")
    assert np.isfinite(st[("a", "b")]["r"]).all()
    assert set(norm) >= {"available", "raw", "rel"} and "pass" not in json.dumps(norm)


def test_f4_support_makes_draw_intervals_unavailable():
    rng = np.random.default_rng(6)
    S, N = M.synthetic_cells(roles(rng), np.zeros((Q, D)), 1.0, rng)
    N = N.copy()
    N[0, 1:] = 0  # role 0 has prompts on question 0 only
    halves = [((np.arange(Q) < 20, np.ones(5, bool)), (np.arange(Q) >= 20, np.ones(5, bool)))]
    sides = M.build_sides(S * N[..., None], N, halves, support="common")
    assert len(sides[0][1].q) == 0  # side B has no common question
    rr, W = M.make_draws(C, Q, n=50)
    st = M.panel({"a": sides}, rr, W, [("a", "a")])
    assert M.summary(st[("a", "a")]["r"])["lo"] is None


def test_four_role_grid_runs():
    rng = np.random.default_rng(7)
    R = rng.standard_normal((4, D))
    cks = {k: M.synthetic_cells(R, np.zeros((Q, D)), 1.0, rng) for k in "ab"}
    halves = M.halves_from(F.make_halves(range(Q), n=5))
    sides = {k: M.build_sides(*v, halves) for k, v in cks.items()}
    rr, W = M.make_draws(4, Q, n=20)
    st = M.panel(sides, rr, W, [("a", "b")])
    assert np.isfinite(st[("a", "b")]["r"][0])
    assert M.p_value(st[("a", "b")]["r"][0], M.null(sides["a"], sides["b"], Q, n_perm=99)) >= 1 / 24 - 0.02
