"""The two chunking rules, on a synthetic cloud: does "distance" still do what it did?

Plan: plans/2026-09-08-steering-rebuild.md (Interventions).

WHY THIS FILE EXISTS. `pick_centroids` grew a `chunk` argument, and the run that
produced 2026-09-10T15-46-fig4 used the code as it stood BEFORE that argument.
The only way "the default is unchanged" is a fact rather than an intention is to
keep a copy of the old algorithm and assert the new one agrees with it, index for
index, over a grid of eps and k. `test_default_matches_old_implementation` is
that copy; it is deliberately a transcription of `git show HEAD:...` and not a
call into the module, because an oracle that shares code with the thing it
checks cannot catch the thing worth catching.

THE CLOUD IS SYNTHETIC ON PURPOSE. The real centroid cloud lives on NFS and
takes ~267 s to load, which is long enough that a test nobody runs is the likely
outcome. Nothing asserted here is a property of the real cloud: they are
properties of the partition -- equal-width spans leave chunks empty, equal-count
groups do not, and at k=1 the two are the same partition -- and a seeded
gaussian exercises all of them in milliseconds.

No pytest: the venv does not have it and this is not worth a dependency. Plain
asserts in functions, one `main` that runs them all.

Usage:
    .venv/bin/python -m steering.test_chunking
"""
from __future__ import annotations

import sys

import numpy as np

from steering.manifold_paths import (END_MARGIN, PersonaPath, chunk_diagnostics,
                                     pick_centroids, select_cylinder)


# --------------------------------------------------------------------------
# The oracle: `pick_centroids` exactly as it stood at HEAD, before `chunk`
# --------------------------------------------------------------------------

def old_pick_centroids(C, P0, P1, eps, k=8, mode="absolute"):
    """Verbatim from `git show HEAD:steering/manifold_paths.py`. Do not tidy."""
    idx_all, u_all, r_all = select_cylinder(C, P0, P1, eps, mode)
    if len(idx_all) == 0:
        return np.zeros(0), np.zeros((0, C.shape[1])), np.zeros(0, dtype=int)

    edges = np.linspace(END_MARGIN, 1.0 - END_MARGIN, int(k) + 1)
    u_sel = u_all[idx_all]
    span_of = np.clip(np.searchsorted(edges, u_sel, side="right") - 1, 0, int(k) - 1)

    chosen = []
    for b in range(int(k)):
        cand = idx_all[span_of == b]
        if len(cand) == 0:
            continue
        chosen.append(int(cand[np.argmin(r_all[cand])]))   # nearest the chord
    chosen = np.array(sorted(chosen, key=lambda i: u_all[i]), dtype=int)
    return u_all[chosen], np.asarray(C)[chosen], chosen


# --------------------------------------------------------------------------
# Synthetic clouds
# --------------------------------------------------------------------------

def cloud(seed=0, n=60, ambient=7, spread=1.0):
    """A gaussian blob and a chord across it.

    The endpoints are two of the cloud's own points, as in the real cases
    (`path_cases.pick_endpoints` builds every pair chord from centroids), so the
    endpoint-duplication that END_MARGIN exists to stop is actually exercised.
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(scale=spread, size=(n, ambient))
    return C, C[0].copy(), C[1].copy()


def clumped_cloud(seed=0, n=60, ambient=7):
    """The same, but with the candidates piled at one end of the chord.

    Equal-width spans and equal-count groups only differ when the candidates are
    unevenly spread along the chord, which is the normal case on the real cloud
    and never the case for a symmetric blob.
    """
    rng = np.random.default_rng(seed)
    P0 = np.zeros(ambient)
    P1 = np.zeros(ambient)
    P1[0] = 10.0
    t = rng.beta(1.5, 6.0, size=n)                    # piled near P0
    C = np.outer(t, P1) + rng.normal(scale=0.4, size=(n, ambient))
    return np.vstack([P0[None, :], P1[None, :], C]), P0, P1


EPS_GRID = (0.0, 0.3, 0.8, 1.5, 3.0)
K_GRID = (1, 2, 3, 5, 8, 13, 64)


# --------------------------------------------------------------------------
# 1. Regression: the default is the old code
# --------------------------------------------------------------------------

def test_default_matches_old_implementation():
    for seed in range(6):
        for C, P0, P1 in (cloud(seed), clumped_cloud(seed)):
            for eps in EPS_GRID:
                for k in K_GRID:
                    for mode in ("absolute", "relative"):
                        u_o, Y_o, i_o = old_pick_centroids(C, P0, P1, eps, k, mode)
                        u_n, Y_n, i_n = pick_centroids(C, P0, P1, eps, k=k,
                                                       mode=mode)
                        where = "seed=%d eps=%r k=%d mode=%s" % (seed, eps, k, mode)
                        assert np.array_equal(i_o, i_n), "indices differ: " + where
                        assert np.array_equal(u_o, u_n), "u differs: " + where
                        assert np.array_equal(Y_o, Y_n), "Y differs: " + where


# --------------------------------------------------------------------------
# 2. k=1: one chunk is one chunk, however you cut it
# --------------------------------------------------------------------------

def test_k1_modes_agree():
    for seed in range(20):
        for C, P0, P1 in (cloud(seed), clumped_cloud(seed)):
            for eps in (0.5, 1.0, 2.0, 4.0):
                _, _, i_d = pick_centroids(C, P0, P1, eps, k=1, chunk="distance")
                _, _, i_n = pick_centroids(C, P0, P1, eps, k=1, chunk="density")
                assert np.array_equal(i_d, i_n), (
                    "k=1 disagreement at seed=%d eps=%r: %r vs %r"
                    % (seed, eps, i_d, i_n))
                assert len(i_d) <= 1


# --------------------------------------------------------------------------
# 3. eps = 0: the negative control, in both modes
# --------------------------------------------------------------------------

def test_eps_zero_is_empty_in_both_modes():
    C, P0, P1 = cloud(3)
    for chunk in ("distance", "density"):
        u, Y, idx = pick_centroids(C, P0, P1, 0.0, k=8, chunk=chunk)
        assert len(idx) == 0 and len(u) == 0, chunk
        assert Y.shape == (0, C.shape[1]), (chunk, Y.shape)
        recs, radius = chunk_diagnostics(C, P0, P1, 0.0, k=8, chunk=chunk)
        assert recs == [] and radius == 0.0, chunk
        path = PersonaPath(P0, P1, C, eps=0.0, k=8, chunk=chunk)
        assert path.n_centroids == 0, chunk
        assert abs(path.detour_ratio - 1.0) < 1e-9, (chunk, path.detour_ratio)


# --------------------------------------------------------------------------
# 4. density fills every bin
# --------------------------------------------------------------------------

def test_density_bin_counts():
    for seed in range(8):
        for C, P0, P1 in (cloud(seed), clumped_cloud(seed)):
            for eps in (0.5, 1.0, 2.0, 4.0):
                n_cand = len(select_cylinder(C, P0, P1, eps)[0])
                for k in K_GRID:
                    _, _, idx = pick_centroids(C, P0, P1, eps, k=k, chunk="density")
                    assert len(idx) == min(k, n_cand), (
                        "density lost a bin: seed=%d eps=%r k=%d n_cand=%d got %d"
                        % (seed, eps, k, n_cand, len(idx)))
                    assert len(set(idx.tolist())) == len(idx), "knot used twice"


def test_distance_can_leave_bins_empty():
    """Not a requirement on "distance" -- a check that the clumped cloud is a
    real test case. If equal-width spans never went empty here, test 4 would be
    passing vacuously."""
    C, P0, P1 = clumped_cloud(0)
    short = [len(pick_centroids(C, P0, P1, eps, k=8, chunk="distance")[2]) < 8
             for eps in (1.0, 2.0)]
    assert any(short), "clumped cloud never starves a distance span"


# --------------------------------------------------------------------------
# 5. m: points per chunk, density only
# --------------------------------------------------------------------------

def test_m_sets_bin_count():
    for seed in range(6):
        for C, P0, P1 in (cloud(seed), clumped_cloud(seed)):
            for eps in (1.0, 2.0, 4.0):
                n_cand = len(select_cylinder(C, P0, P1, eps)[0])
                if n_cand == 0:
                    continue
                for m in (1, 2, 3, 5, 11, 1000):
                    want = max(1, -(-n_cand // m))          # ceil, integer-only
                    _, _, idx = pick_centroids(C, P0, P1, eps, k=8,
                                               chunk="density", m=m)
                    assert len(idx) == want, (
                        "m=%d on %d candidates gave %d bins, want %d"
                        % (m, n_cand, len(idx), want))
                # k is IGNORED when m is given: same answer for any k.
                a = pick_centroids(C, P0, P1, eps, k=2, chunk="density", m=3)[2]
                b = pick_centroids(C, P0, P1, eps, k=99, chunk="density", m=3)[2]
                assert np.array_equal(a, b), "m did not override k"


def test_m_with_distance_raises():
    C, P0, P1 = cloud(4)
    for call in (lambda: pick_centroids(C, P0, P1, 1.0, k=8, chunk="distance", m=3),
                 lambda: chunk_diagnostics(C, P0, P1, 1.0, k=8, chunk="distance", m=3),
                 lambda: PersonaPath(P0, P1, C, eps=1.0, chunk="distance", m=3),
                 # eps=0 exits early with an empty set; the argument is still wrong.
                 lambda: pick_centroids(C, P0, P1, 0.0, k=8, chunk="distance", m=3)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("m with chunk='distance' was accepted")
    for bad_m in (0, -2):
        try:
            pick_centroids(C, P0, P1, 1.0, k=8, chunk="density", m=bad_m)
        except ValueError:
            pass
        else:
            raise AssertionError("m=%r was accepted" % (bad_m,))


# --------------------------------------------------------------------------
# 6. the diagnostic describes the route that was actually built
# --------------------------------------------------------------------------

def test_diagnostics_match_pick():
    for seed in range(6):
        for C, P0, P1 in (cloud(seed), clumped_cloud(seed)):
            for eps in (0.5, 1.0, 2.0, 4.0):
                for k in (1, 3, 8, 13):
                    for chunk in ("distance", "density"):
                        _, _, idx = pick_centroids(C, P0, P1, eps, k=k, chunk=chunk)
                        recs, radius = chunk_diagnostics(C, P0, P1, eps, k=k,
                                                         chunk=chunk)
                        where = ("seed=%d eps=%r k=%d chunk=%s"
                                 % (seed, eps, k, chunk))
                        assert radius == eps, where
                        picked = [r["picked"] for r in recs]
                        assert sorted(picked) == sorted(idx.tolist()), (
                            "picked set differs: " + where)
                        assert [r["bin"] for r in recs] == sorted(
                            r["bin"] for r in recs), "bins out of order: " + where
                        for rec in recs:
                            assert rec["picked_rank_r"] == 0, (
                                "picked was not the nearest: " + where)
                            assert rec["picked"] in rec["cand_idx"].tolist(), where
                            assert (len(rec["cand_u"]) == len(rec["cand_idx"])
                                    == len(rec["cand_r"]) == len(rec["cand_cos"])), where
                            assert (np.diff(rec["cand_u"]) >= 0).all(), (
                                "candidates not in chord order: " + where)
                            assert 0 <= rec["picked_rank_cos"] < len(rec["cand_idx"])


def test_diagnostics_radius_follows_mode():
    C, P0, P1 = cloud(2)
    L = float(np.linalg.norm(P1 - P0))
    assert chunk_diagnostics(C, P0, P1, 0.7, mode="absolute")[1] == 0.7
    assert abs(chunk_diagnostics(C, P0, P1, 0.7, mode="relative")[1] - 0.7 * L) < 1e-12


# --------------------------------------------------------------------------
# 7. a typo is an error, not a default
# --------------------------------------------------------------------------

def test_bad_chunk_raises():
    C, P0, P1 = cloud(1)
    for bad in ("Distance", "equal", None, 3):
        for call in (lambda: pick_centroids(C, P0, P1, 1.0, chunk=bad),
                     lambda: chunk_diagnostics(C, P0, P1, 1.0, chunk=bad),
                     lambda: PersonaPath(P0, P1, C, eps=1.0, chunk=bad),
                     lambda: pick_centroids(C, P0, P1, 0.0, chunk=bad)):
            try:
                call()
            except ValueError as exc:
                assert repr(bad) in str(exc), (
                    "error does not name the value received: %s" % exc)
            else:
                raise AssertionError("chunk=%r was accepted" % (bad,))


# --------------------------------------------------------------------------
# 8. PersonaPath: the default route is untouched, and the report says which
# --------------------------------------------------------------------------

def test_persona_path_default_unchanged_and_reported():
    C, P0, P1 = clumped_cloud(7)
    base = PersonaPath(P0, P1, C, eps=2.0, k=8)
    explicit = PersonaPath(P0, P1, C, eps=2.0, k=8, chunk="distance")
    assert np.array_equal(base.centroid_idx, explicit.centroid_idx)
    assert base.detour_ratio == explicit.detour_ratio

    rep = base.report()
    assert rep["chunk"] == "distance" and rep["m"] is None, rep
    dens = PersonaPath(P0, P1, C, eps=2.0, k=8, chunk="density", m=4).report()
    assert dens["chunk"] == "density" and dens["m"] == 4, dens
    # The knots are still real centroids and the ends are still pinned.
    assert dens["knot_error"] < 1e-8, dens["knot_error"]


TESTS = [test_default_matches_old_implementation,
         test_k1_modes_agree,
         test_eps_zero_is_empty_in_both_modes,
         test_density_bin_counts,
         test_distance_can_leave_bins_empty,
         test_m_sets_bin_count,
         test_m_with_distance_raises,
         test_diagnostics_match_pick,
         test_diagnostics_radius_follows_mode,
         test_bad_chunk_raises,
         test_persona_path_default_unchanged_and_reported]


def main():
    failed = []
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed.append(fn.__name__)
            print("FAIL  %-45s %s" % (fn.__name__, exc))
        except Exception as exc:                      # a raise is also a failure
            failed.append(fn.__name__)
            print("ERROR %-45s %r" % (fn.__name__, exc))
        else:
            print("pass  %s" % fn.__name__)
    print()
    if failed:
        print("FAIL: %d of %d tests failed: %s" % (len(failed), len(TESTS), failed))
        sys.exit(1)
    print("PASS: %d/%d" % (len(TESTS), len(TESTS)))


if __name__ == "__main__":
    main()
