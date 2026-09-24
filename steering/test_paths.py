"""The path contract, on a synthetic cloud: the properties both steering arms rest on.

Plan: plans/2026-09-08-steering-rebuild.md (Interventions).

WHY THIS FILE EXISTS. `interventions.py` calls the controls "structural, not
asserted": delta(0) = 0 and delta(1) = P1 - P0 for both arms, and the linear arm
is exactly alpha*(P1 - P0). Structural is true only for as long as nobody edits
`manifold_paths.py`, and until now nothing ran those properties outside the GPU
smoke gate. A code review also found the manifold arm CLIPPING alpha to [0, 1]
while the linear arm extrapolated -- two arms of one comparison silently
answering different questions past alpha = 1. It now raises; this file keeps it
raising.

Also here: the hidden_states -> decoder-layer off-by-one
(`hook_layer_for_hidden_state`), because getting it wrong steers one layer away
from the geometry that defines the targets and nothing downstream notices.

THE CLOUD IS SYNTHETIC ON PURPOSE, for the reason test_chunking.py gives: the
real cloud takes minutes to load and a test nobody runs catches nothing. Every
property checked is a property of the construction, not of the data.

No pytest in the venv, so: plain asserts in `test_*` functions (pytest collects
them unchanged where it is installed) and one `main` that runs them all.

Usage:
    .venv/bin/python steering/test_paths.py
    .venv/bin/python -m steering.test_paths
"""
from __future__ import annotations

import os
import sys

# Runnable as a plain script as well as with -m: put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from steering.manifold_paths import (FixedEndSpline, LinearPath,  # noqa: E402
                                     PersonaPath)


# --------------------------------------------------------------------------
# Synthetic cloud
# --------------------------------------------------------------------------

def cloud(seed=0, n=80, ambient=9):
    """Points scattered along a chord P0 -> P1 with lateral noise.

    Spread along the chord so the eps-tube actually catches interior centroids
    and the manifold arm is a real curve, not the eps = 0 fallback -- a test of
    PersonaPath that only ever sees the straight-line fallback proves nothing
    about the spline.
    """
    rng = np.random.default_rng(seed)
    P0 = rng.normal(size=ambient) * 3.0           # off the origin, as real P0 is
    P1 = P0 + rng.normal(size=ambient) * 4.0
    t = rng.uniform(0.02, 0.98, size=n)
    C = P0[None, :] + np.outer(t, P1 - P0) + rng.normal(scale=0.8, size=(n, ambient))
    return C, P0, P1


EPS = 2.0          # absolute tube radius; catches plenty of the cloud above


def arms(seed=0, k=6):
    C, P0, P1 = cloud(seed)
    lin = LinearPath(P0, P1)
    man = PersonaPath(P0, P1, C, EPS, k=k, lam=0.0, mode="absolute")
    assert man.n_centroids > 0, "cloud too sparse: manifold arm fell back to the chord"
    return lin, man, P0, P1


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError("expected %s, nothing raised" % exc.__name__)


# --------------------------------------------------------------------------
# 1. The endpoint controls, both arms
# --------------------------------------------------------------------------

def test_delta_zero_at_alpha_0():
    for seed in range(4):
        lin, man, P0, P1 = arms(seed)
        for name, p in (("linear", lin), ("manifold", man)):
            d = p.delta(0.0)
            assert d.shape == (1, P0.shape[0]), "%s delta shape %r" % (name, d.shape)
            assert np.abs(d).max() == 0.0, "%s delta(0) = %g, not exactly 0" % (
                name, np.abs(d).max())


def test_delta_is_chord_at_alpha_1():
    for seed in range(4):
        lin, man, P0, P1 = arms(seed)
        want = P1 - P0
        # Linear: exact by construction (1.0 * (P1-P0)).
        assert np.array_equal(lin.delta(1.0)[0], want), "linear delta(1) != P1-P0"
        # Manifold: the ends are pinned, so the only error is the arc-length
        # inversion landing on the last knot -- rounding, not approximation.
        err = np.abs(man.delta(1.0)[0] - want).max()
        assert err < 1e-9 * max(1.0, np.abs(want).max()), (
            "manifold delta(1) misses P1-P0 by %g" % err)


def test_linear_delta_is_alpha_times_chord():
    lin, _, P0, P1 = arms(0)
    a = np.array([0.0, 0.1, 0.37, 0.5, 0.9, 1.0])
    assert np.array_equal(lin.delta(a), a[:, None] * (P1 - P0)[None, :])
    # The linear arm EXTRAPOLATES, and must keep doing so: dose_escalation.py
    # pushes it to 30 chords. Past 1 is its job, not the manifold arm's.
    for big in (2.0, 6.0, 30.0):
        assert np.array_equal(lin.delta(big)[0], big * (P1 - P0))


# --------------------------------------------------------------------------
# 2. The manifold arm refuses alpha outside [0, 1]
# --------------------------------------------------------------------------

def test_manifold_raises_outside_unit_interval():
    _, man, _, _ = arms(0)
    for bad in (-0.1, 1.0 + 1e-9, 2.0, float("nan"), [0.5, 1.5]):
        _raises(lambda: man.at_alpha(bad))
        _raises(lambda: man.delta(bad))
        _raises(lambda: man.spline.at_alpha(bad))
    e = _raises(lambda: man.delta(2.0))
    assert "LinearPath" in str(e), "message must point the caller at LinearPath: %s" % e
    # ...and still accepts the closed interval, ends included.
    man.at_alpha([0.0, 0.5, 1.0])


def test_manifold_fallback_raises_too():
    """eps = 0 selects no centroid and the arm delegates to a LinearPath.

    The accepted range must not depend on whether the tube caught anything.
    """
    C, P0, P1 = cloud(0)
    empty = PersonaPath(P0, P1, C, 0.0, k=6)
    assert empty.n_centroids == 0 and empty.spline is None
    _raises(lambda: empty.at_alpha(2.0))
    _raises(lambda: empty.delta(-1.0))


# --------------------------------------------------------------------------
# 3. Shape of the manifold curve
# --------------------------------------------------------------------------

def test_knot_error_is_zero():
    for seed in range(4):
        for param in ("projection", "length", "centripetal"):
            C, P0, P1 = cloud(seed)
            p = PersonaPath(P0, P1, C, EPS, k=6, lam=0.0, param=param)
            assert p.n_centroids > 0
            assert p.knot_error() < 1e-9, "seed=%d param=%s knot error %g" % (
                seed, param, p.knot_error())


def test_arc_length_monotone():
    _, man, _, _ = arms(0)
    s = man.spline._arc["s"] if man.spline._arc else man.spline._build_arc()["s"]
    assert (np.diff(s) > 0).all(), "cumulative arc length not strictly increasing"
    # alpha IS normalised arc position: equal alpha steps are equal arc steps.
    # Polyline through dense samples approximates arc from below, so the check
    # is on the spread of the step lengths, not on their sum.
    pts = man.at_alpha(np.linspace(0.0, 1.0, 401))
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert (steps > 0).all(), "at_alpha went backwards or stalled"
    assert steps.max() / steps.min() < 1.05, (
        "alpha is not arc position: step ratio %.3f" % (steps.max() / steps.min()))
    assert man.detour_ratio >= 1.0 - 1e-9, "curve shorter than its chord"


def test_fixed_end_spline_hits_ends_at_any_lam():
    C, P0, P1 = cloud(1)
    x = np.linspace(0.0, 1.0, 8)
    Y = np.vstack([P0, C[:6], P1])
    for lam in (0.0, 1e-3, 1.0, 100.0):
        sp = FixedEndSpline(x, Y, lam)
        ends = sp.at_alpha([0.0, 1.0])
        assert np.abs(ends[0] - P0).max() < 1e-9, "start drifted at lam=%g" % lam
        assert np.abs(ends[1] - P1).max() < 1e-9, "end drifted at lam=%g" % lam


# --------------------------------------------------------------------------
# 4. hidden_states index -> decoder layer
# --------------------------------------------------------------------------

def test_hook_layer_for_hidden_state():
    try:
        from steering.activation_steering import hook_layer_for_hidden_state
    except ImportError as exc:                     # torch missing from this env
        print("  SKIP test_hook_layer_for_hidden_state: cannot import "
              "steering.activation_steering (%s)" % exc)
        return
    # hidden_states[L] is the OUTPUT of model.layers[L-1]; [0] is the embedding.
    for L in (1, 19, 25, 32, 36):
        assert hook_layer_for_hidden_state(L) == L - 1
    for bad in (0, -1):
        _raises(lambda: hook_layer_for_hidden_state(bad))


TESTS = [test_delta_zero_at_alpha_0,
         test_delta_is_chord_at_alpha_1,
         test_linear_delta_is_alpha_times_chord,
         test_manifold_raises_outside_unit_interval,
         test_manifold_fallback_raises_too,
         test_knot_error_is_zero,
         test_arc_length_monotone,
         test_fixed_end_spline_hits_ends_at_any_lam,
         test_hook_layer_for_hidden_state]


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
