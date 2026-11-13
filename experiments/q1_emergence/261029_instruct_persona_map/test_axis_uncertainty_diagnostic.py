"""Numerical checks for the separate, descriptive A axis diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("axis_original", HERE / "analysis.py")
D = load("axis_diagnostic", HERE / "axis_uncertainty_diagnostic.py")


def fixture(seed=4):
    S = np.random.default_rng(seed).normal(size=(5, 40, 5, 9))
    halves = A.make_halves(list(range(40)), n=5)
    rr, W = A.M.make_draws(4, 40, n=31, seed=17)
    return S, halves, rr, W


@pytest.mark.parametrize("mode", ["joint", "roles_only", "questions_only"])
def test_scores_match_original_axis_calculation(mode):
    S, halves, rr, W = fixture()
    r = np.tile(rr[0], (len(rr), 1)) if mode == "questions_only" else rr
    w = np.ones_like(W) if mode == "roles_only" else W
    actual = D.axis_draws(S, halves, rr, W, mode)
    expected, _ = A.axis_statistics(S, np.ones(S.shape[:3]), halves, r, w)
    assert D.interval(actual) == pytest.approx(expected["cosine"], abs=2e-12)
    # Each draw is compared independently, including repeated role multiplicities.
    for k in [0, 1, 7, 19, 31]:
        single, _ = A.axis_statistics(S, np.ones(S.shape[:3]), halves, r[k:k+1], w[k:k+1])
        assert actual[k] == pytest.approx(single["cosine"]["point"], abs=2e-12)


def test_draws_are_translation_and_scale_invariant():
    S, halves, rr, W = fixture()
    expected = D.axis_draws(S, halves, rr, W)
    actual = D.axis_draws(3 * S + 10000, halves, rr, W)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_fixed_default_offset_has_no_role_or_question_spread():
    S, halves, rr, W = fixture()
    S[:] = 0
    S[-1, :, :, 0] = 2
    for mode in ["joint", "roles_only", "questions_only"]:
        np.testing.assert_allclose(D.axis_draws(S, halves, rr, W, mode), 1, atol=1e-12)


def test_identical_questions_remove_question_only_spread():
    S, halves, rr, W = fixture()
    S[:] = S[:, :1]
    values = D.axis_draws(S, halves, rr, W, "questions_only")
    assert np.std(values[1:]) < 1e-12
    assert np.std(D.axis_draws(S, halves, rr, W, "roles_only")[1:]) > .001


def test_identical_roles_remove_role_only_spread():
    S, halves, rr, W = fixture()
    S[:-1] = S[0]
    values = D.axis_draws(S, halves, rr, W, "roles_only")
    assert np.std(values[1:]) < 1e-12
    assert np.std(D.axis_draws(S, halves, rr, W, "questions_only")[1:]) > .001


def test_influence_matches_direct_leave_out_recalculations():
    S, halves, rr, W = fixture()
    result = D.influences(S, halves)
    for role in range(4):
        subset = np.delete(S, role, axis=0)
        r = np.arange(3)[None, :]
        expected = D.axis_draws(subset, halves, r, np.ones((1, 40)))[0]
        assert result["leave_role_out"][role] == pytest.approx(expected, abs=1e-12)
    for q in range(40):
        reduced = [tuple((np.delete(qm, q), wm) for qm, wm in half) for half in halves]
        expected = D.axis_draws(np.delete(S, q, axis=1), reduced, rr[:1], np.ones((1, 39)))[0]
        assert result["leave_question_out"][q] == pytest.approx(expected, abs=1e-12)
    for w in range(5):
        values = []
        for half in halves:
            axes = []
            for qm, wm in half:
                dw = wm.copy()
                dw[w] = False
                axes.append(S[-1][qm][:, dw].mean((0, 1)) - S[:-1][:, qm][:, :, wm].mean((0, 1, 2)))
            values.append(np.dot(*axes) / np.prod([np.linalg.norm(x) for x in axes]))
        assert result["leave_default_wording_out"][w] == pytest.approx(np.mean(values), abs=1e-12)


def test_diagnostic_rejects_nonfinite_or_undefined_axes():
    S, halves, rr, W = fixture()
    S[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        D.axis_draws(S, halves, rr, W)
    with pytest.raises(ValueError, match="undefined"):
        D.axis_draws(np.zeros_like(S), halves, rr, W)


def test_diagnostic_rejects_a_draw_with_no_question_support():
    S, halves, rr, W = fixture()
    W[1] = 0
    W[1, np.flatnonzero(halves[0][0][0])[0]] = 40
    with pytest.raises(ValueError, match="empty resampled half"):
        D.axis_draws(S, halves, rr, W)
