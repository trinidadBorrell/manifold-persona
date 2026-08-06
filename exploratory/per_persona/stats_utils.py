"""Correlation and multiple-comparison helpers shared across this study.

These lived in ``04_id_vs_axis.py`` and ``06_axis_ladder.py``. Because both
filenames began with a digit they could not be imported by name, so the ladder
reached for its dependency through ``importlib.util.spec_from_file_location``.
Collecting them here is what let those load-by-path blocks go away — the
functions are unchanged, and the ladder's rung 2 still runs the *same*
``partial_corr`` that produced the published `id_vs_axis` numbers rather than a
copy of it.

``partial_corr`` (one control) and ``partial_corr_multi`` (any number) are kept
as separate functions rather than merged: the single-control form is the one
whose published output must stay reproducible, and ``partial_corr_multi``
asserts against it at run time in the ladder.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        adj[i] = prev
    return adj


def partial_corr(x, y, z):
    """Pearson r between x and y with z linearly removed from both."""
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, k = len(x), 1
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - k - 2) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df=n - k - 2))


def partial_corr_multi(x, y, Z):
    """Pearson r between x and y with the columns of Z linearly removed.

    Generalises :func:`partial_corr`. Verified in the ladder to agree with it to
    ~1e-12 when Z has one column, so every rung is the same estimator at a
    different k and the ladder is internally consistent.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if Z is None or (hasattr(Z, "shape") and Z.shape[1] == 0):
        r = float(np.corrcoef(x, y)[0, 1])
        n, k = len(x), 0
    else:
        Z = np.asarray(Z, float)
        A = np.column_stack([np.ones(len(x)), Z])
        rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        r = float(np.corrcoef(rx, ry)[0, 1])
        n, k = len(x), Z.shape[1]
    if not np.isfinite(r) or abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - k - 2) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df=n - k - 2))


def boot_ci(x, y, Z, rng, n_boot: int = 2000):
    """Bootstrap 95% CI for a (partial) correlation, resampling ROLES.

    The role is the unit of observation — a role's 200 points share its
    instruction set and the shared question set, so resampling points would
    manufacture precision that does not exist.
    """
    n = len(x)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            out[b] = partial_corr_multi(x[idx], y[idx],
                                        Z[idx] if Z is not None else None)[0]
        except Exception:  # noqa: BLE001 — degenerate resample
            out[b] = np.nan
    out = out[np.isfinite(out)]
    if out.size < 100:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
