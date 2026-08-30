"""Correlation and multiple-comparison helpers shared across this study.

These lived in ``04_id_vs_axis.py`` and ``06_axis_ladder.py``. Both filenames
began with a digit, so they could not be imported by name. The ladder
therefore reached for its dependency through
``importlib.util.spec_from_file_location``. Collecting the functions here let
those load-by-path blocks go away. The functions are unchanged, and the
ladder's rung 2 still runs the *same* ``partial_corr`` that produced the
published `id_vs_axis` numbers rather than a copy of it.

``partial_corr`` (one control) and ``partial_corr_multi`` (any number) are
kept as separate functions rather than merged. The single-control form is the
one whose published output must stay reproducible. ``partial_corr_multi``
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
    """Pearson r between x and y with z linearly removed from both.

    ``z`` is ONE control column [n]; pass a stack to partial_corr_multi,
    which computes the correct df for k controls."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("partial_corr takes one control column; "
                         "use partial_corr_multi for a [n, k] stack")
    Z = np.column_stack([np.ones(len(x)), z])
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


def linfit(x, y) -> dict:
    """Least-squares line PLUS the significance of its slope.

    Every trend line drawn in this study used to be a bare ``np.polyfit``, which
    gives a slope and no way to tell whether it is distinguishable from flat.
    This returns the fit and its test together so a figure cannot show a line
    without showing whether the line means anything.

    ``p`` is the two-sided test of slope = 0, which for a simple regression is
    identical to the test of Pearson r = 0. It is a RAW p-value: where the same
    figure family runs many tests at once, prefer the Benjamini-Hochberg ``q``
    the ladder already computes (`q_global_ctrl_all`). See :func:`fmt_p`.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan,
                "p": np.nan, "r2": np.nan, "n": int(x.size)}
    res = stats.linregress(x, y)
    return {"slope": float(res.slope), "intercept": float(res.intercept),
            "r": float(res.rvalue), "p": float(res.pvalue),
            "r2": float(res.rvalue ** 2), "n": int(x.size)}


def fmt_p(p, label: str = "p") -> str:
    """Compact p/q for a figure title: 'p<1e-10', 'p=0.023', 'p=0.51'.

    Small values are shown as an upper bound rather than as 3.7e-41, which is a
    number nobody reads and which invites over-reading a difference between two
    astronomically small p-values that means nothing.
    """
    if p is None or not np.isfinite(p):
        return f"{label}=n/a"
    if p < 1e-10:
        return f"{label}<1e-10"
    if p < 0.001:
        return f"{label}={p:.1e}"
    return f"{label}={p:.3f}"
