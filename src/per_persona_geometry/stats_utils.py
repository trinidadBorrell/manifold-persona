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
    """Benjamini-Hochberg adjusted p-values (same order as input).

    NON-FINITE p-VALUES ARE NOT TESTS. They come back as NaN and are excluded
    from the ranking and from the multiplicity count `n`. Leaving them in was
    wrong twice over: `np.argsort` sorts NaN to the END, so a NaN took the
    largest-p slot and pulled `prev` around with it, and counting undefined
    rows in `n` inflated every real q in the family.
    """
    p = np.asarray(p, dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return adj
    ps = p[ok]
    n = len(ps)
    order = np.argsort(ps)
    tmp = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, ps[i] * n / (n - rank + 1))
        tmp[i] = prev
    adj[ok] = tmp
    return adj


def residualise(x, Z):
    """`x` with the columns of `Z` (plus an intercept) linearly removed.

    Exposed because a permutation null for a PARTIAL correlation has to permute
    this, not the raw predictor — see the callers in study_entropy/study_ladder.
    """
    x = np.asarray(x, float)
    # atleast_2d BEFORE the width test: the old `.shape[1]` raised IndexError on
    # a 1-D Z — a single covariate passed as a flat array — even though the
    # column_stack below would have handled it. In-tree callers pass 2-D or
    # None, but this is exported for outside permutation nulls to call.
    Z = None if Z is None else np.atleast_2d(np.asarray(Z, float))
    if Z is not None and Z.shape[0] == 1 and len(x) != 1:
        Z = Z.T          # a flat Z is one covariate, not one row of many
    if Z is None or Z.shape[1] == 0:
        return x - x.mean()
    A = np.column_stack([np.ones(len(x)), Z])
    return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]


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
        # THROUGH `residualise`, the same function the Freedman-Lane nulls in
        # study_entropy/study_ladder permute. A private copy here would put the
        # observed statistic and the null it is compared against on two code
        # paths that are only believed to be identical.
        Z = np.asarray(Z, float)
        rx, ry = residualise(x, Z), residualise(y, Z)
        r = float(np.corrcoef(rx, ry)[0, 1])
        n, k = len(x), Z.shape[1]
    if not np.isfinite(r):
        # NaN r means the correlation is UNDEFINED (a degenerate residual, a
        # zero-variance column). p = 0.0 announced it as the most significant
        # result in the family: it took q = 0 under BH and sorted to the top of
        # every FDR listing. NaN says what actually happened.
        return r, float("nan")
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - k - 2) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df=n - k - 2))


MIN_BOOT_FRAC = 0.90


def boot_ci(x, y, Z, rng, n_boot: int = 2000, min_frac: float = MIN_BOOT_FRAC):
    """Bootstrap 95% CI for a (partial) correlation, resampling ROLES.

    The role is the unit of observation — a role's 200 points share its
    instruction set and the shared question set, so resampling points would
    manufacture precision that does not exist.

    A CI IS ONLY REPORTED WHEN ALMOST EVERY RESAMPLE SURVIVED. The old floor
    was 100 of 2,000 — five percent — so an estimator that blew up on 95% of
    resamples still produced a confident-looking "95% CI", built from the
    conditioning-friendly minority and therefore far too narrow. Below
    `min_frac` the honest answer is that the interval is not estimable.
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
    if out.size < max(100, int(np.ceil(min_frac * n_boot))):
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
