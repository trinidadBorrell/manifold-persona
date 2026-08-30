"""Persistent homology of one role's cloud — Betti numbers as geometry metrics.

Why this exists
---------------
Every other metric in this study measures how *big* a role's manifold is (how
many directions its points occupy). None of them can tell a disc from an
annulus: a cloud with a hole through it and a cloud without one can have
identical intrinsic dimension, identical participation ratio and identical
spectrum. Betti numbers are the cheapest available statement about *shape*
rather than size, so they are what this module adds to the panel.

    Betti-0 = number of connected components  (is the role's behaviour one
              coherent mode, or several disjoint ones?)
    Betti-1 = number of independent loops      (does the behaviour close back on
              itself — e.g. a cyclic response register?)
    Betti-2 = number of enclosed voids         (is there a hollow region the
              behaviour surrounds but never enters — a sphere-like shell rather
              than a solid ball?)

The threshold, and why it is relative
-------------------------------------
A Vietoris-Rips filtration returns every feature that ever appears, most of
which are one-simplex noise that dies immediately. Counting them all would make
Betti-1 a measure of sample size. We therefore count only features whose
*lifetime* (death - birth) exceeds ``LIFETIME_FRAC`` of the role's own PCA-50
diameter.

That fraction has to be **relative to each role's own scale**, not absolute.
Roles differ several-fold in cloud radius (that is itself a panel covariate).
An absolute lifetime cut-off would count more loops for large roles purely
because their distances are larger. That would manufacture exactly the scale
confound the rest of this run is built to control for. Dividing by the role's
diameter makes the count scale-free, so Betti and radius are free to be
uncorrelated.

The raw diagrams are always returned alongside the counts and are saved by
the caller. The threshold can then be re-cut later without re-running ripser
(which is the expensive part).

Both directions of the threshold are controlled in ``calib_estimators.py``. A
planted noisy circle must yield Betti-1 = 1, and a planted Gaussian blob must
yield Betti-1 = 0 under this same rule. Without the second, the rule would
manufacture loops and every role would look topologically interesting.
"""
from __future__ import annotations

import os

import numpy as np

LIFETIME_FRAC = 0.10       # feature counts iff lifetime > 10% of cloud diameter

# Highest homology dimension ripser computes: 1 = components + loops, 2 = also
# enclosed voids. Default stays 2, which is right at the 200-point budget this
# study was built on (measured 0.37 s/role).
#
# It is settable because the cost is not smooth in cloud size — it falls off a
# cliff once a role has more than ~900 points, since H2 has to consider every
# triple. Measured on this data, PCA-50, per role:
#
#       points   maxdim=1   maxdim=2
#          200     0.01 s      0.37 s
#          900     0.20 s     46.11 s
#         1200     0.39 s    ~700    s
#
# At the 240-question budget (1200 points/role) that is ~55 h for the panel
# against ~0.03 h, so the question-budget sweep runs this at 1.
#
# Setting it to 1 DOES cost four panel metrics — H2_total_persistence,
# H2_max_lifetime, H2_max_lifetime_frac, persistence_entropy_H2. A note here
# used to claim "H2 is NOT in PANEL_COLS, so raising this changes no panel
# metric"; that is wrong, all four are in PANEL_COLS and reach the ladder, and
# H2_total_persistence carries r = -0.560 against axis_proj at 40 questions.
# `betti_from_diagrams` enumerates whatever dimensions ripser returns and
# `geometry_columns` keeps only columns actually present, so a lower value
# degrades cleanly to a shorter panel rather than failing.
MAXDIM = int(os.environ.get("MP_RIPSER_MAXDIM", 2))


def _diameter(X: np.ndarray) -> float:
    """Max pairwise distance, computed without materialising the full matrix
    twice. 200 points makes this trivially cheap."""
    from scipy.spatial.distance import pdist
    d = pdist(X)
    return float(d.max()) if d.size else 0.0


def persistence(X: np.ndarray) -> dict:
    """Vietoris-Rips H0/H1 of one cloud. Returns diagrams plus scale.

    ``X`` is expected already PCA-reduced (the caller uses PCA-50, matching
    plan #1's D_AMBIENT) — ripser on the full 2048-dim points would be both
    slower and dominated by directions carrying no variance.
    """
    from ripser import ripser as _ripser
    X = np.asarray(X, dtype=np.float64)
    dgms = _ripser(X, maxdim=MAXDIM)["dgms"]
    return {"dgms": dgms, "diameter": _diameter(X)}


def betti_from_diagrams(dgms, diameter: float,
                        frac: float = LIFETIME_FRAC) -> dict:
    """Count long-lived features per homology dimension, plus lifetime summaries.

    H0's single infinite bar (the component that never dies) is counted, so a
    connected cloud gives Betti-0 = 1 rather than 0. Infinite deaths are
    replaced by the diameter, at which scale every point is connected to every
    other and the filtration is complete.
    """
    thr = frac * diameter
    out = {}
    for dim, d in enumerate(dgms):
        if len(d) == 0:
            out[f"betti{dim}"] = 0
            out[f"H{dim}_total_persistence"] = 0.0
            out[f"H{dim}_max_lifetime"] = 0.0
            out[f"H{dim}_max_lifetime_frac"] = 0.0
            out[f"persistence_entropy_H{dim}"] = 0.0
            continue
        birth, death = d[:, 0].copy(), d[:, 1].copy()
        death[~np.isfinite(death)] = diameter      # the essential class
        life = death - birth
        out[f"betti{dim}"] = int((life > thr).sum())
        out[f"H{dim}_total_persistence"] = float(life.sum())
        out[f"H{dim}_max_lifetime"] = float(life.max())
        # Scale-free companion to max_lifetime. Previously computed only for H1,
        # and only in metrics.py; done here so every dimension gets one and the
        # rule lives with the thing it normalises.
        out[f"H{dim}_max_lifetime_frac"] = (float(life.max() / diameter)
                                            if diameter > 0 else np.nan)
        out[f"persistence_entropy_H{dim}"] = persistence_entropy(life)
    return out


def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of the normalised bar lifetimes.

    p_i = life_i / sum(life), then -sum p_i log p_i. Says whether persistence is
    CONCENTRATED in a few long bars or SPREAD over many short ones — a
    shape-of-the-barcode statistic that no count or lifetime captures.

    A cloud with one dominant loop and a haze of noise scores low; one with many
    comparable features scores high (up to log n for n equal bars). It is
    scale-free: multiplying every lifetime by a constant leaves the p_i
    unchanged, which is what makes it safe to compare across roles whose
    diameters differ 3x.
    """
    life = np.asarray(lifetimes, float)
    life = life[np.isfinite(life) & (life > 0)]
    if life.size == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def topology_metrics(X: np.ndarray, frac: float = LIFETIME_FRAC) -> tuple:
    """(metrics dict, diagrams) for one role's PCA-reduced cloud.

    The panel consumes the metrics; the caller persists the diagrams so the
    threshold stays re-cuttable.
    """
    p = persistence(X)
    m = betti_from_diagrams(p["dgms"], p["diameter"], frac)
    m["cloud_diameter"] = p["diameter"]
    return m, p["dgms"]
