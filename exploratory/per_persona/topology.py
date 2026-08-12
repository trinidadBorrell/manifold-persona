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

import numpy as np

LIFETIME_FRAC = 0.10       # feature counts iff lifetime > 10% of cloud diameter
MAXDIM = 1                 # H0 and H1 only; H2 on 200 pts x 276 roles is unaffordable


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
            continue
        birth, death = d[:, 0].copy(), d[:, 1].copy()
        death[~np.isfinite(death)] = diameter      # the essential class
        life = death - birth
        out[f"betti{dim}"] = int((life > thr).sum())
        out[f"H{dim}_total_persistence"] = float(life.sum())
        out[f"H{dim}_max_lifetime"] = float(life.max())
    return out


def topology_metrics(X: np.ndarray, frac: float = LIFETIME_FRAC) -> tuple:
    """(metrics dict, diagrams) for one role's PCA-reduced cloud.

    The panel consumes the metrics; the caller persists the diagrams so the
    threshold stays re-cuttable.
    """
    p = persistence(X)
    m = betti_from_diagrams(p["dgms"], p["diameter"], frac)
    m["cloud_diameter"] = p["diameter"]
    return m, p["dgms"]
