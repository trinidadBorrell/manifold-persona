"""Density and sampling: how are a role's points SPREAD inside its cloud?

Every other family describes the shape the points trace. This one describes the
points themselves — are they laid down evenly, or clumped with empty regions
between them? Two clouds can have identical dimension, spectrum and curvature
while one is uniform and the other is a handful of tight knots.

EVERYTHING HERE IS COMPUTED ON THE CLOUD RESCALED TO UNIT RMS RADIUS
--------------------------------------------------------------------
This is load-bearing, not tidiness. A k-nearest-neighbour distance scales
linearly with the cloud, and a log-density in d dimensions shifts by -d*log(s)
under a rescale by s — at d = 50 that is a factor of 50 on the log scale. Left
raw, both columns would be a restatement of `log_var`, which already correlates
+0.731 with `axis_proj`; the ladder would then show a strong, entirely
meaningless relationship.

Normalising first makes them scale-free BY CONSTRUCTION rather than by a linear
control applied afterwards. `test_scale_invariance.py` asserts it rather than
trusting this comment. The unnormalised mean is still stored as
`knn_dist_abs_mean` so the normalisation can be checked, but it is deliberately
kept out of PANEL_COLS.

WHY kNN DISTANCE AND KDE BOTH, WHEN THEY MEASURE THE SAME THING
---------------------------------------------------------------
They do not measure it equally well here. **Kernel density estimation is not
reliable above roughly 6-8 dimensions** and this is 50: the volume of a
kernel's support grows so fast that essentially no sample falls inside it, and
the estimate degenerates towards a spike at each data point. Scott's rule makes
this explicit — at n = 200, d = 50 it gives a bandwidth factor of
n^(-1/(d+4)) = 0.90, i.e. almost no smoothing at all.

So the bandwidth here is the cloud's own median k-NN distance instead, which at
least adapts to the data, and the KDE column is reported WITH the warning
attached rather than as an equal. `knn_dist_*` is the density estimate that
survives high dimension (it is the Loftsgaarden-Quesenberry estimator in
disguise: density ~ k / (n * V_d * r_k^d)), and it is the one to trust. If the
two disagree in the ladder, the kNN one is right.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

K_NEIGHBOURS = 10          # matches metrics.LOCAL_K, so patches are the same size


def unit_rms(X: np.ndarray) -> np.ndarray:
    """Centre and rescale a cloud to RMS radius 1. See the module docstring."""
    Xc = np.asarray(X, float)
    Xc = Xc - Xc.mean(0)
    r = np.sqrt((Xc ** 2).sum() / len(Xc))
    return Xc / max(r, 1e-12)


def knn_distances(X: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour (self excluded).

    Small where points are packed, large where they are isolated — so this is an
    inverse density, one value per point. Euclidean, matching every other
    within-role metric (see md/SPACES.md).
    """
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, -1]                      # column 0 is the point itself, at 0


def kde_logdensity(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian KDE log-density at each point, evaluated on the sample itself.

    Read the module docstring before using this: at 50 dimensions it is not a
    trustworthy density estimate, and it is here because it was asked for and
    because its disagreement with `knn_dist_*` is itself informative.
    """
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=max(bandwidth, 1e-6), kernel="gaussian").fit(X)
    return kde.score_samples(X)


def density_metrics(X: np.ndarray, k: int = K_NEIGHBOURS) -> tuple:
    """(scalar summaries, per-point values) for one cloud.

    The summaries go in the panel; the per-point arrays are saved by the caller
    so the family's `distributions.png` can show what the summaries threw away.
    """
    P = unit_rms(X)
    knn = knn_distances(P, k)
    knn_abs = knn_distances(np.asarray(X, float) - np.asarray(X, float).mean(0), k)
    bw = float(np.median(knn))           # data-adaptive; see the docstring
    logd = kde_logdensity(P, bw)

    mean_knn = float(np.mean(knn))
    out = {
        "knn_dist_mean": mean_knn,
        # cv, not sd: the mean is ~1 by construction after normalising, but the
        # ratio is the scale-free statement and stays honest if k changes.
        "knn_dist_cv": float(np.std(knn) / mean_knn) if mean_knn > 0 else np.nan,
        "knn_dist_abs_mean": float(np.mean(knn_abs)),
        "kde_logdens_mean": float(np.mean(logd)),
        # sd, not cv: log-density is signed and routinely crosses zero, so a
        # coefficient of variation would blow up on the sign change.
        "kde_logdens_sd": float(np.std(logd)),
        "_kde_bandwidth": bw,
    }
    return out, {"knn_dist": knn, "kde_logdens": logd}
