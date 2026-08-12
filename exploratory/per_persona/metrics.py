"""The geometry panel: every metric computed for ONE point cloud.

Extracted from the old ``05_geometry_panel.py``, which held three things at
once — the positive control, the real panel and the design null. All three run
the identical function on different clouds. That identity is the whole reason
the controls mean anything. So the function lives in one shared module, and
the three callers (`calib_estimators.py`, `study_panel.py`,
`study_design_null.py`) import it rather than each owning a copy.

The panel covers six intrinsic-dimension estimators plus thirteen more
metrics. They span topology, curvature, spectral shape, local heterogeneity
and the extraction-design decomposition. "Assistant-like roles have
lower-dimensional manifolds" and "assistant-like roles have geometrically
different manifolds" are different claims. Only a panel can tell them apart.

``PANEL_COLS`` is the single source of truth for which columns are geometry.
Downstream scripts import it instead of re-deriving the list from a JSON file's
key order.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from manifold.idim import id_estimates
from manifold.local_id import local_id, cv
from manifold.pipeline import fit_manifold
from manifold.tps import reconstruction

from common import design_fractions, pca_stats
from topology import topology_metrics

# --- fixed by the plan; not swept ------------------------------------------
D_PCA = 50           # working space per role, matching plan #1's D_AMBIENT
K_INTRINSIC = 3      # TPS intrinsic dim, matching plan #1 so curvature compares
N_ANCHORS = 40       # k-means centroids used as spline anchors (see below)
LOCAL_K = 10         # neighbours for the local-ID patches
SPECTRUM_RANKS = 50  # ranks used for the log-log decay fit
SEED = 0

# The six dimension estimators `id_vs_axis.py` tests, in its order, so the
# regression check in the ladder can compare like with like.
ID_COLS = ["TwoNN", "MLE", "lPCA", "PCA_participation_ratio",
           "PCA_dim_90pct", "PCA_dim_95pct"]

PANEL_COLS = ID_COLS + [
    "betti0", "betti1", "H1_total_persistence", "H1_max_lifetime",
    "H1_max_lifetime_frac",
    "spline_r2", "curvature_gain",
    "eig_decay_exponent", "effective_rank",
    "local_id_mean", "local_id_cv",
    "instr_frac", "quest_frac", "interaction_frac",
]


def _spectrum(Xc: np.ndarray) -> dict:
    """Decay exponent and effective rank of the covariance spectrum.

    Two shape statistics that are NOT dimension counts. The decay exponent
    says how fast variance falls off across directions (a steep power law
    means a few directions dominate). The effective rank (exp of the spectral
    entropy) says how evenly variance is spread. Both are continuous, so
    unlike PCA_dim_90pct they cannot be pinned to the design's additive rank.
    """
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = lam[lam > 0]
    if lam.size < 3:
        return {"eig_decay_exponent": np.nan, "effective_rank": np.nan}
    m = min(SPECTRUM_RANKS, lam.size)
    x, y = np.log(np.arange(1, m + 1)), np.log(lam[:m])
    slope = float(np.polyfit(x, y, 1)[0])
    p = lam / lam.sum()
    return {"eig_decay_exponent": slope,
            "effective_rank": float(np.exp(-(p * np.log(p)).sum()))}


def _curvature(P: np.ndarray) -> dict:
    """Spline R^2 and curvature gain of one cloud, in its own PCA-50 space.

    Mirrors plan #1's construction exactly. Anchors are group means. The
    fitted TPS surface is scored on ALL the cloud's raw points. The flat
    baseline is a k-dim PCA plane on the same points with the same fixed
    mean. So the gain is curvature, not extra fitting freedom in the
    denominator.

    Anchors are 40 k-means centroids rather than the 40 question-means. Real
    roles, design-null clouds and planted control clouds are then treated
    IDENTICALLY. The planted clouds have no question factor, and a metric
    whose definition changed between the real data and its own control would
    be uncontrolled.
    """
    n_anchor = min(N_ANCHORS, len(P) - 1)
    km = KMeans(n_clusters=n_anchor, n_init=4, random_state=SEED).fit(P)
    anchors = km.cluster_centers_
    gmean = P.mean(0)
    try:
        mani = fit_manifold(anchors, k=K_INTRINSIC, seed=SEED)
        spline_r2 = float(reconstruction(P, mani, gmean).r2)
    except Exception as e:  # noqa: BLE001 — TPS solve is brittle on degenerate clouds
        print(f"    [curvature] TPS failed: {e}")
        return {"spline_r2": np.nan, "curvature_gain": np.nan}
    pca = PCA(n_components=min(K_INTRINSIC, P.shape[1]), random_state=SEED).fit(P)
    recon = pca.inverse_transform(pca.transform(P))
    ssr = float(np.sum((P - recon) ** 2))
    tss = float(np.sum((P - gmean) ** 2))
    return {"spline_r2": spline_r2, "curvature_gain": spline_r2 - (1.0 - ssr / tss)}


def panel_metrics(X: np.ndarray, instr=None, quest=None,
                  keep_diagrams: bool = False) -> tuple:
    """Every panel metric for ONE cloud. Returns (metrics, persistence diagrams).

    ``instr``/``quest`` are optional: the design fractions only exist for clouds
    that actually sit on the two-factor grid (real roles and design-null draws).
    Planted control clouds have no grid, and get NaN there rather than a
    fabricated value.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    out = {}

    # ONE shared working space for every geometric metric: a FIXED PCA-50.
    #
    # Why denoise at all: the role clouds carry an isotropic noise floor spread
    # over all 2048 dimensions. Neighbour-based estimators measure small
    # distances, which is exactly where that floor dominates. On raw points
    # TwoNN reads a true d=3 as 8-11 and MLE as 4.1-4.5 (see
    # docs/notes/calibration-history.md).
    # Projecting onto the top-variance subspace strips most of the floor.
    #
    # Why a FIXED count and not "retain 90/95% of variance": a variance fraction
    # is not the same operation on different clouds. On the planted control
    # clouds PCA-90% keeps ~9 components; on real role clouds it keeps a median
    # of 43 (range 32-82). A control run that way would validate an operation
    # the real data never receives. The retained count would also become a
    # per-role variable confounded with cloud structure. A fixed 50 applies the
    # identical map to real, design-null and planted clouds alike.
    #
    # Why 50 specifically: plan #1 fixed D_AMBIENT=50 for the between-role
    # manifold, so the whole repo shares one ambient convention.
    d = min(D_PCA, min(X.shape) - 1)
    P = PCA(n_components=d, random_state=SEED).fit_transform(X)

    out.update(id_estimates(P))
    out.update(pca_stats(P))
    out.update(_spectrum(P - P.mean(0)))
    topo, dgms = topology_metrics(P)
    out.update({k: v for k, v in topo.items() if k in PANEL_COLS
                or k in ("cloud_diameter",)})
    # Continuous topology readout. The thresholded betti1 is identically 0 on
    # this cloud (max H1 lifetime reaches only 9.4% of diameter, under the 10%
    # rule), so the count has no variance to correlate with anything. The
    # lifetime fraction it is derived from does vary, needs no threshold, and
    # is what the ladder actually uses. betti0/betti1 stay in the panel as
    # descriptive columns.
    out["H1_max_lifetime_frac"] = (out["H1_max_lifetime"] / out["cloud_diameter"]
                                   if out.get("cloud_diameter") else np.nan)
    out.update(_curvature(P))

    lid = local_id(P, k=LOCAL_K)
    out["local_id_mean"] = float(np.nanmean(lid))
    out["local_id_cv"] = float(cv(lid))

    if instr is not None and quest is not None:
        out.update(design_fractions(X, instr, quest))
    else:
        out.update({"instr_frac": np.nan, "quest_frac": np.nan,
                    "interaction_frac": np.nan})

    # scale covariates — these are controls, not responses
    out["log_var"] = float(np.log((Xc ** 2).sum()))
    out["rms_radius"] = float(np.sqrt((Xc ** 2).sum() / len(X)))
    return out, (dgms if keep_diagrams else None)


def geometry_columns(df) -> list:
    """The panel columns present in `df` and carrying variance.

    ``betti1`` is identically 0 on this cloud, so it is a constant column that
    a scaler would drop anyway. Excluding it here makes the predictor list
    explicit instead of implicit.
    """
    return [c for c in PANEL_COLS if c in df.columns and df[c].std() > 0]
