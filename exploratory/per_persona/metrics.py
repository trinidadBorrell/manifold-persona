"""The geometry panel: every metric computed for ONE point cloud.

Extracted from the old ``05_geometry_panel.py``, which held three things at once
— the positive control, the real panel and the design null. All three run the
identical function on different clouds, and that identity is the whole reason
the controls mean anything, so it lives in one shared module and the three
callers (`calib_estimators.py`, `study_panel.py`, `study_design_null.py`) import
it rather than each owning a copy.

The panel covers six intrinsic-dimension estimators plus thirteen more metrics
spanning topology, curvature, spectral shape, local heterogeneity and the
extraction-design decomposition — because "assistant-like roles have
lower-dimensional manifolds" and "assistant-like roles have geometrically
different manifolds" are different claims, and only a panel can tell them apart.

``PANEL_COLS`` is the single source of truth for which columns are geometry.
Downstream scripts import it instead of re-deriving the list from a JSON file's
key order.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from manifold.idim import id_estimates
from manifold.pipeline import fit_manifold
from manifold.tps import reconstruction

from common import design_fractions, pca_stats
from stats_utils import linfit
from curvature import curvature_metrics
from density import density_metrics
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

# Grouped to match families.py, which is the authority on which family a column
# belongs to. `families.check_coverage(PANEL_COLS)` fails loudly if the two ever
# drift apart, so this list and that mapping cannot silently disagree.
# Computed but NOT panelled (2026-08-04, user's request). They stay in ID_COLS
# because that is the historical six `id_vs_axis.py` reported and the ladder's
# regression check compares against them; they are simply not treated as panel
# metrics any more.
#
#   PCA_dim_90pct  same cumulative-variance rule as PCA_dim_95pct at another
#                  cutoff — two cutoffs of one rule are not two measurements.
#   lPCA           a THRESHOLD COUNT, not a dimension: eigenvalues above 5% of
#                  the largest. Its "local" name is misleading (it is fitted
#                  globally here), and it reads the same eigenvalues as
#                  PCA_dim_95pct under a different rule.
DROPPED_FROM_PANEL = ("PCA_dim_90pct", "lPCA")
PANEL_COLS = [c for c in ID_COLS if c not in DROPPED_FROM_PANEL] + [
    # --- spectral shape ---
    "eig_decay_exponent", "effective_rank",
    # --- density and sampling ---
    "knn_dist_mean", "knn_dist_cv", "kde_logdens_mean", "kde_logdens_sd",
    # --- topology: lifetimes and barcode shape, per homology dim ---
    # The thresholded counts betti0/1/2 were removed 2026-08-04: betti1 was 0
    # for 270 of 275 roles and betti2 for all of them, and betti0 was never
    # topology at all (it is 1 + the count of long MST edges, a sparsity
    # measure wearing a topology name). The continuous columns below carry the
    # signal and need no threshold.
    "H0_total_persistence",
    "H1_total_persistence", "H1_max_lifetime", "H1_max_lifetime_frac",
    "H2_total_persistence", "H2_max_lifetime", "H2_max_lifetime_frac",
    "persistence_entropy_H0", "persistence_entropy_H1", "persistence_entropy_H2",
    # --- curvature: discrete, on the kNN graph ---
    "orc_mean", "orc_sd", "frc_mean", "frc_sd",
    # --- extraction design (NOT geometry) ---
    "instr_frac", "quest_frac", "interaction_frac",
]


def _spectrum(Xc: np.ndarray) -> dict:
    """Decay exponent and effective rank of the covariance spectrum.

    Two shape statistics that are NOT dimension counts: the decay exponent says
    how fast variance falls off across directions (a steep power law means a few
    directions dominate), and the effective rank (exp of the spectral entropy)
    says how evenly variance is spread. Both are continuous, so unlike
    PCA_dim_90pct they cannot be pinned to the design's additive rank.
    """
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = lam[lam > 0]
    if lam.size < 3:
        return {"eig_decay_exponent": np.nan, "effective_rank": np.nan}
    m = min(SPECTRUM_RANKS, lam.size)
    x, y = np.log(np.arange(1, m + 1)), np.log(lam[:m])
    # The exponent is the SLOPE of a straight line through a log-log spectrum,
    # so it is only an exponent if the spectrum is actually a power law. That
    # was never checked until 2026-08-04: measured across the 276 roles the fit
    # R^2 runs 0.905-0.991, with 107 roles below 0.95. `eig_decay_r2` carries
    # that per role so the exponent can be read with its own trustworthiness,
    # and `eig_decay_p` tests the slope against flat. Neither is a panel metric
    # -- they are diagnostics of another metric, not measurements of a persona.
    fit = linfit(x, y)
    p = lam / lam.sum()
    return {"eig_decay_exponent": fit["slope"],
            "eig_decay_r2": fit["r2"],
            "eig_decay_p": fit["p"],
            "effective_rank": float(np.exp(-(p * np.log(p)).sum()))}


def _curvature(P: np.ndarray) -> dict:
    """Spline R^2 and curvature gain of one cloud, in its own PCA-50 space.

    Mirrors plan #1's construction exactly — anchors are group means, the fitted
    TPS surface is scored on ALL the cloud's raw points, and the flat baseline is
    a k-dim PCA plane on the same points with the same fixed mean, so the gain is
    curvature and not extra fitting freedom in the denominator.

    Anchors are 40 k-means centroids rather than the 40 question-means, so that
    real roles, design-null clouds and planted control clouds are treated
    IDENTICALLY: the planted clouds have no question factor, and a metric whose
    definition changed between the real data and its own control would be
    uncontrolled.
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
    """Every panel metric for ONE cloud.

    Returns ``(metrics, diagrams, pointwise)``:

      metrics    the scalar columns that go in the panel
      diagrams   persistence diagrams, only when ``keep_diagrams``
      pointwise  per-point and per-edge arrays (knn_dist, kde_logdens, orc,
                 frc) that the scalar columns are summaries
                 of. The family `distributions.png` figures need these — a
                 metric whose MEAN is flat while its SPREAD widens is a real
                 finding that no scalar column can show.

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
    # distances, which is exactly where that floor dominates, so on raw points
    # TwoNN reads a true d=3 as 8-11 and MLE as 4.1-4.5 (see CALIBRATION.md).
    # Projecting onto the top-variance subspace strips most of the floor.
    #
    # Why a FIXED count and not "retain 90/95% of variance": a variance fraction
    # is not the same operation on different clouds. On the planted control
    # clouds PCA-90% keeps ~9 components; on real role clouds it keeps a median
    # of 43 (range 32-82). A control run that way would validate an operation
    # the real data never receives, and the retained count would itself become a
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

    # Topology. `topology_metrics` now returns every homology dimension up to
    # MAXDIM=2 including the scale-free lifetime fractions and the barcode
    # entropies, so nothing needs deriving here any more.
    #
    # The thresholded counts (betti1, betti2) are ~0 on this cloud: the longest
    # H1 bar reaches 9.4% of diameter and the longest H2 bar ~2%, both under the
    # 10% rule. The continuous columns they derive from do vary, need no
    # threshold, and are what the ladder actually uses; the counts stay as
    # descriptive columns and the ladder flags them degenerate if constant.
    topo, dgms = topology_metrics(P)
    out.update({k: v for k, v in topo.items()
                if k in PANEL_COLS or k == "cloud_diameter"})

    # `_curvature` (spline_r2, curvature_gain) is NO LONGER CALLED — removed
    # from the panel 2026-08-04 after it was traced to the extraction grid.
    # N_ANCHORS is 40 and the grid has 40 questions, so the k-means anchors
    # recover the question groups almost exactly (ARI 0.65-0.96 per role). The
    # spline surface therefore explains between-question variance by
    # construction and `spline_r2` came out at r = +0.926 with `quest_frac` and
    # r = -0.987 with `MLE` — the same measurement, not an independent one.
    # The function is kept below so the diagnostic can be re-run.
    dens, dens_pw = density_metrics(P, k=LOCAL_K)
    out.update({k: v for k, v in dens.items() if not k.startswith("_")})
    curv, curv_pw = curvature_metrics(P, k=LOCAL_K)
    out.update(curv)

    if instr is not None and quest is not None:
        out.update(design_fractions(X, instr, quest))
    else:
        out.update({"instr_frac": np.nan, "quest_frac": np.nan,
                    "interaction_frac": np.nan})

    # scale covariates — these are controls, not responses. NOTE these are the
    # only quantities here computed on the RAW cloud rather than the PCA-50
    # working space; md/SPACES.md records that asymmetry.
    out["log_var"] = float(np.log((Xc ** 2).sum()))
    out["rms_radius"] = float(np.sqrt((Xc ** 2).sum() / len(X)))

    pointwise = {**dens_pw, **curv_pw}
    return out, (dgms if keep_diagrams else None), pointwise


def geometry_columns(df) -> list:
    """The panel columns present in `df` and carrying variance.

    ``betti1`` is identically 0 on this cloud (plan amendment A4), so it is a
    constant column that a scaler would drop anyway; excluding it here makes the
    predictor list explicit instead of implicit.
    """
    return [c for c in PANEL_COLS if c in df.columns and df[c].std() > 0]
