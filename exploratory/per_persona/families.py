"""Which family each metric belongs to — the single source of truth.

The panel grew to 30-odd columns added at different times for different reasons,
and as a flat list it hides the one thing a reader needs: which metrics are
alternative measurements of the SAME property and which are independent
evidence. Five dimension estimators agreeing is one finding, not five.

Seven families. The first five are geometry; 06 (extraction design) and 07
(cloud alignment) are not, and are labelled so.

A family may carry an optional ``"predictors"`` key restricting which closeness
measures its figures are drawn against. Only 07 uses it, because two of its
metrics ARE two of the predictors — see that entry.

Nothing else in the study hardcodes a family list. `figures_families.py` asserts
that every ladder metric (``metrics.PANEL_COLS`` plus
``study_ladder.EXTRA_METRICS``) lands in exactly one family, so a metric added
later cannot silently vanish from every family figure.
"""
from __future__ import annotations

FAMILIES = {
    "intrinsic_dimension": {
        "n": 1,
        "title": "Intrinsic dimensionality",
        "question": "How many directions does this persona's behaviour actually use?",
        "metrics": ["TwoNN", "MLE", "PCA_dim_95pct",
                    "PCA_participation_ratio"],
        "reading":
            "Four estimates of ONE number, d, and they do not agree: MLE and "
            "TwoNN read neighbour distances and go negative, PCA_dim_95pct is a "
            "cumulative-variance count and goes negative, participation ratio "
            "is a moment of the whole spectrum and goes POSITIVE. Read the "
            "family as one contested finding, never as four votes. This is also "
            "the family that collapses hardest on the cloud-level closeness "
            "measures. `lPCA`, `PCA_dim_90pct` and `local_id_mean`/`_cv` were "
            "removed 2026-08-04.",
    },
    "spectral_shape": {
        "n": 2,
        "title": "Spectral shape",
        "question": "How is variance distributed across directions — evenly, or "
                    "concentrated in a few?",
        "metrics": ["effective_rank", "eig_decay_exponent"],
        "reading":
            "Split out of intrinsic dimensionality because neither counts "
            "directions: they describe the SHAPE of the eigenvalue spectrum. "
            "They also behave unlike the dimension estimators — `effective_rank` "
            "flips sign between the raw and controlled rungs (-0.108 to +0.308), "
            "which no genuine dimension estimator here does.",
    },
    "density_sampling": {
        "n": 3,
        "title": "Density and sampling",
        "question": "How are the points spread inside the cloud — uniformly, or "
                    "clumped with empty regions?",
        "metrics": ["knn_dist_mean", "knn_dist_cv",
                    "kde_logdens_mean", "kde_logdens_sd"],
        "reading":
            "The only family that asks about the POINTS rather than the shape "
            "they trace. Every column is computed on the cloud rescaled to unit "
            "RMS radius, because raw kNN distance and raw log-density are pure "
            "cloud size — unnormalised they would simply re-measure `log_var`. "
            "Trust `knn_dist_*` over `kde_logdens_*`: kernel density estimation "
            "is not reliable above ~6-8 dimensions and this is 50.",
    },
    "topology": {
        "n": 4,
        "title": "Topology (TDA)",
        "question": "Does the cloud have holes — loops it encircles, voids it "
                    "surrounds?",
        "metrics": ["H0_total_persistence",
                    "H1_total_persistence", "H1_max_lifetime",
                    "H1_max_lifetime_frac",
                    "H2_total_persistence", "H2_max_lifetime",
                    "H2_max_lifetime_frac",
                    "persistence_entropy_H0", "persistence_entropy_H1",
                    "persistence_entropy_H2"],
        "reading":
            "The only family that can tell a disc from an annulus: two clouds "
            "can have identical dimension, spread and spectrum while one has a "
            "hole through it. Vietoris-Rips only — the alpha complex was "
            "dropped because it needs a Delaunay triangulation, which is "
            "infeasible past ~7 dimensions. The thresholded counts betti0/1/2 "
            "were REMOVED 2026-08-04 — betti1 was 0 for 270 of 275 roles, "
            "betti2 for all of them, and betti0 was a sparsity measure wearing "
            "a topology name. What remains is continuous and threshold-free. "
            "The H2 columns are all design-explained: void structure here is "
            "the extraction grid, not the persona.",
    },
    "curvature": {
        "n": 5,
        "title": "Curvature",
        "question": "Is the manifold bent, and which way?",
        "metrics": ["orc_mean", "orc_sd", "frc_mean", "frc_sd"],
        "reading":
            "Two DISCRETE curvatures of the 10-nearest-neighbour graph, defined "
            "edge by edge with no surface fitted anywhere. Positive means "
            "neighbourhoods overlap more than flat space allows; negative means "
            "they diverge. They are not on a common scale (orc sits near 0, frc "
            "near -28), so only their ORDERING across roles is comparable, and "
            "they disagree in sign — read that as evidence about how fragile "
            "curvature is on this data, not as two confirmations. "
            "`spline_r2` and `curvature_gain` were REMOVED 2026-08-04: with 40 "
            "k-means anchors on a 40-question grid the anchors recovered the "
            "questions (ARI 0.65-0.96), so the spline re-expressed the design "
            "(r = +0.93 with quest_frac, -0.99 with MLE). Note the same "
            "confound touches what is left — orc_mean falls from +0.596 to "
            "+0.107 once quest_frac and cloud scale are controlled.",
    },
    "extraction_design": {
        "n": 6,
        "title": "Extraction design (NOT geometry)",
        "question": "How much of a role's within-cloud variance is forced by the "
                    "5x40 grid we built, rather than by the persona?",
        "metrics": ["instr_frac", "quest_frac", "interaction_frac"],
        "reading":
            "These are not geometry and must never be quoted as if they were. "
            "They are kept because `interaction_frac` is the number that says a "
            "per-persona manifold exists at all — it is 17.7% here against 0.6% "
            "on the prompt cloud, which is why this study is possible and the "
            "earlier one was a negative result.",
    },
    "alignment_metrics": {
        "n": 7,
        "title": "Cloud alignment (also predictors)",
        "question": "How much of `default`'s question-similarity structure "
                    "survives the persona?",
        "metrics": ["mknn_align", "cka"],
        # These two are ALSO two of the four predictors, so most of the
        # predictor grid is a self-pair (r = 1 by construction) and would draw
        # as a blank ladder row and a perfect diagonal scatter. The family is
        # therefore restricted to the two CENTROID-level predictors, which is
        # exactly the comparison the family exists to make: does where a role
        # sits predict how its cloud is organised?
        "predictors": ["axis_proj", "cos_centroid"],
        "reading":
            "Not geometry — these are closeness measures read backwards, as "
            "properties of the role's 40-question cloud. Added 2026-08-05. The "
            "family's whole content is how far each row travels from raw to "
            "controlled: `cka` +0.756 -> +0.315 and `mknn_align` +0.727 -> "
            "+0.221 against `axis_proj`, so roughly three quarters of the "
            "apparent agreement between centroid position and cloud "
            "organisation was cloud size and mean-vector length. Both hold "
            "about twice as well against `cos_centroid` (+0.453, +0.467), which "
            "is what the construction predicts: both are scale-invariant by "
            "build and both reference `default`'s centroid, which is what "
            "`cos_centroid` measures and what `axis_proj` does not. The pair's "
            "own correlation (+0.909 -> +0.785, the strongest row in the whole "
            "ladder) is the calibration that says the controls are not simply "
            "flattening everything; it is drawn in "
            "`global/fig13_closeness_as_metric`, not here. See ANALYSIS.md 5b.",
    },
}

# Metrics computed and stored but deliberately NOT in PANEL_COLS: they would add
# ladder rows without adding information. Recorded here so they are documented
# rather than merely absent.
STORED_NOT_PANELLED = {
    "knn_dist_abs_mean": "kNN distance in original units — pure cloud size, kept "
                         "only so the normalisation can be checked",
    "orc_p10": "Ollivier-Ricci 10th percentile — the tail the mean hides",
    "orc_p90": "Ollivier-Ricci 90th percentile",
    "cloud_diameter": "filtration scale for the topology columns",
    "eig_decay_r2": "R^2 of the log-log line whose slope IS eig_decay_exponent "
                    "— says whether that slope is fitted to something actually "
                    "straight. 0.905-0.991 across roles; 107 of 276 below 0.95.",
    "eig_decay_p": "significance of that slope against flat",
    "H0_max_lifetime": "identical to cloud_diameter by construction — H0's "
                       "longest bar IS the essential class, drawn to the "
                       "diameter. Dropped from the panel 2026-08-04.",
    "H0_max_lifetime_frac": "constant 1.0 for the same reason. Dropped.",
    "rms_radius": "scale covariate",
    "cos_centroid_raw": "uncentred centroid cosine — see closeness.py",
    "cka_cosine": "cosine-kernel CKA — r = 0.997 with the linear one",
    "PCA_dim_90pct": "same cumulative-variance rule as PCA_dim_95pct at a "
                     "different cutoff. Still computed (ID_COLS is the "
                     "historical six) but dropped from the panel 2026-08-04.",
    "spline_r2": "R^2 of a thin-plate spline through 40 k-means anchors. "
                 "Removed 2026-08-04 — the 40 anchors recover the 40 questions "
                 "(ARI 0.65-0.96), so it re-expressed quest_frac (r = +0.926) "
                 "and MLE (r = -0.987). No longer computed.",
    "curvature_gain": "spline_r2 minus a flat 3-D plane's R^2. Removed with it; "
                      "also a repo-local construction with no citation.",
    "betti0": "1 + the count of MST edges longer than 10% of the diameter — a "
              "SPARSITY measure, not topology. Removed from the panel 2026-08-04.",
    "betti1": "loops surviving the 10% rule: 0 for 270 of 275 roles. Removed.",
    "betti2": "voids surviving the 10% rule: 0 for every role. Removed.",
    "lPCA": "count of eigenvalues above 5% of the largest — a threshold count, "
            "not a dimension, and fitted globally despite the 'local' in its "
            "name. Dropped from the panel 2026-08-04.",
}

# Families whose metrics are summaries of a per-point or per-edge distribution.
# These get an extra `distributions.png` showing what the summary threw away.
POINTWISE = {
    "density_sampling": {"knn_dist": "distance to the 10th nearest neighbour",
                         "kde_logdens": "KDE log-density"},
    "curvature": {"orc": "Ollivier-Ricci curvature (per edge)",
                  "frc": "Forman-Ricci curvature (per edge)"},
}


def ordered() -> list:
    """(key, spec) pairs in family order."""
    return sorted(FAMILIES.items(), key=lambda kv: kv[1]["n"])


def folder(key: str) -> str:
    """Directory name for a family: `01_intrinsic_dimension`, ... — numbered so
    the reading order survives an alphabetical file listing."""
    return f"{FAMILIES[key]['n']:02d}_{key}"


def of_metric(metric: str) -> str | None:
    for k, spec in FAMILIES.items():
        if metric in spec["metrics"]:
            return k
    return None


def check_coverage(panel_cols) -> None:
    """Every panel column in exactly one family, and no family inventing one.

    Loud on purpose. The failure this prevents is a metric added to
    ``metrics.PANEL_COLS`` months from now that quietly appears in the global
    ladder and in no family figure at all.
    """
    assigned = [m for spec in FAMILIES.values() for m in spec["metrics"]]
    dupes = {m for m in assigned if assigned.count(m) > 1}
    missing = [m for m in panel_cols if m not in assigned]
    unknown = [m for m in assigned if m not in panel_cols]
    problems = []
    if dupes:
        problems.append(f"in more than one family: {sorted(dupes)}")
    if missing:
        problems.append(f"in PANEL_COLS but no family: {missing}")
    if unknown:
        problems.append(f"in a family but not in PANEL_COLS: {unknown}")
    if problems:
        raise ValueError("families.py is out of sync with metrics.PANEL_COLS — "
                         + "; ".join(problems))
