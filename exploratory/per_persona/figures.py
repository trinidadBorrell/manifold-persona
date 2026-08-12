"""Every figure for the geometry-vs-axis run.

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Outputs table)

Grouped the way the study is: fig02-fig05, fig07, fig08, fig10 and fig11 are the
STUDY; fig09 is the design CONFOUND.

fig00/fig01 (the estimator calibration and topology controls) were removed on
2026-08-04 at the user's request. `calib_estimators.py` still produces their data
if run on its own; nothing here reads it any more.

Figure titles no longer carry an EXPLORATORY tag (user request, 2026-07-31). The
fence now rests on `"exploratory": true` in every data file, the no-verdict
banner at the top of REPORT.md, and the plan itself.

Every plot function is defensive (repo convention): a failure logs and returns
so it cannot destroy a run that has already produced its numbers.

Usage:
    .venv/bin/python exploratory/per_persona/figures.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import C_REAL, C_DESIGN, C_GAUSS, C_INSTR, C_QUEST, C_INTER
from stats_utils import linfit, fmt_p
from study_ladder import EXTRA_METRICS

DPI = 300
# The six strongest distinct correlates that remain. `curvature_gain`,
# `spline_r2` and `local_id_mean` were all here until 2026-08-04 and were all
# removed; the list is now the strongest survivor from each of four families.
# Four, not six. `orc_sd` and `interaction_frac` were dropped 2026-08-05:
# interaction_frac is not geometry (and rests on a saturated design — see
# md/ANALYSIS.md), and orc_sd falls to +0.077 once that term is controlled.
HEADLINE = ["H0_total_persistence", "MLE", "persistence_entropy_H0",
            "H1_total_persistence"]

# The core metrics the user singled out (2026-08-04): the pure-geometry columns,
# no topology and no design fractions. Ordered shape-first, then the dimension
# estimators, then spectrum shape. `lPCA` was in this list until later the same
# day, when it was dropped along with `PCA_dim_90pct` and the local-ID pair.
CORE_METRICS = ["TwoNN", "MLE", "PCA_dim_95pct", "PCA_participation_ratio",
                "effective_rank", "eig_decay_exponent",
                "orc_mean", "orc_sd"]

# Spelled-out y-axis labels: the column names are terse and a reader coming to a
# figure cold cannot tell that e.g. `curvature_gain` is a difference of two R^2s.
YLABELS = {
    "MLE": "intrinsic dimension (MLE)\ncalibrated, valid 3 ≤ d ≤ 10",
    "interaction_frac": "interaction share of\nwithin-role variance",
    "H1_total_persistence": "total H1 persistence\n(topological 'loopiness')",
    "H0_total_persistence": "total H0 persistence\n(total MST edge length)",
    "persistence_entropy_H0": "persistence entropy H0\n(concentrated vs spread bars)",
    "orc_mean": "Ollivier-Ricci curvature\n(mean over kNN-graph edges)",
    "orc_sd": "Ollivier-Ricci curvature\n(spread over kNN-graph edges)",
    "frc_mean": "Forman-Ricci curvature\n(mean over kNN-graph edges)",
    "frc_sd": "Forman-Ricci curvature\n(spread over kNN-graph edges)",
    "quest_frac": "question share of\nwithin-role variance",
    "instr_frac": "instruction share of\nwithin-role variance",
    "TwoNN": "intrinsic dimension (TwoNN)\nUNCALIBRATED",
    "eig_decay_exponent": "eigenvalue decay exponent",
    "effective_rank": "effective rank\n(exp of spectral entropy)",
    "mknn_align": "mknn_align\nshared nearest neighbours with `default`",
    "cka": "cka\nsame question-similarity structure as `default`",
}

# Must match study_ladder.RUNGS. Three rungs since response length was dropped
# (2026-08-03): mean-pooled activations divide the token count out, so length
# was never a mechanism that moved a role along the axis.
RUNGS = [("r_raw", "raw"), ("r_ctrl_logvar", "| log_var"),
         ("r_ctrl_all", "| log_var + mean_norm")]
RUNG_COLORS = [C_GAUSS, C_INSTR, C_REAL]
RUNG_OFFSET = 0.17

# The raw-only view of the same data: one dot, one CI, no rungs. Useful when the
# question is "how big is this correlation" rather than "does it survive its
# controls" — the three-rung version answers the second and makes the first
# harder to read.
RUNGS_RAW = [("r_raw", "raw (uncontrolled)")]
RAW_COLORS = [C_REAL]

# Hidden from the fig02 forest (user request, 2026-08-05). The three design
# fractions are not geometry — they are the variance decomposition of the
# prompt grid, and on the forest they read as the strongest "metrics" on the
# board. They still exist in the ladder CSV and in fig09; this only removes
# them from the forest panels.
DROP_FROM_FOREST = ("instr_frac", "quest_frac", "interaction_frac",
                    # Added as ladder ROWS on 2026-08-05 (see study_ladder.
                    # EXTRA_METRICS). Kept off the geometry forest for two
                    # reasons: they are closeness measures, not geometry, and
                    # each is degenerate in its own panel, which would leave a
                    # blank cell a reader has to decode. fig13 is theirs.
                    "mknn_align", "cka")


def grid_shape(n: int) -> tuple:
    """(nrow, ncol) for `n` panels, chosen to leave as few blanks as possible.

    Up to four panels go in a SINGLE ROW — a 4-metric family drawn on a 3-wide
    grid wastes a cell and, worse, splits four things that should be compared
    side by side across two rows. Beyond that, pick whichever of 3 or 4 columns
    leaves fewer empty cells, breaking ties toward 4 (wider, fewer rows).
    """
    if n <= 4:
        return 1, max(n, 1)
    best = min((( -(-n // c) * c - n, -c, c) for c in (3, 4)))
    ncol = best[2]
    return -(-n // ncol), ncol


def _offsets(rungs):
    """Vertical offset per rung, centred on the row, and where the CI bar goes."""
    m = len(rungs)
    off = [(j - (m - 1) / 2) * RUNG_OFFSET for j in range(m)]
    return off, off[-1]

# The four predictors are four ways of asking "how Assistant-like is this
# role?", and they are NOT interchangeable. Spelled out on the panels because
# the bare column names hide what a reader must know: the first two collapse a
# role to one point, the last two compare its whole 40-question response cloud
# against `default`'s. All four run the same way (high = Assistant-like).
PRED_TITLES = {
    "axis_proj":
        "axis_proj  —  position along the Assistant Axis\n"
        "$\\mathrm{mean}(role)\\cdot\\hat{a}$  =  magnitude × direction\n"
        "centroid-level · high = Assistant-like",
    "cos_centroid":
        "cos_centroid  —  angle to `default`'s centroid\n"
        "mean-centred across the 276 roles\n"
        "centroid-level · high = Assistant-like",
    "mknn_align":
        "mknn_align  —  shared nearest neighbours\n"
        "does the role order the 40 questions like `default`?\n"
        "cloud-level · high = Assistant-like",
    "cka":
        "cka  —  centred kernel alignment\n"
        "same question-similarity structure as `default`\n"
        "cloud-level · high = Assistant-like",
}


def defensive(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            print(f"  [fig] {fn.__name__} FAILED (logged, run continues):")
            traceback.print_exc()
    return wrap


def _save(fig, name, run_dir, group: str = ""):
    """Write a cross-family figure under figures/global/.

    `group` puts the variants of one figure number in their own folder — fig03
    has four (one per closeness measure), fig02 has four (forest/axisproj x
    full/raw). Loose files for figures that have only one version.
    """
    p = Path(run_dir) / "figures" / "global" / group / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p.name)


# --------------------------------------------------------------------------- #
# STUDY                                                                        #
# --------------------------------------------------------------------------- #
@defensive
def fig02_forest(lad, run_dir, L, rungs=RUNGS, colors=RUNG_COLORS,
                 sort_on="r_ctrl_all", name="fig02_ladder_forest",
                 null_col="shuffle_max_abs_r_p95", preds=None,
                 drop=DROP_FROM_FOREST):
    """The centrepiece: does each metric's correlation survive its controls?

    One panel per closeness measure, rows aligned, so a metric that only holds
    under one definition of "close to the Assistant" is visible as a row that
    moves between panels.

    With ``rungs=RUNGS_RAW`` this becomes the raw-only view: one dot and its CI,
    no controls, metrics ordered by raw |r|.

    ``preds`` restricts which closeness measures get a panel (default: all four,
    in the ladder's own order) and ``drop`` which rows are hidden. The two are
    coupled in practice: a metric that is also a predictor is degenerate in its
    own panel, so showing `cka`/`mknn_align` as rows only makes sense once the
    panels are restricted to the centroid-level pair. See the fig02c call.
    """
    ci = rungs[-1][0].removeprefix("r_")           # CI belongs to the last rung
    offs, ci_y = _offsets(rungs)
    avail = list(dict.fromkeys(lad.predictor))     # ladder's own order
    preds = avail if preds is None else [p for p in avail if p in preds]
    sub = lad[(lad.predictor == preds[0]) & (~lad.degenerate.fillna(False))
              & (~lad.metric.isin(drop))]
    order = sub.reindex(sub[sort_on].abs().sort_values().index)["metric"].tolist()
    fig, axes = plt.subplots(1, len(preds), sharey=True,
                             figsize=(5.7 * len(preds), 0.42 * len(order) + 4.2))
    for ax, pred in zip(np.atleast_1d(axes), preds):
        s = lad[(lad.predictor == pred)].set_index("metric").reindex(order)
        thr = float(s[null_col].dropna().iloc[0]) if \
            s[null_col].notna().any() else np.nan
        y = np.arange(len(order))
        if np.isfinite(thr):
            ax.axvspan(-thr, thr, color="0.90", zorder=0,
                       label="axis-shuffle null (95th pct of max|r|)")
        ax.axvline(0, color="k", lw=0.8, zorder=1)
        for j, (col, lab) in enumerate(rungs):
            ax.scatter(s[col], y + offs[j], s=26, color=colors[j], zorder=3,
                       label=lab if pred == preds[0] else None)
        ax.hlines(y + ci_y, s[f"ci_lo_{ci}"], s[f"ci_hi_{ci}"],
                  color=colors[-1], lw=1.6, zorder=2)
        for v in (-0.30, 0.30):
            ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("correlation with predictor")
        ax.set_title(PRED_TITLES.get(pred, pred), fontsize=9.5, linespacing=1.35)
        ax.set_xlim(-1, 1)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    # No suptitle (user request, 2026-08-05): the per-panel titles already name
    # the predictor, and the filename names the variant. The freed space goes
    # to the rows, which is what anyone is here to read.
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, f"{name}_L{L}.png", run_dir, "fig02_ladder")


@defensive
def fig02b_forest_axisproj(lad, run_dir, L, rungs=RUNGS, colors=RUNG_COLORS,
                           sort_on="r_ctrl_all", name="fig02b_ladder_axisproj",
                           null_col="shuffle_max_abs_r_p95"):
    """The ladder restricted to axis_proj alone.

    fig02 shows three predictors side by side, which is the right object when the
    question is "is this about direction or magnitude?". It is the wrong object
    when someone just wants to read the result: three panels of near-identical
    rows invite the eye to compare panels instead of reading rungs. This is the
    same data, one predictor, wide enough to label.
    """
    # Was used to hide betti0/betti1 from this view; both were removed from the
    # panel entirely on 2026-08-04, so it is empty. Kept as the hook for
    # suppressing a row here without touching the panel.
    DROP_FROM_AXISPROJ = ()
    ci = rungs[-1][0].removeprefix("r_")
    offs, ci_y = _offsets(rungs)
    s = lad[(lad.predictor == "axis_proj")
            & (~lad.degenerate.fillna(False))
            & (~lad.metric.isin(DROP_FROM_AXISPROJ))].copy()
    order = s.reindex(s[sort_on].abs().sort_values().index)["metric"].tolist()
    s = s.set_index("metric").reindex(order)
    thr = float(s[null_col].dropna().iloc[0]) if \
        s[null_col].notna().any() else np.nan
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 0.46 * len(order) + 2.6))
    if np.isfinite(thr):
        ax.axvspan(-thr, thr, color="0.90", zorder=0,
                   label=f"axis-shuffle null (95th pct of max|r| = {thr:.2f})")
    ax.axvline(0, color="k", lw=0.8, zorder=1)
    for v in (-0.30, 0.30):
        ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
    ax.text(0.30, len(order) - 0.2, " |r| = 0.30", color=C_DESIGN, fontsize=8, va="top")

    for j, (col, lab) in enumerate(rungs):
        ax.scatter(s[col], y + offs[j], s=34, color=colors[j], zorder=3, label=lab)
    ax.hlines(y + ci_y, s[f"ci_lo_{ci}"], s[f"ci_hi_{ci}"],
              color=colors[-1], lw=1.8, zorder=2)

    # Value labels on the last rung — the number people actually quote.
    for yi, v in zip(y, s[rungs[-1][0]]):
        ax.annotate(f"{v:+.2f}", (v, yi + ci_y), fontsize=7.5,
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color=colors[-1], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("correlation with axis_proj   (high = Assistant-like)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, f"{name}_L{L}.png", run_dir, "fig02_ladder")


@defensive
def fig03_scatter(panel, lad, run_dir, L, pred="axis_proj", name="fig03"):
    """The six headline metrics as scatter, against one closeness measure.

    Drawn once per predictor (fig03, fig03b, fig03c, fig03d) so the reader can
    see the same six relationships under each definition of "close to the
    Assistant" rather than trusting that they look alike.
    """
    metrics = [m for m in HEADLINE if m in panel.columns]
    d = panel[panel.role != "default"]
    dflt = panel[panel.role == "default"]
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.7 * ncol, 4.0 * nrow))
    for ax, m in zip(axes.ravel(), metrics):
        row = lad[(lad.predictor == pred) & (lad.metric == m)]
        ax.scatter(d[pred], d[m], s=11, alpha=0.45, color=C_REAL, linewidths=0)
        f = linfit(d[pred], d[m])
        sl, ic = f["slope"], f["intercept"]
        xs = np.linspace(d[pred].min(), d[pred].max(), 50)
        ax.plot(xs, sl * xs + ic, color=C_DESIGN, lw=2)
        if len(dflt):
            dx, dy = float(dflt[pred].iloc[0]), float(dflt[m].iloc[0])
            if not (xs[0] <= dx <= xs[-1]):
                ex = np.linspace(min(xs[0], dx), max(xs[-1], dx), 50)
                ax.plot(ex, sl * ex + ic, ls=":", lw=1.2, color=C_DESIGN)
            ax.scatter([dx], [dy], s=110, marker="*", color="#111111", zorder=6,
                       label="default (excluded from fit)")
            ax.legend(fontsize=7, loc="best")
        r = row.iloc[0] if len(row) else None
        # BH q over every test in the controlled rung, not a raw p: this figure
        # is one of 4 predictors x 31 metrics and a raw p would be read as if
        # it were the only test run.
        ax.set_title(f"{m}\nraw r={r.r_raw:+.3f}  fully-controlled={r.r_ctrl_all:+.3f}"
                     f"  ({fmt_p(r.get('q_global_ctrl_all'), 'q')})"
                     if r is not None else m, fontsize=9)
        ax.set_xlabel(f"{pred}  (high = Assistant-like)", fontsize=8)
        ax.set_ylabel(YLABELS.get(m, m), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f"Headline metrics vs assistant-likeness — {pred}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99 - 0.08 / nrow])
    _save(fig, f"{name}_headline_scatter_{pred}_L{L}.png", run_dir, "fig03_headline")


@defensive
def fig04_design_null(nullj, run_dir, L):
    metrics = [m for m in nullj["real"] if nullj["real"][m]["median"] is not None]
    order = sorted(metrics, key=lambda m: not nullj["design_explained"].get(m, False))
    fig, ax = plt.subplots(figsize=(9, 0.40 * len(order) + 2.4))
    for i, m in enumerate(order):
        r, n = nullj["real"][m], nullj["design_null"][m]
        rng = max(abs(r["median"]), abs(n["median"]), 1e-9)
        ax.hlines(i + 0.16, n["q25"] / rng, n["q75"] / rng, color=C_DESIGN, lw=6,
                  alpha=.65)
        ax.hlines(i - 0.16, r["q25"] / rng, r["q75"] / rng, color=C_REAL, lw=6,
                  alpha=.85)
        ax.plot(n["median"] / rng, i + 0.16, "|", color="k", ms=9)
        ax.plot(r["median"] / rng, i - 0.16, "|", color="k", ms=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([m + ("  [DESIGN-EXPLAINED]" if
                             nullj["design_explained"].get(m) else "") for m in order],
                       fontsize=8)
    ax.set_xlabel("IQR, scaled per metric (real vs persona-free design null)")
    ax.plot([], [], lw=6, color=C_REAL, label="real 276 roles")
    ax.plot([], [], lw=6, color=C_DESIGN, label="design null (100 draws)")
    ax.legend(fontsize=8)
    fig.suptitle("Real vs design null", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"fig04_design_null_L{L}.png", run_dir)


@defensive
def fig05_regression(reg, run_dir, L):
    """Geometry against the boring baseline: how big and how long is the cloud?

    The baseline used to be response length. It is now cloud scale — log_var and
    mean_norm — which is a far harder baseline to beat, since mean_norm is a
    factor of the target by construction (axis_proj = mean_norm x cos_axis).
    """
    names = ["M0_intercept", "M1_scale_only", "M2_geometry_only",
             "M3_geometry_plus_scale"]
    lbl = ["intercept\nonly", "scale\nonly", "geometry\nonly", "geometry\n+ scale"]
    vals = [reg["models"][n]["cv_r2"] for n in names]
    lo = [reg["models"][n]["cv_r2_ci"][0] for n in names]
    hi = [reg["models"][n]["cv_r2_ci"][1] for n in names]
    ins = [reg["models"][n]["in_sample_r2"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(names))
    ax.bar(x - 0.19, vals, 0.36, color=C_REAL, label="cross-validated R²",
           yerr=[np.array(vals) - lo, np.array(hi) - np.array(vals)],
           capsize=4, ecolor="0.3")
    ax.bar(x + 0.19, ins, 0.36, color=C_GAUSS, alpha=.8, label="in-sample R² (overfits)")
    ax.axhline(0.25, color=C_DESIGN, ls=":", lw=1.5, label="plan's 'would matter' bar")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=9)
    ax.set_ylabel("R² predicting assistant-axis position")
    ax.legend(fontsize=8)
    fig.suptitle("Geometry vs cloud scale", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"fig05_regression_L{L}.png", run_dir)


@defensive
def fig11_core_metrics(panel, lad, run_dir, L, pred="axis_proj"):
    """The nine core metrics on one page, against one closeness measure.

    fig03 shows six headline metrics chosen for being the strongest correlates.
    This is the flat list the user asked for instead: the pure-geometry columns,
    no topology and no design fractions, in a fixed order that does not depend
    on the result. Same page, same x-axis, so the nine are directly comparable.
    """
    metrics = [m for m in CORE_METRICS if m in panel.columns]
    d = panel[panel.role != "default"]
    dflt = panel[panel.role == "default"]
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, sharex=True, squeeze=False,
                             figsize=(4.9 * ncol, 4.1 * nrow))
    axes = np.atleast_2d(axes)
    for ax, m in zip(axes.ravel(), metrics):
        row = lad[(lad.predictor == pred) & (lad.metric == m)]
        ax.scatter(d[pred], d[m], s=12, alpha=0.45, color=C_REAL, linewidths=0)
        f = linfit(d[pred], d[m])
        sl, ic = f["slope"], f["intercept"]
        xs = np.linspace(d[pred].min(), d[pred].max(), 50)
        ax.plot(xs, sl * xs + ic, color=C_DESIGN, lw=2)
        if len(dflt):
            dx, dy = float(dflt[pred].iloc[0]), float(dflt[m].iloc[0])
            if not (xs[0] <= dx <= xs[-1]):
                ex = np.linspace(min(xs[0], dx), max(xs[-1], dx), 50)
                ax.plot(ex, sl * ex + ic, ls=":", lw=1.2, color=C_DESIGN)
            ax.scatter([dx], [dy], s=120, marker="*", color="#111111", zorder=6)
        r = row.iloc[0] if len(row) else None
        ax.set_title(f"{m}\nraw r={r.r_raw:+.3f}   controlled={r.r_ctrl_all:+.3f}"
                     f"   ({fmt_p(r.get('q_global_ctrl_all'), 'q')})"
                     if r is not None else m, fontsize=9)
        ax.set_ylabel(YLABELS.get(m, m), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{pred}   (high = Assistant-like)", fontsize=9)
    fig.suptitle(f"The core geometry metrics vs {pred}\n"
                 "★ = `default`, excluded from every fit; dotted = fit extrapolated "
                 "to reach it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"fig11_core_metrics_{pred}_L{L}.png", run_dir, "fig11_core_metrics")


@defensive
def fig12_metric_clusters(panel, lad, run_dir, L, n_clusters=4):
    """Do the metric FAMILIES match what the data is actually measuring?

    The families say where a number came from (a dimension estimator, a
    persistence diagram, a curvature). They are a description of METHOD. This
    asks the empirical question instead: which metrics move together across the
    275 roles? Correlation matrix, hierarchically clustered, families marked
    down the side so the mismatch is visible rather than asserted.

    Design fractions are excluded — this is the geometry panel only, matching
    md/ANALYSIS.md.
    """
    import families as F
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    design = set(F.FAMILIES["extraction_design"]["metrics"])
    usable = lad[~lad.degenerate.fillna(False)]
    # The ladder now also carries the closeness measures as rows (study_ladder.
    # EXTRA_METRICS). They belong to no family and are not geometry, so they
    # cannot join a clustering of the geometry panel — F.of_metric would return
    # None and the family colour strip would fail. Dropped by name, not by
    # "has no family", so a genuine coverage gap still crashes here.
    cols = [m for m in dict.fromkeys(usable.metric)
            if m not in design and m not in EXTRA_METRICS]
    d = panel[panel.role != "default"]
    C = d[cols].corr()

    Z = linkage(squareform((1 - C.abs()).values, checks=False), method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    lab = fcluster(Z, n_clusters, "maxclust")
    cols_o = [cols[i] for i in order]
    lab_o = [lab[i] for i in order]
    Co = C.loc[cols_o, cols_o]

    ax_r = (usable[usable.predictor == "axis_proj"]
            .set_index("metric")["r_ctrl_all"].reindex(cols_o))
    fam_of = {m: F.of_metric(m) for m in cols_o}
    fam_colors = {k: plt.cm.tab10(i) for i, (k, _) in enumerate(F.ordered())}

    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 11), gridspec_kw={"width_ratios": [1, 14, 3]})

    # family colour strip — the taxonomy, for comparison against the clustering
    axes[0].imshow([[0]], visible=False)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(len(cols_o) - .5, -.5)
    for i, m in enumerate(cols_o):
        axes[0].add_patch(plt.Rectangle((0, i - .5), 1, 1,
                                        color=fam_colors[fam_of[m]]))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("family", fontsize=9)

    im = axes[1].imshow(Co.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(cols_o)))
    axes[1].set_xticklabels(cols_o, rotation=90, fontsize=7.5)
    axes[1].set_yticks(range(len(cols_o)))
    axes[1].set_yticklabels(cols_o, fontsize=7.5)
    # cluster boundaries
    for i in range(1, len(lab_o)):
        if lab_o[i] != lab_o[i - 1]:
            axes[1].axhline(i - .5, color="k", lw=2)
            axes[1].axvline(i - .5, color="k", lw=2)
    plt.colorbar(im, ax=axes[1], fraction=.025, label="Pearson r across 275 roles")
    axes[1].set_title(f"metric-by-metric correlation, clustered into {n_clusters} groups",
                      fontsize=10)

    y = np.arange(len(cols_o))
    axes[2].barh(y, ax_r.values, color=[C_REAL if v < 0 else C_DESIGN
                                        for v in ax_r.values])
    axes[2].axvline(0, color="k", lw=.8)
    for v in (-0.30, 0.30):
        axes[2].axvline(v, color="0.6", ls=":", lw=1)
    axes[2].set_ylim(len(cols_o) - .5, -.5); axes[2].set_yticks([])
    axes[2].set_xlim(-1, 1); axes[2].set_xlabel("r with axis_proj\n(controlled)", fontsize=9)
    axes[2].set_title("effect", fontsize=9)

    handles = [plt.Line2D([], [], marker="s", ls="", color=fam_colors[k],
                          label=spec["title"]) for k, spec in F.ordered()
               if k != "extraction_design"]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Do the metric families match what is actually being measured?\n"
                 "black squares = empirical clusters · colour strip = the family "
                 "taxonomy · they do not line up", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    _save(fig, f"fig12_metric_clusters_L{L}.png", run_dir)


@defensive
def fig10_closeness_measures(panel, agree, run_dir, L):
    """The four closeness measures against each other.

    fig02 shows what each measure CORRELATES WITH. This shows what each measure
    IS: its distribution over the 276 roles, and how far it agrees with the
    other three. Without it, four ladder panels that look alike could be four
    views of one number and a reader could not tell.
    """
    preds = list(PRED_TITLES)
    d = panel[panel.role != "default"]
    dflt = panel[panel.role == "default"]
    n = len(preds)
    fig, axes = plt.subplots(n, n, figsize=(3.1 * n, 3.0 * n))
    for i, yi in enumerate(preds):
        for j, xj in enumerate(preds):
            ax = axes[i, j]
            if i == j:
                ax.hist(d[yi], bins=32, color=C_REAL, alpha=.85)
                if len(dflt):
                    ax.axvline(float(dflt[yi].iloc[0]), color="#111111", lw=1.6)
                ax.set_yticks([])
                ax.set_title(yi, fontsize=10, fontweight="bold")
            elif i > j:
                ax.scatter(d[xj], d[yi], s=9, alpha=.45, color=C_REAL, linewidths=0)
                f = linfit(d[xj], d[yi])
                xs = np.linspace(d[xj].min(), d[xj].max(), 40)
                ax.plot(xs, f["slope"] * xs + f["intercept"], color=C_DESIGN, lw=1.5)
                ax.set_title(fmt_p(f["p"]), fontsize=7.5, color="0.35")
                if len(dflt):
                    ax.scatter([float(dflt[xj].iloc[0])], [float(dflt[yi].iloc[0])],
                               s=110, marker="*", color="#111111", zorder=6)
            else:
                r = float(agree["pearson"][xj][yi])
                rho = float(agree["spearman"][xj][yi])
                # Colour by agreement so the two-family split is visible from
                # across the room rather than by reading twelve numbers.
                ax.set_facecolor(plt.cm.Blues(0.12 + 0.55 * abs(r)))
                ax.text(.5, .58, f"r = {r:+.3f}", ha="center", va="center",
                        fontsize=15, fontweight="bold", transform=ax.transAxes)
                ax.text(.5, .34, f"rho = {rho:+.3f}", ha="center", va="center",
                        fontsize=11, color="0.25", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            if i == n - 1 and i != j:
                ax.set_xlabel(xj, fontsize=9)
            if j == 0 and i != j:
                ax.set_ylabel(yi, fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("The four closeness measures — distributions, pairwise agreement, "
                 "and where `default` sits (★ / vertical line)\n"
                 "centroid-level: axis_proj, cos_centroid   ·   "
                 "cloud-level: mknn_align, cka", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"fig10_closeness_measures_L{L}.png", run_dir)


@defensive
def fig07_magnitude_direction(panel, run_dir, L):
    d = panel[panel.role != "default"]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    s = ax.scatter(d.axis_proj, d.cos_axis, c=d.mean_norm, s=26, cmap="magma",
                   linewidths=0)
    plt.colorbar(s, ax=ax, label="‖mean(role)‖  (magnitude)")
    r = np.corrcoef(d.axis_proj, d.cos_axis)[0, 1]
    ax.set_xlabel("axis_proj  =  ‖mean(role)‖ × cos_axis   (mixes magnitude + direction)")
    ax.set_ylabel("cos_axis   (direction only)")
    ax.annotate(f"r = {r:+.3f}", xy=(0.03, 0.95), xycoords="axes fraction",
                fontsize=11, fontweight="bold")
    fig.suptitle("Magnitude vs direction", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"fig07_magnitude_vs_direction_L{L}.png", run_dir)


@defensive
def fig08_families(fam, run_dir, L, pred="axis_proj", name="fig08"):
    """Family-level trend, against one closeness measure.

    The 15 Ward families are fixed; only the x-axis changes between fig08 and
    its b/c/d variants. If the per-role trend is real it should survive being
    aggregated to 15 groups AND survive the choice of closeness measure.
    """
    x = f"mean_{pred}"
    fs = pd.DataFrame(fam["families"]).sort_values(x, ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, col, lbl in ((axes[0], "median_MLE", "median MLE (calibrated dimension)"),
                         (axes[1], "median_orc_mean", "median Ollivier-Ricci curvature")):
        ax.scatter(fs[x], fs[col], s=fs.n_roles * 6, alpha=.7,
                   color=C_REAL, linewidths=0)
        for _, r in fs.iterrows():
            ax.annotate(r.example_roles[0], (r[x], r[col]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
        f = linfit(fs[x], fs[col])
        xs = np.linspace(fs[x].min(), fs[x].max(), 40)
        ax.plot(xs, f["slope"] * xs + f["intercept"], color=C_DESIGN, lw=1.6)
        ax.set_xlabel(f"family mean {pred}")
        ax.set_ylabel(lbl)
        sp = fam["family_rank_vs_geometry"][pred][col]
        # n = 15 families here, not 275 roles — a low p is much easier to miss
        # at this sample size, so both tests are shown.
        ax.set_title(f"Spearman rho = {sp['spearman_rho']:+.3f} "
                     f"({fmt_p(sp['p'])})   |   OLS {fmt_p(f['p'])}, n={f['n']}",
                     fontsize=8.5)
    fig.suptitle(f"Role families — ranked by {pred}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, f"{name}_families_{pred}_L{L}.png", run_dir, "fig08_families")


# --------------------------------------------------------------------------- #
# CONFOUND                                                                     #
# --------------------------------------------------------------------------- #
@defensive
def fig09_contrast(con, run_dir, L):
    """Variance decomposition of the RESPONSE cloud only.

    Earlier revisions put the 25-point prompt cloud beside it. That comparison
    now lives in the report's table instead: this run's analysis is entirely on
    the 5x40 response cloud, and showing a second cloud invited the figure to be
    read as a result about both. IQR whiskers across the 276 roles replace the
    second cloud as the thing that gives each bar context.
    """
    c_ = con["clouds"].get("response_5x40", {"error": 1})
    if "error" in c_:
        print("  [fig] fig09 skipped: response cloud unavailable")
        return
    terms = ["instr_frac", "quest_frac", "interaction_frac"]
    labels = ["instruction\nphrasing", "question", "interaction\n(persona × situation)"]
    cols = [C_INSTR, C_QUEST, C_INTER]
    med = [c_[t]["median"] for t in terms]
    lo = [c_[t]["median"] - c_[t]["q25"] for t in terms]
    hi = [c_[t]["q75"] - c_[t]["median"] for t in terms]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(terms))
    ax.bar(x, med, 0.6, color=cols, yerr=[lo, hi], capsize=6, ecolor="0.25")
    for xi, vi, h in zip(x, med, hi):
        ax.text(xi, vi + h + .022, f"{vi:.3f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("share of within-role variance\n(median over 276 roles, "
                  "whiskers = IQR)", fontsize=10)
    ax.set_ylim(0, max(m + h for m, h in zip(med, hi)) * 1.18)
    ax.axhline(0, color="k", lw=.8)
    fig.suptitle("Within-role variance, response cloud", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"fig09_response_variance_L{L}.png", run_dir)


@defensive
def fig13_closeness_as_metric(lad, run_dir, L, rungs=RUNGS, colors=RUNG_COLORS):
    """`mknn_align` and `cka` read as METRICS rather than as predictors.

    The same three rungs the geometry panel gets. The point of the figure is the
    DISTANCE each row travels from grey to blue: if the cloud-level measures
    agreed with the centroid-level ones about anything other than cloud size,
    the rungs would sit on top of each other. They do not.

    The two ladder-only rows (each measure against the other) are drawn as the
    reference band: that pair keeps +0.785 under the same controls, so the
    erosion elsewhere is not the controls destroying everything they touch.
    """
    METRICS = list(EXTRA_METRICS)
    ci = rungs[-1][0].removeprefix("r_")
    offs, ci_y = _offsets(rungs)
    pairs = [(m, p) for m in METRICS
             for p in ["axis_proj", "cos_centroid", "mknn_align", "cka"]
             if p != m]
    s = lad.set_index(["metric", "predictor"])
    rowsp = [(m, p) for m, p in pairs if (m, p) in s.index
             and not bool(s.loc[(m, p), "degenerate"])]
    if not rowsp:
        print("  [fig] fig13 skipped: cloud measures not in the ladder")
        return
    # Centroid-level predictors first, then the measure-vs-measure reference.
    rowsp.sort(key=lambda t: (t[1] in METRICS, t[0]))
    labels = [f"{m}\nvs {p}" for m, p in rowsp]
    y = np.arange(len(rowsp))[::-1].astype(float)

    thr_col = "shuffle_max_abs_r_p95"
    fig, ax = plt.subplots(figsize=(10.5, 0.85 * len(rowsp) + 2.8))
    thr = float(lad[thr_col].dropna().iloc[0]) if lad[thr_col].notna().any() \
        else np.nan
    if np.isfinite(thr):
        ax.axvspan(-thr, thr, color="0.90", zorder=0,
                   label=f"axis-shuffle null (95th pct of max|r| ≈ {thr:.2f})")
    ax.axvline(0, color="k", lw=0.8, zorder=1)
    for v in (-0.30, 0.30):
        ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
    # Separates the two centroid-level blocks from the measure-vs-measure pair.
    n_cent = sum(1 for _, p in rowsp if p not in METRICS)
    if 0 < n_cent < len(rowsp):
        ax.axhline(y[n_cent] + 0.5, color="0.6", lw=0.8, ls="--")

    for j, (col, lab) in enumerate(rungs):
        v = [s.loc[(m, p), col] for m, p in rowsp]
        ax.scatter(v, y + offs[j], s=48, color=colors[j], zorder=3, label=lab)
    ax.hlines(y + ci_y,
              [s.loc[(m, p), f"ci_lo_{ci}"] for m, p in rowsp],
              [s.loc[(m, p), f"ci_hi_{ci}"] for m, p in rowsp],
              color=colors[-1], lw=2.0, zorder=2)
    for yi, (m, p) in zip(y, rowsp):
        raw, ctl = s.loc[(m, p), "r_raw"], s.loc[(m, p), rungs[-1][0]]
        ax.annotate(f"{raw:+.2f} → {ctl:+.2f}", (ctl, yi + ci_y), fontsize=8,
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    color=colors[-1], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, linespacing=1.3)
    ax.set_ylim(-0.7, len(rowsp) - 0.3)
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("correlation  (metric vs predictor)")
    ax.set_title("The cloud-level measures read as metrics\n"
                 "how much of their agreement with the centroid-level "
                 "measures is cloud size?", fontsize=10, linespacing=1.35)
    # Below the axes, not inside it: the bottom two rows sit at r ≈ 0.8, which
    # is exactly where an in-axes legend lands.
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, f"fig13_closeness_as_metric_L{L}.png", run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    args = ap.parse_args()
    run, L = Path(args.outdir), args.label_layer
    D = run / "data"
    j = lambda n: json.load(open(D / n))          # noqa: E731

    panel = pd.read_csv(D / f"per_role_panel_L{L}.csv")
    lad = pd.read_csv(D / f"ladder_L{L}.csv")
    nullj = j(f"design_null_L{L}.json")
    reg, fam = j(f"regression_L{L}.json"), j(f"families_L{L}.json")
    con = j(f"prompt_vs_response_L{L}.json")
    agree = j(f"predictor_agreement_L{L}.json")

    # One suffix per closeness measure. axis_proj keeps the bare figure number;
    # the rest are b/c/d in the order the user asked for, which is NOT the
    # ladder's predictor order — the predictor name is in every filename so the
    # two can never be confused.
    suffixed = [("axis_proj", ""), ("cos_centroid", "b"),
                ("cka", "c"), ("mknn_align", "d")]
    suffixed = [(p, s) for p, s in suffixed if p in set(lad.predictor)]

    fig02_forest(lad, run, L)
    fig02b_forest_axisproj(lad, run, L)
    # Same two plots, raw correlation only: one dot, one CI, no control rungs.
    fig02_forest(lad, run, L, rungs=RUNGS_RAW, colors=RAW_COLORS, sort_on="r_raw",
                 name="fig02raw_ladder_forest",
                 null_col="shuffle_max_abs_r_p95_raw")
    fig02b_forest_axisproj(lad, run, L, rungs=RUNGS_RAW, colors=RAW_COLORS,
                           sort_on="r_raw", name="fig02braw_ladder_axisproj",
                           null_col="shuffle_max_abs_r_p95_raw")
    # The same forest with `mknn_align` and `cka` restored as ROWS (user
    # request, 2026-08-05). Restricted to the two CENTROID-level predictors:
    # each cloud measure is degenerate in its own panel, and the comparison the
    # figure is for is centroid position vs cloud organisation. The design
    # fractions stay hidden.
    fig02_forest(lad, run, L, name="fig02c_ladder_forest_alignment",
                 preds=["axis_proj", "cos_centroid"],
                 drop=tuple(m for m in DROP_FROM_FOREST
                            if m not in EXTRA_METRICS))
    for pred, sfx in suffixed:
        fig03_scatter(panel, lad, run, L, pred=pred, name=f"fig03{sfx}")
    fig04_design_null(nullj, run, L)
    fig05_regression(reg, run, L)
    for pred, sfx in suffixed:
        fig11_core_metrics(panel, lad, run, L, pred=pred)
    fig12_metric_clusters(panel, lad, run, L)
    fig10_closeness_measures(panel, agree, run, L)
    fig07_magnitude_direction(panel, run, L)
    for pred, sfx in suffixed:
        fig08_families(fam, run, L, pred=pred, name=f"fig08{sfx}")
    fig09_contrast(con, run, L)
    fig13_closeness_as_metric(lad, run, L)
    print("figures done")


if __name__ == "__main__":
    main()
