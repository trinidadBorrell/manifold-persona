"""Every figure for the geometry-vs-axis run.

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Outputs table)

Grouped the way the study is: fig00/fig01 are the CALIBRATION controls,
fig02-fig05 and fig07/fig08 are the STUDY, fig09 is the design CONFOUND.

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

DPI = 300
HEADLINE = ["curvature_gain", "local_id_mean", "MLE", "spline_r2",
            "interaction_frac", "H1_total_persistence"]

# Spelled-out y-axis labels: the column names are terse and a reader coming to a
# figure cold cannot tell that e.g. `curvature_gain` is a difference of two R^2s.
YLABELS = {
    "curvature_gain": "curvature gain\n(spline R² − flat-plane R²)",
    "local_id_mean": "mean local intrinsic dimension\n(10-NN patches)",
    "MLE": "intrinsic dimension (MLE)\ncalibrated, valid 3 ≤ d ≤ 10",
    "spline_r2": "spline R²\n(fit of a curved surface)",
    "interaction_frac": "interaction share of\nwithin-role variance",
    "H1_total_persistence": "total H1 persistence\n(topological 'loopiness')",
    "quest_frac": "question share of\nwithin-role variance",
    "instr_frac": "instruction share of\nwithin-role variance",
    "betti0": "Betti-0 (long-lived components)",
    "TwoNN": "intrinsic dimension (TwoNN)\nUNCALIBRATED",
    "local_id_cv": "local-ID coefficient of variation",
    "eig_decay_exponent": "eigenvalue decay exponent",
    "effective_rank": "effective rank\n(exp of spectral entropy)",
}

# Must match study_ladder.RUNGS. Three rungs since response length was dropped
# (2026-08-03): mean-pooled activations divide the token count out, so length
# was never a mechanism that moved a role along the axis.
RUNGS = [("r_raw", "raw"), ("r_ctrl_logvar", "| log_var"),
         ("r_ctrl_all", "| log_var + mean_norm")]
RUNG_COLORS = [C_GAUSS, C_INSTR, C_REAL]
RUNG_OFFSET = 0.17
LAST_RUNG_Y = (len(RUNGS) - 1 - 1.0) * RUNG_OFFSET   # where the CI bar sits

# The three predictors are three ways of asking "how Assistant-like is this
# role?", and they are NOT interchangeable. Spelled out on the panels because
# the bare column names hide the one thing a reader must know: the third one
# runs the opposite way, so its whole panel is a mirror image.
PRED_TITLES = {
    "axis_proj":
        "axis_proj  —  position along the Assistant Axis\n"
        "$\\mathrm{mean}(role)\\cdot\\hat{a}$  =  magnitude × direction\n"
        "high = Assistant-like",
    "cos_axis":
        "cos_axis  —  direction only\n"
        "$\\cos$ angle to the axis, magnitude divided out\n"
        "high = Assistant-like",
    "dist_default":
        "dist_default  —  distance from `default`\n"
        "unlike the Assistant in ANY direction\n"
        "high = UN-Assistant-like  →  signs flip",
}


def defensive(fn):
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            print(f"  [fig] {fn.__name__} FAILED (logged, run continues):")
            traceback.print_exc()
    return wrap


def _save(fig, name, run_dir):
    p = Path(run_dir) / "figures" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p.name)


# --------------------------------------------------------------------------- #
# CALIBRATION                                                                  #
# --------------------------------------------------------------------------- #
@defensive
def fig00_calibration(cal, run_dir, L):
    df = pd.DataFrame(cal["calibration"])
    # All six, gated ones first, so a reader can see WHY three passed and three
    # did not rather than being told.
    gated = list(cal["verdict"]["gated_on"])
    ests = gated + [e for e in ("MLE", "TwoNN", "lPCA", "PCA_participation_ratio",
                                "PCA_dim_90pct", "PCA_dim_95pct")
                    if e not in gated]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True)
    axes = axes.ravel()
    radii = sorted(df.radius.unique())
    for ax, est in zip(axes, ests):
        for i, r in enumerate(radii):
            g = df[df.radius == r].sort_values("planted_d")
            ax.plot(g.planted_d, g[est], "o-", ms=5,
                    color=plt.cm.viridis(i / max(1, len(radii) - 1)),
                    label=f"radius {r:.1f}")
        lim = [0, max(13, df[est].max() * 1.05)]
        ax.plot(lim, lim, "k--", lw=1, label="identity")
        ax.axvspan(2.5, 10.5, color="0.9", zorder=0)
        is_gated = est in gated
        err = cal["verdict"].get("worst_relative_error_d_le_10", {}).get(est)
        errs = f"worst {err*100:.0f}% (d≤10)" if err is not None else ""
        ax.set_title(f"{est}\n{'CALIBRATED' if is_gated else 'FAILS'} · {errs}",
                     fontsize=9, color="#0072B2" if is_gated else "#C0392B")
        ax.set_xlabel("planted dimension")
        ax.set_ylim(0, lim[1])
    for a in (axes[0], axes[3]):
        a.set_ylabel("recovered dimension")
    axes[0].legend(fontsize=7)
    fig.suptitle("Calibration: recovered vs planted dimension", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, f"fig00_calibration_L{L}.png", run_dir)


@defensive
def fig01_topology_controls(cal, run_dir, L):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, want in ((axes[0], "circle", 1), (axes[1], "blob", 0)):
        dgms = [np.asarray(d) for d in cal[f"{key}_dgms"]]
        diam = cal[key]["cloud_diameter"]
        h1 = dgms[1]
        if len(h1):
            ax.scatter(h1[:, 0], h1[:, 1], s=22, color=C_REAL, zorder=3)
        lim = [0, diam * 1.05]
        ax.plot(lim, lim, "k-", lw=1)
        ax.plot(lim, [x + 0.10 * diam for x in lim], "--", color=C_DESIGN, lw=1.2,
                label="10% of diameter (the rule)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        got = cal[key]["betti1"]
        ax.set_title(f"planted {key}: Betti-1 = {got} (expected {want}) "
                     f"{'PASS' if got == want else 'FAIL'}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Topology controls", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, f"fig01_topology_controls_L{L}.png", run_dir)


# --------------------------------------------------------------------------- #
# STUDY                                                                        #
# --------------------------------------------------------------------------- #
@defensive
def fig02_forest(lad, run_dir, L):
    """The centrepiece: does each metric's correlation survive its controls?"""
    preds = ["axis_proj", "cos_axis", "dist_default"]
    sub = lad[(lad.predictor == "axis_proj") & (~lad.degenerate.fillna(False))]
    order = sub.reindex(sub.r_ctrl_all.abs().sort_values().index)["metric"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.42 * len(order) + 4.2), sharey=True)
    for ax, pred in zip(axes, preds):
        s = lad[(lad.predictor == pred)].set_index("metric").reindex(order)
        thr = float(s["shuffle_max_abs_r_p95"].dropna().iloc[0]) if \
            s["shuffle_max_abs_r_p95"].notna().any() else np.nan
        y = np.arange(len(order))
        if np.isfinite(thr):
            ax.axvspan(-thr, thr, color="0.90", zorder=0,
                       label="axis-shuffle null (95th pct of max|r|)")
        ax.axvline(0, color="k", lw=0.8, zorder=1)
        for j, (col, lab) in enumerate(RUNGS):
            ax.scatter(s[col], y + (j - 1.0) * RUNG_OFFSET, s=26,
                       color=RUNG_COLORS[j], zorder=3,
                       label=lab if pred == "axis_proj" else None)
        ax.hlines(y + LAST_RUNG_Y, s["ci_lo_ctrl_all"], s["ci_hi_ctrl_all"],
                  color=C_REAL, lw=1.6, zorder=2)
        for v in (-0.30, 0.30):
            ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("correlation with predictor")
        ax.set_title(PRED_TITLES.get(pred, pred), fontsize=9.5, linespacing=1.35)
        ax.set_xlim(-1, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("The correlation ladder", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _save(fig, f"fig02_ladder_forest_L{L}.png", run_dir)


@defensive
def fig02b_forest_axisproj(lad, run_dir, L):
    """The ladder restricted to axis_proj alone.

    fig02 shows three predictors side by side, which is the right object when the
    question is "is this about direction or magnitude?". It is the wrong object
    when someone just wants to read the result: three panels of near-identical
    rows invite the eye to compare panels instead of reading rungs. This is the
    same data, one predictor, wide enough to label.
    """
    # Dropped from this view only (fig02 still shows all 20):
    #   betti1 — 0 for all but a handful of roles, so its correlation is noise
    #            about a near-constant and the row is meaningless.
    #   betti0 — NOT broken: it equals 1 + #{MST edges > 10% of diameter}, a real
    #            sparsity measure (r = -0.570). Dropped at the user's request
    #            because "Betti-0" reads as topology and it is not measuring
    #            topology. Put it back by editing DROP_FROM_AXISPROJ.
    DROP_FROM_AXISPROJ = ("betti0", "betti1")
    s = lad[(lad.predictor == "axis_proj")
            & (~lad.degenerate.fillna(False))
            & (~lad.metric.isin(DROP_FROM_AXISPROJ))].copy()
    order = s.reindex(s.r_ctrl_all.abs().sort_values().index)["metric"].tolist()
    s = s.set_index("metric").reindex(order)
    thr = float(s["shuffle_max_abs_r_p95"].dropna().iloc[0]) if \
        s["shuffle_max_abs_r_p95"].notna().any() else np.nan
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 0.46 * len(order) + 2.6))
    if np.isfinite(thr):
        ax.axvspan(-thr, thr, color="0.90", zorder=0,
                   label=f"axis-shuffle null (95th pct of max|r| = {thr:.2f})")
    ax.axvline(0, color="k", lw=0.8, zorder=1)
    for v in (-0.30, 0.30):
        ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
    ax.text(0.30, len(order) - 0.2, " |r| = 0.30", color=C_DESIGN, fontsize=8, va="top")

    for j, (col, lab) in enumerate(RUNGS):
        ax.scatter(s[col], y + (j - 1.0) * RUNG_OFFSET, s=34,
                   color=RUNG_COLORS[j], zorder=3, label=lab)
    ax.hlines(y + LAST_RUNG_Y, s["ci_lo_ctrl_all"], s["ci_hi_ctrl_all"],
              color=C_REAL, lw=1.8, zorder=2)

    # Value labels on the fully-controlled point — the number people actually quote.
    for yi, v in zip(y, s["r_ctrl_all"]):
        ax.annotate(f"{v:+.2f}", (v, yi + LAST_RUNG_Y), fontsize=7.5,
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color=C_REAL, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("correlation with axis_proj   (high = Assistant-like)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.suptitle("The correlation ladder — assistant-axis projection", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, f"fig02b_ladder_axisproj_L{L}.png", run_dir)


@defensive
def fig03_scatter(panel, lad, run_dir, L):
    d = panel[panel.role != "default"]
    dflt = panel[panel.role == "default"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, m in zip(axes.ravel(), HEADLINE):
        row = lad[(lad.predictor == "axis_proj") & (lad.metric == m)]
        ax.scatter(d.axis_proj, d[m], s=11, alpha=0.45, color=C_REAL, linewidths=0)
        sl, ic = np.polyfit(d.axis_proj, d[m], 1)
        xs = np.linspace(d.axis_proj.min(), d.axis_proj.max(), 50)
        ax.plot(xs, sl * xs + ic, color=C_DESIGN, lw=2)
        if len(dflt):
            dx, dy = float(dflt.axis_proj.iloc[0]), float(dflt[m].iloc[0])
            if not (xs[0] <= dx <= xs[-1]):
                ex = np.linspace(min(xs[0], dx), max(xs[-1], dx), 50)
                ax.plot(ex, sl * ex + ic, ls=":", lw=1.2, color=C_DESIGN)
            ax.scatter([dx], [dy], s=110, marker="*", color="#111111", zorder=6,
                       label="default (excluded from fit)")
            ax.legend(fontsize=7, loc="best")
        r = row.iloc[0] if len(row) else None
        ax.set_title(f"{m}\nraw r={r.r_raw:+.3f}  fully-controlled={r.r_ctrl_all:+.3f}"
                     if r is not None else m, fontsize=9)
        ax.set_xlabel("assistant-axis projection  (high = Assistant-like)", fontsize=8)
        ax.set_ylabel(YLABELS.get(m, m), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Headline metrics vs assistant-likeness", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"fig03_headline_scatter_L{L}.png", run_dir)


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
def fig08_families(fam, run_dir, L):
    fs = pd.DataFrame(fam["families"]).sort_values("mean_axis_proj", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, col, lbl in ((axes[0], "median_MLE", "median MLE (calibrated dimension)"),
                         (axes[1], "median_curvature_gain", "median curvature gain")):
        ax.scatter(fs.mean_axis_proj, fs[col], s=fs.n_roles * 6, alpha=.7,
                   color=C_REAL, linewidths=0)
        for _, r in fs.iterrows():
            ax.annotate(r.example_roles[0], (r.mean_axis_proj, r[col]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
        sl, ic = np.polyfit(fs.mean_axis_proj, fs[col], 1)
        xs = np.linspace(fs.mean_axis_proj.min(), fs.mean_axis_proj.max(), 40)
        ax.plot(xs, sl * xs + ic, color=C_DESIGN, lw=1.6)
        ax.set_xlabel("family mean assistant-axis projection")
        ax.set_ylabel(lbl)
    rho = fam["family_axis_vs_median_MLE"]["spearman_rho"]
    rhoc = fam["family_axis_vs_median_curvature"]["spearman_rho"]
    axes[0].set_title(f"Spearman rho = {rho:+.3f}", fontsize=9)
    axes[1].set_title(f"Spearman rho = {rhoc:+.3f}", fontsize=9)
    fig.suptitle("Role families", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, f"fig08_families_L{L}.png", run_dir)


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
    cal, nullj = j(f"calibration_L{L}.json"), j(f"design_null_L{L}.json")
    reg, fam = j(f"regression_L{L}.json"), j(f"families_L{L}.json")
    con = j(f"prompt_vs_response_L{L}.json")

    fig00_calibration(cal, run, L)
    fig01_topology_controls(cal, run, L)
    fig02_forest(lad, run, L)
    fig02b_forest_axisproj(lad, run, L)
    fig03_scatter(panel, lad, run, L)
    fig04_design_null(nullj, run, L)
    fig05_regression(reg, run, L)
    fig07_magnitude_direction(panel, run, L)
    fig08_families(fam, run, L)
    fig09_contrast(con, run, L)
    print("figures done")


if __name__ == "__main__":
    main()
