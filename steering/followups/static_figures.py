"""Static PNG figures for REPORT.md, drawn from the figure CSVs of a finished run.

Palette: the dataviz skill's validated categorical theme, slots in fixed order, one
arm = one colour everywhere. Nine arms exceed the eight slots, so Study A (7 arms) and
Study B (2 arms + the 2 Study A references) are drawn as separate figures.

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.static_figures
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from steering.followups.common import OUT_ROOT  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
A_ARMS = ["linear", "manifold_v1", "manifold_v2", "nearest", "target_vector",
          "replace_linear", "replace_manifold"]
COL = dict(zip(A_ARMS, SLOTS))
COL.update(linear_B=SLOTS[7], nearest_B="#52514e")
ROUTES = ["validator>vampire", "validator>bard", "assistant>vampire", "assistant>bard"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True})


def slug(r):
    return r.replace(">", "-")


def title(r):
    return r.replace(">", " → ")


def _legend(fig, arms):
    h = [plt.Line2D([], [], color=COL[a], lw=2, marker="o", ms=4) for a in arms]
    fig.legend(h, arms, loc="lower center", ncol=len(arms), fontsize=8, bbox_to_anchor=(0.5, -0.01))


def fig_pca(root, out):
    arms = ["linear", "nearest", "replace_linear", "target_vector"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, r in zip(axs.flat, ROUTES):
        d = pd.read_csv(root / "figures" / f"figA-pca-{slug(r)}.csv")
        cen = d[d.layer == "centroid"]
        oth = cen[cen.role == "other"]
        ax.scatter(oth.pc1, oth.pc2, s=6, color="#b9c0ca", zorder=1, lw=0)
        for a in arms:
            ip = d[(d.study == "A") & (d.arm == a) & (d.role == "intended")].sort_values("alpha")
            if len(ip):
                ax.plot(ip.pc1, ip.pc2, ls="--", lw=1.4, color=COL[a], alpha=.8, zorder=2)
            rz = d[(d.study == "A") & (d.arm == a) & (d.role == "realised") & (d.layer == "text")]
            rz = rz[rz.q_idx < 5].groupby("alpha")[["pc1", "pc2"]].mean().reset_index()
            ax.plot(rz.pc1, rz.pc2, "-o", ms=4, color=COL[a], zorder=3,
                    markeredgecolor=SURF, markeredgewidth=1)
            last = rz.iloc[-1]
            ax.annotate("α=%g" % last.alpha, (last.pc1, last.pc2), fontsize=7, color=INK2,
                        xytext=(4, 3), textcoords="offset points")
        kn = cen[cen.role == "knot"]
        ax.scatter(kn.pc1, kn.pc2, s=28, facecolor=SURF, edgecolor=COL["nearest"], lw=1.5, zorder=4)
        for role, mk in (("source", "o"), ("target", "D")):
            p = cen[cen.role == role].iloc[0]
            ax.scatter(p.pc1, p.pc2, s=70, marker=mk, color=INK, zorder=5, edgecolor=SURF)
            ax.annotate(f"{p.label} ({role})", (p.pc1, p.pc2), fontsize=8, color=INK,
                        xytext=(6, 6), textcoords="offset points", weight="bold")
        ax.set_title(title(r))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    _legend(fig, arms)
    fig.suptitle("Study A: intended paths (dashed) and where the text actually lands (solid, mean of 5 "
                 "identity questions per α)\nPCA of the 275 role centroids; open circles = nearest-rule "
                 "knots. PCA is a view: distances are 4096-d.", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "fig1_pca_paths_studyA.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dist(root, study, arms, fname, suptitle):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for j, r in enumerate(ROUTES):
        d = pd.read_csv(root / "figures" / f"figB-dist-{slug(r)}.csv")
        d = d[(d.measure == "text") & (d.to == "target") & (d.kind == "data")]
        for i, metric in enumerate(("euclid", "cosine_centred")):
            ax = axs[i, j]
            for a in arms:
                s = d[(d.metric == metric) & (d.arm == a) & (d.study == study[a])].sort_values("alpha")
                if s.empty:
                    continue
                ax.fill_between(s.alpha, s.ci_lo, s.ci_hi, color=COL[a], alpha=.12, lw=0)
                ax.plot(s.alpha, s["mean"], "-o", ms=3.5, color=COL[a],
                        markeredgecolor=SURF, markeredgewidth=.8)
            if i == 0:
                ax.set_title(title(r))
            ax.set_ylabel("Euclidean distance to target" if metric == "euclid"
                          else "centred cosine to target")
            if i == 1:
                ax.set_xlabel("α")
    _legend(fig, arms)
    fig.suptitle(suptitle, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(root / "figures" / "png" / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _target(root, study, arms, fname, suptitle):
    ci = pd.read_csv(root / "figures" / "figC-target-ci.csv")
    judge = pd.concat([pd.read_csv(f) for f in glob.glob(str(root / "figures" / "figC-judge-*.csv"))])
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1.3]))
    for j, r in enumerate(ROUTES):
        ax, bx = axs[0, j], axs[1, j]
        for a in arms:
            s = ci[(ci.route == r) & (ci.arm == a) & (ci.study == study[a])].sort_values("alpha")
            if s.empty:
                continue
            ax.fill_between(s.alpha, 100 * s.ci_lo, 100 * s.ci_hi, color=COL[a], alpha=.10, lw=0)
            ax.plot(s.alpha, 100 * s.target, "-o", ms=3.5, color=COL[a],
                    markeredgecolor=SURF, markeredgewidth=.8)
            g = judge[(judge.route == r) & (judge.arm == a) & (judge.study == study[a])].sort_values("alpha")
            bx.plot(g.alpha, 100 * g.share_collapsed, "-", color=COL[a], lw=1.4)
            bad = g[g.canary_acc < 0.8]
            bx.scatter(bad.alpha, 100 * bad.share_collapsed, marker="x", color=COL[a], s=22, zorder=3)
        ax.axhline(15, color=INK2, lw=.8, ls=":")
        ax.set_ylim(-3, 103)
        ax.set_title(title(r))
        ax.set_ylabel("% judged as target")
        bx.set_ylim(-3, 103)
        bx.set_ylabel("% collapsed")
        bx.set_xlabel("α")
    _legend(fig, arms)
    fig.suptitle(suptitle, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(root / "figures" / "png" / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_dose(root, out):
    d = pd.read_csv(root / "figures" / "figC-dose.csv")
    arms = ["linear", "manifold_v1", "manifold_v2", "nearest", "target_vector", "linear_B", "nearest_B"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, r in zip(axs, ROUTES):
        for a in arms:
            s = d[(d.route == r) & (d.arm == a) & (d.alpha > 0)].sort_values("push")
            ax.plot(s.push, s.target, "-o", ms=4, lw=1.2, color=COL[a],
                    markeredgecolor=SURF, markeredgewidth=.8)
        ax.set_title(title(r))
        ax.set_xlabel("push size ‖Δ‖ (activation units)")
    axs[0].set_ylabel("% judged as target")
    _legend(fig, arms)
    fig.suptitle("Post hoc: target share against actual push size, additive arms (Study A and B). "
                 "Most same-α differences between arms follow push; direction still matters for vampire.",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out / "fig5_target_vs_push.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_followup(root, out):
    f = pd.read_csv(root / "followup_F" / "summary.csv")
    base = pd.read_csv(root / "figures" / "figC-target-ci.csv")
    base = base[(base.arm == "replace_linear") & (base.alpha == 1.0) & (base.study == "A")]
    rows = [dict(cond="rank 64, t=1", route=r.route, target=100 * r.target) for r in base.itertuples()]
    rows += [dict(cond="rank %d, t=%g" % (r.rank, r.alpha), route=r.route, target=r.target)
             for r in f.itertuples()]
    t = pd.DataFrame(rows)
    order = ["rank 16, t=1", "rank 64, t=1", "rank 256, t=1", "rank 64, t=1.25",
             "rank 64, t=1.5", "rank 64, t=2"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, r in zip(axs, ROUTES):
        s = t[t.route == r].set_index("cond").reindex(order)
        ax.bar(range(len(order)), s.target, color=COL["replace_linear"], width=.62)
        for i, v in enumerate(s.target):
            ax.text(i, v + 2, "%.0f" % v, ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_title(title(r))
        ax.set_ylim(0, 110)
    axs[0].set_ylabel("% judged as target")
    fig.suptitle("Follow-up F (exploratory): paper-style replacement, varying subspace rank and "
                 "extrapolating past the target (t > 1). Canary 3/3 and collapse ≤ 7% in every cell.",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig6_followupF_replace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(OUT_ROOT)
    out = root / "figures" / "png"
    out.mkdir(parents=True, exist_ok=True)
    fig_pca(root, out)
    sa = {a: "A" for a in A_ARMS}
    _dist(root, sa, A_ARMS, "fig2_distance_studyA.png",
          "Study A: distance of the response text to the target centroid vs α "
          "(mean of 5 questions, bootstrap 95% CI band)")
    _target(root, sa, A_ARMS, "fig3_target_share_studyA.png",
            "Study A: % of identity answers judged as the target (top, 95% CI over questions; dotted = "
            "15-pt bar) and % collapsed (bottom; × = canary failed)")
    sb = dict(linear="A", target_vector="A", linear_B="B", nearest_B="B")
    arms_b = ["linear", "target_vector", "linear_B", "nearest_B"]
    _target(root, sb, arms_b, "fig4a_target_share_studyB.png",
            "Study B (start = measured position h0(q)) vs Study A linear and target_vector: "
            "% judged as target (top) and % collapsed (bottom; × = canary failed)")
    _dist(root, sb, arms_b, "fig4b_distance_studyB.png",
          "Study B vs Study A: distance of the response text to the target centroid vs α")
    fig_dose(root, out)
    fig_followup(root, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
