"""E7 figures, generated from stored results.json (no recomputation).

  fig_e7_2x2.png      -- headline 2x2 (cloud x layer): all four estimators per
                         cell on a shared axis, resample interval on TwoNN,
                         per-cloud TwoNN bands showing the basis separation,
                         prompt_L26 PR outlier annotated.
  fig_e7_null_gap.png -- real TwoNN vs 5 matched-marginal Gaussian null draws
                         per cell, log y-axis.

Usage (from repo root):
    MPLCONFIGDIR=/tmp/mpl .venv/bin/python output/e7_id_reconciliation/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(OUT_DIR, "results.json")))

# palette (dataviz reference, light mode; first three slots validate all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

CELLS = [("prompt_L19", "prompt tokens · layer 19"),
         ("prompt_L26", "prompt tokens · layer 26"),
         ("resp_L19", "response tokens · layer 19"),
         ("resp_L26", "response tokens · layer 26")]

plt.rcParams.update({
    "font.size": 8, "text.color": INK, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.75)


def fig_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=True)
    # per-cloud TwoNN range (both implementations, both layers) -> the two bands
    bands = {}
    for cloud in ("prompt", "resp"):
        vals = [RES[f"{cloud}_L{l}"][k] for l in (19, 26)
                for k in ("twonn_handrolled", "twonn_skdim")]
        bands[cloud] = (min(vals), max(vals))

    marker_kw = dict(ms=7, mew=1.4, zorder=4)
    for ax, (cell, title) in zip(axes.flat, CELLS):
        r = RES[cell]
        style_axes(ax)
        for lo, hi in bands.values():
            ax.axhspan(lo, hi, color=GRID, alpha=0.45, lw=0, zorder=1)
        ax.set_title(title, loc="left", fontsize=9, fontweight="semibold", pad=6)

        pts = [(0, r["twonn_handrolled"], BLUE, "o", BLUE),      # filled
               (1, r["twonn_skdim"], BLUE, "o", SURFACE),        # open
               (2, r["mle_k20"], ORANGE, "s", ORANGE),
               (3, r["pca"]["participation_ratio"], AQUA, "^", AQUA)]
        bs = r.get("twonn_subsample", r.get("twonn_bootstrap"))
        ci = bs.get("subsample_interval95", bs.get("ci95"))
        ax.plot([0, 0], ci, color=BLUE, lw=1.4, zorder=3,
                solid_capstyle="butt")
        for capy in ci:
            ax.plot([-0.07, 0.07], [capy, capy], color=BLUE, lw=1.2, zorder=3)
        for x, y, col, mk, face in pts:
            ax.plot(x, y, mk, color=col, mfc=face, **marker_kw)
            ax.annotate(f"{y:.2f}", (x, y), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=6.8, color=INK2)

    # band labels + basis-gap note (top-left cell only; bands repeat in all)
    ax0 = axes[0, 0]
    ax0.text(-0.52, sum(bands["resp"]) / 2, "response-cloud TwoNN range",
             fontsize=6.5, color=MUTED, va="center")
    ax0.text(-0.52, bands["prompt"][0] - 0.45, "prompt-cloud TwoNN range",
             fontsize=6.5, color=MUTED, va="top")
    ax0.text(1.75, (bands["prompt"][1] + bands["resp"][0]) / 2,
             "token-basis gap ≈ 4–5 dims", fontsize=7, color=INK2,
             style="italic", ha="center", va="center")

    # the prompt_L26 PR outlier (estimator disagreement, not hidden)
    ax1 = axes[0, 1]
    pr26 = RES["prompt_L26"]["pca"]["participation_ratio"]
    ax1.annotate("PR reads 12.4 — inside the response band.\n"
                 "Variance-based PR is anisotropy-inflated;\n"
                 "neighbor-based estimates read 6.5–7.3.",
                 (3, pr26), xytext=(0.12, 15.1), fontsize=6.8, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                 shrinkB=6))

    axes[0, 0].set_ylim(5, 17.5)
    axes[0, 0].set_yticks([6, 8, 10, 12, 14, 16])
    for ax in axes[1]:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["TwoNN\n(hand-rolled)", "TwoNN\n(skdim)",
                            "MLE\n(k=20 only*)", "PCA\nparticipation ratio"],
                           fontsize=7.5)
    ax0.set_xlim(-0.6, 3.75)
    fig.supylabel("intrinsic-dimension estimate", fontsize=9, color=INK2)
    fig.suptitle("Intrinsic dimension across the 2×2: token cloud × read layer",
                 fontsize=12, fontweight="semibold", x=0.5, y=0.99)
    fig.text(0.5, 0.945, "Rows separate by ~4–5 dims; columns move each cloud by ≤1 dim. "
             "Gray bands: per-cloud TwoNN range across both layers.\n"
             "*the advertised k=10 run silently reused k=20 (Result 2); "
             "the resample interval is a ≈63% subsample, not a bootstrap (Result 2).",
             ha="center", va="top", fontsize=7.5, color=INK2)

    handles = [
        Line2D([], [], marker="o", color=BLUE, ls="", ms=7, mew=1.4, label="TwoNN, hand-rolled"),
        Line2D([], [], marker="o", color=BLUE, mfc=SURFACE, ls="", ms=7, mew=1.4, label="TwoNN, skdim"),
        Line2D([], [], marker="s", color=ORANGE, ls="", ms=7, mew=1.4, label="Levina–Bickel MLE"),
        Line2D([], [], marker="^", color=AQUA, ls="", ms=7, mew=1.4, label="PCA participation ratio"),
        Line2D([], [], color=BLUE, lw=1.4, label="95% resample interval"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.01, 0.045, 1, 0.90))
    path = os.path.join(OUT_DIR, "fig_e7_2x2.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_null_gap():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    style_axes(ax)
    jit = [-0.14, -0.07, 0.0, 0.07, 0.14]
    ratios = []
    for i, (cell, _) in enumerate(CELLS):
        r = RES[cell]
        nulls = r["null_twonn"]
        real = r["twonn_handrolled"]
        ratio = (sum(nulls) / len(nulls)) / real
        ratios.append(ratio)
        ax.plot([i + j for j in jit], nulls, "o", color=MUTED, mfc=SURFACE,
                ms=5, mew=1.2, zorder=3)
        ax.plot(i, real, "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=4)
        # gap connector + ratio label at the geometric midpoint
        ax.plot([i, i], [real * 1.12, min(nulls) * 0.93], ls=":", lw=1,
                color=AXIS, zorder=2)
        ax.text(i + 0.09, (real * min(nulls)) ** 0.5, f"×{ratio:.0f}",
                fontsize=8.5, color=INK2, va="center")
        ax.annotate(f"{real:.1f}", (i, real), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=INK2)
        ax.annotate(f"≈{sum(nulls) / len(nulls):.0f}", (i, max(nulls)),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=7, color=INK2)

    ax.set_yscale("log")
    ax.set_ylim(5, 300)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_xticks(range(4))
    ax.set_xticklabels([t.replace(" tokens · layer ", " · L")
                        for _, t in CELLS], fontsize=8)
    ax.set_ylabel("TwoNN intrinsic dimension (log scale)", fontsize=9)
    ax.set_title("Every cell sits 12–16× below its matched-marginal Gaussian null",
                 loc="left", fontsize=11.5, fontweight="semibold", pad=10)
    handles = [
        Line2D([], [], marker="o", color=BLUE, ls="", ms=8, mec=SURFACE,
               mew=1.5, label="real cloud (hand-rolled TwoNN)"),
        Line2D([], [], marker="o", color=MUTED, mfc=SURFACE, ls="", ms=5,
               mew=1.2, label="matched-marginal Gaussian null (5 draws)"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
              fontsize=8, bbox_to_anchor=(0.5, 1.02), columnspacing=1.2,
              handletextpad=0.4)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_e7_null_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("null/real ratios:", [f"{x:.1f}" for x in ratios])
    return path


if __name__ == "__main__":
    for p in (fig_2x2(), fig_null_gap()):
        print("wrote", p)
