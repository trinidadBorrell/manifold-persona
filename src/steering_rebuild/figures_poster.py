"""Poster figures: the two arms side by side, and their head-to-head difference.

`figures_steering.py` draws one arm per file, which is right for a report and
wrong for a poster: the comparison the run exists to make ends up split across
two images that a reader has to hold in their head. These are laid out for a
wall — one shared y-scale, larger type, and the arms adjacent.

    python -m steering_rebuild.figures_poster --judged judged_L19.parquet --outdir figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from steering_rebuild.judge import PLOT_CATEGORIES, PLOT_LABEL, RESIDUAL_CATEGORIES  # noqa: E402

PALETTE = {"assistant": "#4C72B0", "human_role": "#2E7D32",
           "nonhuman_role": "#55A868", "weird_role": "#C44E52",
           "nonsensical": "#8172B3"}
ARM_TITLE = {"linear_axis": "Linear — straight along the Assistant Axis",
             "manifold_axis": "Manifold — along the fitted role curve"}
BOOT, SEED = 2000, 0


def _stats(df, arm):
    """Per-alpha category fractions, bootstrapped over ROLES (the unit)."""
    sub = pd.concat([df[df.arm == arm], df[df.arm == "unsteered"]])
    out = {}
    for a, g in sub.groupby("alpha"):
        roles = sorted(g.role.dropna().unique())
        mat = np.zeros((len(roles), len(PLOT_CATEGORIES)))
        for i, r in enumerate(roles):
            s = g.loc[g.role == r, "judge_score"].dropna()
            if not len(s):
                continue
            vc = s.value_counts(normalize=True)
            mat[i] = [vc.get(c, 0.0) for c in PLOT_CATEGORIES]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(roles), size=(BOOT, len(roles)))
        reps = mat[idx].mean(1)
        out[float(a)] = (mat.mean(0), np.percentile(reps, 2.5, 0), np.percentile(reps, 97.5, 0))
    return out


def _panel(ax, st, title, show_legend, fs):
    x = np.array(sorted(st))
    for j, cat in enumerate(PLOT_CATEGORIES):
        y = np.array([st[a][0][j] for a in x])
        lo = np.array([st[a][1][j] for a in x])
        hi = np.array([st[a][2][j] for a in x])
        ax.plot(x, y, "-o", ms=5, lw=2.4, color=PALETTE[cat],
                label=PLOT_LABEL[cat], zorder=3)
        ax.fill_between(x, lo, hi, color=PALETTE[cat], alpha=0.18, lw=0)
    ax.axvline(0, color="#888888", lw=1.0, zorder=0)
    ax.set_xlim(-1.06, 0.30)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=fs + 1, pad=8)
    ax.set_xlabel("steering along the Assistant Axis\n(fraction of avg. residual norm)", fontsize=fs)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=fs - 1)
    if show_legend:
        ax.legend(fontsize=fs - 1, loc="upper left", framealpha=0.92)


def poster_arms(df, out: Path, fs=12):
    """Both arms, shared scale, adjacent — the comparison in one image."""
    arms = [a for a in ("linear_axis", "manifold_axis") if (df.arm == a).any()]
    fig, axes = plt.subplots(1, len(arms), figsize=(7.2 * len(arms), 6.0), squeeze=False)
    for k, arm in enumerate(arms):
        ax = axes[0][k]
        _panel(ax, _stats(df, arm), ARM_TITLE.get(arm, arm), k == 0, fs)
        if k == 0:
            ax.set_ylabel("fraction of responses", fontsize=fs + 1)
        else:
            ax.set_yticklabels([])
    resid = df.judge_score.isin(RESIDUAL_CATEGORIES).mean()
    fig.suptitle("Persona under activation steering — Qwen3-8B, layer 19, 50 roles",
                 fontsize=fs + 5, y=0.985)
    fig.text(0.5, 0.015,
             "each point = mean over 50 roles (20 rollouts each); band = bootstrap 95%% CI over roles.  "
             "ambiguous+other = %.1f%% overall, not plotted." % (100 * resid),
             ha="center", fontsize=fs - 2, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    return out


def poster_headtohead(df, out: Path, fs=12):
    """The three quantities that decide whether the manifold arm is better."""
    groups = [("stayed the Assistant", ["assistant"], "#4C72B0"),
              ("adopted the role (human+nonhuman)", ["human_role", "nonhuman_role"], "#2E7D32"),
              ("broke down (nonsensical)", ["nonsensical"], "#8172B3")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=True)
    for ax, (title, cats, col) in zip(axes, groups):
        for arm, ls, mk in (("linear_axis", "-", "o"), ("manifold_axis", "--", "s")):
            sub = pd.concat([df[df.arm == arm], df[df.arm == "unsteered"]])
            xs, ys = [], []
            for a, g in sub.groupby("alpha"):
                per = g.groupby("role").judge_score.apply(
                    lambda s: s.isin(cats).mean() if len(s.dropna()) else np.nan)
                xs.append(float(a)); ys.append(float(np.nanmean(per)))
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], ls, marker=mk, ms=6, lw=2.6,
                    color=col, alpha=1.0 if arm == "linear_axis" else 0.55,
                    label="linear" if arm == "linear_axis" else "manifold")
        ax.axvline(0, color="#888888", lw=1.0, zorder=0)
        ax.set_title(title, fontsize=fs + 1)
        ax.set_xlabel("fraction of avg. residual norm", fontsize=fs)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=fs, loc="best")
        ax.tick_params(labelsize=fs - 1)
    axes[0].set_ylabel("fraction of responses", fontsize=fs + 1)
    axes[0].set_ylim(0, 1.0)
    fig.suptitle("Same dose, different path: does following the manifold help?",
                 fontsize=fs + 5, y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--layer", type=int, default=19)
    a = ap.parse_args()
    df = pd.read_parquet(a.judged)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    p1 = poster_arms(df, out / ("poster01_arms_L%d.png" % a.layer))
    p2 = poster_headtohead(df, out / ("poster02_headtohead_L%d.png" % a.layer))
    print("wrote", p1); print("wrote", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
