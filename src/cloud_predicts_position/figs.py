"""Figures and tables for Part B (plan 2026-10-29, sections 3.5 and 4.4) into output/29-oct/partB/.

  F1_scatter_primary.png       axis_proj (x) vs metric (y) for the nine primary metrics: one random ordering at i = 200,
                               training personas filled, held-out hollow, the fitted line
  F2_learning_curves.png       test R^2 vs i for the primary metrics, with the permutation 95th percentile and the scale-only model
  F3_forest.png                Spearman of every metric with axis_proj (all personas), raw and scale-adjusted, by family
  F4_questions_fit.png         Experiment 2: metric from Q[:i] (filled) with its fitted line, metric from matched held-out
                               questions (hollow), primary metrics x i in {10, 40, 120}
  F5_reliability.png           Experiment 2: split-half reliability vs i, primary metrics and the position itself
  F6_rmse_ratio.png            Experiment 2: RMSE_held / RMSE_train vs i
  tables.md, summary.json

    python -m cloud_predicts_position.figs --layer 25 --view avg [--centring q]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cloud_predicts_position import data as D
from cloud_predicts_position.exp1 import I_GRID as I1, SCALE, ols_fit_predict, metric_columns
from cloud_predicts_position.panel import FAMILIES, PRIMARY

INK, INK2 = "#0b0b0b", "#52514e"
C_REAL, C_NULL, C_SCALE, C_HELD = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
FAMILY_NAMES = dict(S="scale", D="intrinsic dimension", P="spectral shape", N="density", T="topology", C="curvature",
                    A="anisotropy to the assistant axis", G="design decomposition", R="alignment with the default cloud", I="consistency")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})


def family_of(c):
    return next(k for k, v in FAMILIES.items() if c in v)


def fig_scatter(df, out, i=200, seed=0):
    df = df[df.role != D.DEFAULT].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    tr, te = order[:i], order[i:]
    y = df.axis_proj.to_numpy(float)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5))
    for ax, c in zip(axes.ravel(), PRIMARY):
        m = df[c].to_numpy(float)
        ok = np.isfinite(m)
        a, b = np.polyfit(y[tr][ok[tr]], m[tr][ok[tr]], 1)
        xs = np.linspace(y.min(), y.max(), 50)
        ax.plot(xs, a * xs + b, color=INK2, linewidth=1)
        ax.scatter(y[tr], m[tr], s=12, color=C_REAL, linewidths=0, label=f"training personas (n = {i})")
        ax.scatter(y[te], m[te], s=16, facecolors="none", edgecolors=C_HELD, linewidths=0.9, label=f"held-out personas (n = {len(te)})")
        yhat = ols_fit_predict(m[tr][:, None], y[tr], m[te][:, None])
        r2 = 1 - ((y[te] - yhat) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum()
        rho = spearmanr(y[ok], m[ok])[0]
        ax.set_title(f"{c}  [{FAMILY_NAMES[family_of(c)]}]\nSpearman {rho:.2f} (all), test R² {r2:.2f}", fontsize=8, color=INK)
        ax.set_ylabel(c)
    for ax in axes[-1]:
        ax.set_xlabel("projection of the persona centroid\non the assistant axis (+ = toward the default assistant)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "F1_scatter_primary.png")
    plt.close(fig)


def fig_learning(curves, out):
    cv, ridge, perm_metric = curves["curves"], curves["ridge"], curves["perm_metric"]
    I = curves["i_grid"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), PRIMARY):
        if c not in cv:
            ax.set_visible(False); continue
        med = [np.median(cv[c][str(i)]["r2"]) for i in I]
        lo = [np.percentile(cv[c][str(i)]["r2"], 2.5) for i in I]
        hi = [np.percentile(cv[c][str(i)]["r2"], 97.5) for i in I]
        sc = [np.median(cv[c][str(i)]["r2_scale"]) for i in I]
        adj = [np.median(cv[c][str(i)]["r2_adj"]) for i in I]
        ax.fill_between(I, lo, hi, color=C_REAL, alpha=0.15, linewidth=0)
        ax.plot(I, med, color=C_REAL, linewidth=2, marker="o", markersize=3, label="metric alone, median test R² (2.5–97.5% over orderings)")
        ax.plot(I, sc, color=C_SCALE, linewidth=1.4, linestyle="--", label="scale-only model (log variance, centroid norm, response length)")
        ax.plot(I, adj, color=C_HELD, linewidth=1.4, linestyle="-.", label="metric + scale")
        ax.axhline(np.percentile(perm_metric[c], 95), color=C_NULL, linewidth=1, linestyle=":", label="permutation null, 95th percentile (i = 200)")
        ax.set_title(f"{c}  [{FAMILY_NAMES[family_of(c)]}]", fontsize=8, color=INK)
        ax.set_ylim(-0.3, 1)
    for ax in axes[-1]:
        ax.set_xlabel("training personas i")
    for ax in axes[:, 0]:
        ax.set_ylabel("test R² of the assistant-axis projection")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    rid = ", ".join(f"i = {i}: {ridge[str(i)]['r2_median']:.2f}" for i in I)
    fig.suptitle(f"ridge over the whole panel, median test R²: {rid}", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "F2_learning_curves.png")
    plt.close(fig)


def fig_forest(df, out):
    df = df[df.role != D.DEFAULT]
    cols = metric_columns(df)
    y = df.axis_proj.to_numpy(float)
    S = df[SCALE].to_numpy(float)
    rows = []
    rng = np.random.default_rng(0)
    for c in cols:
        m = df[c].to_numpy(float); ok = np.isfinite(m)
        rho = spearmanr(y[ok], m[ok])[0]
        # scale-adjusted: residualise both on the scale covariates
        A = np.hstack([np.ones((ok.sum(), 1)), S[ok]])
        ry = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
        rm = m[ok] - A @ np.linalg.lstsq(A, m[ok], rcond=None)[0]
        rho_adj = spearmanr(ry, rm)[0]
        boots = [spearmanr(y[ok][b], m[ok][b])[0] for b in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(300))]
        rows.append(dict(metric=c, family=family_of(c), rho=rho, lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), rho_adj=rho_adj))
    t = pd.DataFrame(rows)
    t["fam_order"] = t.family.map({k: i for i, k in enumerate(FAMILIES)})
    t = t.sort_values(["fam_order", "rho"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(t) + 1.5))
    ypos = np.arange(len(t))
    ax.hlines(ypos, t.lo, t.hi, color=C_REAL, linewidth=1.2)
    ax.scatter(t.rho, ypos, color=C_REAL, s=18, zorder=3, label="Spearman with axis projection (95% bootstrap over personas)")
    ax.scatter(t.rho_adj, ypos, facecolors="none", edgecolors=C_SCALE, s=22, zorder=3, label="after residualising both on the scale covariates")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{'★ ' if m in PRIMARY else ''}{m}" for m in t.metric], fontsize=7)
    for fam, g in t.groupby("family", sort=False):
        ax.text(1.02, float(np.mean(g.index.to_numpy())), FAMILY_NAMES[fam], fontsize=7, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Spearman ρ with the projection of the persona centroid on the assistant axis")
    ax.set_xlim(-1, 1)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "F3_forest.png")
    plt.close(fig)
    return t


def fig_exp2(cells, summ, mdir, out, view, L, centring):
    I_show = [10, 40, 120]
    # F4: scatter per primary metric x i, from seed 0
    ax_file = np.load(D.OUT / f"axis_{view}_L{L}.npz", allow_pickle=True)
    pos = pd.Series(ax_file["axis_proj"].astype(float), index=ax_file["roles"])
    fig, axes = plt.subplots(len(PRIMARY), len(I_show), figsize=(3.2 * len(I_show), 2.2 * len(PRIMARY)))
    for col, i in enumerate(I_show):
        f = mdir / f"metrics_{view}_L{L}_s0_i{i}.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f); g = g[(g.centring == centring) & (g.role != D.DEFAULT)]
        tr = g[g.split == "train"].set_index("role"); te = g[g.split == "test"].set_index("role")
        common = tr.index.intersection(te.index)
        y = pos.loc[common].to_numpy()
        for row, c in enumerate(PRIMARY):
            ax = axes[row, col]
            mt, me = tr.loc[common, c].to_numpy(float), te.loc[common, c].to_numpy(float)
            ok = np.isfinite(mt) & np.isfinite(me)
            if ok.sum() < 10:
                continue
            b, a = np.polyfit(y[ok], mt[ok], 1)
            xs = np.linspace(y.min(), y.max(), 50)
            ax.plot(xs, a + b * xs, color=INK2, linewidth=1)
            ax.scatter(y[ok], mt[ok], s=8, color=C_REAL, linewidths=0, label=f"metric from the first {i} questions")
            ax.scatter(y[ok], me[ok], s=11, facecolors="none", edgecolors=C_HELD, linewidths=0.7, label=f"metric from {i} held-out questions")
            if row == 0:
                ax.set_title(f"i = {i} questions", color=INK)
            if col == 0:
                ax.set_ylabel(c, fontsize=7.5)
            if row == len(PRIMARY) - 1:
                ax.set_xlabel("assistant-axis projection (all questions)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "F4_questions_fit.png")
    plt.close(fig)
    # F5 reliability, F6 rmse ratio
    s = summ[summ.centring == centring]
    for fname, col, ylab, ref in (("F5_reliability.png", "reliability", "split-half reliability of the metric\n(Pearson r, first i questions vs i held-out questions)", 0.8),
                                  ("F6_rmse_ratio.png", "rmse_ratio", "RMSE of the held-out points on the fitted line\n/ RMSE of the training points", 1.2)):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        cmap = plt.get_cmap("tab10")
        for j, c in enumerate(PRIMARY):
            g = s[s.metric == c].sort_values("i")
            if len(g):
                ax.plot(g.i, g[col], marker="o", markersize=3, linewidth=1.4, color=cmap(j % 10), label=c)
        if col == "reliability":
            g = s[s.metric == "axis_proj_itself"].sort_values("i")
            ax.plot(g.i, g[col], color=INK, linewidth=2.2, linestyle="--", label="the position itself")
        ax.axhline(ref, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 60, 80, 120]); ax.set_xticklabels([5, 10, 20, 40, 60, 80, 120])
        ax.set_xlabel("number of questions i")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)


def tables(t1, forest, summ, out, tag):
    lines = [f"# Part B — cloud metrics vs assistant-axis position ({tag})\n"]
    lines.append("## Experiment 1 — hold out personas (i = 200 training personas, 100 orderings; permutation null 1,000)\n")
    lines.append("| metric | family | primary | test R² (median, 2.5–97.5%) | Spearman test | reverse R² (metric from position) | scale-only R² | metric + scale R² | ΔR² over scale | perm. 95th pct | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in t1.iterrows():
        lines.append(f"| {r.metric} | {FAMILY_NAMES[r.family]} | {'★' if r.primary else ''} | {r.r2_test_median:.2f} ({r.r2_test_lo:.2f}–{r.r2_test_hi:.2f}) | {r.spearman_test:.2f} | {r.r2_reverse:.2f} | {r.r2_scale_only:.2f} | {r.r2_metric_plus_scale:.2f} | {r.delta_r2_over_scale:+.2f} | {r.perm_p95_this_metric:.2f} | {r.verdict} |")
    lines.append("\n## Experiment 1 — learning curve, median test R² by training size\n")
    lines.append("| metric | " + " | ".join(f"i = {i}" for i in I1) + " |\n|---|" + "---|" * len(I1))
    for _, r in t1[t1.primary].iterrows():
        lines.append(f"| {r.metric} | " + " | ".join(f"{r[f'r2_i{i}']:.2f}" for i in I1) + " |")
    if summ is not None:
        lines.append("\n## Experiment 2 — hold out questions (10 orderings)\n")
        piv = summ[summ.metric.isin(PRIMARY + ["axis_proj_itself"])].pivot(index="metric", columns="i", values="reliability")
        lines.append("Split-half reliability (median over orderings):\n")
        lines.append("| metric | " + " | ".join(f"i = {i}" for i in piv.columns) + " |\n|---|" + "---|" * len(piv.columns))
        for m, row in piv.iterrows():
            lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
        piv2 = summ[summ.metric.isin(PRIMARY)].pivot(index="metric", columns="i", values="rmse_ratio")
        lines.append("\nRMSE of the held-out points on the fitted line / RMSE of the training points:\n")
        lines.append("| metric | " + " | ".join(f"i = {i}" for i in piv2.columns) + " |\n|---|" + "---|" * len(piv2.columns))
        for m, row in piv2.iterrows():
            lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
        fp = summ.dropna(subset=["smallest_i_passing"]).groupby("metric").smallest_i_passing.min() if "smallest_i_passing" in summ else None
        if fp is not None and len(fp):
            lines.append("\nSmallest i passing the rule (reliability ≥ 0.8 and RMSE ratio ≤ 1.2): " + ", ".join(f"{m} {int(v)}" for m, v in fp.items()))
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="avg")
    ap.add_argument("--centring", default="q")
    ap.add_argument("--out", default="output/29-oct/partB")
    a = ap.parse_args(argv)
    tag = f"{a.view}_L{a.layer}_{a.centring}"
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(D.OUT / f"panel_{a.view}_L{a.layer}.csv")
    df = panel[panel.centring == a.centring].copy()
    fig_scatter(df, out)
    forest = fig_forest(df, out)
    t1 = pd.read_csv(D.OUT / "exp1" / f"exp1_{tag}.csv")
    curves = json.loads((D.OUT / "exp1" / f"exp1_{tag}_curves.json").read_text())
    fig_learning(curves, out)
    summ_f = D.OUT / "exp2" / f"exp2_{a.view}_L{a.layer}_summary.csv"
    summ = pd.read_csv(summ_f) if summ_f.exists() else None
    if summ is not None:
        cells = pd.read_csv(D.OUT / "exp2" / f"exp2_{a.view}_L{a.layer}_cells.csv")
        fig_exp2(cells, summ, D.OUT / "exp2", out, a.view, a.layer, a.centring)
    tables(t1, forest, summ[summ.centring == a.centring] if summ is not None else None, out, tag)
    json.dump(dict(exp1=t1.to_dict("records"), ridge=curves["ridge"]), open(out / "summary.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
