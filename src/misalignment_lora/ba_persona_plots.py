"""Persona-level plots for one model family at one layer (default: Qwen2.5-7B, layer 19).

  plot 1: EM share of the persona (x) vs each cloud metric (y)        -> partPersona_<fam>/plot1_em_vs_metric/L<L>/
  plot 2: closeness to the assistant (x) vs each cloud metric (y)     -> partTrends_<fam>/{own_axis,base_axis}/L<L>/
  plot 3: closeness to the assistant (x) vs EM per persona (y)        -> partPersona_<fam>/plot3_em_vs_position/{own_axis,base_axis}/

Closeness to the assistant, two versions:
  own_axis  = persona centroid projected on that model's own refit assistant axis (relative to that model);
  base_axis = persona centroid of the fine-tuned model projected on the BASE model's axis (base centring).
One colour per model: base + 3 bad-advice fine-tunes + 2 controls.

  python -m misalignment_lora.ba_persona_plots --family q7 --layer 19
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from misalignment_lora import ba_figs as B
from misalignment_lora import em_persona as P

EM, D = P.EM, P.D


def tables(fam, L):
    d = B.FAMS[fam]
    base_name = f"base_{fam}" if fam != "q3" else "base"
    roles_b, Cb, ab, pb = B.F.ax_npz(EM / d["base"], L)
    T = {}
    names = [base_name] + [m for m in B.models_of(fam) if (D / m / "judgements.jsonl").exists()]
    for m in names:
        pp = P.per_persona(m).set_index("role")
        if m == base_name:
            src, pf, pbase = EM / d["base"], pb, pb
            roles_f = roles_b
        else:
            src = EM / f"cpp_{m}"
            roles_f, Cf, _, pf = B.F.ax_npz(src, L)
            pbase = (Cf - Cb.mean(0)) @ ab
        pp["pos_own"] = pd.Series(pf, index=roles_f).reindex(pp.index)
        pp["pos_base"] = pd.Series(pbase, index=roles_f).reindex(pp.index)
        pan = B.F.panel_q(src, L)
        T[m] = pp.join(pan.select_dtypes("number").drop(columns=["layer", "n_roles", "axis_proj"], errors="ignore"), how="left")
    return base_name, T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="q7")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--out-root", default="output/10-nov")
    a = ap.parse_args(argv)
    from pathlib import Path
    B.OUT = Path(a.out_root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cloud_predicts_position.figs import FAMILY_NAMES, family_of as _fo
    from cloud_predicts_position.panel import SHOWN, KEEP

    fam, L = a.family, a.layer
    nm = B.FAMS[fam]["name"]
    base_name, T = tables(fam, L)
    col = {m: (B.COL["base"] if m == base_name else B.COL[B.cond_of(m)]) for m in T}
    lab = {m: ("base" if m == base_name else B.CLAB[B.cond_of(m)]) for m in T}
    first = T[base_name]
    cols = [c for c in SHOWN if c in first.columns and all(c in t.columns and t[c].std() > 0 for t in T.values())]

    def fam_of(c):
        try:
            return _fo(c)
        except StopIteration:
            return "X"
    FN = dict(FAMILY_NAMES, X="other (response length, raw norm)")
    rng = np.random.default_rng(0)

    def rho(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[ok], y[ok])[0] if ok.sum() > 5 else np.nan

    def draw(ax, xcol, c, kind, xl):
        for m, t in T.items():
            x, y = t[xcol].to_numpy(float), t[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            xj = x + rng.normal(0, 0.012, len(x)) if kind == "em" else x
            ax.scatter(xj, y, s=4, color=col[m], alpha=0.25, linewidths=0)
            if kind == "em":
                edges = [-0.01, 0.001, 0.2, 0.4, 1.01]
            else:
                edges = np.quantile(x, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            pts = [(x[idx == i].mean(), np.median(y[idx == i])) for i in range(len(edges) - 1) if (idx == i).sum() >= 5]
            if pts:
                ax.plot(*zip(*pts), color=col[m], lw=1.8, label=f"{lab[m]} (ρ {rho(x, y):+.2f})")
        ax.set_title(f"{c} [{FN[fam_of(c)]}]", fontsize=8)
        ax.set_xlabel(xl, fontsize=7)

    def sheet(xcol, kind, xl, outdir, grid_path, title):
        outdir.mkdir(parents=True, exist_ok=True)
        for c in cols:
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            draw(ax, xcol, c, kind, xl); ax.set_ylabel(c); ax.legend(frameon=False, fontsize=6.5)
            fig.suptitle(f"{nm}, layer {L}", fontsize=8.5)
            fig.tight_layout(); fig.savefig(outdir / f"{c}.png", dpi=110); plt.close(fig)
        prim = [c for c in KEEP if c in cols]
        fig, axes = plt.subplots(3, 4, figsize=(20, 11))
        for ax, c in zip(axes.ravel(), prim):
            draw(ax, xcol, c, kind, xl)
        axes[0, 0].legend(frameon=False, fontsize=6)
        fig.suptitle(f"{nm}, layer {L}: {title}; lines = median per {'EM-share level' if kind == 'em' else 'position decile'}, one colour per model", fontsize=9)
        fig.tight_layout(); fig.savefig(grid_path, dpi=120); plt.close(fig)

    pp = B.OUT / f"partPersona_{fam}"
    tr = B.OUT / f"partTrends_{fam}"
    # plot 1
    sheet_em = pp / "plot1_em_vs_metric"
    sheet("em_share", "em", "EM share of the persona's 8 answers (jittered)", sheet_em / f"L{L}",
          sheet_em / f"F_primary_metrics_vs_em_L{L}.png", "metric vs EM share per persona")
    # plot 2 (both axes)
    for tag, xcol, xl in (("own_axis", "pos_own", "projection on this model's own assistant axis"),
                          ("base_axis", "pos_base", "projection on the base model's assistant axis")):
        sheet(xcol, "pos", xl, tr / tag / f"L{L}", tr / tag / f"F17_primary_metrics_vs_position_L{L}.png", "metric vs closeness to the assistant")
    # plot 3 (both axes)
    order = list(T)
    for tag, xcol, xl in (("own_axis", "pos_own", "persona's projection on this model's own refit assistant axis (+ = closer)"),
                          ("base_axis", "pos_base", "persona's projection on the base model's assistant axis (+ = closer)")):
        o = pp / "plot3_em_vs_position" / tag; o.mkdir(parents=True, exist_ok=True)
        for y, ylab in (("em_share", "EM share of the persona's 8 answers"), ("aligned_mean", "mean alignment score of the persona's 8 answers (0-100)")):
            nr = int(np.ceil(len(order) / 3))
            fig, axes = plt.subplots(nr, 3, squeeze=False, figsize=(13.8, 3.8 * nr), sharex=True, sharey=True)
            for ax, m in zip(axes.ravel(), order):
                t = T[m]
                yy = t[y] + (rng.normal(0, 0.01, len(t)) if y == "em_share" else 0)
                ax.scatter(t[xcol], yy, s=8, color=col[m], linewidths=0, alpha=0.8)
                ax.set_title(f"{lab[m]}: Spearman {rho(t[xcol].to_numpy(float), t[y].to_numpy(float)):+.2f}", fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel(xl, fontsize=7)
            for ax in axes[:, 0]:
                ax.set_ylabel(ylab, fontsize=7)
            fig.suptitle(f"{nm}, layer {L}: EM per persona vs closeness to the assistant ({tag.replace('_', ' ')})", fontsize=9)
            fig.tight_layout(); fig.savefig(o / f"F15_{y}_vs_position_L{L}.png", dpi=110); plt.close(fig)
    # spearman tables
    rows = [dict(model=m, metric=c, rho_em_share=rho(t.em_share.to_numpy(float), t[c].to_numpy(float)),
                 rho_aligned_mean=rho(t.aligned_mean.to_numpy(float), t[c].to_numpy(float))) for m, t in T.items() for c in cols]
    pd.DataFrame(rows).to_csv(sheet_em / f"spearman_L{L}.csv", index=False)
    print("wrote", pp, tr)


if __name__ == "__main__":
    main()
