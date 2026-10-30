"""partExtra: how much PGA direction i differs from PCA direction i.

For each layer the 276 persona vectors are written in their exact PCA coordinates (frame_last_L<L>.npz), so PCA i is the
unit vector e_i and PGA i is the column u_i of V. Two residuals per i:
  from PCA i          r_i   = |u_i - (u_i . e_i) e_i|^2 = 1 - cos^2(u_i, e_i)      (0 = same line, 1 = orthogonal)
  from span(PCA 1..i) R_i   = 1 - sum_{j<=i} (u_i . e_j)^2                          (share of PGA i outside the first i PCA directions)
Nulls: Gaussian and shuffled clouds with the same per-axis variances, graph and PGA refit (20 draws each).

    python -m pga_linearity.residuals --out output/29-oct/partExtra
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

from curvature_null.run import null
from pm_geometry.components import knn_graph, pga

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
INK, INK2 = "#15232e", "#4e5c68"
C = {19: "#2a78d6", 25: "#d6453f", 32: "#169a6b"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})


def residuals(V, kmax=10):
    r, R = [], []
    for i in range(kmax):
        u = V[:, i]
        r.append(1 - u[i] ** 2)
        R.append(1 - (u[: i + 1] ** 2).sum())
    return np.array(r), np.array(R)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="output/29-oct/partExtra"); ap.add_argument("--n-null", type=int, default=20)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, bands = [], {}
    for L in (19, 25, 32):
        f = np.load(PM / "PGALIN" / f"frame_last_L{L}.npz", allow_pickle=True)
        Z, V = f["Z"].astype(np.float64), f["V"].astype(np.float64)
        r, R = residuals(V)
        rng = np.random.default_rng(0)
        nr, nR = [], []
        for kind in ("gauss", "shuffle"):
            for _ in range(a.n_null):
                Zn = null(Z.astype(np.float32), kind, rng).astype(np.float64); Zn -= Zn.mean(0)
                P = pga(Zn, knn_graph(Zn)); rr, RR = residuals(P["V"]); nr.append(rr); nR.append(RR)
        nr, nR = np.array(nr), np.array(nR)
        bands[L] = dict(r=r, R=R, r_lo=np.percentile(nr, 5, 0), r_hi=np.percentile(nr, 95, 0), r_md=np.median(nr, 0),
                        R_lo=np.percentile(nR, 5, 0), R_hi=np.percentile(nR, 95, 0), R_md=np.median(nR, 0))
        for i in range(10):
            rows.append(dict(layer=L, i=i + 1, cos_pga_i_pca_i=round(float(V[i, i]), 3), residual_from_pca_i=round(float(r[i]), 3),
                             residual_from_pca_1_to_i=round(float(R[i]), 3), null_residual_from_pca_i_median=round(float(np.median(nr[:, i])), 3),
                             null_residual_from_pca_i_5_95=f"{np.percentile(nr[:, i], 5):.2f}–{np.percentile(nr[:, i], 95):.2f}",
                             null_residual_from_pca_1_to_i_median=round(float(np.median(nR[:, i])), 3)))
        print("layer", L, "done", flush=True)
    t = pd.DataFrame(rows); t.to_csv(out / "pga_vs_pca_residuals.csv", index=False)
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    k = np.arange(1, 11)
    for ax, key, title in ((axes[0], "r", "residual of PGA i from PCA i:  1 − cos²(PGA i, PCA i)"), (axes[1], "R", "residual of PGA i from span(PCA 1..i):  share of PGA i outside the first i PCA directions")):
        for L in (19, 25, 32):
            b = bands[L]
            ax.fill_between(k, b[f"{key}_lo"], b[f"{key}_hi"], color=C[L], alpha=0.12, linewidth=0)
            ax.plot(k, b[f"{key}_md"], color=C[L], linewidth=1, linestyle="--")
            ax.plot(k, b[key], color=C[L], linewidth=2, marker="o", markersize=4, label=f"layer {L}")
        ax.set_xticks(k); ax.set_xlabel("direction i"); ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("residual (0 = identical direction, 1 = orthogonal)")
    axes[0].legend(frameon=False, fontsize=8, title="solid: real personas · dashed + band: flat nulls (median, 5–95%)", title_fontsize=7.5)
    fig.suptitle("How far each Fletcher PGA direction is from the matching PCA direction (276 Qwen3-8B persona vectors, last answer token)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "pga_vs_pca_residuals.png"); plt.close(fig)
    # table png
    for L in (19, 25, 32):
        tl = t[t.layer == L][["i", "cos_pga_i_pca_i", "residual_from_pca_i", "null_residual_from_pca_i_5_95", "residual_from_pca_1_to_i", "null_residual_from_pca_1_to_i_median"]]
        fig, ax = plt.subplots(figsize=(11, 3.9), dpi=170); ax.axis("off")
        tab = ax.table(cellText=tl.values.tolist(), colLabels=["direction i", "cos(PGA i, PCA i)", "residual of PGA i\nfrom PCA i", "same residual on\nflat nulls (5–95%)", "residual of PGA i\nfrom span(PCA 1..i)", "same, flat nulls\n(median)"],
                       loc="upper center", cellLoc="center", colWidths=[0.1, 0.16, 0.17, 0.2, 0.19, 0.18])
        tab.auto_set_font_size(False); tab.set_fontsize(9.5)
        for (r_, c_), cell in tab.get_celld().items():
            cell.set_edgecolor("#c9d1d9"); cell.set_height(0.085)
            if r_ == 0: cell.set_facecolor("#dfe6ee"); cell.set_text_props(weight="bold"); cell.set_height(0.16)
            else: cell.set_facecolor("#f7f9fb" if r_ % 2 else "#ffffff")
        ax.set_title(f"Layer {L}: residual of each PGA direction from the matching PCA direction (1 − cos²) and from the first i PCA directions", fontsize=9.5, color=INK2, loc="left", pad=6)
        fig.savefig(out / f"pga_vs_pca_residuals_table_L{L}.png", bbox_inches="tight"); plt.close(fig)
    print(t[t.layer == 25][["i", "cos_pga_i_pca_i", "residual_from_pca_i", "null_residual_from_pca_i_5_95", "residual_from_pca_1_to_i", "null_residual_from_pca_1_to_i_median"]].to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
