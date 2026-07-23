"""UMAP + Assistant-Axis recovery for the character-role cloud.

Produces:
  - 2D UMAP colored by (a) cluster and (b) our assistant-axis projection,
    with the `default` (Assistant baseline) points highlighted.
  - 3D interactive UMAP (HTML) colored by axis projection, hover = role.
  - PCA(1,2) colored by axis projection; reports cosine(PC1, assistant_axis)
    to test whether the leading component IS the Assistant Axis (arXiv:2601.10387).

Assistant axis is computed in our own embedding space as
mean(default points) - mean(all points) -- see common.assistant_axis.

Usage:
    .venv/bin/python exploratory/assistant_axis/03_umap_axis.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from manifold_persona.common import (load_points, center, savefig, resolve_run_dir,
                    assistant_axis, project, DEFAULT_COLOR,
                    ward_families, clusters_path, distinct_colors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--cluster_col", default="kmeans_pca95")
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X_raw, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    Xs = center(X_raw)

    # Assistant-axis projection (computed on the centered space, per the paper).
    axis = assistant_axis(Xs, meta)
    proj = project(Xs, axis)
    is_default = meta["role"].values == "default"

    # Cluster labels if available (role-mean parquet; align by role, not position).
    clus_path = clusters_path(args.view, layer)
    cluster_labels = None
    if clus_path.exists():
        cdf = pd.read_parquet(clus_path)
        if args.cluster_col in cdf.columns:
            cluster_labels = (cdf.set_index("role")[args.cluster_col]
                              .reindex(meta["role"].values).values)

    print(f"Fitting UMAP 2D/3D on {Xs.shape} ...")
    r2 = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                   metric="cosine", random_state=args.seed)
    XY2 = r2.fit_transform(Xs)
    r3 = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                   metric="cosine", random_state=args.seed)
    XY3 = r3.fit_transform(Xs)

    # --- 2D panels: axis projection + cluster ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    sc = axes[0].scatter(XY2[:, 0], XY2[:, 1], c=proj, s=7, cmap="coolwarm", linewidths=0)
    axes[0].scatter(XY2[is_default, 0], XY2[is_default, 1], s=40, facecolors="none",
                    edgecolors=DEFAULT_COLOR, linewidths=1.2, label="default (Assistant)")
    axes[0].set_title(f"UMAP 2D — assistant-axis projection (layer {layer})")
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=axes[0], label="→ more Assistant-like")

    if cluster_labels is not None:
        cats = sorted(set(cluster_labels))
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(cats):
            m = cluster_labels == c
            color = "#bbbbbb" if c == -1 else cmap(i % 20)
            axes[1].scatter(XY2[m, 0], XY2[m, 1], s=7, color=color, linewidths=0,
                            label=f"c{c}" if c != -1 else "noise")
        axes[1].set_title(f"UMAP 2D — cluster ({args.cluster_col})")
        if len(cats) <= 20:
            axes[1].legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    savefig(fig, f"03_umap2d_{args.view}_L{layer}.png", run_dir)

    # --- PCA(1,2) + Assistant-Axis alignment ---
    # Main space is centered (covariance PCA) = the paper's §2.1.3 normalization.
    pca = PCA(n_components=5, random_state=0).fit(Xs)
    XP = pca.transform(Xs)
    pc1 = pca.components_[0]
    cos_pc1_axis = float(abs(np.dot(pc1, axis) / (np.linalg.norm(pc1) * np.linalg.norm(axis) + 1e-8)))
    # Contrast: had we z-scored instead, the feature reweighting destroys the
    # alignment. Compute it once so the report can justify the centering choice.
    Xz = (X_raw - X_raw.mean(0, keepdims=True))
    sdz = X_raw.std(0, keepdims=True); sdz[sdz == 0] = 1.0
    Xz = Xz / sdz
    axis_z = Xz[is_default].mean(0) - Xz.mean(0)
    axis_z = axis_z / (np.linalg.norm(axis_z) + 1e-8)
    pc1_z = PCA(n_components=1, random_state=0).fit(Xz).components_[0]
    cos_pc1_axis_zscored = float(abs(np.dot(pc1_z, axis_z) /
                                     (np.linalg.norm(pc1_z) * np.linalg.norm(axis_z) + 1e-8)))
    print(f"cos(PC1,axis) centered(paper)={cos_pc1_axis:.2f}  z-scored={cos_pc1_axis_zscored:.2f}")
    fig2, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    sc2 = ax.scatter(XP[:, 0], XP[:, 1], c=proj, s=7, cmap="coolwarm", linewidths=0)
    ax.scatter(XP[is_default, 0], XP[is_default, 1], s=40, facecolors="none",
               edgecolors=DEFAULT_COLOR, linewidths=1.2, label="default (Assistant)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
    ax.set_title(f"PCA(1,2) colored by assistant-axis projection\n"
                 f"|cos(PC1, assistant_axis)| = {cos_pc1_axis:.2f}")
    fig2.colorbar(sc2, ax=ax, label="→ more Assistant-like")
    fig2.tight_layout()
    savefig(fig2, f"03_pca12_{args.view}_L{layer}.png", run_dir)

    # --- role means colored by Ward family (each point = one role) ---
    fam = ward_families(Xs, k=15)[0]
    uniqf = sorted(set(fam))
    fcols = {f: c for f, c in zip(uniqf, distinct_colors(len(uniqf)))}
    pt_colors = [fcols[f] for f in fam]
    figr, axr = plt.subplots(1, 2, figsize=(14, 6.5))
    axr[0].scatter(XY2[:, 0], XY2[:, 1], s=18, c=pt_colors, linewidths=0, alpha=0.9)
    axr[0].set_title(f"UMAP 2D by Ward family (layer {layer})\n"
                     f"{len(uniqf)} families — one point per role")
    axr[0].set_xticks([]); axr[0].set_yticks([])
    axr[1].scatter(XP[:, 0], XP[:, 1], s=18, c=pt_colors, linewidths=0, alpha=0.9)
    axr[1].set_title(f"PCA(1,2) by Ward family (layer {layer})")
    axr[1].set_xlabel("PC1"); axr[1].set_ylabel("PC2")
    figr.tight_layout()
    savefig(figr, f"03_by_family_{args.view}_L{layer}.png", run_dir)

    # --- density clusterings (HDBSCAN, DBSCAN) on the UMAP layout ---
    # Shows *why* §2 says there are no density clusters: noise (grey) dominates.
    if clus_path.exists():
        cdf_all = pd.read_parquet(clus_path).set_index("role")
        dens_cols = [("hdbscan_pca95", "HDBSCAN"), ("dbscan_pca95", "DBSCAN")]
        avail = [(c, t) for c, t in dens_cols if c in cdf_all.columns]
        if avail:
            figd, axd = plt.subplots(1, len(avail), figsize=(7 * len(avail), 6.2),
                                     squeeze=False)
            for ax, (col, title) in zip(axd[0], avail):
                lab = cdf_all[col].reindex(meta["role"].values).values
                cats = sorted(set(lab))
                n_clu = len([c for c in cats if c >= 0])
                noise = float((lab < 0).mean())
                cmap = plt.get_cmap("tab20")
                for i, c in enumerate([c for c in cats if c >= 0]):
                    m = lab == c
                    ax.scatter(XY2[m, 0], XY2[m, 1], s=20, color=cmap(i % 20),
                               linewidths=0, label=f"c{c} (n={int(m.sum())})")
                mn = lab < 0
                if mn.any():
                    ax.scatter(XY2[mn, 0], XY2[mn, 1], s=14, color="#cccccc",
                               linewidths=0, label=f"noise ({noise*100:.0f}%)")
                ax.set_title(f"{title} (pca95) on UMAP — {n_clu} clusters, "
                             f"{noise*100:.0f}% noise")
                ax.set_xticks([]); ax.set_yticks([])
                if n_clu <= 12:
                    ax.legend(markerscale=1.5, fontsize=7, loc="best")
            figd.tight_layout()
            savefig(figd, f"03_density_clusters_{args.view}_L{layer}.png", run_dir)

    # --- 3D interactive ---
    import plotly.express as px
    df3 = pd.DataFrame({"x": XY3[:, 0], "y": XY3[:, 1], "z": XY3[:, 2],
                        "axis_projection": proj, "role": meta["role"].values,
                        "is_default": is_default})
    fig3 = px.scatter_3d(df3, x="x", y="y", z="z", color="axis_projection",
                         color_continuous_scale="RdBu", hover_data=["role", "is_default"],
                         title=f"UMAP 3D — assistant-axis projection (layer {layer})")
    fig3.update_traces(marker=dict(size=2.5, opacity=0.75))
    fig3.write_html(str(run_dir / f"03_umap3d_axis_{args.view}_L{layer}.html"))
    print("wrote", run_dir / f"03_umap3d_axis_{args.view}_L{layer}.html")

    # --- 3D interactive in PCA space (PC1,2,3), colored by assistant-axis proj ---
    var = pca.explained_variance_ratio_
    dfpca = pd.DataFrame({"PC1": XP[:, 0], "PC2": XP[:, 1], "PC3": XP[:, 2],
                          "axis_projection": proj, "role": meta["role"].values,
                          "is_default": is_default})
    figp = px.scatter_3d(dfpca, x="PC1", y="PC2", z="PC3", color="axis_projection",
                         color_continuous_scale="RdBu", hover_data=["role", "is_default"],
                         title=(f"PCA(1,2,3) — assistant-axis projection (layer {layer})  "
                                f"[var: {var[0]*100:.0f}/{var[1]*100:.0f}/{var[2]*100:.0f}%,  "
                                f"|cos(PC1,axis)|={cos_pc1_axis:.2f}]"))
    figp.update_traces(marker=dict(size=2.5, opacity=0.75))
    figp.update_layout(scene=dict(xaxis_title=f"PC1 ({var[0]*100:.0f}%)",
                                  yaxis_title=f"PC2 ({var[1]*100:.0f}%)",
                                  zaxis_title=f"PC3 ({var[2]*100:.0f}%)"))
    figp.write_html(str(run_dir / f"03_pca3d_axis_{args.view}_L{layer}.html"))
    print("wrote", run_dir / f"03_pca3d_axis_{args.view}_L{layer}.html")

    # --- Save axis-projection ranking of roles (mean per role) ---
    dfp = pd.DataFrame({"role": meta["role"].values, "proj": proj})
    role_proj = dfp.groupby("role")["proj"].mean().sort_values()
    ranking = {"cos_pc1_axis": cos_pc1_axis,
               "cos_pc1_axis_zscored": cos_pc1_axis_zscored,
               "pca_var_top3": [float(x) for x in pca.explained_variance_ratio_[:3]],
               "most_assistant_like": role_proj.tail(15)[::-1].round(3).to_dict(),
               "least_assistant_like": role_proj.head(15).round(3).to_dict(),
               "default_rank_from_top": int((role_proj.rank(ascending=False)["default"]))
               if "default" in role_proj.index else None}
    json.dump(ranking, open(run_dir / f"03_axis_ranking_{args.view}_L{layer}.json", "w"), indent=2)
    role_proj.to_csv(run_dir / f"03_role_axis_projection_{args.view}_L{layer}.csv")
    print(f"cos(PC1, axis)={cos_pc1_axis:.2f}  default rank from top: {ranking['default_rank_from_top']}")


if __name__ == "__main__":
    main()
