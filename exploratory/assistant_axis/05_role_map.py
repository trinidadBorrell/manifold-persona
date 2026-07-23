"""Map of the 276 roles: UMAP and PCA of the per-role centroids.

One point per role (centroid of all its points), laid out with UMAP and PCA and
colored by the Ward family from 04_role_families.py. Produces a static 2-panel
figure (UMAP-2D | PCA(1,2)) with role labels + interactive 3D HTMLs (UMAP + PCA)
with the role name on hover.

Usage:
    .venv/bin/python exploratory/assistant_axis/05_role_map.py
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import umap

from manifold_persona.common import (load_points, center, savefig, resolve_run_dir,
                    role_centroids, distinct_colors)


def family_map(run_dir, view, layer):
    fs = sorted(glob.glob(str(run_dir / f"04_role_families_{view}_L*.json")))
    if fs:
        fam = json.load(open(fs[-1]))["families"]
        return {name: int(f) for f, d in fam.items() for name in d["roles"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n_families", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    Xs = center(X)
    names, C = role_centroids(Xs, meta)     # 276 × hidden
    print(f"{len(names)} role centroids, layer {layer}")

    fmap = family_map(run_dir, args.view, layer)
    if fmap:
        fam = np.array([fmap.get(n, -1) for n in names])
    else:
        Z = linkage(C, method="ward")
        fam = fcluster(Z, t=args.n_families, criterion="maxclust")

    U2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, metric="cosine",
                   random_state=args.seed).fit_transform(C)
    U3 = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.3, metric="cosine",
                   random_state=args.seed).fit_transform(C)
    pca = PCA(n_components=3, random_state=0).fit(C)
    P = pca.transform(C); var = pca.explained_variance_ratio_

    uniqf = sorted(set(fam))
    cols = distinct_colors(len(uniqf))
    cmap = {f: cols[i] for i, f in enumerate(uniqf)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    panels = [(U2, "UMAP-2D of role centroids", "UMAP-1", "UMAP-2"),
              (P[:, :2], "PCA(1,2) of role centroids",
               f"PC1 ({var[0]*100:.0f}%)", f"PC2 ({var[1]*100:.0f}%)")]
    for ax, (XY, ttl, xl, yl) in zip(axes, panels):
        for f in uniqf:
            m = fam == f
            ax.scatter(XY[m, 0], XY[m, 1], s=22, color=cmap[f], linewidths=0)
        for i, nm in enumerate(names):
            ax.annotate(nm, (XY[i, 0], XY[i, 1]), fontsize=4.5, alpha=0.75)
        ax.set_title(f"{ttl} (layer {layer}, colored by family)")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    savefig(fig, f"05_role_map_{args.view}_L{layer}.png", run_dir)

    import plotly.express as px
    for coords, pfx, pretty, ax3 in [
        (U3, "umap", "UMAP", ("UMAP-1", "UMAP-2", "UMAP-3")),
        (P,  "pca",  "PCA",  (f"PC1 ({var[0]*100:.0f}%)", f"PC2 ({var[1]*100:.0f}%)",
                              f"PC3 ({var[2]*100:.0f}%)")),
    ]:
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
                           "role": names, "family": [f"fam{f}" for f in fam]})
        fig3 = px.scatter_3d(df, x="x", y="y", z="z", color="family", hover_name="role",
                             hover_data={"x": False, "y": False, "z": False, "family": True},
                             title=f"{pretty} 3D of role centroids (layer {layer})")
        fig3.update_traces(marker=dict(size=4, opacity=0.85))
        fig3.update_layout(scene=dict(xaxis_title=ax3[0], yaxis_title=ax3[1], zaxis_title=ax3[2]))
        out = run_dir / f"05_{pfx}3d_role_map_{args.view}_L{layer}.html"
        fig3.write_html(str(out)); print("wrote", out)


if __name__ == "__main__":
    main()
