"""Cluster maps + interpretability (Q4).

For EVERY clustering (HDBSCAN, DBSCAN, GMM, KMeans-optimal, KMeans k=12, KMeans
k=24 — all run in the PCA-95% space by 02_clustering.py) draw the role cloud in
FOUR embeddings: UMAP-2D, UMAP-3D, PCA-2D, PCA-3D. That is 6 methods × 4 views
= the 24 panels requested. In every panel:
  - points are colored by cluster (noise = grey),
  - the `default` Assistant persona is marked with a black ★ and labelled, so you
    can see where the Assistant sits in each partition,
  - a handful of representative role names are annotated (cluster-centroid-nearest
    + the global axis extremes) for orientation.

Also writes a per-method **interpretability** markdown/JSON: each cluster's size,
mean assistant-axis projection, whether it holds `default`, and its representative
member roles at both axis poles — enough to read each cluster semantically.

Usage:
    .venv/bin/python exploratory/assistant_axis/06_cluster_maps.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
import umap

from common import (load_points, center, savefig, resolve_run_dir,
                    assistant_axis, project, clusters_path)

METHODS = [  # (report key, parquet column suffix base, pretty title)
    ("hdbscan", "hdbscan", "HDBSCAN"),
    ("dbscan", "dbscan", "DBSCAN"),
    ("gmm", "gmm", "GMM"),
    ("kmeans", "kmeans", "KMeans (optimal k)"),
    ("kmeans_k12", "kmeansk12", "KMeans k=12"),
    ("kmeans_k24", "kmeansk24", "KMeans k=24"),
]


def _label_idx(labels, XY, roles, is_default, per_cluster=1, extremes=2, proj=None):
    """Indices of points to annotate: default + centroid-nearest per cluster +
    global axis extremes. Kept small so panels stay readable."""
    keep = set(np.where(is_default)[0].tolist())
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        centroid = XY[idx].mean(0)
        near = idx[np.argsort(((XY[idx] - centroid) ** 2).sum(1))[:per_cluster]]
        keep.update(near.tolist())
    if proj is not None:
        order = np.argsort(proj)
        keep.update(order[:extremes].tolist())
        keep.update(order[-extremes:].tolist())
    return sorted(keep)


def _scatter(ax, XY, labels, is_default, roles, proj, dim, title):
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    pos = [c for c in cats if c >= 0]
    for i, c in enumerate(pos):
        m = labels == c
        color = cmap(i % 20)
        args = [XY[m, 0], XY[m, 1]] + ([XY[m, 2]] if dim == 3 else [])
        ax.scatter(*args, s=14, color=color, linewidths=0, alpha=0.85)
    mn = labels < 0
    if mn.any():
        args = [XY[mn, 0], XY[mn, 1]] + ([XY[mn, 2]] if dim == 3 else [])
        ax.scatter(*args, s=10, color="#cccccc", linewidths=0, alpha=0.7)
    # default ★
    d = is_default
    if d.any():
        args = [XY[d, 0], XY[d, 1]] + ([XY[d, 2]] if dim == 3 else [])
        ax.scatter(*args, s=180, marker="*", color="k", edgecolors="w",
                   linewidths=0.8, zorder=5)
    # role-name labels
    for i in _label_idx(labels, XY, roles, is_default, proj=proj):
        coord = list(XY[i, :dim])
        txt = roles[i] + (" (Assistant)" if is_default[i] else "")
        if dim == 3:
            ax.text(coord[0], coord[1], coord[2], txt, fontsize=6,
                    color="k" if is_default[i] else "#333")
        else:
            ax.annotate(txt, coord, fontsize=6.5,
                        color="k" if is_default[i] else "#333",
                        fontweight="bold" if is_default[i] else "normal")
    ax.set_title(title, fontsize=10)
    if dim == 2:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def semantic_lines(labels, roles, proj, top=6):
    lines, rows = [], []
    for c in sorted(set(labels)):
        if c < 0:
            n = int((labels < 0).sum())
            lines.append(f"- **noise** (n={n}): points HDBSCAN/DBSCAN could not assign to any dense cluster")
            continue
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(-proj[idx])]
        names = [roles[i] for i in order]
        has_def = "default" in names
        head = ", ".join(names[:top])
        tail = ", ".join(names[-min(top, len(names)):][::-1])
        badge = " · **contains `default` (the Assistant)**" if has_def else ""
        lines.append(
            f"- **c{c}** (n={len(idx)}, mean-axis={proj[idx].mean():+.2f}){badge}: "
            f"Assistant-end members — {head}"
            + (f"; opposite pole — {tail}" if len(names) > top else ""))
        rows.append({"cluster": int(c), "size": int(len(idx)),
                     "mean_axis_proj": round(float(proj[idx].mean()), 3),
                     "has_default": has_def,
                     "assistant_end": head, "far_end": tail})
    return lines, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X_raw, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    Xs = center(X_raw)
    roles = meta["role"].values
    is_default = roles == "default"
    axis = assistant_axis(Xs, meta)
    proj = project(Xs, axis)
    pca_tag = f"pca{int(args.pca_var*100)}"
    tag = f"{args.view}_L{layer}"

    clus_path = clusters_path(args.view, layer)
    if not clus_path.exists():
        raise SystemExit(f"missing {clus_path}; run 02_clustering.py first")
    cdf = pd.read_parquet(clus_path).set_index("role")

    # Embeddings computed once on the centered means.
    print("Fitting UMAP 2D/3D + PCA ...")
    XY2 = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    metric="cosine", random_state=args.seed).fit_transform(Xs)
    XY3 = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    metric="cosine", random_state=args.seed).fit_transform(Xs)
    pca = PCA(n_components=3, random_state=0).fit(Xs)
    XP = pca.transform(Xs)
    var = pca.explained_variance_ratio_
    embeds = [("umap2d", XY2, 2, "UMAP 2D"), ("umap3d", XY3, 3, "UMAP 3D"),
              ("pca2d", XP, 2, f"PCA 2D (PC1 {var[0]*100:.0f}%, PC2 {var[1]*100:.0f}%)"),
              ("pca3d", XP, 3, "PCA 3D (PC1–3)")]

    interp = {}
    for mkey, base, pretty in METHODS:
        col = f"{base}_{pca_tag}"
        if col not in cdf.columns:
            print(f"  skip {mkey}: no column {col}")
            continue
        labels = cdf[col].reindex(roles).values

        # 2x2 figure: the 4 embeddings for this method
        fig = plt.figure(figsize=(15, 12))
        for j, (ekey, XY, dim, etitle) in enumerate(embeds):
            ax = fig.add_subplot(2, 2, j + 1, projection="3d" if dim == 3 else None)
            _scatter(ax, XY, labels, is_default, roles, proj, dim, etitle)
        n_clu = len([c for c in set(labels) if c >= 0])
        noise = float((labels < 0).mean())
        fig.suptitle(f"{pretty} — {n_clu} clusters, {noise*100:.0f}% noise  "
                     f"[{pca_tag}, {args.view}, layer {layer}]  ★ = default (Assistant)",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        savefig(fig, f"06_{mkey}_maps_{tag}.png", run_dir)
        plt.close(fig)

        lines, rows = semantic_lines(labels, roles, proj)
        dcl = [r for r in rows if r["has_default"]]
        interp[mkey] = {"pretty": pretty, "n_clusters": n_clu, "noise_frac": noise,
                        "default_cluster": dcl[0]["cluster"] if dcl else None,
                        "lines": lines, "rows": rows}

    json.dump(interp, open(run_dir / f"06_cluster_interpretability_{tag}.json", "w"), indent=2)

    # A readable markdown fragment the report can inline.
    md = ["## Cluster interpretability (per method)\n"]
    for mkey, base, pretty in METHODS:
        if mkey not in interp:
            continue
        d = interp[mkey]
        where = (f"`default` sits in cluster **c{d['default_cluster']}**"
                 if d["default_cluster"] is not None else "`default` fell in the noise set")
        md.append(f"### {pretty} — {d['n_clusters']} clusters, {d['noise_frac']*100:.0f}% noise")
        md.append(f"{where}.\n")
        md.extend(d["lines"])
        md.append("")
    (run_dir / f"06_cluster_interpretability_{tag}.md").write_text("\n".join(md))
    print("wrote interpretability md + 24 panels")


if __name__ == "__main__":
    main()
