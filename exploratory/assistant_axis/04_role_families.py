"""Role families: what groups together, and how Assistant-like each group is.

Two views:
  (A) Hierarchical clustering of the 276 per-role centroids (Ward linkage) ->
      dendrogram + a cut into K families, with each family's member roles listed.
      This is the analogue of "traits per cluster" for the archetype cloud.
  (B) For a point-cloud clustering column, the composition of each cluster:
      size, top member roles, %default, and mean assistant-axis projection.

Usage:
    .venv/bin/python exploratory/assistant_axis/04_role_families.py --cluster_col kmeans_pca50
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from common import (load_points, center, savefig, resolve_run_dir,
                    assistant_axis, project, role_centroids, clusters_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--cluster_col", default="kmeans_pca95")
    ap.add_argument("--n_families", type=int, default=15)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X_raw, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    Xs = center(X_raw)
    axis = assistant_axis(Xs, meta)
    proj = project(Xs, axis)
    tag = f"{args.view}_L{layer}"

    # ---- (A) Hierarchical role families from per-role centroids ----
    roles, C = role_centroids(Xs, meta)
    Z = linkage(C, method="ward")
    fam = fcluster(Z, t=args.n_families, criterion="maxclust")
    role_proj = pd.Series(proj, index=meta["role"].values).groupby(level=0).mean()

    families = {}
    for f in sorted(set(fam)):
        members = [roles[i] for i in range(len(roles)) if fam[i] == f]
        members_sorted = sorted(members, key=lambda r: -role_proj.get(r, 0))
        families[int(f)] = {
            "size": len(members),
            "mean_axis_proj": float(np.mean([role_proj[r] for r in members])),
            "roles": members_sorted,
        }
    # order families by how Assistant-like they are
    fam_order = sorted(families, key=lambda f: -families[f]["mean_axis_proj"])
    print(f"\n== {args.n_families} role families (by mean assistant-axis projection) ==")
    for f in fam_order:
        d = families[f]
        head = ", ".join(d["roles"][:10])
        print(f"  fam{f} (n={d['size']}, proj={d['mean_axis_proj']:+.2f}): {head}"
              + (" ..." if d["size"] > 10 else ""))
    json.dump({"families": families, "order_by_assistant_like": fam_order},
              open(run_dir / f"04_role_families_{tag}.json", "w"), indent=2)

    # Dendrogram (truncated) colored at the family cut
    fig, ax = plt.subplots(figsize=(13, 5.5))
    dendrogram(Z, truncate_mode="lastp", p=args.n_families * 2, ax=ax,
               color_threshold=Z[-(args.n_families - 1), 2], no_labels=False,
               leaf_rotation=90, leaf_font_size=7)
    ax.set_title(f"Role-centroid hierarchy (Ward), cut into {args.n_families} families — layer {layer}")
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    savefig(fig, f"04_role_dendrogram_{tag}.png", run_dir)

    # ---- (B) Composition of a role-mean clustering (each row = one role) ----
    clus_path = clusters_path(args.view, layer)
    if clus_path.exists():
        cdf = pd.read_parquet(clus_path)
        if args.cluster_col in cdf.columns:
            df = cdf.copy()
            df["proj"] = df["role"].map(pd.Series(proj, index=meta["role"].values))
            df = df[df[args.cluster_col] >= 0]
            rows = []
            for c, sub in df.groupby(args.cluster_col):
                members = sub.sort_values("proj", ascending=False)["role"].tolist()
                rows.append({
                    "cluster": int(c), "size": len(sub),
                    "has_default": bool((sub["role"] == "default").any()),
                    "mean_axis_proj": round(float(sub["proj"].mean()), 3),
                    "roles": ", ".join(members[:8]) + (" …" if len(members) > 8 else ""),
                })
            comp = pd.DataFrame(rows).sort_values("mean_axis_proj", ascending=False)
            comp.to_csv(run_dir / f"04_cluster_composition_{args.cluster_col}_{tag}.csv", index=False)
            print(f"\n== Cluster composition ({args.cluster_col}) ==")
            print(comp.to_string(index=False))

            # Bar: mean axis projection per cluster
            fig2, ax2 = plt.subplots(figsize=(max(6, 0.5 * len(comp)), 4.5))
            order = comp.sort_values("mean_axis_proj")
            colors = plt.get_cmap("coolwarm")(
                (order["mean_axis_proj"] - order["mean_axis_proj"].min()) /
                (order["mean_axis_proj"].max() - order["mean_axis_proj"].min() + 1e-9))
            ax2.bar(range(len(order)), order["mean_axis_proj"], color=colors)
            ax2.set_xticks(range(len(order)))
            ax2.set_xticklabels([f"c{c}" for c in order["cluster"]], rotation=0, fontsize=8)
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_ylabel("mean assistant-axis projection")
            ax2.set_title(f"How Assistant-like is each cluster? ({args.cluster_col}, layer {layer})")
            fig2.tight_layout()
            savefig(fig2, f"04_cluster_axis_{args.cluster_col}_{tag}.png", run_dir)


if __name__ == "__main__":
    main()
