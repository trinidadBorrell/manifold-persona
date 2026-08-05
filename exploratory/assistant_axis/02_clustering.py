"""Clustering of the character-role persona cloud (assistant-axis study).

Each point is ONE role-mean vector (mean across answers; ~276 points). Clustering
runs in a **PCA-95%** feature space (PCA keeping 95% of the variance — denoises
the dim>>N regime) and, for reference, the full centered space. We report
**internal** validity only (silhouette, Davies–Bouldin, cluster counts, noise
fraction). We deliberately do **not** score against a Ward "family" partition:
after aggregation each role is one point, so any family labelling is just another
clustering of the same points — scoring against it is circular. The Ward
hierarchy is kept purely descriptively in 04 (dendrogram) / 05 (map).

Per-method **interpretability**: for every clustering we save each cluster's
size, mean assistant-axis projection, whether it contains the `default`
Assistant, and its representative roles (nearest the centroid + axis extremes),
so clusters can be read semantically.

Saves assignments to data/.../clusters_rolemean_<view>_L<layer>.parquet.

Usage:
    .venv/bin/python exploratory/assistant_axis/02_clustering.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan

from manifold_persona.common import (load_points, center, savefig, resolve_run_dir,
                    assistant_axis, project, clusters_path)


SIL_SAMPLE = 2500


def _sil(X, labels):
    n = len(labels)
    return float(silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, n), random_state=0))


def internal_scores(X, labels):
    mask = labels >= 0
    out = {"n_clusters": int(len(set(labels[mask]))),
           "noise_frac": float((~mask).mean())}
    if len(set(labels[mask])) < 2 or mask.sum() < 3:
        out.update({"silhouette": None, "davies_bouldin": None})
        return out
    out.update({"silhouette": _sil(X[mask], labels[mask]),
                "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask]))})
    return out


def valid_k_range(n, lo=3, hi=31):
    """k values that KMeans+silhouette can actually accept for n points.

    silhouette_score requires 2 <= n_labels <= n-1, so k must stop at n-1. The
    default point cloud is one centroid per ROLE (load_points aggregates), so n
    is the role count -- 276 for a full run, but as few as a handful for a smoke
    test with --roles. Without this clamp the hardcoded range(3, 31) crashes.
    """
    return range(lo, min(hi, n))


def choose_k(X, k_range):
    sils = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        sils[k] = _sil(X, km.labels_)
    if not sils:
        raise ValueError(f"no valid k for {len(X)} points; need at least 4")
    return max(sils, key=sils.get), sils


def cluster_composition(labels, roles, proj, top=8):
    """Semantic read of each cluster: size, mean axis proj, has_default,
    representative roles (highest + lowest axis projection = the poles that make
    the cluster interpretable)."""
    comp = []
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        sub = sorted(idx, key=lambda i: -proj[i])
        names = [roles[i] for i in sub]
        comp.append({
            "cluster": int(c),
            "size": int(len(idx)),
            "mean_axis_proj": round(float(proj[idx].mean()), 3),
            "has_default": bool("default" in names),
            "top_assistant_like": names[:top],
            "least_assistant_like": names[-min(top, len(names)):][::-1],
        })
    comp.sort(key=lambda d: -d["mean_axis_proj"])
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pca_var", type=float, default=0.95,
                    help="variance fraction to keep for the PCA clustering space")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X_raw, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    Xs = center(X_raw)
    roles = meta["role"].values

    # PCA-95%: number of components carrying `pca_var` of the variance.
    s = np.linalg.svd(Xs, full_matrices=False, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    d_var = int(np.searchsorted(cum, args.pca_var) + 1)
    Xp = PCA(n_components=d_var, random_state=0).fit_transform(Xs)
    pca_tag = f"pca{int(args.pca_var*100)}"

    axis = assistant_axis(Xs, meta)
    proj = project(Xs, axis)
    print(f"view={args.view} layer={layer} N={Xs.shape[0]} roles={meta['role'].nunique()} "
          f"{pca_tag}={d_var} comps ({args.pca_var:.0%} var)")

    spaces = {"centered": Xs, pca_tag: Xp}
    report = {"_meta": {"view": args.view, "layer": layer, "n": int(Xs.shape[0]),
                        "n_roles": int(meta["role"].nunique()),
                        "pca_var": args.pca_var, "pca_dim": d_var,
                        "cluster_space": pca_tag}}
    assignments = {}
    compositions = {}
    k_range = valid_k_range(Xs.shape[0])

    for name, X in spaces.items():
        report[name] = {}
        best_k, sils = choose_k(X, k_range)
        km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X)
        report[name]["kmeans"] = {"k": best_k, "silhouette_by_k": sils, **internal_scores(X, km.labels_)}
        assignments[f"kmeans_{name}"] = km.labels_

        for kfix in [k for k in (12, 24) if k < X.shape[0]]:
            kmf = KMeans(n_clusters=kfix, n_init=10, random_state=0).fit(X)
            report[name][f"kmeans_k{kfix}"] = {"k": kfix, **internal_scores(X, kmf.labels_)}
            assignments[f"kmeansk{kfix}_{name}"] = kmf.labels_

        gmm = GaussianMixture(n_components=best_k, random_state=0).fit(X)
        gl = gmm.predict(X)
        report[name]["gmm"] = {"k": best_k, **internal_scores(X, gl)}
        assignments[f"gmm_{name}"] = gl

        hdb = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3).fit(X)
        report[name]["hdbscan"] = internal_scores(X, hdb.labels_)
        assignments[f"hdbscan_{name}"] = hdb.labels_

        nn = NearestNeighbors(n_neighbors=5).fit(X)
        d, _ = nn.kneighbors(X)
        eps = float(np.median(d[:, -1]))
        db = DBSCAN(eps=eps, min_samples=3).fit(X)
        report[name]["dbscan"] = {"eps": eps, **internal_scores(X, db.labels_)}
        assignments[f"dbscan_{name}"] = db.labels_

        print(f"[{name}] kmeans k={best_k} sil={report[name]['kmeans']['silhouette']:.3f} | "
              f"hdbscan clusters={report[name]['hdbscan'].get('n_clusters')} "
              f"noise={report[name]['hdbscan'].get('noise_frac'):.0%}")

    # --- Interpretability: per-method cluster composition on the PCA space ---
    method_cols = {"kmeans": f"kmeans_{pca_tag}", "kmeans_k12": f"kmeansk12_{pca_tag}",
                   "kmeans_k24": f"kmeansk24_{pca_tag}", "gmm": f"gmm_{pca_tag}",
                   "hdbscan": f"hdbscan_{pca_tag}", "dbscan": f"dbscan_{pca_tag}"}
    method_cols = {m: c for m, c in method_cols.items() if c in assignments}
    for m, col in method_cols.items():
        compositions[m] = cluster_composition(assignments[col], roles, proj)
        dcl = [c for c in compositions[m] if c["has_default"]]
        report[pca_tag][m]["default_cluster"] = dcl[0]["cluster"] if dcl else None
    json.dump(compositions, open(run_dir / f"02_cluster_composition_{args.view}_L{layer}.json", "w"), indent=2)

    json.dump(report, open(run_dir / f"clustering_{args.view}_L{layer}.json", "w"), indent=2)
    assign_df = meta.copy()
    for col, labels in assignments.items():
        assign_df[col] = labels
    assign_path = clusters_path(args.view, layer)
    assign_df.to_parquet(assign_path, index=False)
    print("wrote", assign_path)

    # Plot: silhouette-vs-k + per-method silhouette (no external family metric) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sils = report[pca_tag]["kmeans"]["silhouette_by_k"]
    ks = sorted(int(k) for k in sils)
    axes[0].plot(ks, [sils[k] if k in sils else sils[str(k)] for k in ks], "-o", color="#0072B2")
    axes[0].axvline(report[pca_tag]["kmeans"]["k"], color="#D55E00", ls="--",
                    label=f"best k={report[pca_tag]['kmeans']['k']}")
    axes[0].set_xlabel("k (KMeans)"); axes[0].set_ylabel("silhouette")
    axes[0].set_title(f"KMeans model selection ({pca_tag})"); axes[0].legend()

    methods = [m for m in ("kmeans", "kmeans_k12", "kmeans_k24", "gmm", "hdbscan", "dbscan")
               if m in report[pca_tag]]
    sil_vals = [report[pca_tag][m].get("silhouette") or 0 for m in methods]
    nois = [report[pca_tag][m].get("noise_frac") or 0 for m in methods]
    x = np.arange(len(methods))
    axes[1].bar(x - 0.2, sil_vals, 0.4, label="silhouette", color="#009E73")
    axes[1].bar(x + 0.2, nois, 0.4, label="noise fraction", color="#CC79A7")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].set_ylabel("score"); axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Internal validity per method ({pca_tag})"); axes[1].legend()
    fig.tight_layout()
    savefig(fig, f"02_clustering_{args.view}_L{layer}.png", run_dir)


if __name__ == "__main__":
    main()
