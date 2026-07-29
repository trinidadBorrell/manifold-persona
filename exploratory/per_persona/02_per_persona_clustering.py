"""Clustering inside each role — 276 separate clustering studies.

Mirrors exploratory/assistant_axis/02_clustering.py (same methods, same internal
scores, same PCA-95% working space) but run once per role on that role's own 25
points, and — unlike the assistant-axis version — scored against **real external
labels**. That is the one thing this study can do that the between-role study
cannot: after role-mean aggregation each role is a single point and any grouping
of those points is circular (assistant_axis/02 says so explicitly), whereas here
every point carries a known instruction index and a known question index. So we
score ARI / NMI against both, exactly as exploratory/persona_vectors/02 scores
against trait and polarity.

The question is therefore not "does a role cluster?" — 25 points always split —
but **what does it cluster BY**: the phrasing of the persona instruction, or the
question being answered? And is that split any different from the design null,
which has the same 5x5 grid and no persona at all?

k is capped at 8, not 30: on 25 points a k of 30 is undefined and even k=12
leaves 2 points per cluster. Methods that need density (HDBSCAN/DBSCAN) run with
min_cluster_size=2 for the same reason.

Usage:
    .venv/bin/python exploratory/per_persona/02_per_persona_clustering.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)
import hdbscan

from common import (load_role_clouds, resolve_run_dir, savefig, design_null_draws,
                    band, single_threaded, C_REAL, C_DESIGN, C_INSTR, C_QUEST)

K_RANGE = range(2, 9)          # 25 points: k=8 already means ~3 points per cluster
METHODS = ["kmeans", "kmeans_k5", "gmm", "hdbscan", "dbscan"]


def internal_scores(X, labels) -> dict:
    mask = labels >= 0
    out = {"n_clusters": int(len(set(labels[mask]))), "noise_frac": float((~mask).mean())}
    if len(set(labels[mask])) < 2 or mask.sum() < 3:
        return {**out, "silhouette": None, "davies_bouldin": None}
    return {**out,
            "silhouette": float(silhouette_score(X[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask]))}


def external_scores(labels, instr, quest) -> dict:
    """ARI / NMI of a clustering against the two known design factors."""
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return {k: None for k in ("ari_instr", "nmi_instr", "ari_quest", "nmi_quest")}
    return {"ari_instr": float(adjusted_rand_score(instr[mask], labels[mask])),
            "nmi_instr": float(normalized_mutual_info_score(instr[mask], labels[mask])),
            "ari_quest": float(adjusted_rand_score(quest[mask], labels[mask])),
            "nmi_quest": float(normalized_mutual_info_score(quest[mask], labels[mask]))}


def choose_k(X):
    sils = {k: float(silhouette_score(X, KMeans(n_clusters=k, n_init=10,
                                                random_state=0).fit(X).labels_))
            for k in K_RANGE}
    return max(sils, key=sils.get), sils


def cluster_one(Xr, instr, quest, pca_var=0.95) -> dict:
    """Run the full method suite on ONE role's 25 points. Returns a flat dict."""
    Xc = Xr - Xr.mean(0)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    d = max(2, int(np.searchsorted(cum, pca_var) + 1))
    # svd_solver="full": with only 25 rows the exact SVD is cheaper than the
    # randomized one sklearn would otherwise pick, and it keeps numpy's
    # Accelerate BLAS from raising spurious overflow warnings inside
    # randomized_svd (results were finite either way, the flags are bogus).
    X = PCA(n_components=min(d, min(Xc.shape) - 1), svd_solver="full",
            random_state=0).fit_transform(Xc)

    best_k, sils = choose_k(X)
    labs = {"kmeans": KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X).labels_,
            "kmeans_k5": KMeans(n_clusters=5, n_init=10, random_state=0).fit(X).labels_,
            "gmm": GaussianMixture(n_components=best_k, random_state=0).fit(X).predict(X),
            "hdbscan": hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit(X).labels_}
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    eps = float(np.median(dist[:, -1]))
    labs["dbscan"] = DBSCAN(eps=eps, min_samples=2).fit(X).labels_

    out = {"pca_dim": int(X.shape[1]), "best_k": int(best_k),
           "sil_best_k": float(sils[best_k])}
    # How well do the *ground-truth* partitions themselves score? This is the
    # ceiling: if the instruction partition has a higher silhouette than any
    # discovered clustering, the geometry simply IS the instruction grouping.
    for nm, lab in (("truth_instr", instr), ("truth_quest", quest)):
        sc = internal_scores(X, np.asarray(lab))
        out[f"{nm}_silhouette"] = sc["silhouette"]
    for m, lab in labs.items():
        for k, v in {**internal_scores(X, lab), **external_scores(lab, instr, quest)}.items():
            out[f"{m}_{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--n_null", type=int, default=100)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    t0 = time.time()
    roles, clouds, factors, manifest = load_role_clouds(args.view, args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    print(f"view={args.view} layer={layer} roles={len(roles)} points/role=25")

    rows = []
    with single_threaded():
        for i, r in enumerate(roles):
            instr, quest = factors[r]
            rows.append({"role": r, **cluster_one(clouds[r], instr, quest, args.pca_var)})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(roles)} roles  ({time.time()-t0:.1f}s)")
    df = pd.DataFrame(rows)

    print(f"design null: {args.n_null} draws ...")
    draws = list(design_null_draws(clouds, factors, args.n_null))   # eigh: wants threads
    with single_threaded():
        null_rows = [cluster_one(Xn, i_, q_, args.pca_var) for Xn, i_, q_ in draws]
    dfn = pd.DataFrame(null_rows)

    num_cols = [c for c in df.columns if c != "role" and df[c].dtype.kind in "fi"]
    results = {"_meta": {"view": args.view, "layer": layer, "n_roles": len(roles),
                         "points_per_role": 25, "k_range": [K_RANGE.start, K_RANGE.stop - 1],
                         "pca_var": args.pca_var, "n_null": args.n_null,
                         "runtime_s": round(time.time() - t0, 1)},
               "real": {c: band(df[c]) for c in num_cols},
               "design_null": {c: band(dfn[c]) for c in num_cols if c in dfn}}

    df.to_csv(run_dir / f"02_per_role_clustering_{args.view}_L{layer}.csv", index=False)
    json.dump(results, open(run_dir / f"02_per_persona_clustering_{args.view}_L{layer}.json", "w"),
              indent=2, default=float)

    print("\n== what does each role's 25-point cloud cluster BY? (medians over 276 roles) ==")
    print(f"  {'method':12s} {'k':>4s} {'sil':>7s} {'ARI instr':>10s} {'ARI quest':>10s}"
          f" | {'null ARI instr':>14s}")
    for m in METHODS:
        rb, nb = band(df[f"{m}_ari_instr"]), band(dfn[f"{m}_ari_instr"])
        kb, sb, qb = band(df[f"{m}_n_clusters"]), band(df[f"{m}_silhouette"]), band(df[f"{m}_ari_quest"])
        fmt = lambda b: "  n/a" if b["median"] is None else f"{b['median']:.3f}"
        print(f"  {m:12s} {kb['median']:4.0f} {fmt(sb):>7s} {fmt(rb):>10s} {fmt(qb):>10s}"
              f" | {fmt(nb):>14s}")
    ti, tq = band(df["truth_instr_silhouette"]), band(df["truth_quest_silhouette"])
    print(f"\n  ground-truth partition silhouette: instruction {ti['median']:.3f} | "
          f"question {tq['median']:.3f}")

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # A: chosen k across roles, real vs design null.
    ax = axes[0]
    ks = sorted(K_RANGE)
    w = 0.38
    real_c = [float((df["best_k"] == k).sum()) / len(df) for k in ks]
    null_c = [float((dfn["best_k"] == k).sum()) / len(dfn) for k in ks]
    ax.bar(np.array(ks) - w / 2, real_c, w, color=C_REAL, label="276 real roles")
    ax.bar(np.array(ks) + w / 2, null_c, w, color=C_DESIGN, label="design null")
    ax.set_xlabel("k chosen by silhouette")
    ax.set_ylabel("fraction of roles")
    ax.set_title("Selected number of clusters per role\n(25 points, k searched over 2-8)")
    ax.legend(fontsize=8)

    # B: what the clusters recover — instruction vs question.
    ax = axes[1]
    rng = np.random.default_rng(0)
    for j, m in enumerate(METHODS):
        for col, off, c in ((f"{m}_ari_instr", -0.17, C_INSTR),
                            (f"{m}_ari_quest", 0.17, C_QUEST)):
            v = df[col].dropna().values
            ax.scatter(np.full(len(v), j) + off + rng.uniform(-0.07, 0.07, len(v)), v,
                       s=6, alpha=0.3, color=c, linewidths=0)
            b = band(df[col])
            if b["median"] is not None:
                ax.plot([j + off - 0.13, j + off + 0.13], [b["median"]] * 2, color=c, lw=2.5)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("ARI vs the known factor")
    ax.set_title("What the within-role clusters recover")
    ax.plot([], [], "o", color=C_INSTR, label="ARI vs instruction phrasing")
    ax.plot([], [], "o", color=C_QUEST, label="ARI vs question")
    ax.legend(fontsize=8, loc="upper right")

    # C: the ceiling — how separable are the two ground-truth partitions?
    ax = axes[2]
    ax.scatter(df["truth_quest_silhouette"], df["truth_instr_silhouette"], s=12,
               alpha=0.5, color=C_REAL, linewidths=0, label="real role")
    ax.scatter(dfn["truth_quest_silhouette"], dfn["truth_instr_silhouette"], s=12,
               alpha=0.5, color=C_DESIGN, linewidths=0, marker="^", label="design null")
    lo = float(min(df["truth_quest_silhouette"].min(), df["truth_instr_silhouette"].min())) - 0.05
    hi = float(max(df["truth_quest_silhouette"].max(), df["truth_instr_silhouette"].max())) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#888888", ls="--", lw=1)
    ax.set_xlabel("silhouette of the question partition")
    ax.set_ylabel("silhouette of the instruction partition")
    ax.set_title("Separability of the two design factors\n(above the diagonal = instruction dominates)")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    savefig(fig, f"02_per_persona_clustering_{args.view}_L{layer}.png", run_dir)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
