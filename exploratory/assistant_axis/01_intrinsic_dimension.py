"""Intrinsic dimension of the character-role persona cloud (assistant-axis study).

Same estimators as the persona-vectors study (scikit-dimension: TwoNN, MLE,
lPCA, MOM, TLE, CorrInt + PCA baselines), applied to the 276-role point cloud.
Second panel: PCA cumulative variance-explained curve (mirrors the assistant-axis
repo's pca.py analysis).

Usage:
    .venv/bin/python exploratory/assistant_axis/01_intrinsic_dimension.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import skdim

from common import load_points, center, savefig, resolve_run_dir


def pca_participation_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = s ** 2
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def pca_cumvar(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    var = s ** 2
    return np.cumsum(var) / var.sum()


ESTIMATORS = {
    "TwoNN":   (lambda: skdim.id.TwoNN(),   "global"),
    "MLE":     (lambda: skdim.id.MLE(K=20), "pw_mean"),
    "lPCA":    (lambda: skdim.id.lPCA(),    "global"),
    "MOM":     (lambda: skdim.id.MOM(),     "global"),
    "TLE":     (lambda: skdim.id.TLE(),     "global"),
    "CorrInt": (lambda: skdim.id.CorrInt(), "global"),
}


def _read(est, how):
    return float(np.nanmean(est.dimension_pw_)) if how == "pw_mean" else float(est.dimension_)


def estimate_id_only(X):
    """The 6 neighbour/PCA ID estimators (no PCA-baseline extras)."""
    out = {}
    for name, (factory, how) in ESTIMATORS.items():
        try:
            v = _read(factory().fit(X), how)
            out[name] = None if (v != v or v <= 0) else v
        except Exception as e:
            out[name] = None
            print(f"  [{name}] failed: {e}")
    return out


def estimate_all(X):
    out = estimate_id_only(X)
    cum = pca_cumvar(X)
    out["PCA_participation_ratio"] = pca_participation_ratio(X)
    out["PCA_dim_90pct"] = int(np.searchsorted(cum, 0.90) + 1)
    return out, cum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    X, meta, manifest = load_points(view=args.view, layer=args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    print(f"view={args.view} layer={layer} ambient={X.shape[1]} N={X.shape[0]} roles={meta['role'].nunique()}")

    g, cum = estimate_all(X)
    # --- Robustness to the dim>>N regime (Q1): re-estimate ID on a PCA-95%
    # projection. The neighbour-based estimators act on local distances, which
    # live on the manifold, so ambient dim should barely matter; PCA-first
    # confirms this (it typically shifts estimates DOWN ~1-2 dims by dropping
    # noise directions, never up). We report both. The true limiter is small N,
    # not ambient dim: with N=276 and ID~10 these estimators are mildly
    # DOWN-biased, so read them as soft LOWER bounds on the true ID.
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    from sklearn.decomposition import PCA
    Xc = center(X)
    X95 = PCA(n_components=d95, random_state=0).fit_transform(Xc)
    g_pca95 = estimate_id_only(X95)
    print(f"\nPCA-95% dim = {d95} components (of max {min(X.shape)-1})")
    results = {"_meta": {"view": args.view, "layer": layer, "ambient": int(X.shape[1]),
                         "n": int(X.shape[0]), "n_roles": int(meta["role"].nunique()),
                         "pca95_dim": d95},
               "global": g, "global_pca95": g_pca95}
    print("\n== Global intrinsic dimension (full ambient vs PCA-95%) ==")
    for k, v in g.items():
        v95 = g_pca95.get(k)
        extra = f"   | pca95: {v95}" if v95 is not None else ""
        print(f"  {k:26s} {v}{extra}")

    out_json = run_dir / f"intrinsic_dimension_{args.view}_L{layer}.json"
    json.dump(results, open(out_json, "w"), indent=2)
    print("wrote", out_json)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    id_names = [k for k in ESTIMATORS if g.get(k) is not None]
    xs = np.arange(len(id_names))
    full_v = [g[k] for k in id_names]
    pca_v = [g_pca95.get(k) or 0 for k in id_names]
    axes[0].bar(xs - 0.2, full_v, 0.4, color="#0072B2", label=f"full (d={X.shape[1]})")
    axes[0].bar(xs + 0.2, pca_v, 0.4, color="#56B4E9", label=f"PCA-95% (d={d95})")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(id_names, rotation=40, ha="right")
    axes[0].set_ylabel("estimated dimension"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"Intrinsic dimension: full ambient vs PCA-95%\n(layer {layer}, N={X.shape[0]})")

    axes[1].plot(range(1, len(cum) + 1), cum, color="#009E73")
    for thr in (0.7, 0.8, 0.9, 0.95):
        d = int(np.searchsorted(cum, thr) + 1)
        axes[1].axhline(thr, ls="--", lw=0.8, color="grey")
        axes[1].annotate(f"{int(thr*100)}% : {d}d", (len(cum), thr), ha="right", va="bottom", fontsize=8)
    axes[1].set_xlabel("PCA component"); axes[1].set_ylabel("cumulative variance")
    axes[1].set_xlim(0, min(len(cum), 120)); axes[1].set_ylim(0, 1)
    axes[1].set_title("PCA cumulative variance explained")
    fig.tight_layout()
    savefig(fig, f"01_intrinsic_dimension_{args.view}_L{layer}.png", run_dir)


if __name__ == "__main__":
    main()
