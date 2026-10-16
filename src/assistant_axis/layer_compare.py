"""Compare the whole persona manifold across depth: hidden states 19, 25, 32.

Runs the SAME estimators `01_intrinsic_dimension.py` uses, on the SAME object (the 275
fully-role-playing role centroids), at three depths, so the numbers are directly comparable.

It reads the cached centroid matrices rather than the clouds. That is not a shortcut: the cached
`C` is exactly what `01_intrinsic_dimension.py` computes after the role-mean collapse, and reading
it avoids three NFS reads of 5.4 GB arrays to recompute a (275, 4096) matrix. The caches were
written by the geometry builds, from the clouds, with the same `fully_only` exclusion.

Reported per layer:
  - six intrinsic-dimension estimators (TwoNN, MLE, lPCA, MOM, TLE, CorrInt)
  - PCA: explained variance of PC1-3, and the number of PCs for 90% of the variance
  - |cos(PC1, assistant_axis)| -- whether the dominant direction IS the axis
  - the axis ordering: which roles sit at each end, and where the study pairs fall
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import skdim

ESTIMATORS = {
    "TwoNN":   (lambda: skdim.id.TwoNN(),   "global"),
    "MLE":     (lambda: skdim.id.MLE(K=20), "pw_mean"),
    "lPCA":    (lambda: skdim.id.lPCA(),    "global"),
    "MOM":     (lambda: skdim.id.MOM(),     "global"),
    "TLE":     (lambda: skdim.id.TLE(),     "global"),
    "CorrInt": (lambda: skdim.id.CorrInt(), "global"),
}
PAIRS = [("validator", "vampire"), ("validator", "bard"), ("validator", "amateur")]


def _read(est, how):
    return float(np.nanmean(est.dimension_pw_)) if how == "pw_mean" else float(est.dimension_)


def estimate_id(X):
    out = {}
    for name, (factory, how) in ESTIMATORS.items():
        try:
            out[name] = round(_read(factory().fit(X), how), 3)
        except Exception as e:                    # noqa: BLE001 - report, never kill the sweep
            out[name] = None
            print(f"    {name}: FAILED ({e!r})")
    return out


def analyse(tag, cache):
    d = np.load(cache, allow_pickle=True)
    C = np.asarray(d["C"], dtype=np.float64)
    names = [str(x) for x in d["names"]]
    axis = np.asarray(d["axis_unit"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    ap = np.asarray(d["axis_proj"], dtype=np.float64)
    idx = {n: i for i, n in enumerate(names)}

    Xc = C - C.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(len(C) - 1, 1)
    evr = var / var.sum()
    cum = np.cumsum(evr)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    cos_pc1 = float(abs(Vt[0] @ axis))

    print(f"\n=== hidden state {tag} ===  centroids {C.shape}")
    ids = estimate_id(C)
    print("  intrinsic dimension: " + "  ".join(f"{k}={v}" for k, v in ids.items()))
    print(f"  PCA: PC1 {100*evr[0]:.1f}%  PC2 {100*evr[1]:.1f}%  PC3 {100*evr[2]:.1f}%"
          f"   PCs for 90% = {n90} of {len(evr)}")
    print(f"  |cos(PC1, assistant_axis)| = {cos_pc1:.3f}")
    order = np.argsort(ap)
    print("  axis near:", ", ".join(names[i] for i in order[-5:][::-1]))
    print("  axis far: ", ", ".join(names[i] for i in order[:5]))
    rank = {n: int(np.where(order[::-1] == idx[n])[0][0]) + 1 for n in names}
    pair_info = {}
    for A, B in PAIRS:
        pair_info[f"{A}>{B}"] = {"rank_A": rank[A], "rank_B": rank[B],
                                 "chord": round(float(np.linalg.norm(C[idx[B]] - C[idx[A]])), 2)}
        print(f"  {A}>{B}: rank {rank[A]} -> {rank[B]} of {len(names)}"
              f"  chord {pair_info[f'{A}>{B}']['chord']}")
    return {"layer": tag, "n_centroids": int(C.shape[0]), "hidden": int(C.shape[1]),
            "id": ids, "evr": [round(float(v), 5) for v in evr[:10]],
            "cumvar_10": [round(float(v), 5) for v in cum[:10]],
            "pcs_for_90pct": n90, "cos_pc1_axis": round(cos_pc1, 4),
            "axis_range": [round(float(ap[order[0]]), 2), round(float(ap[order[-1]]), 2)],
            "near": [names[i] for i in order[-8:][::-1]], "far": [names[i] for i in order[:8]],
            "pairs": pair_info,
            "roles": names, "axis_proj": [round(float(v), 3) for v in ap]}


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--caches", required=True,
                    help="comma list of tag=path, e.g. 19=/a/b.npz,25=/c/d.npz")
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()
    results = []
    for item in a.caches.split(","):
        tag, path = item.split("=", 1)
        results.append(analyse(tag, path))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
