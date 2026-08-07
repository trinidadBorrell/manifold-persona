"""E7: intrinsic-dimension reconciliation — the 2x2 (cloud x layer) design.

Two prior ID estimates disagree: the response-token cloud at layer 19 gave
ID ~ 13 (hand-rolled TwoNN 12.8, MLE 15.7-17.0, PCA participation ratio 13.1),
while the collaborator's prompt-token cloud at layer 26 gave ID ~ 4-7. Token
basis AND depth differ, so the gap is confounded. This script runs the full
2x2 {prompt cloud, response cloud} x {layer 19, layer 26} on the RAW
6,900-point cloud with three estimator families per cell, so the gap can be
attributed to basis, depth, both, or neither.

Estimators per cell (matching the prior run's methodology):
  1. TwoNN (Facco et al. 2017): skdim.id.TwoNN (default discard_fraction=0.1)
     AND a hand-rolled TwoNN — MLE form d = n / sum(log(r2/r1)) over all points
     with r1 > 0, no discard — the prior run was the hand-rolled one.
  2. MLE (Levina-Bickel) at K=10 and K=20 via skdim.id.MLE (dimension_pw_ mean,
     i.e. mean over per-point local estimates — same read-out as the repo's
     exploratory/assistant_axis/01_intrinsic_dimension.py).
  3. PCA on the centered cloud: participation ratio (sum l)^2 / sum l^2,
     n components for 90% / 95% variance, top-5 per-PC variance shares.
Plus a 100-resample bootstrap 95% CI on the hand-rolled TwoNN, and a reduced
null: 5 draws of a matched-marginal Gaussian (same per-dim mean + variance,
diagonal covariance), hand-rolled TwoNN on each — null should be >> real.

Bootstrap note: each resample draws n indices with replacement and the
estimator runs on the UNIQUE set (~63.2% of points; duplicates have r1 = 0 and
carry no TwoNN information). This biases the CI slightly low relative to the
full-sample point estimate (TwoNN drifts down with smaller n), which is also
what the prior run showed (CI 11.0-12.7 vs point 12.8).

Usage (from repo root):
    .venv/bin/python output/e7_id_reconciliation/run_e7.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import time

import numpy as np
import skdim

OUT_DIR = "output/e7_id_reconciliation"
CLOUD_DIRS = {"prompt": "data/embeddings_roles",
              "resp": "data/embeddings_roles_resp"}
LAYERS = (19, 26)
SEED = 0
N_BOOT = 100
N_NULL = 5


def load_cell(cloud: str, layer: int):
    os.environ["MP_ROLE_DIR"] = CLOUD_DIRS[cloud]
    from manifold_persona.common import load_points
    X, meta, man = load_points(view="prompt_avg", layer=layer, aggregate="none")
    return np.asarray(X, dtype=np.float64), meta, man


def pairwise_sq(X: np.ndarray) -> np.ndarray:
    """Full [n,n] squared-euclidean distance matrix, float64, diag = inf."""
    G = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, np.inf)
    return D2


def nn12(D2: np.ndarray):
    """(r1, r2) euclidean distances to the 1st and 2nd NN per row."""
    part = np.partition(D2, 1, axis=1)[:, :2]
    part.sort(axis=1)
    return np.sqrt(part[:, 0]), np.sqrt(part[:, 1])


def twonn_handrolled(D2: np.ndarray) -> float:
    """Facco TwoNN, MLE form: d = n / sum(log(r2/r1)), points with r1>0."""
    r1, r2 = nn12(D2)
    mask = r1 > 0
    logs = np.log(r2[mask] / r1[mask])
    return float(mask.sum() / logs.sum())


def subsample_twonn(D2: np.ndarray, n_boot: int, rng) -> dict:
    """Dedup resampling gives a ~0.632n WITHOUT-replacement subsample per draw,
    so this is a descriptive subsample interval, NOT a bootstrap CI
    (review finding #8): TwoNN drifts with n, so coverage language would lie."""
    n = D2.shape[0]
    vals, uniq_ns = [], []
    for _ in range(n_boot):
        u = np.unique(rng.integers(0, n, n))
        vals.append(twonn_handrolled(D2[np.ix_(u, u)]))
        uniq_ns.append(len(u))
    v = np.asarray(vals)
    return {"mean": float(v.mean()),
            "subsample_interval95": [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))],
            "n_boot": n_boot, "mean_unique_n": float(np.mean(uniq_ns))}


def null_twonn(X: np.ndarray, n_null: int, rng) -> list:
    """Matched-marginal Gaussian (per-dim mean+var, diagonal cov) TwoNN draws."""
    mu, sd = X.mean(0), X.std(0)
    out = []
    for _ in range(n_null):
        Z = mu + sd * rng.standard_normal(X.shape)
        out.append(twonn_handrolled(pairwise_sq(Z)))
    return out


def pca_metrics(X: np.ndarray) -> dict:
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = s ** 2
    shares = eig / eig.sum()
    cum = np.cumsum(shares)
    return {"participation_ratio": float((eig.sum() ** 2) / (eig ** 2).sum()),
            "n_pc_90": int(np.searchsorted(cum, 0.90) + 1),
            "n_pc_95": int(np.searchsorted(cum, 0.95) + 1),
            "top5_var_shares": [float(x) for x in shares[:5]]}


def run_cell(cloud: str, layer: int) -> dict:
    t0 = time.time()
    X, meta, _ = load_cell(cloud, layer)
    n, d = X.shape
    print(f"\n=== cell {cloud} L{layer}: n={n} ambient={d} "
          f"roles={meta['role'].nunique()} ===", flush=True)
    res = {"n": n, "ambient": d}

    # exact-duplicate accounting (r1 == 0 rows are dropped by the estimator)
    D2 = pairwise_sq(X)
    r1, _ = nn12(D2)
    res["n_zero_r1"] = int((r1 == 0).sum())

    # 1) TwoNN, both implementations
    res["twonn_handrolled"] = twonn_handrolled(D2)
    res["twonn_skdim"] = float(skdim.id.TwoNN().fit(X).dimension_)
    print(f"  TwoNN hand-rolled={res['twonn_handrolled']:.2f} "
          f"skdim={res['twonn_skdim']:.2f}", flush=True)

    # bootstrap CI on the hand-rolled TwoNN
    rng = np.random.default_rng(SEED)
    res["twonn_subsample"] = subsample_twonn(D2, N_BOOT, rng)
    ci = res["twonn_subsample"]["subsample_interval95"]
    print(f"  subsample interval (~0.63n, not a CI) [{ci[0]:.2f}, {ci[1]:.2f}]", flush=True)

    # 2) Levina-Bickel MLE (skdim, per-point mean)
    # skdim's default neighborhood_based mode ignores the constructor K;
    # the neighbour count must go to fit() (review finding #3).
    for k in (10, 20):
        est = skdim.id.MLE().fit(X, n_neighbors=k)
        res[f"mle_k{k}"] = float(np.nanmean(est.dimension_pw_))
        print(f"  MLE k={k}: {res[f'mle_k{k}']:.2f}", flush=True)

    # 3) PCA
    res["pca"] = pca_metrics(X)
    p = res["pca"]
    print(f"  PCA PR={p['participation_ratio']:.2f} n90={p['n_pc_90']} "
          f"n95={p['n_pc_95']} PC1={p['top5_var_shares'][0]:.3f}", flush=True)

    # reduced null
    del D2
    res["null_twonn"] = null_twonn(X, N_NULL, np.random.default_rng(SEED + 1))
    print(f"  null TwoNN (5 draws): "
          f"{', '.join(f'{v:.1f}' for v in res['null_twonn'])}", flush=True)

    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def main():
    results = {"_meta": {"design": "2x2 cloud x layer, raw 6900-point cloud",
                         "view": "prompt_avg", "aggregate": "none",
                         "seed": SEED, "n_boot": N_BOOT, "n_null": N_NULL,
                         "estimators": {
                             "twonn_handrolled": "d = n/sum(log(r2/r1)), r1>0, no discard",
                             "twonn_skdim": "skdim.id.TwoNN default (discard_fraction=0.1)",
                             "mle": "skdim.id.MLE(K), mean of dimension_pw_",
                             "pca": "covariance PCA on centered cloud"}}}
    for cloud in ("prompt", "resp"):
        for layer in LAYERS:
            results[f"{cloud}_L{layer}"] = run_cell(cloud, layer)
    out = os.path.join(OUT_DIR, "results.json")
    json.dump(results, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
