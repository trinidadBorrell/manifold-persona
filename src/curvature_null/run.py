"""Is persona space curved, or does it only look curved? Real points against flat nulls (plan:
output/20-oct-AI-pilled/PLAN.md). One Condor job per mode/layer, CPU only.

    python -m curvature_null.run --e1 $PM/E1 --out $PM/CURV --mode means   --layer 28
    python -m curvature_null.run --e1 $PM/E1 --out $PM/CURV --mode answers --layer 28
    python -m curvature_null.run --out $PM/CURV --mode control

Nulls, refit to each sample (same N, same PCA variances):
  gauss    z_ij ~ N(0, var_j)                                   flat by construction
  shuffle  each PCA coordinate permuted across points on its own  keeps each direction's exact
           distribution, breaks every bend that links two directions
Measures (graph rule = pm_geometry.components.knn_graph: k = 10, union, k += 2 until connected):
  A  shortest path / straight distance: median over all pairs (N <= 1000) or 200 sources x all,
     over the far half of those pairs, and per straight-distance decile; trend = decile 10 / decile 5
     (a fixed-k graph has a stretch > 1 even in flat space; on a curved surface the ratio also grows
     with distance, arc / chord)
  B  A at N = 276, 1000, 3000, 11040 per-answer points
  C  [primary] 5-fold CV R^2 of a ridge quadratic fit of PC k on PCs 1-10 except k (k = 1-10)
  D  max-match |Spearman rho| of PGA k against PCA 1-50, k = 1-50 (N = 276 only)
  E  TwoNN (Facco 2017, the same formula as components.twonn)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr

from pm_geometry import geometry as G
from pm_geometry.components import NEUTRAL, K_GRAPH, persona_sets, knn_graph, pga

N_SWEEP = [276, 1000, 3000, 11040]
N_SRC = 200
K_PGA = 50


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def answers(e1, L):
    """Every role answer minus the no-prompt answer to the same question: (11040, 4096)."""
    meta = pd.read_csv(e1 / "meta.csv")
    H = np.load(e1 / ("last_L%d.npy" % L), mmap_mode="r")
    neu = (meta.role == NEUTRAL).to_numpy()
    pos = {q: i for i, q in zip(np.flatnonzero(neu), meta.question_idx[neu])}
    keep = np.flatnonzero(~neu)
    base = np.asarray(H[[pos[q] for q in meta.question_idx.to_numpy()[keep]]], dtype=np.float32)
    return np.asarray(H[keep], dtype=np.float32) - base


def pca(X):
    p = G.PCA99.fit(X.astype(np.float64))
    return p.to_pca(X.astype(np.float64)).astype(np.float32)


def null(Z, kind, rng):
    if kind == "gauss":
        return (rng.standard_normal(Z.shape) * Z.std(0)).astype(np.float32)
    return np.stack([rng.permutation(c) for c in Z.T], 1)


# ---------------------------------------------------------------------------------------------
# graph measures (A, B, E)
# ---------------------------------------------------------------------------------------------

def sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
    return np.maximum(d, 0)


def knn(Z, kmax, chunk=1000):
    """Brute-force kNN in chunks: (idx, dist) of the kmax nearest other points."""
    n = len(Z)
    I = np.zeros((n, kmax), int)
    Dk = np.zeros((n, kmax), np.float32)
    for s in range(0, n, chunk):
        d = sqdist(Z[s:s + chunk], Z)
        d[np.arange(len(d)), np.arange(s, s + len(d))] = np.inf
        j = np.argpartition(d, kmax, 1)[:, :kmax]
        dj = np.take_along_axis(d, j, 1)
        o = np.argsort(dj, 1)
        I[s:s + chunk], Dk[s:s + chunk] = np.take_along_axis(j, o, 1), np.sqrt(np.take_along_axis(dj, o, 1))
    return I, Dk


def twonn_from(d1, d2):
    mu = np.sort(d2 / np.maximum(d1, 1e-12))
    mu = mu[np.isfinite(mu) & (mu > 1)]
    n = len(mu)
    keep = int(0.9 * n)
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


def graph_stats(Z, rng, k=K_GRAPH):
    n = len(Z)
    I, Dk = knn(Z, min(n - 1, 40))
    while True:
        r = np.repeat(np.arange(n), k)
        W = csr_matrix((Dk[:, :k].ravel().astype(np.float64), (r, I[:, :k].ravel())), shape=(n, n))
        W = W.maximum(W.T)
        if connected_components(W, directed=False)[0] == 1 or k >= I.shape[1]:
            break
        k += 2
    src = np.arange(n) if n <= 1000 else np.sort(rng.choice(n, N_SRC, replace=False))
    geo = dijkstra(W, directed=False, indices=src)
    D = np.sqrt(sqdist(Z[src], Z)).astype(np.float64)
    if n <= 1000:
        iu = np.triu_indices(n, 1)
        geo, D = geo[iu], D[iu]
    else:
        m = np.ones_like(D, bool)
        m[np.arange(len(src)), src] = False
        geo, D = geo[m], D[m]
    rt = geo / D
    far = D > np.median(D)
    dec = np.minimum((np.argsort(np.argsort(D)) * 10) // len(D), 9)
    by_dec = [float(np.median(rt[dec == i])) for i in range(10)]
    return dict(n=n, d=int(Z.shape[1]), k=int(k), p50=float(np.median(rt)), p50_far=float(np.median(rt[far])),
                p90=float(np.percentile(rt, 90)), by_decile=by_dec, trend=by_dec[9] / by_dec[4],
                twonn=twonn_from(Dk[:, 0], Dk[:, 1]))


def with_nulls(Z, rng, n_null):
    out = dict(real=graph_stats(Z, rng))
    for kind in ("gauss", "shuffle"):
        out[kind] = [graph_stats(null(Z, kind, rng), rng) for _ in range(n_null)]
    return out


# ---------------------------------------------------------------------------------------------
# C: bends linking directions; D: PGA vs PCA
# ---------------------------------------------------------------------------------------------

def quad(F):
    i, j = np.triu_indices(F.shape[1])
    return np.hstack([F, F[:, i] * F[:, j]])


def ridge_cv_r2(X, y, rng, folds=5, alphas=np.logspace(-3, 3, 13)):
    """Out-of-fold R^2; alpha picked by an inner 5-fold split of each training fold."""
    def fit(Xa, ya, a):
        mx, sx, my = Xa.mean(0), Xa.std(0) + 1e-12, ya.mean()
        A = (Xa - mx) / sx
        w = np.linalg.solve(A.T @ A + a * len(A) * np.eye(A.shape[1]), A.T @ (ya - my))
        return lambda Xb: ((Xb - mx) / sx) @ w + my

    def cv(Xa, ya, a, f):
        p = np.zeros_like(ya)
        for t in f:
            tr = np.setdiff1d(np.arange(len(ya)), t)
            p[t] = fit(Xa[tr], ya[tr], a)(Xa[t])
        return p

    fo = np.array_split(rng.permutation(len(y)), folds)
    pred = np.zeros_like(y)
    for t in fo:
        tr = np.setdiff1d(np.arange(len(y)), t)
        inner = np.array_split(rng.permutation(len(tr)), folds)
        best = min(alphas, key=lambda a: ((cv(X[tr], y[tr], a, inner) - y[tr]) ** 2).sum())
        pred[t] = fit(X[tr], y[tr], best)(X[t])
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def bends(Z, rng, m=10):
    m = min(m, Z.shape[1])
    Z = Z[:, :m].astype(np.float64)
    return [ridge_cv_r2(quad(np.delete(Z, k, 1)), Z[:, k], rng) for k in range(m)]


def pga_match(Z):
    Z = Z.astype(np.float64)
    P = pga(Z, knn_graph(Z))
    K = min(K_PGA, Z.shape[1])
    rho = np.abs(spearmanr(P["coords"][:, :K], Z[:, :K])[0][:K, K:])
    return dict(max_rho=rho.max(1).tolist(), best_pca=(rho.argmax(1) + 1).tolist(),
                frechet_mean=int(P["mu"]))


# ---------------------------------------------------------------------------------------------
# synthetic controls
# ---------------------------------------------------------------------------------------------

def control(kind, n, dim, noise_share, rng, D=178, theta=np.deg2rad(100)):
    """A dim-sphere cap (polar angle <= theta, radius 1), or the same sample flattened by the log
    map at the pole (geodesic distance from the pole kept, so flat with the same extent), rotated
    into D dims, plus isotropic noise carrying `noise_share` of the total variance."""
    P = np.zeros((0, dim + 1))
    while len(P) < n:
        g = rng.standard_normal((4 * n, dim + 1))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        P = np.vstack([P, g[g[:, -1] >= np.cos(theta)]])
    P = P[:n]
    if kind == "flat":
        ang = np.arccos(np.clip(P[:, -1], -1, 1))
        t = P[:, :-1] / np.maximum(np.linalg.norm(P[:, :-1], axis=1, keepdims=True), 1e-12)
        P = np.hstack([ang[:, None] * t, np.zeros((n, 1))])
    P -= P.mean(0)
    Q, _ = np.linalg.qr(rng.standard_normal((D, dim + 1)))
    X = P @ Q.T
    sig = P.var(0).sum() * noise_share / (1 - noise_share)
    X += rng.standard_normal(X.shape) * np.sqrt(sig / D)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["means", "answers", "control"], required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--n-null-big", type=int, default=5, help="null draws at N >= 3000")
    ap.add_argument("--sweep", default=",".join(map(str, N_SWEEP)))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.layer)
    sweep = [int(x) for x in a.sweep.split(",")]
    nn = lambda n: a.n_null if n < 3000 else a.n_null_big

    if a.mode == "means":
        f = out / ("means_L%d.json" % a.layer)
        roles, sets, _ = persona_sets(Path(a.e1), "last", a.layer)
        Z = pca(sets["all"])
        res = dict(layer=a.layer, var=Z.var(0).tolist()[:50], graph=with_nulls(Z, rng, a.n_null),
                   bends=dict(real=bends(Z, rng), shuffle=[bends(null(Z, "shuffle", rng), rng) for _ in range(a.n_null)]),
                   pga=dict(real=pga_match(Z), gauss=[pga_match(pca(null(Z, "gauss", rng))) for _ in range(a.n_null)],
                            shuffle=[pga_match(pca(null(Z, "shuffle", rng))) for _ in range(a.n_null)]))
        res["frechet_mean_role"] = roles[res["pga"]["real"]["frechet_mean"]]
        g0 = knn_graph(Z.astype(np.float64))
        iu = np.triu_indices(len(Z), 1)
        res["check"] = dict(p50_knn_graph=float(np.median(g0["dist"][iu] / g0["D"][iu])),
                            twonn_components=__import__("pm_geometry.components", fromlist=["twonn"]).twonn(Z.astype(np.float64)))
    elif a.mode == "answers":
        f = out / ("answers_L%d.json" % a.layer)
        X = answers(Path(a.e1), a.layer)
        res = dict(layer=a.layer, sweep={})
        for n in sweep:
            sub = X if n >= len(X) else X[np.sort(rng.choice(len(X), n, replace=False))]
            Z = pca(sub)
            res["sweep"][str(n)] = with_nulls(Z, rng, nn(n))
            print(n, Z.shape, json.dumps(res["sweep"][str(n)]["real"]), flush=True)
        Z = pca(X)
        res["bends"] = dict(real=bends(Z, rng), shuffle=[bends(null(Z, "shuffle", rng), rng) for _ in range(a.n_null_big)])
    else:
        f = out / "control.json"
        res = {}
        for kind in ("sphere", "flat"):
            for dim in (2, 8):
                for ns in (0.01, 0.1):
                    key = "%s_d%d_noise%.2f" % (kind, dim, ns)
                    res[key] = {}
                    for n in sweep:
                        Z = pca(control(kind, n, dim, ns, rng))
                        res[key][str(n)] = with_nulls(Z, rng, nn(n))
                        if n == 276:
                            res[key]["bends"] = dict(real=bends(Z, rng),
                                                     shuffle=[bends(null(Z, "shuffle", rng), rng) for _ in range(a.n_null)])
                        print(key, n, json.dumps(res[key][str(n)]["real"]), flush=True)
    f.write_text(json.dumps(res))
    print("wrote", f, flush=True)


if __name__ == "__main__":
    main()
