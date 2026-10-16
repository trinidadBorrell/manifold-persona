"""Part 2 -- persona component analysis on the E1 extraction (plan 2026-10-08, 2.0-2.4).

How many components / principal personas does the persona set need, which are they, and is that
a one-layer or a multi-layer property? Run per representation and layer:

  representations  last : last answer token minus the neutral answer (PersonaManifold eq. 1)
                   mean : answer-token mean minus the neutral answer (our view; the subtraction is a
                          common shift, so it changes no distance)
  points           276 persona vectors (one per role, mean over 5 system prompts x 40 questions)

  flatness   path length along the kNN graph / straight-line distance, all pairs; Isomap vs PCA
             residual variance by dimension
  ID         Levina-Bickel MLE and TwoNN on the 276 points (a lower bound at this N) and on 5,000
             per-answer points (answer minus its neutral answer)
  PCA        global PCA: variance explained by k components
  PGA        Fletcher et al. 2004, Alg. 2 on the data graph:
               mu = graph Frechet mean (eq. 5): the persona minimising sum_i d_g(x, x_i)^2
               Log_mu(x_i) = d_g(mu, x_i) * unit(z_i - mu), z_i = the node two hops along the
                 shortest mu->x_i path (the path's initial direction; x_i itself if closer)
               S = 1/N sum u_i u_i^T; eigenpairs = principal geodesic directions and variances
  BSA        Pennec 2018, sample-limited barycentric subspaces: reference personas R, greedy
             forward (nested flags, s6.4); unexplained variance = sum_i squared distance from y_i
             to the affine span of R (the exponential barycentric subspace of a flat embedding),
             in Isomap coordinates (classical MDS of the graph distances) -> "geodesic"; in PCA
             coordinates -> "linear"
  stability  the same on persona vectors from question halves A/B: principal-persona overlap
             and principal-angle cosine between the top-3 PGA directions

Multi-layer (2.4):
  M0  best single layer (by PGA split-half stability, ties -> fewer components)
  M1  per-layer-scaled concatenation over a band of layers, then the same graph pipeline:
      d^2 = sum_l |x^l - y^l|^2 / s_l^2, s_l = median pairwise distance at layer l (a Euclidean
      product metric); band = contiguous layers whose linear CKA with M0's layer is >= 0.9
  M2  linear: PCA on the same stacked vectors; each component's layer profile = the norm of its
      loading block per layer [deviation from the plan's PARAFAC2: no tensor library in the
      cluster env; stacked PCA is the linear multi-layer baseline with the same readout]

    python -m steering.personamanifold.components --e1 E1 --out P2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.personamanifold import geometry as G

NEUTRAL = "__neutral__"
K_GRAPH = 10
VAR_TARGET = 0.90
MAX_K = 40
BSA_MAX_K = int(__import__('os').environ.get('PM_BSA_MAX_K', '40'))


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def persona_sets(e1: Path, view: str, L: int):
    """Persona vectors from all questions, and from question halves A/B."""
    meta = pd.read_csv(e1 / "meta.csv")
    H = np.load(e1 / ("%s_L%d.npy" % (view, L)), mmap_mode="r")
    neu = (meta.role == NEUTRAL).to_numpy()
    q0 = meta.question_idx[neu].to_numpy()
    H0 = H[neu].astype(np.float64)
    qs = sorted(set(q0))
    halves = {"all": set(qs), "A": set(qs[0::2]), "B": set(qs[1::2])}
    out = {}
    for name, keep in halves.items():
        m = (~neu) & meta.question_idx.isin(keep).to_numpy()
        roles, X = G.persona_vectors(H[m], meta.role[m].to_numpy(), meta.question_idx[m].to_numpy(), H0, q0)
        out[name] = X
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(~neu)
    pick = rng.choice(cand, size=min(3000, len(cand)), replace=False)
    pos = {q: i for i, q in enumerate(q0)}
    cloud = H[np.sort(pick)].astype(np.float64) - H0[[pos[q] for q in meta.question_idx.to_numpy()[np.sort(pick)]]]
    return roles, out, cloud


# ---------------------------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------------------------

def knn_graph(Z, k=K_GRAPH):
    """Euclidean kNN graph (edge if either picks the other), Dijkstra distances + predecessors.
    k grows until the graph is connected."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    n = len(Z)
    while True:
        idx = np.argsort(D, 1)[:, 1:k + 1]
        r = np.repeat(np.arange(n), k)
        c = idx.ravel()
        W = csr_matrix((D[r, c], (r, c)), shape=(n, n))
        W = W.maximum(W.T)
        if connected_components(W, directed=False)[0] == 1 or k >= n - 1:
            break
        k += 2
    dist, pred = dijkstra(W, directed=False, return_predecessors=True)
    return dict(D=D, dist=dist, pred=pred, k=k)


def path(g, a, b):
    p, j = [b], b
    while j != a:
        j = g["pred"][a, j]
        p.append(j)
    return p[::-1]


def ratios(g):
    iu = np.triu_indices(len(g["D"]), 1)
    return g["dist"][iu] / g["D"][iu]


def mds(Dm, m=None):
    """Classical MDS (Isomap's embedding step); keeps positive-eigenvalue dims."""
    n = len(Dm)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (Dm ** 2) @ J
    lam, V = np.linalg.eigh(B)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    pos = lam > 1e-9 * lam[0]
    lam, V = lam[pos], V[:, pos]
    if m:
        lam, V = lam[:m], V[:, :m]
    return V * np.sqrt(lam)


def residual_curve(Dm, Y, kmax=MAX_K):
    """Isomap's diagnostic: 1 - R^2(Dm, pairwise distances of Y[:, :k]) for k = 1..kmax."""
    iu = np.triu_indices(len(Dm), 1)
    out = []
    for k in range(1, min(kmax, Y.shape[1]) + 1):
        Dk = np.sqrt(((Y[:, None, :k] - Y[None, :, :k]) ** 2).sum(-1))[iu]
        out.append(1 - np.corrcoef(Dm[iu], Dk)[0, 1] ** 2)
    return out


def twonn(Z):
    """Facco et al. 2017: slope through the origin of -log(1-F(mu)) on log(mu), top 10% dropped."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Z).query(Z, k=3)
    mu = np.sort(d[:, 2] / np.maximum(d[:, 1], 1e-12))
    mu = mu[np.isfinite(mu) & (mu > 1)]
    n = len(mu)
    keep = int(0.9 * n)
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


# ---------------------------------------------------------------------------------------------
# PGA (Fletcher 2004) on the graph
# ---------------------------------------------------------------------------------------------

def pga(Z, g, hop=2):
    mu = int(np.argmin((g["dist"] ** 2).sum(1)))           # eq. 5 over data points
    U = np.zeros_like(Z)
    for i in range(len(Z)):
        if i == mu:
            continue
        p = path(g, mu, i)
        z = Z[p[min(hop, len(p) - 1)]]
        v = z - Z[mu]
        U[i] = g["dist"][mu, i] * v / np.linalg.norm(v)
    S = U.T @ U / len(Z)
    lam, V = np.linalg.eigh(S)
    o = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[o], 0, None), V[:, o]
    return dict(mu=mu, U=U, lam=lam, V=V, coords=U @ V)


def n_for(curve_explained, target=VAR_TARGET):
    c = np.asarray(curve_explained)
    hit = np.flatnonzero(c >= target)
    return int(hit[0] + 1) if len(hit) else None


# ---------------------------------------------------------------------------------------------
# BSA (Pennec 2018), sample-limited, greedy forward
# ---------------------------------------------------------------------------------------------

def _affine_resid(Y, R):
    base = Y[R[0]]
    if len(R) == 1:
        E = Y - base
    else:
        A = (Y[R[1:]] - base).T                            # (m, k)
        Q, _ = np.linalg.qr(A)
        E = (Y - base) - ((Y - base) @ Q) @ Q.T
    return float((E ** 2).sum())


def bsa(Y, start, kmax=MAX_K):
    """Greedy nested flag starting at `start` (the Frechet mean). Returns the reference order and
    the explained fraction 1 - sigma2_out / total after each addition (k = 1 is just the mean)."""
    total = float(((Y - Y.mean(0)) ** 2).sum())
    R = [start]
    expl = [1 - _affine_resid(Y, R) / total]
    cand = set(range(len(Y))) - {start}
    while len(R) < min(kmax + 1, len(Y)):
        best = min(cand, key=lambda j: _affine_resid(Y, R + [j]))
        R.append(best)
        cand.discard(best)
        expl.append(1 - _affine_resid(Y, R) / total)
    return R, expl


# ---------------------------------------------------------------------------------------------
# one representation
# ---------------------------------------------------------------------------------------------

def analyse(X, roles, with_stability=None, kmax=MAX_K):
    """X (N, D) persona vectors -> every Part 2 number for this set."""
    pca = G.PCA99.fit(X)
    Z = pca.to_pca(X)
    tot = (Z ** 2).sum() / len(Z)
    pca_expl = np.cumsum((Z ** 2).sum(0) / len(Z) / tot)[:kmax]
    g = knn_graph(Z)
    rt = ratios(g)
    Y = mds(g["dist"])
    P = pga(Z, g)
    pga_expl = np.cumsum(P["lam"] / P["lam"].sum())[:kmax]
    R_geo, bsa_geo = bsa(Y, P["mu"], max(kmax, BSA_MAX_K))
    R_lin, bsa_lin = bsa(Z, P["mu"], max(kmax, BSA_MAX_K))
    res = dict(
        D_prime=int(Z.shape[1]), graph_k=int(g["k"]),
        path_over_straight=dict(p50=float(np.median(rt)), p90=float(np.percentile(rt, 90)),
                                frac_above_1_2=float((rt > 1.2).mean())),
        isomap_residual=residual_curve(g["dist"], Y),
        pca_residual=residual_curve(g["D"], Z),
        id_lb_points=G.levina_bickel(Z), id_twonn_points=twonn(Z),
        pca_explained=pca_expl.tolist(), pga_explained=pga_expl.tolist(),
        bsa_geodesic_explained=bsa_geo, bsa_linear_explained=bsa_lin,
        n90=dict(pca=n_for(pca_expl), pga=n_for(pga_expl),
                 bsa_geodesic=(n_for(bsa_geo[1:]) if len(bsa_geo) > 1 else None),
                 bsa_linear=(n_for(bsa_lin[1:]) if len(bsa_lin) > 1 else None)),
        frechet_mean=roles[P["mu"]],
        bsa_geodesic_personas=[roles[i] for i in R_geo[:16]],
        bsa_linear_personas=[roles[i] for i in R_lin[:16]],
        pga_ends=[dict(component=k + 1, var_share=float(P["lam"][k] / P["lam"].sum()),
                       minus=[roles[i] for i in np.argsort(P["coords"][:, k])[:6]],
                       plus=[roles[i] for i in np.argsort(P["coords"][:, k])[::-1][:6]])
                  for k in range(min(5, len(P["lam"])))],
        pca_ends=[dict(component=k + 1, var_share=float(pca.var_ratio[k] / pca.var_ratio.sum()),
                       minus=[roles[i] for i in np.argsort(Z[:, k])[:6]],
                       plus=[roles[i] for i in np.argsort(Z[:, k])[::-1][:6]])
                  for k in range(min(5, Z.shape[1]))],
    )
    if with_stability is not None:
        XA, XB = with_stability
        PA, PB = [pga(Zh, knn_graph(Zh)) for Zh in (pca.to_pca(XA), pca.to_pca(XB))]
        cos = np.linalg.svd(PA["V"][:, :3].T @ PB["V"][:, :3], compute_uv=False)
        RA = bsa(mds(knn_graph(pca.to_pca(XA))["dist"]), P["mu"], 10)[0]
        RB = bsa(mds(knn_graph(pca.to_pca(XB))["dist"]), P["mu"], 10)[0]
        res["stability"] = dict(pga_top3_principal_cos=[float(c) for c in cos],
                                bsa_top10_overlap=len(set(RA[1:]) & set(RB[1:])) / 10)
    return res, dict(pca=pca, Z=Z, g=g, P=P, R_geo=R_geo)


def pdist(X):
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    return np.sqrt(D2[np.triu_indices(len(X), 1)])


def linear_cka(A, B):
    """Linear CKA (Kornblith et al. 2019) in Gram form: <Ka, Kb>_F / (|Ka|_F |Kb|_F), centred."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    Ka, Kb = A @ A.T, B @ B.T
    return float((Ka * Kb).sum() / (np.linalg.norm(Ka) * np.linalg.norm(Kb)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="12-32")
    ap.add_argument("--views", default="last,mean")
    ap.add_argument("--per-layer-only", action="store_true", help="write layers/<view>_L<L>.json and stop")
    a = ap.parse_args(argv)
    e1, out = Path(a.e1), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (e1 / "meta.csv").exists():
        from steering.personamanifold.build import consolidate
        consolidate(e1)
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    summary = dict(layers=layers, views={})
    for view in a.views.split(","):
        per_layer, Xs, XAs, XBs = {}, {}, {}, {}
        for L in layers:
            roles, sets, cloud = persona_sets(e1, view, L)
            Xs[L], XAs[L], XBs[L] = sets["all"], sets["A"], sets["B"]
            cache = out / "layers" / ("%s_L%d.json" % (view, L))
            if cache.exists():
                per_layer[L] = json.loads(cache.read_text())
                continue
            res, _ = analyse(sets["all"], roles, (sets["A"], sets["B"]))
            pc = G.PCA99.fit(cloud)
            Zc = pc.to_pca(cloud)
            res.update(id_lb_answers=G.levina_bickel(Zc[:2000], 10, 30), id_twonn_answers=twonn(Zc))
            per_layer[L] = res
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(res, default=float))
            print(view, L, json.dumps(dict(n90=res["n90"], stab=res["stability"],
                                           id=[round(res["id_lb_points"], 1), round(res["id_twonn_answers"], 1)],
                                           ratio=res["path_over_straight"])), flush=True)
        if a.per_layer_only:
            continue
        # ---- multi-layer ----
        def stab_key(L):
            s = per_layer[L]["stability"]
            return (np.mean(s["pga_top3_principal_cos"]) + s["bsa_top10_overlap"]) / 2
        best = max(layers, key=stab_key)
        cka = [[linear_cka(Xs[i], Xs[j]) for j in layers] for i in layers]
        bi = layers.index(best)
        band = [best]
        for d in (-1, 1):
            j = bi + d
            while 0 <= j < len(layers) and cka[bi][j] >= 0.9:
                band.append(layers[j])
                j += d
        band = sorted(band)

        def stack(D, Ls):
            s = {L: np.median(pdist(D[L])) for L in Ls}
            return np.concatenate([D[L] / s[L] for L in Ls], 1)
        M1, _ = analyse(stack(Xs, band), roles, (stack(XAs, band), stack(XBs, band)))
        M1all, _ = analyse(stack(Xs, layers), roles, (stack(XAs, layers), stack(XBs, layers)))
        S = stack(Xs, band)
        pca = G.PCA99.fit(S)
        prof = [[float(np.linalg.norm(pca.V[k].reshape(len(band), -1)[i])) for i in range(len(band))]
                for k in range(5)]
        M2 = dict(n90=n_for(np.cumsum(pca.var_ratio / pca.var_ratio.sum())), layer_profiles=prof, band=band,
                  stability=dict(pca_top3_principal_cos=[float(c) for c in np.linalg.svd(
                      G.PCA99.fit(stack(XAs, band)).V[:3] @ G.PCA99.fit(stack(XBs, band)).V[:3].T,
                      compute_uv=False)]))
        summary["views"][view] = dict(per_layer=per_layer, best_layer=best, band=band, cka=cka,
                                      M0=per_layer[best], M1=M1, M1_all_layers=M1all, M2=M2)
        print(view, "best", best, "band", band, "M1 n90", M1["n90"], "M2 n90", M2["n90"], flush=True)
        (out / ("components_%s.json" % a.views.replace(",", "_"))).write_text(json.dumps(summary, default=float))
    if not a.per_layer_only:
        (out / ("components_%s.json" % a.views.replace(",", "_"))).write_text(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
