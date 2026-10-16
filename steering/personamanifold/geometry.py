"""PersonaManifold geometry and steering vectors (Xu, Xu, Wu, arXiv 2609.34571).

Equation numbers are the paper's. Where the paper is silent, the released library
(github.com/airaer1998/PersonaManifold @33e181c, src/personamanifold/geometry.py) decides, and
each such choice is marked [repo]. Choices neither source makes are marked [ours].

  eq. 1   x_i = mean_j (h_ij - h_0j)                       persona_vectors
  s3.1    PCA keeping 99% cumulative variance, centred      PCA99   (mean restored on the way back [repo])
  s3.2    d* = consensus of Marchenko-Pastur and Levina-Bickel   intrinsic_dim
  eq. 2   C_i = 1/k sum_{N_k(i)} (x_j - x_i)(x_j - x_i)^T   offsets from x_i, not the local mean [repo]
  eq. 3   g_i = sum_{a<=d*} v_a v_a^T / (lambda_a + eps)    eps = 1e-4 (Table A.1)
  eq. 4   w_ij = 1/2 (|x_j - x_i|_{g_i} + |x_i - x_j|_{g_j}),  kNN graph, k = 5 d* (Table A.1)
  s3.3.1  Dijkstra path, smoothed by a cubic spline         natural cubic, centripetal knots [ours];
                                                            the repo uses the polyline instead
  s3.3.2  points at equal arc length
  eq. 6   delta(t) = alpha * P^+ g(t) / |P^+ g(t)| * s(t),  s(t) = (1-t)|x_1| + t|x_2|
          |x| taken on the reconstructed endpoints P^+ g(0), P^+ g(1) [repo]
  s4.1    Euclidean-Linear baseline: g(t) = (1-t) x_1 + t x_2, same injection rule
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-4
K_PER_DSTAR = 5
VAR_KEEP = 0.99


# ---------------------------------------------------------------------------------------------
# eq. 1 and PCA
# ---------------------------------------------------------------------------------------------

def persona_vectors(H, role, q_idx, H0, q0):
    """eq. 1. H: (n, D) answers, role/q_idx per row; H0: (M, D) neutral answers with ids q0.
    Returns (roles, X) with X[i] = mean over role i's rows of (h - h0[its question])."""
    pos = {q: i for i, q in enumerate(q0)}
    base = H0[[pos[q] for q in q_idx]]
    diff = H.astype(np.float64) - base.astype(np.float64)
    roles = sorted(set(role))
    role = np.asarray(role)
    X = np.stack([diff[role == r].mean(0) for r in roles])
    return roles, X


@dataclass
class PCA99:
    mean: np.ndarray            # (D,)
    V: np.ndarray               # (D', D) rows = components
    var_ratio: np.ndarray

    @classmethod
    def fit(cls, X, keep=VAR_KEEP):
        mu = X.mean(0)
        Xc = X - mu
        if len(X) < X.shape[1]:                       # n < D: eigendecompose the n x n Gram matrix
            lam, U = np.linalg.eigh(Xc @ Xc.T)
            o = np.argsort(lam)[::-1]
            lam, U = np.clip(lam[o], 0, None), U[:, o]
            S = np.sqrt(lam)
            pos = S > S[0] * 1e-10
            Vt = np.zeros((len(S), X.shape[1]))
            Vt[pos] = (U[:, pos].T @ Xc) / S[pos, None]
        else:
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = S ** 2 / (S ** 2).sum()
        rank = int((S > S[0] * 1e-10).sum())
        d = min(int(np.searchsorted(np.cumsum(r), keep) + 1), rank)
        return cls(mu, Vt[:d], r[:d])

    def to_pca(self, X):
        return (X - self.mean) @ self.V.T

    def to_activation(self, Z):                       # P^+ with the mean restored [repo]
        return Z @ self.V + self.mean


# ---------------------------------------------------------------------------------------------
# s3.2 intrinsic dimension
# ---------------------------------------------------------------------------------------------

def _knn(Z, k):
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)      # Gram form: no (N, N, D) array
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, 1)[:, :k]
    return idx, np.sqrt(np.take_along_axis(D2, idx, 1))


def levina_bickel(Z, k1=5, k2=20):
    """Levina & Bickel 2004 eq. 8, averaged over points then over k in [k1, k2] (k-1 normaliser)."""
    _, T = _knn(Z, k2)
    est = []
    for k in range(k1, k2 + 1):
        lg = np.log(T[:, k - 1:k] / T[:, :k - 1])
        est.append(np.mean(1.0 / lg.mean(1)))
    return float(np.mean(est))


def marchenko_pastur(Z, k=None):
    """[ours: the paper names the test, gives no details] Per point, eigenvalues of the kNN
    covariance; signal = eigenvalues above the MP upper edge sigma^2 (1 + sqrt(D'/k))^2, with
    sigma^2 the median of the bottom half of eigenvalues. Returns the median count over points."""
    n, Dp = Z.shape
    k = k or min(n - 1, max(30, n // 4))
    idx, _ = _knn(Z, k)
    counts = []
    for i in range(n):
        Y = Z[idx[i]] - Z[idx[i]].mean(0)
        lam = np.sort(np.linalg.eigvalsh(Y.T @ Y / k) if Dp <= k else
                      np.linalg.eigvalsh(Y @ Y.T / k))[::-1]
        lam = lam[lam > 1e-12 * lam[0]]
        sig2 = np.median(lam[len(lam) // 2:])
        edge = sig2 * (1 + np.sqrt(Dp / k)) ** 2
        counts.append(int((lam > edge).sum()))
    return int(np.median(counts))


def intrinsic_dim(Z):
    lb, mp = levina_bickel(Z), marchenko_pastur(Z)
    return dict(levina_bickel=lb, marchenko_pastur=mp, d_star=int(round((lb + mp) / 2)))


# ---------------------------------------------------------------------------------------------
# eqs. 2-4: metric tensors and the geodesic graph
# ---------------------------------------------------------------------------------------------

@dataclass
class Graph:
    Z: np.ndarray               # (N, D') PCA coordinates
    G: list                     # metric tensors, each (D', D')
    W: "object"                 # scipy csr, symmetric edge weights
    dist: np.ndarray            # (N, N) geodesic distances
    pred: np.ndarray            # (N, N) Dijkstra predecessors
    k: int
    d_star: int


def build_graph(Z, d_star, k=None, eps=EPS):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(Z)
    k = min(k or K_PER_DSTAR * d_star, n - 1)
    idx, _ = _knn(Z, k)
    G = []
    for i in range(n):
        off = Z[idx[i]] - Z[i]                                  # eq. 2, offsets from x_i [repo]
        lam, vec = np.linalg.eigh(off.T @ off / k)
        top = np.argsort(lam)[::-1][:d_star]
        G.append((vec[:, top] / (lam[top] + eps)) @ vec[:, top].T)   # eq. 3
    edges = {}                                                  # edge if either picks the other [repo]
    for i in range(n):
        for j in idx[i]:
            e = (min(i, j), max(i, j))
            if e not in edges:
                d = Z[j] - Z[i]
                edges[e] = max(0.5 * (np.sqrt(max(d @ G[i] @ d, 0)) +
                                      np.sqrt(max(d @ G[j] @ d, 0))), 1e-12)   # eq. 4
    (r, c), v = zip(*edges.keys()), list(edges.values())
    W = csr_matrix((v + v, (list(r) + list(c), list(c) + list(r))), shape=(n, n))
    dist, pred = dijkstra(W, directed=False, return_predecessors=True)
    return Graph(Z, G, W, dist, pred, k, d_star)


def shortest_path(g, a, b):
    path, j = [b], b
    while j != a:
        j = g.pred[a, j]
        if j < 0:
            raise ValueError("no path %d -> %d (graph disconnected)" % (a, b))
        path.append(j)
    return path[::-1]


# ---------------------------------------------------------------------------------------------
# s3.3 the path and eq. 6
# ---------------------------------------------------------------------------------------------

def smooth_path(P, ts, n_dense=2000):
    """Natural cubic through the path nodes P (L+1, D'), centripetal knots, resampled at equal
    arc length; returns the points at arc fractions ts. Two nodes -> the straight segment."""
    from scipy.interpolate import CubicSpline
    if len(P) == 2:
        return np.stack([(1 - t) * P[0] + t * P[1] for t in ts])
    u = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1) ** 0.5)])
    cs = CubicSpline(u / u[-1], P, bc_type="natural", axis=0)
    s = np.linspace(0, 1, n_dense)
    Y = cs(s)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Y, axis=0), axis=1))])
    return np.stack([Y[np.searchsorted(arc, t * arc[-1]).clip(0, n_dense - 1)] for t in ts])


def steering_deltas(pca, gamma, ts, alpha=1.0):
    """eq. 6 for the points gamma (K, D') sampled at arc fractions ts (ts[0] = 0, ts[-1] = 1)
    -> (K, D) additive deltas in activation space."""
    A = pca.to_activation(gamma)
    nrm = np.linalg.norm(A, axis=1)
    ts = np.asarray(ts, dtype=np.float64)
    s = (1 - ts) * nrm[0] + ts * nrm[-1]
    return alpha * A / nrm[:, None] * s[:, None]


def route(pca, g, roles, src, tgt, ts):
    """Both paper arms for one (source, target): geodesic and Euclidean-linear."""
    a, b = roles.index(src), roles.index(tgt)
    nodes = shortest_path(g, a, b)
    Z = g.Z
    geo = smooth_path(Z[nodes], ts)
    lin = np.stack([(1 - t) * Z[a] + t * Z[b] for t in ts])
    dE = float(np.linalg.norm(Z[a] - Z[b]))
    return dict(nodes=[roles[i] for i in nodes], d_g=float(g.dist[a, b]), d_E=dE,
                d_g_euclid_len=float(np.linalg.norm(np.diff(Z[nodes], axis=0), axis=1).sum()),
                ts=list(ts), geo=geo, lin=lin)
