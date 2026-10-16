"""Routes for C (calibration), E1 (steering stability) and E2 (principal direction) — plan 2026-10-09.

Every route is built from ONE persona representation: the E1b extraction (Assistant Axis roles,
prompts and questions; PersonaManifold eq. 1: last answer token minus the no-prompt answer, 20 questions
x 3 prompts per persona). Routes live in the 99% PCA space Z of the 276 persona vectors and become
additive deltas with PersonaManifold eq. 6 (geometry.steering_deltas) — the same rule for every route.

  LIN    straight line A -> B
  PMG    PersonaManifold geodesic: paper graph (metric kNN, k = 5 d*), Dijkstra, cubic spline
  MANk   cubic through k real personas near the A -> B chord (one per equal span, nearest the chord)
  FL     Fletcher straight line: straight line in principal-geodesic coordinates (top K for 90%),
         turned into a chain of the personas nearest to it, cubic through them
  FG     Fletcher principal geodesic: walk the Fletcher graph from A, each step to the neighbour that
         advances most along direction k while changing the other top-K coordinates least
  PCA    straight line along the k-th PCA axis (E2 baseline)

The Fletcher graph (Euclidean kNN, k = 10, raised until connected) and principal-geodesic coordinates
come from pm_geometry.components (Fletcher et al. 2004, Alg. 2 on the data graph).
"""
from __future__ import annotations

import numpy as np

from pm_geometry import components as C
from pm_geometry import geometry as G

TS_FINE = tuple(np.round(np.linspace(0, 1, 11), 2))       # every route is sampled here, then indexed
WALK_C = 2.0                                               # FG / PCA / E2 distance, in sigma_k


class LayerGeom:
    def __init__(self, roles, X, X_abs, pca, Z):
        self.roles, self.X, self.X_abs, self.pca, self.Z = list(roles), X, X_abs, pca, Z
        self.idim = G.intrinsic_dim(Z)
        self.gP = G.build_graph(Z, self.idim["d_star"])                 # paper graph
        self.gF = C.knn_graph(Z)                                         # Fletcher graph
        self.P = C.pga(Z, self.gF)
        lam = self.P["lam"]
        self.K = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.90) + 1)
        self.coords = self.P["U"] @ self.P["V"][:, :self.K]              # principal-geodesic coordinates
        self.sig = np.sqrt(lam[:self.K])
        self.sig_pca = Z.std(0)
        D, k = self.gF["D"], self.gF["k"]
        knn = np.argsort(D, 1)[:, 1:k + 1]
        A = np.zeros(D.shape, bool)
        A[np.repeat(np.arange(len(Z)), k), knn.ravel()] = True
        A |= A.T                                                         # edge if either picks the other
        self.nbrs = [np.flatnonzero(A[i]) for i in range(len(Z))]

    def idx(self, name):
        return self.roles.index(name)

    # ---------------------------------------------------------------------------------------------
    def _spline(self, nodes):
        return G.smooth_path(self.Z[nodes], TS_FINE)

    def lin(self, a, b):
        return np.stack([(1 - t) * self.Z[a] + t * self.Z[b] for t in TS_FINE]), [a, b]

    def pmg(self, a, b):
        nodes = G.shortest_path(self.gP, a, b)
        return self._spline(nodes), nodes

    def man(self, a, b, k):
        za, zb = self.Z[a], self.Z[b]
        v = zb - za
        L2 = v @ v
        u = (self.Z - za) @ v / L2
        perp = np.linalg.norm((self.Z - za) - np.outer(u, v), axis=1)
        knots = []
        for j in range(k):
            lo, hi = j / k, (j + 1) / k
            m = (u > max(lo, 1e-3)) & (u < min(hi, 1 - 1e-3))
            m[[a, b]] = False
            if m.any():
                c = np.flatnonzero(m)
                knots.append(int(c[np.argmin(perp[c])]))
        nodes = [a] + sorted(set(knots), key=lambda i: u[i]) + [b]
        return self._spline(nodes), nodes

    def _chain(self, cpath, a, b):
        """Personas nearest to each point of a curve in principal-geodesic coordinates, A first, B last."""
        nodes = [a]
        for c in cpath:
            j = int(np.argmin(((self.coords - c) ** 2).sum(1)))
            if j != nodes[-1] and j not in nodes:
                nodes.append(j)
        if b is not None:
            nodes = [n for n in nodes if n != b] + [b]
        elif len(nodes) == 1:                                            # no persona left the start
            d = ((self.coords - cpath[-1]) ** 2).sum(1)
            d[a] = np.inf
            nodes.append(int(np.argmin(d)))
        return nodes

    def fl(self, a, b):
        cs = [(1 - t) * self.coords[a] + t * self.coords[b] for t in np.linspace(0, 1, 60)]
        nodes = self._chain(cs, a, b)
        return self._spline(nodes), nodes

    def fg_walk(self, a, k, sign, lam=1.0, max_steps=12):
        """Fletcher principal geodesic from A along direction k (0-based), sign +/-1."""
        target = WALK_C * self.sig[k]
        cur, adv, nodes = a, 0.0, [a]
        e = np.zeros(self.K)
        e[k] = sign
        for _ in range(max_steps):
            best, bs = None, -np.inf
            for j in self.nbrs[cur]:
                if j in nodes:
                    continue
                dc = self.coords[j] - self.coords[cur]
                along = dc @ e
                if along <= 0:
                    continue
                score = along - lam * np.linalg.norm(dc - along * e)
                if score > bs:
                    best, bs = int(j), score
            if best is None:
                break
            adv += (self.coords[best] - self.coords[cur]) @ e
            nodes.append(best)
            cur = best
            if adv >= target:
                break
        if len(nodes) == 1:                                              # no neighbour advanced: nearest
            d = ((self.coords - (self.coords[a] + target * e)) ** 2).sum(1)   # persona to the intended end
            d[a] = np.inf
            nodes.append(int(np.argmin(d)))
        return nodes

    def fg(self, nodes):
        return self._spline(nodes), nodes

    def fl_dir(self, a, k, sign):
        e = np.zeros(self.K)
        e[k] = sign
        cs = [self.coords[a] + t * WALK_C * self.sig[k] * e for t in np.linspace(0, 1, 60)]
        nodes = self._chain(cs, a, None)
        return self._spline(nodes), nodes

    def pca_dir(self, a, k, sign):
        e = np.zeros(self.Z.shape[1])
        e[k] = sign
        pts = np.stack([self.Z[a] + t * WALK_C * self.sig_pca[k] * e for t in TS_FINE])
        d = ((self.Z - pts[-1]) ** 2).sum(1)
        d[a] = np.inf
        end = int(np.argmin(d))
        return pts, [a, end]

    # ---------------------------------------------------------------------------------------------
    def deltas(self, gamma, alpha=1.0):
        """eq. 6 for all TS_FINE points of one route -> (11, D)."""
        return G.steering_deltas(self.pca, gamma, TS_FINE, alpha)


def at(s):
    """Index of step s in TS_FINE."""
    return int(round(float(s) * 10))
