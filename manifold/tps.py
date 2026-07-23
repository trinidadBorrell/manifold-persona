"""Thin-plate-spline manifold — numpy reimplementation of causalab's spline.

We cannot import ``causalab.methods.spline`` here: causalab targets Python 3.12
(runtime ``X | Y`` unions) and this extraction venv is 3.9. So this module:
  - **faithfully ports** ``causalab/methods/spline/tps.py`` (kernel, augmented
    linear solve, evaluate) — line-for-line for the pure-linear case, and
  - **adapts, with hardening,** the projection of
    ``causalab/methods/spline/manifold.py::encode_to_nearest_point``: same
    Gauss-Newton idea and nearest-centroid warm start, but forward-difference
    Jacobian, an intrinsic-coordinate clamp box, and a keep-best-residual rule
    (see ``project``). The hardening is required for the high-D-ambient /
    low-D-surface regime and is applied identically to real and null data.
Restricted to the pure-linear (non-periodic, non-sphere) case we need for roles.

A ``SplineManifold`` interpolates ``control_points`` (k-dim intrinsic coords) ->
``target_points`` (D-dim ambient anchors), giving a smooth k-dim surface. We then
project arbitrary ambient points onto that surface and read off the residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


def thin_plate_kernel(r: np.ndarray, dim: int) -> np.ndarray:
    """phi(r) = r^(dim-2) * log(r), matching causalab tps.thin_plate_kernel.

    dim==2 -> r^2 log r; general -> r^(dim-2) log r. Zero at r~0.
    """
    r_safe = np.clip(r, 1e-10, None)
    kernel = r_safe ** (dim - 2) * np.log(r_safe)
    return np.where(r < 1e-10, 0.0, kernel)


class SplineManifold:
    """TPS surface: control_points (n,k) -> target_points (n,D).

    Intrinsic coords are min/range-normalised to [0,1]^k for conditioning
    (mirrors causalab SplineManifold._normalize_intrinsic); the same transform is
    applied to every query. ``smoothness`` adds lambda*I to the kernel block
    (0 = exact interpolation), exactly as in causalab tps._fit.
    """

    def __init__(self, control_points: np.ndarray, target_points: np.ndarray,
                 smoothness: float = 0.0):
        cp = np.asarray(control_points, dtype=np.float64)
        tp = np.asarray(target_points, dtype=np.float64)
        self.k = cp.shape[1]
        self.D = tp.shape[1]
        self.target_points = tp
        # intrinsic normalisation
        self._cp_min = cp.min(axis=0)
        self._cp_range = np.clip(cp.max(axis=0) - cp.min(axis=0), 1e-8, None)
        cpn = (cp - self._cp_min) / self._cp_range
        self.control_points = cpn
        self._fit(cpn, tp, smoothness)

    def _fit(self, cp: np.ndarray, tp: np.ndarray, smoothness: float) -> None:
        n = cp.shape[0]
        K = thin_plate_kernel(cdist(cp, cp), self.k)
        if smoothness > 0:
            K = K + smoothness * np.eye(n)
        P = np.concatenate([np.ones((n, 1)), cp], axis=1)   # [1, x] polynomial
        m = P.shape[1]
        A = np.zeros((n + m, n + m))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        rhs = np.zeros((n + m, self.D))
        rhs[:n] = tp
        coeffs = np.linalg.solve(A, rhs)
        self._w = coeffs[:n]        # (n, D) rbf weights
        self._poly = coeffs[n:]     # (m, D) polynomial coeffs

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the surface at intrinsic coords u (batch,k) -> (batch,D).

        u is the *unnormalised* intrinsic coordinate; normalisation is applied
        here so callers work in the natural coordinate frame.
        """
        un = (np.asarray(u, dtype=np.float64) - self._cp_min) / self._cp_range
        Kq = thin_plate_kernel(cdist(un, self.control_points), self.k)
        Pq = np.concatenate([np.ones((un.shape[0], 1)), un], axis=1)
        return Kq @ self._w + Pq @ self._poly

    def _decode_norm(self, un: np.ndarray) -> np.ndarray:
        """decode for already-normalised coords (internal GN loop)."""
        Kq = thin_plate_kernel(cdist(un, self.control_points), self.k)
        Pq = np.concatenate([np.ones((un.shape[0], 1)), un], axis=1)
        return Kq @ self._w + Pq @ self._poly

    def project(self, x: np.ndarray, n_iters: int = 8, fd_eps: float = 1e-4,
                damping: float = 1e-3):
        """Gauss-Newton projection onto the surface (adapted from causalab
        encode_to_nearest_point), hardened for a high-D ambient / low-D surface.

        Warm start: the intrinsic coord of the nearest target point. Iterates
        argmin_u ||decode(u) - x||^2 with a *forward*-difference Jacobian
        (causalab uses central; forward halves the decodes), but:
          - intrinsic coords are clamped to a padded box around the control
            points so the TPS is never queried where it extrapolates to huge
            values (the cause of overflow), and
          - we keep the *best* (smallest-residual) coord seen per point, so the
            projection can never be worse than the nearest-centroid warm start
            (guarantees a well-defined, monotone reconstruction).
        """
        x = np.asarray(x, dtype=np.float64)
        b = x.shape[0]
        lo = self.control_points.min(0) - 0.25       # padded box in normalised coords
        hi = self.control_points.max(0) + 0.25
        nn = np.argmin(cdist(x, self.target_points), axis=1)
        un = self.control_points[nn].copy()          # (b, k) normalised
        x_hat = self._decode_norm(un)
        best_un = un.copy()
        best_res = np.linalg.norm(x - x_hat, axis=1)
        for _ in range(n_iters):
            r = x_hat - x
            J = np.zeros((b, self.D, self.k))         # forward diff (reuse x_hat)
            for j in range(self.k):
                uf = un.copy(); uf[:, j] += fd_eps
                J[:, :, j] = (self._decode_norm(uf) - x_hat) / fd_eps
            JtJ = np.einsum("bij,bik->bjk", J, J) + damping * np.eye(self.k)
            Jtr = np.einsum("bij,bi->bj", J, r)
            delta = np.linalg.solve(JtJ, Jtr[..., None])[..., 0]
            un = np.clip(un - delta, lo, hi)
            x_hat = self._decode_norm(un)
            res = np.linalg.norm(x - x_hat, axis=1)
            better = res < best_res
            best_res = np.where(better, res, best_res)
            best_un[better] = un[better]
        return best_res, self._decode_norm(best_un)


@dataclass
class ReconResult:
    ssr: float
    tss: float
    nre: float      # SSR / TSS
    r2: float       # 1 - NRE


def reconstruction(points: np.ndarray, manifold: SplineManifold,
                   global_mean: np.ndarray) -> ReconResult:
    """NRE / R^2 of `points` against a fitted manifold.

    TSS uses a *fixed* global mean (the raw-cloud mean) so the denominator is
    identical for real and permuted runs — only SSR moves.
    """
    resid, _ = manifold.project(points)
    ssr = float(np.sum(resid ** 2))
    tss = float(np.sum(np.sum((points - global_mean) ** 2, axis=1)))
    nre = ssr / tss
    return ReconResult(ssr=ssr, tss=tss, nre=nre, r2=1.0 - nre)
