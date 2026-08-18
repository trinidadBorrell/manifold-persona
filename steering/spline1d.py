"""Natural cubic spline (Reinsch), numpy port of causalab's ``CubicSpline1D``.

Why this exists at all
----------------------
``causalab.methods.spline.SplineManifold`` with ``intrinsic_dim == 1`` does
**not** use the thin-plate spline — it resolves to ``CubicSpline1D``, a Reinsch
(1967) natural cubic spline. The 1-D path *is* the causalab path, so a faithful
port is the honest way to use it here. causalab itself cannot be imported: it
needs Python 3.12 and this venv is 3.9 (RESEARCH.steering.md).

Why 1-D at all
--------------
causalab's ``control_points`` are intrinsic coordinates *handed to it by a
causal model* — a coordinate that means something, known in advance. The H1 run
had no such coordinate for roles, manufactured one from unsupervised PCA, and
had to pick ``k``; DESIGN.md records the post-hoc finding that the assumed k=3
was wrong (true ID ~8-10). This track does not have to manufacture one: the
Assistant-Axis projection is externally defined by the paper's contrast vector
(arXiv:2601.10387 section 3.1) and plays exactly causalab's role. So
``intrinsic_dim = 1`` and there is no dimension to choose anywhere.

Fidelity
--------
Ported from ``causalab/methods/spline/cubic.py``:

- ``_fit_natural``  -> :func:`_fit_natural`   (Reinsch system, gamma_0 = gamma_{n-1} = 0)
- ``_eval_segment`` -> :func:`_eval_segment`  (the (a, b) form, unchanged)
- ``_evaluate_natural`` -> :meth:`CubicSpline1D.evaluate`
  (clamp into the knot range, then add the linear extrapolation correction)

torch -> numpy, float64 throughout. The smoothing parameter ``lam`` is kept so
the port is complete, but this run fixes ``lam = 0`` (plain interpolation):
tuning it against an outcome we have already looked at is exactly what an
exploratory run must not do (plan, Method).
"""
from __future__ import annotations

import numpy as np


def _fit_natural(x: np.ndarray, y: np.ndarray, h: np.ndarray, lam: float):
    """Second derivatives ``gamma`` at all knots, and fitted knot values.

    Reinsch formulation. With natural BCs ``gamma_0 = gamma_{n-1} = 0`` the
    interior second derivatives satisfy ``(R + lam Q^T Q) gamma_int = Q^T y``
    and the smoothed values are ``y_hat = y - lam Q gamma_int``. At ``lam = 0``
    this collapses to plain natural-cubic interpolation (``y_hat = y``).

    Args:
        x: (n,) sorted, strictly increasing knot coordinates.
        y: (n, ambient) knot values.
        h: (n-1,) spacings ``x[1:] - x[:-1]``.
        lam: smoothing parameter; 0 = interpolation.

    Returns:
        (gamma (n, ambient), y_hat (n, ambient)).
    """
    n = x.shape[0]
    ambient = y.shape[1]

    if n == 2:
        # Degenerate: only the two endpoints, both gamma = 0 -> a straight line.
        return np.zeros((n, ambient), dtype=np.float64), y.copy()

    m = n - 2                                  # interior knot count

    # R (m x m, symmetric tridiagonal).
    diag_R = (h[:-1] + h[1:]) / 3.0            # (m,)
    off_R = h[1:-1] / 6.0                      # (m-1,)
    R = np.diag(diag_R) + np.diag(off_R, 1) + np.diag(off_R, -1)

    # Q^T y without materialising Q. Q's column j-1 (interior knot j) has
    #   Q[j-1, j-1] = 1/h[j-1],  Q[j, j-1] = -(1/h[j-1] + 1/h[j]),  Q[j+1, j-1] = 1/h[j]
    inv_h = 1.0 / h                            # (n-1,)
    Qt_y = (y[:-2] * inv_h[:-1, None]
            - y[1:-1] * (inv_h[:-1] + inv_h[1:])[:, None]
            + y[2:] * inv_h[1:, None])         # (m, ambient)

    if lam == 0.0:
        A = R
    else:
        Q = np.zeros((n, m), dtype=np.float64)
        for j in range(m):
            Q[j, j] = inv_h[j]
            Q[j + 1, j] = -(inv_h[j] + inv_h[j + 1])
            Q[j + 2, j] = inv_h[j + 1]
        A = R + lam * (Q.T @ Q)

    gamma_int = np.linalg.solve(A, Qt_y)       # (m, ambient)

    gamma = np.zeros((n, ambient), dtype=np.float64)
    gamma[1:-1] = gamma_int

    if lam == 0.0:
        y_hat = y.copy()
    else:
        Qg = np.zeros_like(y)
        for j in range(m):
            Qg[j] += inv_h[j] * gamma_int[j]
            Qg[j + 1] += -(inv_h[j] + inv_h[j + 1]) * gamma_int[j]
            Qg[j + 2] += inv_h[j + 1] * gamma_int[j]
        y_hat = y - lam * Qg

    return gamma, y_hat


def _eval_segment(u, x_left, x_right, y_l, y_r, g_l, g_r):
    """Standard cubic-spline segment formula in (a, b) form. ``u`` is 1-D."""
    h_seg = (x_right - x_left)[:, None]
    a = (x_right - u)[:, None]
    b = (u - x_left)[:, None]
    return (g_l * a ** 3 / (6.0 * h_seg)
            + g_r * b ** 3 / (6.0 * h_seg)
            + (y_l / h_seg - g_l * h_seg / 6.0) * a
            + (y_r / h_seg - g_r * h_seg / 6.0) * b)


class CubicSpline1D:
    """A curve ``u -> R^ambient`` through control points, natural BCs.

    Duplicate knot coordinates are averaged rather than rejected: two roles can
    share an ``axis_proj`` value to float precision, and dropping one of them
    would silently delete a role from the manifold.
    """

    def __init__(self, u: np.ndarray, y: np.ndarray, lam: float = 0.0):
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64)
        if u.shape[0] != y.shape[0]:
            raise ValueError("u and y must have the same number of rows, got "
                             "%d and %d" % (u.shape[0], y.shape[0]))
        if u.shape[0] < 2:
            raise ValueError("need at least 2 control points, got %d" % u.shape[0])

        order = np.argsort(u, kind="stable")
        u, y = u[order], y[order]

        # Average exact ties so spacings are strictly positive.
        uniq, inv = np.unique(u, return_inverse=True)
        if uniq.shape[0] != u.shape[0]:
            y_avg = np.zeros((uniq.shape[0], y.shape[1]), dtype=np.float64)
            counts = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
            np.add.at(y_avg, inv, y)
            y = y_avg / counts[:, None]
            u = uniq
            if u.shape[0] < 2:
                raise ValueError("all control points share one coordinate")

        self.x = u
        self.h = np.diff(u)
        if np.any(self.h <= 0):
            raise ValueError("knot coordinates must be strictly increasing")
        self.lam = float(lam)
        self._y_obs = y                        # observed knot values, for GCV
        self.gamma, self.y_hat = _fit_natural(self.x, y, self.h, self.lam)
        self.ambient = y.shape[1]

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Evaluate at ``u`` (batch,) -> (batch, ambient).

        Inside the knot range this is the cubic; outside it is the linear
        extrapolation causalab applies, whose boundary slopes simplify because
        ``gamma_0 = gamma_{n-1} = 0``.
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        x, h, gamma, y_hat = self.x, self.h, self.gamma, self.y_hat
        n = x.shape[0]

        u_clamped = np.clip(u, x[0], x[-1])
        idx = np.clip(np.searchsorted(x, u_clamped, side="right") - 1, 0, n - 2)

        out = _eval_segment(u_clamped, x[idx], x[idx + 1],
                            y_hat[idx], y_hat[idx + 1], gamma[idx], gamma[idx + 1])

        h0, hn = h[0], h[-1]
        slope_left = (y_hat[1] - y_hat[0]) / h0 - h0 * gamma[1] / 6.0
        slope_right = (y_hat[-1] - y_hat[-2]) / hn + hn * gamma[-2] / 6.0
        delta_left = np.clip(u - x[0], None, 0.0)[:, None]
        delta_right = np.clip(u - x[-1], 0.0, None)[:, None]
        return out + slope_left[None, :] * delta_left + slope_right[None, :] * delta_right

    def gcv_score(self, u_ctrl: np.ndarray, y_ctrl: np.ndarray) -> float:
        """Generalized cross-validation score for this fit. Lower is better.

            GCV(lam) = (RSS / n) / (1 - tr(H)/n)^2

        `H` is the smoother matrix mapping observed values to fitted values.
        The Reinsch form gives it in closed form as ``H = I - lam Q A^-1 Q^T``
        with ``A = R + lam Q^T Q`` — so the effective degrees of freedom
        ``tr(H)`` costs one m x m solve rather than n refits, and no leave-one-out
        loop is needed.

        GCV is an approximation to leave-one-out prediction error. What matters
        here is the property that makes it usable above the preregistration
        boundary: **it only ever sees the 276 role centroids.** It never sees a
        steered generation, a judge label, or a category fraction, so choosing
        lambda by GCV cannot tune the answer toward a result we like.
        """
        n = self.x.shape[0]
        y_hat = self.evaluate(self.x)
        rss = float(((y_hat - self._y_obs) ** 2).sum())

        if self.lam == 0.0:
            trH = float(n)                     # interpolation: H = I
        else:
            h = self.h
            m = n - 2
            inv_h = 1.0 / h
            diag_R = (h[:-1] + h[1:]) / 3.0
            off_R = h[1:-1] / 6.0
            R = np.diag(diag_R) + np.diag(off_R, 1) + np.diag(off_R, -1)
            Q = np.zeros((n, m), dtype=np.float64)
            for j in range(m):
                Q[j, j] = inv_h[j]
                Q[j + 1, j] = -(inv_h[j] + inv_h[j + 1])
                Q[j + 2, j] = inv_h[j + 1]
            # errstate: numpy 2.0.2 on macOS Accelerate emits spurious FP
            # warnings from any matmul (see steering/geometry.py::_matmul).
            with np.errstate(all="ignore"):
                QtQ = Q.T @ Q
                A = R + self.lam * QtQ
                # tr(H) = n - lam*tr(Q A^-1 Q^T) = n - lam*sum(diag(A^-1 (Q^T Q)))
                trH = float(n - self.lam * np.sum(np.linalg.solve(A, QtQ).diagonal()))
            if not np.isfinite(trH):
                return float("inf")

        denom = 1.0 - trH / n
        if denom <= 1e-12:                     # saturated: no effective smoothing
            return float("inf")
        # Per-coordinate mean so the score does not scale with ambient dim.
        return (rss / (n * self._y_obs.shape[1])) / denom ** 2

    def tangent(self, u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Central-difference tangent dS/du at ``u`` -> (batch, ambient).

        Central rather than the forward difference ``tps.py`` uses: this is a
        1-D curve so the extra evaluation is cheap, and the tangent is the
        steering direction itself, where accuracy matters more than speed.
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        return (self.evaluate(u + eps) - self.evaluate(u - eps)) / (2.0 * eps)


def fit_gcv(u: np.ndarray, y: np.ndarray, grid: np.ndarray = None):
    """Fit with lambda chosen by GCV over a log grid. Returns (spline, report).

    Why a grid and not an optimiser: the search is over one scalar, the grid is
    logged in full so the choice is auditable, and a grid cannot land in a local
    minimum an optimiser would have to be trusted to escape.
    """
    if grid is None:
        grid = np.concatenate([[0.0], np.logspace(-6, 6, 49)])
    scores = []
    for lam in grid:
        try:
            s = CubicSpline1D(u, y, lam=float(lam))
            scores.append(s.gcv_score(u, y))
        except Exception:
            scores.append(float("inf"))
    scores = np.asarray(scores, dtype=np.float64)
    best = int(np.argmin(scores))
    spline = CubicSpline1D(u, y, lam=float(grid[best]))
    report = {"rule": "GCV over a fixed log grid; outcome-blind (sees only centroids)",
              "grid": [float(g) for g in grid],
              "scores": [float(s) for s in scores],
              "lambda": float(grid[best]),
              "gcv": float(scores[best])}
    return spline, report
