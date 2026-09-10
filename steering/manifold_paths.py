"""Chord and manifold paths between two fixed points in activation space.

Plan: plans/2026-09-08-steering-rebuild.md (Interventions).

THE CONSTRUCTION. Given a start point P0 and an end point P1:

  linear path    the straight chord P0 -> P1
  manifold path  a cubic spline with THE SAME ENDPOINTS, routed through the role
                 centroids lying inside a cylinder of radius eps*||P1-P0|| around
                 that chord

Both are then sampled at the same alpha values, where alpha in [0,1] is NORMALISED
ARC POSITION along the path -- not a dose, and not a fraction of the knot
coordinate. Same endpoints, same number of stops, different route.

THREE THINGS THIS MODULE ADDS that exist nowhere else in the repo or in causalab:

1. Endpoint-constrained fitting. `spline1d._fit_natural` computes
   `y_hat = y - lam*Q@gamma_int`, and row 0 of `Q@gamma_int` is
   `inv_h[0]*gamma_int[0]`. So pinning `gamma_int[0] = gamma_int[-1] = 0` makes
   the fitted curve pass through the first and last knots EXACTLY, at any lam.
   Pinning components of gamma in a quadratic objective is just deleting those
   rows and columns from the linear system -- no Lagrange multipliers needed.
   Without this, at production lam the ends drift and "same endpoints" is false.

2. Arc-length reparameterisation. The spline is keyed by a scalar knot
   coordinate; arc length is defined BY the fitted curve, so it can only be
   built after fitting. Quadrature on ||dS/du||, cumulative table, monotone
   inversion.

3. Cylinder selection of centroids. causalab's `_subsample_control_points`
   is evenly-spaced or farthest-point -- blind to any reference chord.

THE COORDINATE. Selection and fitting both use projection onto the A->B CHORD,
never `axis_proj = h.a_hat`. Those are different orderings whenever the chord is
not parallel to the Assistant Axis, which is the normal case: measured on this
cloud, ~72% of a near->far displacement is orthogonal to the axis. Fitting on one
ordering while selecting on the other makes the curve zigzag.
"""

import numpy as np

from steering.spline1d import _fit_natural, _eval_segment, fit_gcv


def _require_finite_2d(C, P0):
    """House rule (geometry.py::_require_finite): refuse bad input, never coerce.

    A NaN centroid would otherwise be silently excluded -- `NaN < radius` is
    False -- so the cylinder would quietly shrink instead of failing.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("C must be (n, ambient), got %r" % (C.shape,))
    if C.shape[1] != np.asarray(P0).shape[-1]:
        raise ValueError("C has ambient %d but the endpoints have %d"
                         % (C.shape[1], np.asarray(P0).shape[-1]))
    if not np.isfinite(C).all():
        raise ValueError("centroids contain non-finite values")
    return C


# A centroid closer than this (as a fraction of chord length) to either
# endpoint is dropped: it duplicates a pinned knot rather than adding a route.
END_MARGIN = 1e-3


# --------------------------------------------------------------------------
# Chord frame
# --------------------------------------------------------------------------

def chord_frame(P0: np.ndarray, P1: np.ndarray):
    """Unit direction and length of the chord P0 -> P1."""
    d = np.asarray(P1, dtype=np.float64) - np.asarray(P0, dtype=np.float64)
    L = float(np.linalg.norm(d))
    if not np.isfinite(L) or L <= 0:
        raise ValueError("degenerate chord: ||P1-P0|| = %r" % L)
    return d / L, L


def chord_coords(C: np.ndarray, P0: np.ndarray, P1: np.ndarray):
    """Position along the chord and perpendicular distance from it.

    Returns:
        u: (n,) fraction along the chord. 0 at P0, 1 at P1. Unbounded outside.
        r: (n,) perpendicular distance from the chord line, in activation units.
    """
    dhat, L = chord_frame(P0, P1)
    rel = np.asarray(C, dtype=np.float64) - np.asarray(P0, dtype=np.float64)
    s = rel @ dhat                              # signed distance along the chord
    u = s / L
    perp = rel - np.outer(s, dhat)
    r = np.linalg.norm(perp, axis=1)
    return u, r


def select_cylinder(C: np.ndarray, P0: np.ndarray, P1: np.ndarray, eps: float,
                    mode: str = "absolute"):
    """Indices of points inside the eps-cylinder, ordered along the chord.

    A point qualifies when its perpendicular distance is below `eps*L` AND its
    projection lands strictly between the endpoints. Both conditions matter: the
    second stops the curve doubling back past P0 or P1, which would break the
    monotone knot coordinate the spline requires.

    eps = 0 selects nothing, which is the design's negative control -- the
    manifold path must then collapse onto the chord exactly.
    """
    if mode not in ("absolute", "relative"):
        raise ValueError("mode must be 'absolute' or 'relative', got %r" % (mode,))
    C = _require_finite_2d(C, P0)
    u, r = chord_coords(C, P0, P1)
    _, L = chord_frame(P0, P1)
    # "relative" scales the tube with the chord, which does NOT transfer between
    # chords of different length: measured on this cloud, eps=0.35 gives a tight
    # tube around a 38-unit far-pair chord and a proportionally huge one around a
    # 5.8-unit midway chord, and the detour ratio (arc/chord) then explodes
    # because the denominator is small. "absolute" fixes the tube radius in
    # activation units so every case gets the same physical neighbourhood.
    radius = float(eps) * L if mode == "relative" else float(eps)
    # Exclude a margin at BOTH ends, not just u in (0,1). The endpoint centroids
    # are themselves members of C and project to u = 0 and u = 1 exactly -- but
    # only in exact arithmetic. In floating point B lands a hair either side of
    # 1.0, and when it lands below it is admitted as an INTERIOR knot while also
    # being the pinned endpoint. Two knots ~1e-16 apart make the cubic's second
    # derivative blow up between them, and the path loops instead of travelling.
    #
    # Measured: of 16 routes, the 7 whose control set contained an endpoint had
    # detour 5.7-9.1; the 9 that did not had 1.24-1.72. Perfect separation.
    # A margin in arc terms also protects against a genuinely distinct centroid
    # sitting almost on top of an endpoint, which fails the same way.
    keep = (r < radius) & (u > END_MARGIN) & (u < 1.0 - END_MARGIN)
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(u[idx])]

    return idx, u, r


# --------------------------------------------------------------------------
# Endpoint-constrained cubic spline
# --------------------------------------------------------------------------

def _fit_fixed_ends(x: np.ndarray, y: np.ndarray, h: np.ndarray, lam: float):
    """Reinsch fit with the first and last knots pinned to their given values.

    `_fit_natural` solves `(R + lam Q'Q) gamma_int = Q'y` and returns
    `y_hat = y - lam Q gamma_int`. Row 0 of `Q gamma_int` is
    `inv_h[0]*gamma_int[0]`; row n-1 is `inv_h[n-2]*gamma_int[m-1]`. So
    `gamma_int[0] = gamma_int[m-1] = 0` is exactly the condition
    `y_hat[0] = y[0]` and `y_hat[-1] = y[-1]`.

    Enforcing it = dropping those two rows/columns from the system.
    """
    n = x.shape[0]
    ambient = y.shape[1]

    if n <= 4 or lam == 0.0:
        # n<=4 leaves no free interior gamma once both ends are pinned (keep =
        # arange(1, m-1) is empty at m=2 as well as m<=1), so the
        # constrained solution IS the interpolant through the knots. Force
        # lam=0 there: smoothing with no free parameter would move the ends
        # (measured drift 2e-2 at n=3, lam=3e-6) and break "same endpoints".
        return _fit_natural(x, y, h, 0.0 if n <= 4 else lam)

    m = n - 2
    diag_R = (h[:-1] + h[1:]) / 3.0
    off_R = h[1:-1] / 6.0
    R = np.diag(diag_R) + np.diag(off_R, 1) + np.diag(off_R, -1)

    inv_h = 1.0 / h
    Qt_y = (y[:-2] * inv_h[:-1, None]
            - y[1:-1] * (inv_h[:-1] + inv_h[1:])[:, None]
            + y[2:] * inv_h[1:, None])

    Q = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        Q[j, j] = inv_h[j]
        Q[j + 1, j] = -(inv_h[j] + inv_h[j + 1])
        Q[j + 2, j] = inv_h[j + 1]
    A = R + lam * (Q.T @ Q)

    keep = np.arange(1, m - 1)                  # free interior gammas
    gamma_int = np.zeros((m, ambient), dtype=np.float64)
    gamma_int[keep] = np.linalg.solve(A[np.ix_(keep, keep)], Qt_y[keep])

    gamma = np.zeros((n, ambient), dtype=np.float64)
    gamma[1:-1] = gamma_int
    y_hat = y - lam * (Q @ gamma_int)
    return gamma, y_hat


class FixedEndSpline:
    """Natural cubic spline through (x, y) that passes exactly through the ends.

    Same evaluation maths as `spline1d.CubicSpline1D` -- this reuses its segment
    evaluator -- but the fit pins the endpoints, and the curve is additionally
    given an arc-length parameterisation.
    """

    def __init__(self, x, y, lam):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError("y must be (n, ambient), got %r" % (y.shape,))
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y disagree: %d vs %d" % (x.shape[0], y.shape[0]))
        order = np.argsort(x)
        x, y = x[order], y[order]

        # Strictly increasing is required. spline1d merges EXACT ties; that is
        # not enough here -- two knots 1e-16 apart are not equal but produce the
        # same blow-up, so merge on a relative tolerance instead.
        #
        # THE FIRST AND LAST KNOTS ARE PINNED: they are P0 and P1, and the whole
        # "same endpoints" control rests on them. A group that touches an
        # endpoint therefore takes THAT endpoint's value, not the group mean --
        # averaging an interloper into row 0 moved the curve's start by 6.2 on a
        # unit-scale test and broke the control silently.
        span = float(x[-1] - x[0]) or 1.0
        tol = 1e-9 * span
        if (np.diff(x) <= tol).any():
            xs, ys = [], []
            i, n = 0, len(x)
            while i < n:
                j = i
                while j + 1 < n and x[j + 1] - x[i] <= tol:
                    j += 1
                if i == 0:                       # group holds the start
                    xs.append(x[0]); ys.append(y[0])
                elif j == n - 1:                 # group holds the end
                    xs.append(x[n - 1]); ys.append(y[n - 1])
                else:
                    xs.append(x[i]); ys.append(y[i:j + 1].mean(0))
                i = j + 1
            x, y = np.array(xs), np.stack(ys)

        if len(x) < 2:
            raise ValueError("need at least 2 distinct knots after merging, got %d"
                             % len(x))
        self.x = x
        self.lam = float(lam)
        h = np.diff(x)
        # _fit_fixed_ends forces lam=0 when there is no free interior gamma.
        # Store what was USED, so a sweep table never prints a lam that no fit saw.
        self.lam_used = 0.0 if len(x) <= 4 else float(lam)
        self.gamma, self.y_hat = _fit_fixed_ends(x, y, h, float(lam))
        self._arc = None

    # -- evaluation -------------------------------------------------------

    def evaluate(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        uc = np.clip(u, self.x[0], self.x[-1])
        k = np.clip(np.searchsorted(self.x, uc, side="right") - 1,
                    0, len(self.x) - 2)
        return _eval_segment(uc, self.x[k], self.x[k + 1],
                             self.y_hat[k], self.y_hat[k + 1],
                             self.gamma[k], self.gamma[k + 1])

    def tangent(self, u, eps=1e-5):
        """dS/du. One-sided at the ends, central inside.

        `evaluate` CLIPS to [x0, x-1], so a central difference straddling an
        endpoint divides a one-sided difference by 2e and returns exactly half
        the true derivative. That halved speed made the arc-length quadrature
        first-order instead of second (rel. error 8e-4 rather than 1e-8 at
        n_grid=800) and let detour_ratio report 0.9994 for a curve that IS the
        chord -- a length below the straight line, which is impossible.
        """
        x0, x1 = float(self.x[0]), float(self.x[-1])
        e = eps * (x1 - x0)
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        lo, hi = u - e, u + e
        at_lo = np.where(lo < x0, u, lo)          # one-sided at the left end
        at_hi = np.where(hi > x1, u, hi)          # one-sided at the right end
        step = (at_hi - at_lo)[:, None]
        return (self.evaluate(at_hi) - self.evaluate(at_lo)) / step

    # -- arc length -------------------------------------------------------

    def _build_arc(self, n_grid=800):
        """Cumulative arc length on a dense grid, and its monotone inverse."""
        u = np.linspace(self.x[0], self.x[-1], n_grid)
        speed = np.linalg.norm(self.tangent(u), axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.diff(u) * 0.5 * (speed[1:] + speed[:-1]))])
        self._arc = {"u": u, "s": s, "L": float(s[-1])}
        return self._arc

    @property
    def arc_length(self):
        return (self._arc or self._build_arc())["L"]

    def at_alpha(self, alpha):
        """Point(s) at normalised arc position alpha in [0, 1]."""
        a = self._arc or self._build_arc()
        alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        u = np.interp(np.clip(alpha, 0.0, 1.0) * a["L"], a["s"], a["u"])
        return self.evaluate(u)

    # -- shape descriptors ------------------------------------------------

    def bending_energy(self):
        """Integral of ||S''(u)||^2 du, exactly.

        S'' is linear on each segment between the knot second derivatives, so
        the integral over a segment of width h is
            h/3 * (||g_l||^2 + g_l.g_r + ||g_r||^2)
        which is exact, not quadrature.
        """
        h = np.diff(self.x)
        gl, gr = self.gamma[:-1], self.gamma[1:]
        term = ((gl * gl).sum(1) + (gl * gr).sum(1) + (gr * gr).sum(1))
        return float((h / 3.0 * term).sum())


# --------------------------------------------------------------------------
# The two paths
# --------------------------------------------------------------------------

class LinearPath:
    """The straight chord. alpha is arc position, which here is also fraction."""

    def __init__(self, P0, P1):
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.P1 = np.asarray(P1, dtype=np.float64)
        _, self.arc_length = chord_frame(self.P0, self.P1)
        self.n_centroids = 0

    def at_alpha(self, alpha):
        a = np.atleast_1d(np.asarray(alpha, dtype=np.float64))[:, None]
        return self.P0[None, :] + a * (self.P1 - self.P0)[None, :]

    def bending_energy(self):
        return 0.0


def endpoint_drift(path, P0, P1):
    """Max distance between the path's own ends and the points it must hit.

    The plan's positive control: this must be ~0 for both arms at every eps.
    """
    ends = path.at_alpha(np.array([0.0, 1.0]))
    return float(max(np.linalg.norm(ends[0] - P0), np.linalg.norm(ends[1] - P1)))


def pick_centroids(C: np.ndarray, P0: np.ndarray, P1: np.ndarray, eps: float,
                   k: int = 8, mode: str = "absolute"):
    """Choose k REAL persona centroids for the curve to pass through.

    Every knot is a fully-role-playing persona centroid -- never an average of
    several. A mean of `soldier` and `observer` is not a persona, and a curve
    through it passes through nothing that exists.

    eps sets the candidate set: centroids inside the tube around the A->B chord.
    From those we take k, one per equal span of the chord, choosing in each span
    the candidate NEAREST THE CHORD. That keeps the knots real, fixed in number,
    and well separated along the route -- the separation is what stops an
    interpolating cubic from swinging between them.

    Returns (u, Y, idx): knot coordinates, centroid vectors, and their indices
    into C. Fewer than k are returned when spans are empty.
    """
    idx_all, u_all, r_all = select_cylinder(C, P0, P1, eps, mode)
    if len(idx_all) == 0:
        return np.zeros(0), np.zeros((0, C.shape[1])), np.zeros(0, dtype=int)

    edges = np.linspace(END_MARGIN, 1.0 - END_MARGIN, int(k) + 1)
    u_sel = u_all[idx_all]
    span_of = np.clip(np.searchsorted(edges, u_sel, side="right") - 1, 0, int(k) - 1)

    chosen = []
    for b in range(int(k)):
        cand = idx_all[span_of == b]
        if len(cand) == 0:
            continue
        chosen.append(int(cand[np.argmin(r_all[cand])]))   # nearest the chord
    chosen = np.array(sorted(chosen, key=lambda i: u_all[i]), dtype=int)
    return u_all[chosen], np.asarray(C)[chosen], chosen


class PersonaPath:
    """A->B through k real persona centroids. Smooth, and exact at every knot.

    lam defaults to 0, which for a natural cubic means the curve INTERPOLATES
    every knot exactly while still being C2-continuous -- smooth, no kinks. The
    thing lam=0 cannot prevent is overshoot between knots that sit far apart, so
    `excursion` reports it and lam stays available for the routes that need it.
    """

    def __init__(self, P0, P1, C, eps, k=8, lam=0.0, mode="absolute",
                 param="centripetal"):
        """param: how the spline's abscissa is built from the knots.

        DEFAULT IS "centripetal". Measured on the real cloud, 16 routes, eps=9:
        mean spline overshoot above the polyline floor was 1.173 for
        "projection" (worst case 18.97), 0.135 for "length", 0.114 for
        "centripetal" (worst case 0.43 for both). Exact interpolation is
        unaffected -- knot error stays ~1e-14 in all three.

        "projection"  u = position along the straight chord. The knot gap is
                      then ||dy||*cos(theta)/L while the curve must travel
                      ||dy||, so mean speed on a segment is L/cos(theta) --
                      unbounded as knots go lateral. As k grows the u-gaps
                      shrink like 1/k but the ambient gaps stay at the cloud's
                      lateral scatter, so the speed ratio grows with k and a
                      C2 cubic overshoots. This is the suspected cause of
                      detour rising 1.07 -> 2.08 from k=2 to k=8.
        "length"      cumulative ambient distance between consecutive knots,
                      normalised. causalab implements exactly this
                      (methods/spline/train.py:216-234 at 154a584) and never
                      enables it anywhere.
        "centripetal" the same with distances raised to 0.5 -- the Catmull-Rom
                      choice, provably free of cusps and self-intersections.
        """
        if param not in ("projection", "length", "centripetal"):
            raise ValueError("param must be projection|length|centripetal, got %r"
                             % (param,))
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.P1 = np.asarray(P1, dtype=np.float64)
        self.eps, self.k, self.lam = float(eps), int(k), float(lam)
        self.param = param
        u, Y, idx = pick_centroids(C, self.P0, self.P1, eps, k, mode)
        self.centroid_idx = idx
        self.n_centroids = int(len(idx))

        if self.n_centroids == 0:
            self.spline = None
            self.knots = np.vstack([self.P0[None, :], self.P1[None, :]])
            self._fallback = LinearPath(self.P0, self.P1)
            _, self.arc_length = chord_frame(self.P0, self.P1)
        else:
            Yf = np.vstack([self.P0[None, :], Y, self.P1[None, :]])
            self.knots = Yf
            if param == "projection":
                x = np.concatenate([[0.0], u, [1.0]])
            else:
                d = np.linalg.norm(np.diff(Yf, axis=0), axis=1)
                if param == "centripetal":
                    d = np.sqrt(d)
                x = np.concatenate([[0.0], np.cumsum(d)])
                x = x / x[-1] if x[-1] > 0 else np.linspace(0, 1, len(Yf))
            self.spline = FixedEndSpline(x, Yf, self.lam)
            self.lam = self.spline.lam_used
            self.arc_length = self.spline.arc_length
            self._fallback = None

    def at_alpha(self, alpha):
        return (self._fallback if self.spline is None else self.spline).at_alpha(alpha)

    def bending_energy(self):
        return 0.0 if self.spline is None else self.spline.bending_energy()

    @property
    def detour_ratio(self):
        return float(self.arc_length / chord_frame(self.P0, self.P1)[1])

    def delta(self, alpha):
        """S(alpha) - S(0): the route as a DISPLACEMENT, not a position.

        `interventions.py::make_linear_contrast_delta_fn` records why this
        matters: replacing h with a centroid-like point collapsed within-cell
        spread 16.05 -> 3.79 and emitted "the the the" forever, because no real
        forward pass sits at a mean -- and a path point IS a mean. The additive
        form is the one that has been validated on this model.
        """
        a = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        return self.at_alpha(a) - self.at_alpha(0.0)

    def report(self, names=None):
        """Everything a manifest row needs, in one place."""
        return {"n_centroids": self.n_centroids, "eps": self.eps, "k": self.k,
                "lam": self.lam, "param": self.param,
                "detour_ratio": self.detour_ratio,
                "polyline_ratio": self.polyline_ratio,
                "overshoot": self.overshoot,
                "knot_error": self.knot_error(),
                "excursion": self.excursion(),
                "centroid_idx": [int(i) for i in self.centroid_idx],
                "centroid_roles": ([names[int(i)] for i in self.centroid_idx]
                                   if names is not None else None)}

    @property
    def polyline_ratio(self):
        """Straight-line-through-the-knots length / chord length.

        THE FLOOR. No curve that interpolates these knots can be shorter than
        the polyline joining them, so detour_ratio - polyline_ratio is exactly
        the overshoot the spline adds. If the two are close, the parameterisation
        is fine and any remaining length is the knots' own fault.
        """
        if self.spline is None:
            return 1.0
        d = np.linalg.norm(np.diff(self.knots, axis=0), axis=1).sum()
        return float(d / chord_frame(self.P0, self.P1)[1])

    @property
    def overshoot(self):
        """How much length the spline adds beyond the unavoidable polyline."""
        return float(self.detour_ratio - self.polyline_ratio)

    def knot_error(self):
        """Max distance from a CHOSEN PERSONA CENTROID to the fitted curve.

        Measured against `self.knots`, the centroids themselves -- not against
        `spline.y_hat`, which is what the fit produced. At lam>0 those differ:
        comparing to y_hat reported 7e-15 while the curve actually missed the
        personas by 12.79. This is the only check that the route visits the
        personas it claims to, so it must measure the thing it claims.
        """
        if self.spline is None:
            return 0.0
        got = self.spline.evaluate(self.spline.x[1:-1])
        want = self.knots[1:-1]
        return float(np.linalg.norm(got - want, axis=1).max()) if len(want) else 0.0

    def excursion(self):
        """How far the curve strays from the chord, vs how far its knots do.

        Ratio > 1 means the curve overshoots past the personas it visits, which
        is the only failure mode lam=0 leaves open.
        """
        if self.spline is None:
            return 1.0
        dhat, L = chord_frame(self.P0, self.P1)
        pts = self.at_alpha(np.linspace(0, 1, 200)) - self.P0
        curve = np.linalg.norm(pts - np.outer(pts @ dhat, dhat), axis=1).max()
        kn = self.knots[1:-1] - self.P0
        knots = np.linalg.norm(kn - np.outer(kn @ dhat, dhat), axis=1).max() if len(kn) else 1.0
        return float(curve / max(knots, 1e-9))
