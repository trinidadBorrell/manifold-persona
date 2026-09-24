"""The `nearest` knot rule and the path built through it (plan exp3, mentor point a).

Plan: plans/2026-09-24-steering-followups.md (Design, exp3).

WHY A THIRD RULE. `manifold_paths.pick_centroids` has two partitions,
"distance" (equal-width spans of the chord) and "density" (equal-count groups),
and both then take ONE knot per chunk. That guarantees spread along the route
but not proximity: a chunk whose only candidates are far off the chord still
contributes a knot, and the spline is dragged out to it. The mentor's
suggestion is the opposite trade -- take the k centroids closest to the A-B
SEGMENT, full stop, and let them fall wherever they fall along it.

DISTANCE TO THE SEGMENT, NOT THE LINE. The projection u is clamped to [0, 1]
before measuring, so a centroid beyond an end is scored by its distance to
that END, not by its perpendicular distance to the infinite line. Without the
clamp, a persona 3 chords past the target but exactly on the line would score
0 and win.

EXCLUSIONS. The two endpoint roles themselves (`exclude_idx`), and any
centroid whose chord coordinate is within END_MARGIN of 0 or 1 -- the same
margin, and for the same reason, as `select_cylinder`: a knot a hair from a
pinned end makes the cubic's second derivative blow up between them and the
path loops. That is a BAND around each end, |u| < END_MARGIN or
|u - 1| < END_MARGIN. It is NOT a one-sided cut u in (margin, 1 - margin):
that cut would make the clamp above meaningless, since nothing outside [0, 1]
could then be picked at all. So a picked knot CAN lie beyond an end (u < 0 or
u > 1); exp3 reports how many do (`n_outside`), because such a knot makes the
route double back.

THE PATH. `NearestPath` is a `PersonaPath` whose knots come from this rule.
PersonaPath picks its own knots inside `__init__` and has no hook for
externally chosen ones, so the subclass replaces only the constructor and
reuses the rest unchanged: `FixedEndSpline` (the endpoint-pinned Reinsch fit
and arc-length table), `at_alpha`, `delta`, `report`, `knot_error`,
`detour_ratio`. The abscissa construction (centripetal by default) is the one
PersonaPath documents; it is restated here in six lines because it is inlined
in PersonaPath.__init__ and cannot be called on its own.
"""
from __future__ import annotations

import numpy as np

from steering.manifold_paths import (END_MARGIN, FixedEndSpline, LinearPath,
                                     PersonaPath, _require_finite_2d, chord_coords,
                                     chord_frame)


def segment_distance(C, P0, P1):
    """(u, d_seg): chord coordinate and Euclidean distance to the SEGMENT [P0, P1]."""
    C = _require_finite_2d(C, P0)
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    u, _ = chord_coords(C, P0, P1)
    uc = np.clip(u, 0.0, 1.0)
    foot = P0[None, :] + uc[:, None] * (P1 - P0)[None, :]
    return u, np.linalg.norm(C - foot, axis=1)


def pick_nearest_segment(C, P0, P1, k, exclude_idx=()):
    """The k centroids nearest the segment [P0, P1], ordered by chord coordinate.

    Same return contract as `manifold_paths.pick_centroids`: (u, Y, idx) --
    knot chord coordinates, centroid vectors, indices into C -- ascending in u.
    Fewer than k only when fewer candidates survive the exclusions.

    Ties in distance are broken by index (stable sort), so the pick is
    deterministic for a given C.
    """
    if int(k) < 0:
        raise ValueError("k must be >= 0, got %r" % (k,))
    C = _require_finite_2d(C, P0)
    u, d = segment_distance(C, P0, P1)
    keep = (np.abs(u) >= END_MARGIN) & (np.abs(u - 1.0) >= END_MARGIN)
    ex = np.asarray(list(exclude_idx), dtype=int)
    if ex.size:
        keep[ex] = False
    cand = np.nonzero(keep)[0]
    if int(k) == 0 or len(cand) == 0:
        return np.zeros(0), np.zeros((0, C.shape[1])), np.zeros(0, dtype=int)
    chosen = cand[np.argsort(d[cand], kind="stable")[:int(k)]]
    chosen = chosen[np.argsort(u[chosen], kind="stable")]
    return u[chosen], C[chosen], chosen


def endpoint_exclusions(C, P0, P1, tol=1e-9):
    """Indices of centroids that coincide with an endpoint (the endpoint roles).

    For Study A the ends ARE role centroids and are excluded by index. For
    Study B, P0 is a measured footprint that matches no centroid, so only the
    target is found here. Matching by value rather than by name means the
    caller cannot pass the wrong role and silently keep an endpoint as a knot.
    """
    C = np.asarray(C, dtype=np.float64)
    out = []
    for P in (P0, P1):
        dd = np.linalg.norm(C - np.asarray(P, dtype=np.float64)[None, :], axis=1)
        out.extend(np.nonzero(dd <= tol * max(1.0, float(np.linalg.norm(P))))[0].tolist())
    return sorted(set(out))


class NearestPath(PersonaPath):
    """P0 -> P1 through the k centroids nearest the segment. chunk = "nearest".

    Everything after knot selection is inherited from PersonaPath, so the
    nearest arm and the distance/density arms differ in WHICH centroids are
    knots and in nothing else -- the property that lets exp3/exp4 compare them.
    """

    def __init__(self, P0, P1, C, k=8, lam=0.0, param="centripetal", exclude_idx=None):
        if param not in ("projection", "length", "centripetal"):
            raise ValueError("param must be projection|length|centripetal, got %r" % (param,))
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.P1 = np.asarray(P1, dtype=np.float64)
        self.eps, self.k, self.lam = float("nan"), int(k), float(lam)
        self.param = param
        self.chunk, self.m = "nearest", None
        if exclude_idx is None:
            exclude_idx = endpoint_exclusions(C, self.P0, self.P1)
        self.exclude_idx = [int(i) for i in exclude_idx]
        u, Y, idx = pick_nearest_segment(C, self.P0, self.P1, k, self.exclude_idx)
        self.knot_u = u
        self.centroid_idx = idx
        self.n_centroids = int(len(idx))

        if self.n_centroids == 0:
            self.spline = None
            self.knots = np.vstack([self.P0[None, :], self.P1[None, :]])
            self._fallback = LinearPath(self.P0, self.P1)
            _, self.arc_length = chord_frame(self.P0, self.P1)
            return
        Yf = np.vstack([self.P0[None, :], Y, self.P1[None, :]])
        self.knots = Yf
        if param == "projection":
            x = np.concatenate([[0.0], u, [1.0]])
        else:                                   # as PersonaPath.__init__
            dd = np.linalg.norm(np.diff(Yf, axis=0), axis=1)
            if param == "centripetal":
                dd = np.sqrt(dd)
            x = np.concatenate([[0.0], np.cumsum(dd)])
            x = x / x[-1] if x[-1] > 0 else np.linspace(0, 1, len(Yf))
        self.spline = FixedEndSpline(x, Yf, self.lam)
        self.lam = self.spline.lam_used
        self.arc_length = self.spline.arc_length
        self._fallback = None


def mean_segment_distance(C, idx, P0, P1):
    """Mean distance of the chosen knots to the segment -- exp3's proximity column."""
    if len(idx) == 0:
        return float("nan")
    _, d = segment_distance(np.asarray(C)[np.asarray(idx, dtype=int)], P0, P1)
    return float(d.mean())


# ---------------------------------------------------------------------------
# One constructor for every knot rule
# ---------------------------------------------------------------------------

KNOT_RULES = ("distance", "density", "nearest")


def build_manifold(rule, P0, P1, C, k, eps=None, exclude_idx=None):
    """The manifold path under one knot rule. The ONE place exps 3, 4, 5, the
    judge's option list and the figures build their routes, so a knot set named
    in a table is the knot set that was steered through.

    distance / density: `PersonaPath` with eps = EPS_ALL (every centroid a
    candidate), centripetal, lam 0 -- the construction the demo routes used
    (jobs_condor/steer_demo.py, EPS = 1e9). nearest: `NearestPath`.
    """
    from steering.followups.common import EPS_ALL
    if rule == "nearest":
        return NearestPath(P0, P1, C, k=k, lam=0.0, param="centripetal",
                           exclude_idx=exclude_idx)
    if rule in ("distance", "density"):
        return PersonaPath(P0, P1, C, EPS_ALL if eps is None else eps, k=k, lam=0.0,
                           mode="absolute", param="centripetal", chunk=rule)
    raise ValueError("rule must be one of %s, got %r" % (KNOT_RULES, rule))
