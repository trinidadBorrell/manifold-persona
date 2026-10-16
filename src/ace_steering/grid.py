"""The cells of the ACE study and the intervention each one applies.

A cell is (step, route, layer, arm, alpha). Per route and layer:

    none      alpha 0              unsteered; stands for linear AND manifold at 0
    linear    12 signed alphas     delta = alpha (mu_B - mu_A), constant
    ace       13 signed alphas     delta(h) = -((h - mu_A).r_hat) r_hat + alpha r   (eq. 5)
    manifold  .25 .5 .75 1         delta = S(alpha) - S(0), manifold_v1 (k = 8)
    ace_manifold  13 signed alphas delta(h) = U U^T (T(alpha) - h), U = the route's own directions
                                   (knots - mu_A and mu_B - mu_A, orthonormalised, <= 9), T(alpha) =
                                   S(alpha) on [0, 1] and the chord continued past either end

= 30 cells (43 with ace_manifold, added 2026-09-30). ACE at alpha 0 is its own cell: it pins the r-coordinate, it is
not a no-op.

Each cell generates N_PROMPTS x len(QUESTIONS) x N_SAMPLES[step] answers
(75 in step 1, 25 in step 2). Step 2 skips the layer-19 cells of the step-1
routes: step 1's sample 0 is the identical draw (same seed, same batch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ace_steering.common import (K, KNOT_RULE, MANIFOLD_ALPHAS, N_PROMPTS, N_SAMPLES, OUT_ROOT,
                                       QUESTIONS, SIGNED_ALPHAS, STEP1_LAYER, STEP1_ROUTES,
                                       STEP5_LAYERS, STEP5_PERCENTILES, geom_path, layers_for,
                                       route_slug, routes_for)

ARMS = ("none", "linear", "ace", "manifold", "ace_manifold")


@dataclass(frozen=True)
class Cell:
    step: int
    route: str
    layer: int
    arm: str
    alpha: float

    @property
    def cell_id(self):
        return "s%d_L%d_%s_%s_a%s" % (self.step, self.layer, route_slug(self.route), self.arm,
                                      ("%+g" % self.alpha).replace(".", "p").replace("+", "p").replace("-", "m"))

    @property
    def n_answers(self):
        return N_PROMPTS * len(QUESTIONS) * N_SAMPLES[self.step]


def route_cells(step, route, layer, arms=None):
    """arms: which arms to include (default: the original four; ace_manifold only when named,
    so jobs already sharded on the 30-cell list keep the same enumeration)."""
    arms = ("none", "linear", "ace", "manifold") if arms is None else tuple(arms)
    out = [Cell(step, route, layer, "none", 0.0)]
    out += [Cell(step, route, layer, "linear", a) for a in SIGNED_ALPHAS if a != 0]
    out += [Cell(step, route, layer, "ace", a) for a in SIGNED_ALPHAS]
    out += [Cell(step, route, layer, "manifold", a) for a in MANIFOLD_ALPHAS if a != 0]
    out += [Cell(step, route, layer, "ace_manifold", a) for a in SIGNED_ALPHAS]
    return [c for c in out if c.arm in arms]


def knots_ok(route, layer):
    return int(route_path(route, layer).n_centroids) == K


def floor_cells(routes=None):
    """Step 5 (follow-up D): activation capping as a floor at STEP5_LAYERS at once.
    alpha holds the percentile p of tau_p; layer is the first floor layer (naming only)."""
    routes = routes_for(5) if routes is None else routes
    out = []
    for route in routes:
        out.append(Cell(5, route, STEP5_LAYERS[0], "none", 0.0))
        for arm in ("floor_linear", "floor_manifold"):
            out += [Cell(5, route, STEP5_LAYERS[0], arm, float(p)) for p in STEP5_PERCENTILES]
    return out


def floor_tau(route, layer, p):
    """tau_p: p-th percentile of <h, r_hat> over the TARGET persona's own role-play rows at this layer."""
    G = geom(layer)
    _, B, cA, cB = G.route(route)
    r_hat = (cB - cA) / np.linalg.norm(cB - cA)
    z = np.load(OUT_ROOT / "clouds" / ("%s_L%d.npz" % (route_slug(route), layer)), allow_pickle=True)
    rows = z["X"][z["role"].astype(str) == B].astype(np.float64)
    if len(rows) < 10:
        raise ValueError("only %d target rows for %s at L%d" % (len(rows), route, layer))
    return float(np.percentile(rows @ r_hat, p)), len(rows)


def build_cells(step, routes=None, layers=None, skip_short=False, arms=None):
    """skip_short: leave out (route, layer) pairs whose manifold_v1 route has fewer than K
    knots (an empty distance chunk), so the rest can run while that case is decided."""
    routes = routes_for(step) if routes is None else routes
    layers = layers_for(step) if layers is None else layers
    if arms is None and int(step) == 3:                  # follow-up A runs all four arms
        arms = ("none", "linear", "ace", "manifold", "ace_manifold")
    cells = []
    for layer in layers:
        for route in routes:
            if int(step) == 2 and layer == STEP1_LAYER and route in STEP1_ROUTES:
                continue                                  # reused from step 1, sample 0
            if skip_short and not knots_ok(route, layer):
                continue
            cells += route_cells(step, route, layer, arms)
    return cells


def smoke_cells(route=STEP1_ROUTES[1], layer=STEP1_LAYER):
    """Every arm at alpha in {0, 1} -- the plan's smoke check."""
    return [Cell(0, route, layer, "none", 0.0), Cell(0, route, layer, "linear", 1.0),
            Cell(0, route, layer, "ace", 0.0), Cell(0, route, layer, "ace", 1.0),
            Cell(0, route, layer, "manifold", 1.0)]


# ---------------------------------------------------------------------------
# Geometry and interventions
# ---------------------------------------------------------------------------

_GEOMS, _PATHS = {}, {}


def geom(layer):
    from steering_followups.common import Geom
    if layer not in _GEOMS:
        _GEOMS[layer] = Geom(geom_path(layer))
    return _GEOMS[layer]


def route_path(route, layer):
    """The manifold_v1 route: PersonaPath, distance chunking, k = 8, eps = everything."""
    from steering_followups.selection import build_manifold
    key = (route, layer)
    if key not in _PATHS:
        G = geom(layer)
        _, _, cA, cB = G.route(route)
        _PATHS[key] = build_manifold(KNOT_RULE, cA, cB, G.C, K)
    return _PATHS[key]


def route_personas(route, layer):
    """The judge's option list for this route at this layer: A, the knots in route order, B."""
    G = geom(layer)
    A, B = route.split(">")
    return [A] + [G.names[int(i)] for i in route_path(route, layer).centroid_idx] + [B]


def route_basis(route, layer):
    """Orthonormal basis of the route's own directions: mu_B - mu_A first, then the knots."""
    G = geom(layer)
    _, _, cA, cB = G.route(route)
    p = route_path(route, layer)
    V = np.vstack([cB - cA] + [G.C[int(i)] - cA for i in p.centroid_idx]).T      # (d, k+1)
    Q, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])
    return Q[:, keep]


def route_target(route, layer, alpha):
    """T(alpha): the spline point on [0, 1]; past either end, the chord continued."""
    G = geom(layer)
    _, _, cA, cB = G.route(route)
    a = float(alpha)
    if a < 0:
        return cA + a * (cB - cA)
    if a > 1:
        return cB + (a - 1) * (cB - cA)
    return np.asarray(route_path(route, layer).at_alpha(a)[0], dtype=np.float64)


def resolve(cell):
    """dict(kind, ...): kind 'none' | 'constant' (vec) | 'ace' (mu_a, mu_b, alpha)."""
    from steering_core.manifold_paths import LinearPath
    G = geom(cell.layer)
    _, _, cA, cB = G.route(cell.route)
    if cell.arm == "none":
        return dict(kind="none", delta_norm=0.0)
    if cell.arm == "linear":
        vec = LinearPath(cA, cB).delta(cell.alpha)[0]
    elif cell.arm == "manifold":
        vec = route_path(cell.route, cell.layer).delta(cell.alpha)[0]
    elif cell.arm == "ace":
        return dict(kind="ace", mu_a=cA, mu_b=cB, alpha=float(cell.alpha),
                    delta_norm=float(abs(cell.alpha) * np.linalg.norm(cB - cA)))
    elif cell.arm in ("floor_linear", "floor_manifold"):
        per_layer = []
        for L in STEP5_LAYERS:
            GL = geom(L)
            _, _, a_, b_ = GL.route(cell.route)
            tau, n = floor_tau(cell.route, L, cell.alpha)
            S = None
            if cell.arm == "floor_manifold":
                S = np.asarray(route_path(cell.route, L).at_alpha(np.linspace(0, 1, 401)), dtype=np.float64)
            per_layer.append(dict(layer=L, mu_a=a_, mu_b=b_, tau=tau, n_target_rows=n, S_grid=S))
        return dict(kind="floor", layers=per_layer, delta_norm=float("nan"))
    elif cell.arm == "ace_manifold":
        U = route_basis(cell.route, cell.layer)
        T = route_target(cell.route, cell.layer, cell.alpha)
        return dict(kind="pin", U=U, target=T, k=int(U.shape[1]),
                    delta_norm=float(np.linalg.norm(T - cA)))
    else:
        raise ValueError(cell.arm)
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if not np.isfinite(vec).all():
        raise ValueError("non-finite delta for %s" % cell.cell_id)
    return dict(kind="constant", vec=vec, delta_norm=float(np.linalg.norm(vec)))


def controls(cells, tol=1e-6):
    """Geometry checks before any GPU time: failures as a list (empty = pass).

    alpha = 1: linear and manifold deltas both equal mu_B - mu_A exactly.
    Every manifold route has k interior knots (distance chunking gives fewer
    when a chunk is empty, which would change the judge's option count).
    """
    bad = []
    for c in cells:
        G = geom(c.layer)
        _, _, cA, cB = G.route(c.route)
        if c.arm in ("linear", "manifold") and c.alpha == 1.0:
            err = float(np.abs(resolve(c)["vec"] - (cB - cA)).max())
            if err > tol:
                bad.append((c.cell_id, "alpha=1 delta != mu_B - mu_A", err))
        if c.arm == "manifold":
            # Fewer than K is allowed (an empty distance chunk; user 2026-09-30: "no problem
            # in mismatch on route points") and recorded in routes.json; none at all is not.
            n = int(route_path(c.route, c.layer).n_centroids)
            if n == 0:
                bad.append((c.cell_id, "route has no knots", n))
    return bad
