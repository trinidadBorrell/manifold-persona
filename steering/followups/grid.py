"""The exp4/exp5 grid: cells, seeds, stages, and the intervention each cell applies.

Plan: plans/2026-09-24-steering-followups.md (Design, exps 4-5; Seeds).

TRACKED ON PURPOSE. The GPU driver (jobs_condor/followups_steer.py) is cluster
infra and stays untracked, but the grid IS the design: which arms, which
alphas, which seed each generation gets, and what vector or point each cell
writes. Keeping it here means the judge, the figures and a reader of the repo
see the same enumeration the GPU saw -- and `--dry-run` can verify the plan's
cell counts on a login node with no torch model loaded.

THE GRID (plan, Design):

  Study A, fixed start P0 = c_A, 4 routes x 7 arms
    additive  linear          delta = alpha (c_B - c_A)
              manifold_v1     S(alpha) - S(0), knots by "distance"  (k = 8)
              manifold_v2     ...                  by "density"
              nearest         ...                  by "nearest"     (new rule, exp3)
              target_vector   delta = alpha (c_B - h0_mean)          (positive control)
    replace   replace_linear  last position := (1-t) c_A + t c_B
              replace_manifold last position := nearest-k spline point at arc t
    alpha = t in {0, .25, .5, .75, 1} for the 5 path arms; linear and
    target_vector also at {1.5, 2, 3, 4}  ->  4 x (5*5 + 2*9) = 172 cells

  Study B, measured start P0 = h0(q), 4 routes x 2 arms x 7 alphas = 56 cells
    additive  linear_B        delta = alpha (c_B - h0(q))
              nearest_B       nearest-k spline h0(q) -> c_B; past alpha = 1 it
                              is the linear continuation (see `_nearest_b`)

  Every cell: 5 identity questions + canary, x 3 samples = 18 generations.

SEEDS. `seed = 20260924 + unit_index*100 + sample_index`, unit = (route, arm,
alpha, question) enumerated in that nesting order -- the plan's "cell_index
... route, then arm, then alpha, then question" read literally, which makes
the seed unit a (cell, question) pair, not a cell. Study B's units continue the
count after Study A's, so no two generations in the run share a seed.

STAGES. Two quantities are measured by the run itself and needed by later
cells, so the grid is generated in dependency order:

  A     every Study A cell that needs no h0: 140 cells
  h0    CPU: from stage A's alpha = 0 ADDITIVE cells (an unsteered model --
        the delta is exactly zero there), write h0_mean per route (identity
        questions) and h0(q) per route and question
  Atv   Study A target_vector at alpha > 0 (needs h0_mean): 32 cells
  B     Study B (needs h0(q)): 56 cells

Atv and B are independent and can run in one submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re
import numpy as np

from steering.followups.common import K_DEFAULT, N_SAMPLES, QUESTIONS, ROUTES, seed_for
from steering.followups.replace import linear_target, manifold_target
from steering.followups.selection import build_manifold
from steering.manifold_paths import LinearPath

PATH_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXT_ALPHAS = PATH_ALPHAS + (1.5, 2.0, 3.0, 4.0)
B_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

# (arm, form, knot rule or None). Order is the seed order.
STUDY_A_ARMS = (
    ("linear", "additive", None),
    ("manifold_v1", "additive", "distance"),
    ("manifold_v2", "additive", "density"),
    ("nearest", "additive", "nearest"),
    ("target_vector", "additive", None),
    ("replace_linear", "replace", None),
    ("replace_manifold", "replace", "nearest"),
)
EXTENDED_ARMS = ("linear", "target_vector")
STUDY_B_ARMS = (
    ("linear_B", "additive", None),
    ("nearest_B", "additive", "nearest"),
)
STAGES = ("A", "Atv", "B", "F")


@dataclass
class Cell:
    study: str
    route: str
    arm: str
    form: str
    rule: object
    alpha: float
    cell_index: int                          # position in the full grid, 0-based
    stage: str
    units: list = field(default_factory=list)   # [(unit_index, q_idx, kind, question)]

    @property
    def cell_id(self):
        return "%s_%03d_%s_%s_a%s" % (self.study, self.cell_index, self.route.replace(">", "-"),
                                      self.arm, ("%g" % self.alpha).replace(".", "p"))

    def seeds(self, n_samples=N_SAMPLES):
        return [(u, s, seed_for(u, s)) for (u, *_rest) in self.units for s in range(n_samples)]


def _stage(study, arm, alpha):
    if study == "B":
        return "B"
    return "Atv" if (arm == "target_vector" and alpha > 0) else "A"


def build_cells(routes=ROUTES, questions=QUESTIONS):
    """The full ordered grid. Unit indices are assigned HERE and nowhere else."""
    cells, unit = [], 0
    for study, arms in (("A", STUDY_A_ARMS), ("B", STUDY_B_ARMS)):
        for route in routes:
            for arm, form, rule in arms:
                if study == "A":
                    alphas = EXT_ALPHAS if arm in EXTENDED_ARMS else PATH_ALPHAS
                else:
                    alphas = B_ALPHAS
                for al in alphas:
                    c = Cell(study, route, arm, form, rule, float(al), len(cells),
                             _stage(study, arm, al))
                    for qi, (kind, q) in enumerate(questions):
                        c.units.append((unit, qi, kind, q))
                        unit += 1
                    cells.append(c)
    return cells


# ---------------------------------------------------------------------------
# Follow-up F (EXPLORATORY, added 2026-09-24 after seeing Study A; plan `mode: explore`,
# one follow-up chunk). Question: the paper-faithful replace arm reaches bard at t=1 but
# never vampire. Is 64 dims too few (rank) or is the dose too small (t > 1, extrapolated
# past c_B along the chord, still written only inside the subspace)?
# ---------------------------------------------------------------------------
FOLLOWUP_ARMS = (("replace_linear_r64", (1.25, 1.5, 2.0)),
                 ("replace_linear_r16", (1.0,)),
                 ("replace_linear_r256", (1.0,)))
FOLLOWUP_CELL0, FOLLOWUP_UNIT0 = 1000, 100000     # disjoint from the main grid's seeds


def followup_cells(routes=ROUTES, questions=QUESTIONS):
    cells, unit = [], FOLLOWUP_UNIT0
    for route in routes:
        for arm, ts in FOLLOWUP_ARMS:
            for t in ts:
                c = Cell("F", route, arm, "replace", None, float(t),
                         FOLLOWUP_CELL0 + len(cells), "F")
                for qi, (kind, q) in enumerate(questions):
                    c.units.append((unit, qi, kind, q))
                    unit += 1
                cells.append(c)
    return cells


def replace_rank(arm):
    m = re.search(r"_r(\d+)$", arm)
    return int(m.group(1)) if m else 64


def smoke_cells():
    """--smoke: 1 route, 2 arms (one additive, one replace), 2 alphas, 1 question.

    Its own enumeration (unit indices from 0), labelled study 'smoke', so a smoke
    row can never be mistaken for a grid row.
    """
    cells, unit = [], 0
    for arm, form, rule in (STUDY_A_ARMS[0], STUDY_A_ARMS[5]):
        for al in (0.0, 1.0):
            c = Cell("smoke", ROUTES[0], arm, form, rule, al, len(cells), "smoke")
            c.units.append((unit, 0, QUESTIONS[0][0], QUESTIONS[0][1]))
            unit += 1
            cells.append(c)
    return cells


# ---------------------------------------------------------------------------
# What each cell writes
# ---------------------------------------------------------------------------

class Interventions:
    """Resolve (cell, question) -> the vector to add or the point to write.

    Paths are built once per (route, rule, start) and cached. h0 inputs:
      h0_mean  {route: (d,)}           for target_vector at alpha > 0
      h0q      {route: (n_q, d)}       for Study B, row = question index
    Asking for one that was not supplied raises -- a cell must never fall back
    to a centroid silently.
    """

    def __init__(self, G, k=K_DEFAULT, h0_mean=None, h0q=None):
        self.G, self.k = G, int(k)
        self.h0_mean = h0_mean or {}
        self.h0q = h0q or {}
        self._paths = {}

    def path(self, route, rule, P0, P1, start_key):
        key = (route, rule, start_key)
        if key not in self._paths:
            # Never route through the source or target persona themselves. With P0 = c_A
            # (Study A) endpoint_exclusions already drops them; with P0 = h0(q) (Study B)
            # c_A is NOT an endpoint and could be picked as a waypoint (review 2026-09-24 #10).
            A, B = route.split(">")
            ex = [self.G.ix[A], self.G.ix[B]] if rule == "nearest" else None
            self._paths[key] = build_manifold(rule, P0, P1, self.G.C, self.k, exclude_idx=ex)
        return self._paths[key]

    def _need(self, table, route, what):
        if route not in table:
            raise KeyError("%s for %s is required by this cell but was not supplied "
                           "(run stage h0 first)" % (what, route))
        return table[route]

    def resolve(self, cell: Cell, q_idx: int) -> dict:
        """dict(form, vec, path_meta). vec is the delta (additive) or target (replace)."""
        A, B, cA, cB = self.G.route(cell.route)
        al, meta = cell.alpha, {"beyond_path": False}
        if cell.study in ("A", "smoke"):
            if cell.arm == "linear":
                vec = LinearPath(cA, cB).delta(al)[0]
            elif cell.arm in ("manifold_v1", "manifold_v2", "nearest"):
                p = self.path(cell.route, cell.rule, cA, cB, "cA")
                vec = p.delta(al)[0]
                meta.update(self._meta(p))
            elif cell.arm == "target_vector":
                if al == 0:
                    vec = np.zeros_like(cA)
                else:
                    vec = LinearPath(self._need(self.h0_mean, cell.route, "h0_mean"), cB).delta(al)[0]
            elif cell.arm == "replace_linear":
                vec = linear_target(cA, cB, al)
            elif cell.arm == "replace_manifold":
                p = self.path(cell.route, cell.rule, cA, cB, "cA")
                vec = manifold_target(p, al)
                meta.update(self._meta(p))
            else:
                raise ValueError(cell.arm)
        elif cell.study == "F":
            # pi(t) = c_A + t (c_B - c_A), t may exceed 1 (extrapolation along the chord)
            vec = cA + al * (cB - cA) if al != 1.0 else cB.copy()
        else:
            h0 = np.asarray(self._need(self.h0q, cell.route, "h0(q)")[q_idx], dtype=np.float64)
            if cell.arm == "linear_B" or (cell.arm == "nearest_B" and al > 1.0):
                # nearest_B past 1: S(1) - S(0) = c_B - h0, continued straight
                # along the chord -- which IS linear_B. The plan's "past alpha=1
                # every additive path arm equals linear", applied to Study B.
                vec = LinearPath(h0, cB).delta(al)[0]
                meta["beyond_path"] = cell.arm == "nearest_B"
            elif cell.arm == "nearest_B":
                p = self.path(cell.route, "nearest", h0, cB, ("h0q", q_idx))
                vec = p.delta(al)[0]
                meta.update(self._meta(p))
            else:
                raise ValueError(cell.arm)
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if not np.isfinite(vec).all():
            raise ValueError("non-finite intervention for %s q%d" % (cell.cell_id, q_idx))
        return dict(form=cell.form, vec=vec, **meta)

    def _meta(self, p):
        return dict(n_knots=int(p.n_centroids), detour=float(p.detour_ratio),
                    knots=" ".join(self.G.names[int(i)] for i in p.centroid_idx))


def geometry_controls(cells, iv: Interventions, tol=1e-4):
    """Plan, Controls: 'Geometry'. Returns a list of failures (empty = pass).

    alpha = 1: every additive arm's delta equals its own P1 - P0.
    replace:   every target equals pi(t), re-derived independently; at t = 1
               replace_linear writes c_B EXACTLY.
    """
    bad = []
    for c in cells:
        for (_, qi, _, _) in c.units[:1] if c.study != "B" else c.units:
            try:
                r = iv.resolve(c, qi)
            except KeyError:
                continue                            # h0 not yet available: checked at its stage
            A, B, cA, cB = iv.G.route(c.route)
            if c.form == "additive" and c.alpha == 1.0:
                if c.arm == "target_vector":
                    want = cB - iv.h0_mean[c.route]
                elif c.study == "B":
                    want = cB - iv.h0q[c.route][qi]
                else:
                    want = cB - cA
                err = float(np.max(np.abs(r["vec"] - want)))
                if err >= tol:
                    bad.append((c.cell_id, qi, "alpha=1 delta != P1-P0", err))
            if c.form == "replace":
                if c.arm == "replace_linear" or c.arm.startswith("replace_linear_r"):
                    want = cA + c.alpha * (cB - cA)
                else:
                    want = iv.path(c.route, "nearest", cA, cB, "cA").at_alpha(c.alpha)[0]
                err = float(np.max(np.abs(r["vec"] - want)))
                if err >= tol:
                    bad.append((c.cell_id, qi, "replace target != pi(t)", err))
                if c.arm == "replace_linear" and c.alpha == 1.0 and not np.array_equal(r["vec"], cB):
                    bad.append((c.cell_id, qi, "replace_linear t=1 is not c_B exactly", 0.0))
    return bad
