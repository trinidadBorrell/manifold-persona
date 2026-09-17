"""exp01: is equal-count (density) chunking of the eps-cylinder a no-op?

Plan: plans/2026-09-17-chunk-selection-ablation.md. Context: steering/README.md.

WHY THIS EXISTS. Run `2026-09-10T15-46-fig4` froze `eps=12, k=8, lam=0,
param="centripetal", mode="absolute"` (`steering/path_cases.py:37-44`) and its
persona-pair arms reached the target persona in <=1.5% of responses. RESULTS.md
S4 names three possible causes; this run addresses exactly one of them -- that
the KNOTS are the wrong points. The production rule cuts the tube into `k`
equal-WIDTH spans of chord projection and takes the candidate nearest the chord
in each. Where candidates bunch along `u`, that leaves spans empty and
over-samples dense stretches. Equal-COUNT bins are the obvious alternative and
have never been tried.

The question is deliberately NOT "which mode is better". The plan's Questions
item 4 records that the deciding metric for path *quality* is unset, so this run
is a characterization: it measures how far apart the two modes' knot sets are
and stops. Every path-quality metric is computed and stored so naming a decider
later costs a read of `data/grid_eps_k.csv` rather than a rerun -- but a metric
chosen after seeing that file is exploratory, permanently.

DECIDING METRIC: knot-set Jaccard between modes at matched (route, eps, k),
median over the eps x k grid, route-clustered bootstrap CI (2,000 resamples,
seed 137). Falsifier: median >= 0.95 means the refinement is dead.

THE UNIT OF OBSERVATION IS THE ROUTE. There are 16 (1 axis + 3 near-Assistant
sources x [3 rank-midway + 2 far targets]). 331,200 activation records is not n,
and the 16 routes are not independent -- the 15 pair routes reuse 3 source and 5
target centroids -- which is why every CI here resamples ROUTES, not rows.

WHAT THIS SCRIPT DOES NOT DO. It writes no figures. `steering/chunk_figures.py`
consumes the CSVs written here. A figure whose numbers exist only inside a PNG
is not a result (steering/README.md, Conventions), so the numbers are written
first and drawn second, by a different process.

Outputs, all under `--out` and nowhere else:

    data/controls.json      the four gates, the Jaccard summary, provenance
    data/grid_eps_k.csv     route x eps x k x chunk, every path metric
    data/jaccard_summary.csv  route x eps x k, the deciding metric per cell
    data/grid_m.csv         route x eps x m, density mode only
    data/fig01_candidates.csv  every candidate in every chunk, one route
    data/jitter_stability.csv  centroid-jitter bootstrap (only with --cloud)
    logs/exp01.log          everything printed
    manifest.json           sha, dirty flag, grids, seeds, versions, status
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from steering.manifold_paths import (
    END_MARGIN, LinearPath, PersonaPath, chord_coords, chord_frame,
    chunk_diagnostics, pick_centroids, select_cylinder,
)
from steering.path_cases import EPS as PROD_EPS, K as PROD_K, LAM, MODE, PARAM
from steering.runmeta import build_manifest, file_sha256, write_manifest

# --------------------------------------------------------------------------
# The grids. Plan, ### Design.
# --------------------------------------------------------------------------
# eps is ABSOLUTE (activation units), so the same number is a different
# RELATIVE neighbourhood on a 38-unit far-pair chord than on a 5.8-unit midway
# chord. That is confound 3 in the plan and the reason nothing here is averaged
# over routes without also being reported per route.
EPS_GRID = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 28.0]
K_GRID = [1, 2, 4, 6, 8, 12, 16]
M_GRID = [1, 2, 3, 5, 8, 12]
M_EPS_GRID = [8.0, 12.0, 20.0]          # the m ablation's eps slice
CHUNKS = ("distance", "density")

# Held fixed across the whole run: lam=0 (interpolate every knot exactly),
# centripetal abscissa, absolute eps. Imported from `path_cases` rather than
# retyped -- a second copy of these numbers is the bug that module exists to
# prevent.
FIXED = {"lam": LAM, "param": PARAM, "mode": MODE}

# Seeds. The grid itself is deterministic; seeds enter in exactly two places
# (plan, ### Seeds).
BOOT_SEED = 137
N_BOOT = 2000
N_JITTER = 20                           # centroid-jitter resamples, seed 137+b

# Control thresholds, verbatim from the plan's ### Baselines and controls.
KNOT_ERROR_MAX = 1e-8
CHORD_COLLAPSE_TOL = 1e-9
N_ALPHA_NEGATIVE = 200

# The cloud is published pre-thinned to a single layer: index 0 IS hidden state
# 19 (`manifest.json: source_layers [19], primary_layer 0`). Not a flag, because
# any other value would silently analyse a layer the cloud does not contain.
LAYER_INDEX = 0

PLAN = "plans/2026-09-17-chunk-selection-ablation.md"

# Plan, exp01 brief: "Blocked if: ... n_fully < 10". Same number `load_cases`
# uses, restated here because the cache path never reaches `load_cases`.
MIN_ROLES = 10

# Machine-local cache of the 267s (2.7 GB over NFS) geometry load. Expected to
# be ABSENT on any other box, in which case --geom-cache or --cloud is required;
# it is a speed cache, never a source of truth, and its sha256 goes in the
# manifest so a run can be tied to the bytes it read.
DEFAULT_GEOM_CACHE = (
    "/tmp/claude-42236/-home-triniborrell-home-projects-manifold-persona/"
    "e69bf2ee-90aa-4723-9372-bea4d010a0c8/scratchpad/geom_cache.npz")
GEOM_CACHE_ENV = "CHUNK_ABLATION_GEOM_CACHE"


# --------------------------------------------------------------------------
# House rules: refuse bad input, never coerce
# --------------------------------------------------------------------------

def _require_finite(a, what):
    """`geometry.py::_require_finite`, same contract.

    A NaN centroid does not raise anywhere downstream -- `NaN < radius` is
    False -- so the cylinder would quietly shrink and every Jaccard on this run
    would be computed over a candidate set nobody chose.
    """
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        raise ValueError("%s contains %d non-finite values" % (what, n))
    return a


class Log:
    """Print and append, in that order, so a crash keeps what got as far as stdout.

    steering/README.md asks for `logs/exp01.log`; this is the only writer.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.fh.write(str(msg) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


class OutDir:
    """Every write goes through here, so `--out` is a boundary and not a habit.

    The plan's Data section calls `data/` read-only and the run dir the only
    writable place. A path that escapes is refused rather than repaired: a
    run that quietly wrote next door is worse than one that died.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        # `.run-active` marks the live run for the output guard; touching it is
        # idempotent when `runmeta.new_run_dir` already made this directory.
        (self.root / ".run-active").touch()

    def path(self, rel):
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("refusing to write outside --out: %s" % p)
        return p

    def write_csv(self, rel, rows, columns):
        """One CSV writer for the whole run, with the column order pinned.

        The key check is not decoration: `pd.DataFrame(rows, columns=...)`
        fills a missing key with NaN, which in a metrics table reads exactly
        like a measured non-finite value. Refuse instead.
        """
        for i, r in enumerate(rows):
            missing = [c for c in columns if c not in r]
            if missing:
                raise ValueError("row %d of %s is missing %s"
                                 % (i, rel, missing))
        p = self.path(rel)
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(p, index=False)
        return p, len(df)

    def write_json(self, rel, obj):
        p = self.path(rel)
        p.write_text(json.dumps(obj, indent=2, default=_jsonable))
        return p


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------
# Geometry: the 275 fully-role-playing centroids and the 16 routes
# --------------------------------------------------------------------------

class Geom:
    """C, the role names, and the 16 (kind, P0, P1, A, B) routes.

    Built either from the slow real load (`--cloud`) or from a cached npz that
    holds exactly what the slow load produced. Both paths end here so nothing
    downstream can tell them apart, and `source` records which one ran.
    """

    def __init__(self, C, names, axis_proj, axis_unit, cases, source, extra=None):
        self.C = _require_finite(C, "role centroids")
        self.names = [str(n) for n in names]
        self.axis_proj = np.asarray(axis_proj, dtype=np.float64)
        self.axis_unit = np.asarray(axis_unit, dtype=np.float64)
        self.cases = cases
        self.source = source
        self.extra = extra or {}
        # Chord projection of all 275 centroids, per route. It depends only on
        # (P0, P1), so recomputing it for each of the 1,568 grid cells is a
        # 275x4096 reduction repeated ~100 times per route for no new
        # information. Cached, not approximated: the values are identical.
        self._u_cache = {}
        if self.C.shape[0] != len(self.names):
            raise ValueError("C has %d rows but %d names"
                             % (self.C.shape[0], len(self.names)))
        # The plan's "Blocked if": n_fully < 10. `load_cases` enforces this on
        # the slow path; a cache built before the labels existed would sail
        # past it, so the gate lives here where BOTH paths meet.
        if self.C.shape[0] < MIN_ROLES:
            raise SystemExit(
                "only %d fully-role-playing centroids (need >= %d). The plan "
                "blocks on this: labels missing, or the cache was built "
                "without them." % (self.C.shape[0], MIN_ROLES))
        if len(self.cases) == 0:
            raise ValueError("no routes")
        for kind, P0, P1, A, B in self.cases:
            _require_finite(P0, "P0 of %s>%s" % (A, B))
            _require_finite(P1, "P1 of %s>%s" % (A, B))

    @property
    def routes(self):
        """(route_name, kind, P0, P1, A, B) per case. The name is `A>B`."""
        return [("%s>%s" % (A, B), kind, P0, P1, A, B)
                for kind, P0, P1, A, B in self.cases]

    def chord_u(self, route, P0, P1):
        """`chord_coords(C, P0, P1)[0]`, memoised per route."""
        if route not in self._u_cache:
            self._u_cache[route] = chord_coords(self.C, P0, P1)[0]
        return self._u_cache[route]


def load_geom_cache(path):
    """The cached geometry. Keys are fixed by the script that wrote it."""
    p = Path(path)
    if not p.exists():
        raise SystemExit("--geom-cache %s does not exist" % p)
    z = np.load(p, allow_pickle=False)
    need = ("C", "names", "axis_proj", "axis_unit", "kinds", "P0", "P1", "A", "B")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise SystemExit(
            "%s is not a chunk-ablation geometry cache: missing %s. Rebuild it "
            "or pass --cloud/--labels for the real load." % (p, missing))
    kinds, P0, P1, A, B = z["kinds"], z["P0"], z["P1"], z["A"], z["B"]
    cases = [(str(kinds[i]), np.asarray(P0[i], dtype=np.float64),
              np.asarray(P1[i], dtype=np.float64), str(A[i]), str(B[i]))
             for i in range(len(kinds))]
    return Geom(z["C"], z["names"], z["axis_proj"], z["axis_unit"], cases,
                source={"kind": "geom_cache", "path": str(p),
                        "sha256": file_sha256(p)},
                extra={"keep": z["keep"].tolist() if "keep" in z.files else None})


def load_geom_cloud(cloud, labels):
    """The real load: 2.7 GB over NFS, ~267 s, then the plan's exclusion rule.

    `path_cases.load_cases` applies `fully_only` -- the EXISTING filter, decided
    long before this plan -- and builds the same 16 routes `run_steering` steers
    along. Reimplementing either here would let this run analyse knots that no
    generation run ever used.
    """
    from steering.geometry import load_geometry
    from steering.path_cases import load_cases

    geom = load_geometry(resp_dir=cloud, layer=LAYER_INDEX, labels_path=labels)
    cs = load_cases(geom)
    cases = [(k, np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64),
              str(a), str(b)) for k, p0, p1, a, b in cs.cases]
    return Geom(cs.C, cs.names, cs.axis_proj, geom.axis_unit, cases,
                source={"kind": "cloud", "cloud": str(cloud),
                        "labels": str(labels), "layer_index": LAYER_INDEX},
                extra={"keep": cs.keep.tolist(), "dropped": list(cs.dropped),
                       "picked": cs.picked, "n_roles_total": len(geom.roles),
                       "geom": geom, "n_fully": int(len(cs.keep))})


def resolve_geometry(args, say):
    """--geom-cache, then $CHUNK_ABLATION_GEOM_CACHE, then the default, then --cloud.

    The cache is preferred when it exists because the alternative is 267 s of
    NFS per invocation and the cache is byte-identical geometry; --cloud always
    wins when explicitly combined with no cache.
    """
    cache = args.geom_cache
    if cache is None and not args.cloud:
        cache = os.environ.get(GEOM_CACHE_ENV) or DEFAULT_GEOM_CACHE
        if not Path(cache).exists():
            cache = None
    if cache is not None:
        say("geometry: cache %s" % cache)
        return load_geom_cache(cache)
    if not args.cloud:
        raise SystemExit(
            "no geometry: pass --geom-cache <npz> or --cloud <resp_dir> "
            "(with --labels). The default cache %s does not exist on this box."
            % DEFAULT_GEOM_CACHE)
    say("geometry: real load from %s (2.7 GB over NFS, ~267 s)" % args.cloud)
    if not args.labels:
        raise SystemExit(
            "--cloud without --labels: load_geometry would fall back to "
            "unfiltered centroids and `fully_only` refuses to proceed. The "
            "exclusion rule is part of the plan, not an option.")
    return load_geom_cloud(args.cloud, args.labels)


# --------------------------------------------------------------------------
# The oracle: the pre-change equal-width selection, re-implemented inline
# --------------------------------------------------------------------------

def oracle_pick_centroids(C, P0, P1, eps, k=8, mode="absolute"):
    """`pick_centroids` EXACTLY as it stood at HEAD, before `chunk=` existed.

    Copied verbatim from `git show HEAD:steering/manifold_paths.py`, not
    paraphrased, and deliberately NOT importing the new `_chunk_candidates`.
    An oracle that shares code with the thing it checks certifies that the code
    is self-consistent, which is not the claim: the claim is that `chunk=
    "distance"` at eps=12, k=8 selects the knots run 2026-09-10T15-46-fig4
    actually steered through.

    `select_cylinder` IS shared, and that is correct -- the candidate set is not
    what changed, the partition of it is.
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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def min_knot_spacing_u(u_all, idx):
    """Smallest gap between consecutive knot `u` values. NaN below 2 knots.

    `u` is chord projection, the coordinate SELECTION uses -- not the spline's
    centripetal abscissa. Two knots close in `u` are the configuration that
    makes an interpolating cubic swing (manifold_paths.FixedEndSpline), so this
    is the supply-side warning for the overshoot the other metrics measure.
    """
    if len(idx) < 2:
        return float("nan")
    us = np.sort(np.asarray(u_all, dtype=np.float64)[np.asarray(idx, dtype=int)])
    return float(np.diff(us).min())


def path_row(geom, route, kind, A, B, P0, P1, eps, k, chunk, m=None):
    """One grid cell: fit the path, read every metric off it.

    `PersonaPath.report` is the single source for the metric names, so a column
    here can never drift from what the class calls the same quantity.
    """
    path = PersonaPath(P0, P1, geom.C, eps, k=k, lam=FIXED["lam"],
                       mode=FIXED["mode"], param=FIXED["param"],
                       chunk=chunk, m=m)
    rep = path.report(geom.names)
    idx = np.asarray(path.centroid_idx, dtype=int)
    return {
        "route": route, "kind": kind, "A": A, "B": B,
        "eps": float(eps), "k": int(k), "chunk": chunk,
        "n_centroids": int(rep["n_centroids"]),
        "centroid_idx": ";".join(str(int(i)) for i in idx),
        "centroid_roles": ";".join(rep["centroid_roles"] or []),
        "detour_ratio": float(rep["detour_ratio"]),
        "polyline_ratio": float(rep["polyline_ratio"]),
        "overshoot": float(rep["overshoot"]),
        "excursion": float(rep["excursion"]),
        "knot_error": float(rep["knot_error"]),
        "min_knot_spacing_u": min_knot_spacing_u(geom.chord_u(route, P0, P1), idx),
        "chord_len": float(chord_frame(P0, P1)[1]),
        # Underscored keys are never written: `write_csv` pins the columns.
        # The knot SET is carried alongside the joined string so step 3 does
        # not have to parse its own output back to compute the deciding metric.
        "_idx": frozenset(int(i) for i in idx),
    }


GRID_COLUMNS = ["route", "kind", "A", "B", "eps", "k", "chunk", "n_centroids",
                "centroid_idx", "centroid_roles", "detour_ratio",
                "polyline_ratio", "overshoot", "excursion", "knot_error",
                "min_knot_spacing_u", "chord_len"]
M_COLUMNS = GRID_COLUMNS + ["m", "n_bins"]


def jaccard(a, b):
    """|A n B| / |A u B|, with TWO EMPTY SETS SCORING 1.0.

    The convention is the plan's (### Metric) and it is not free: at eps=0
    neither mode selects anything, and "they agree" is the honest reading of
    that -- both modes returned the chord. The alternative (0/0 -> NaN, or 0)
    would either drop the negative-control cells from the median or drag it
    down with a disagreement that did not happen.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def route_clustered_median_ci(values_by_route, n_boot=N_BOOT, seed=BOOT_SEED):
    """Median over all cells, with a CI from resampling ROUTES with replacement.

    Rows are not exchangeable: the 15 pair routes reuse 3 source and 5 target
    centroids, so two rows from the same route share a chord, a candidate set,
    and most of their knots. Resampling rows would treat 49 grid cells of one
    route as 49 independent facts and report a CI several times too narrow.
    Resampling routes keeps each replicate a coherent pseudo-experiment: a
    route is in or out with its whole eps x k surface.
    """
    routes = sorted(values_by_route)
    per = [np.asarray(values_by_route[r], dtype=np.float64) for r in routes]
    allv = np.concatenate(per) if per else np.zeros(0)
    if allv.size == 0:
        return {"median": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_routes": 0, "n_cells": 0,
                "n_boot": int(n_boot), "seed": int(seed)}
    point = float(np.median(allv))
    n = len(routes)
    if n < 2:
        # One cluster carries no between-route variability; a CI would be a
        # zero-width fiction (figures_steering._role_bootstrap, same rule).
        return {"median": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_routes": n, "n_cells": int(allv.size),
                "n_boot": int(n_boot), "seed": int(seed)}
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        pick = rng.integers(0, n, n)
        reps[b] = np.median(np.concatenate([per[i] for i in pick]))
    return {"median": point,
            "ci_lo": float(np.percentile(reps, 2.5)),
            "ci_hi": float(np.percentile(reps, 97.5)),
            "n_routes": n, "n_cells": int(allv.size),
            "n_boot": int(n_boot), "seed": int(seed)}


# --------------------------------------------------------------------------
# Step 1 -- the four controls
# --------------------------------------------------------------------------

def control_baseline(geom, say):
    """`chunk="distance"` at the frozen production point reproduces the old code.

    Gates everything: if this fails, `chunk=` changed the DEFAULT behaviour and
    every knot set in run 2026-09-10T15-46-fig4's manifest is a different set
    from what today's code would build, so no comparison here means anything.
    """
    rows, ok = [], True
    for route, _kind, P0, P1, _A, _B in geom.routes:
        _, _, want = oracle_pick_centroids(geom.C, P0, P1, PROD_EPS, k=PROD_K,
                                           mode=FIXED["mode"])
        _, _, got = pick_centroids(geom.C, P0, P1, PROD_EPS, k=PROD_K,
                                   mode=FIXED["mode"], chunk="distance")
        same = (len(want) == len(got)
                and np.array_equal(np.asarray(want, dtype=int),
                                   np.asarray(got, dtype=int)))
        ok &= bool(same)
        rows.append({"route": route, "identical": bool(same),
                     "oracle_idx": [int(i) for i in want],
                     "chunk_distance_idx": [int(i) for i in got]})
    say("  baseline (eps=%g k=%d, oracle vs chunk='distance'): %d/%d routes identical"
        % (PROD_EPS, PROD_K, sum(r["identical"] for r in rows), len(rows)))
    return {"pass": bool(ok), "condition":
            "selected centroid indices identical to the pre-change "
            "pick_centroids for all 16 routes at eps=%g, k=%d" % (PROD_EPS, PROD_K),
            "n_routes": len(rows), "routes": rows}


def control_positive(geom, eps_grid, say):
    """At k=1 both modes are ONE bin over the whole tube, so they must agree.

    One bin is one partition however you cut it, and the rule inside a bin is
    unchanged (nearest the chord), so a Jaccard below 1.0 here means the two
    code paths disagree about the CANDIDATE SET, not about chunking -- which
    would invalidate every other cell in the grid.
    """
    rows, ok = [], True
    for eps in [e for e in eps_grid if e > 0]:
        for route, _kind, P0, P1, _A, _B in geom.routes:
            sets = {}
            for chunk in CHUNKS:
                _, _, idx = pick_centroids(geom.C, P0, P1, eps, k=1,
                                           mode=FIXED["mode"], chunk=chunk)
                sets[chunk] = frozenset(int(i) for i in idx)
            j = jaccard(sets["distance"], sets["density"])
            ok &= (j == 1.0)
            rows.append({"route": route, "eps": float(eps), "jaccard": float(j)})
    worst = min([r["jaccard"] for r in rows], default=float("nan"))
    say("  positive (k=1, eps>0): %d cells, min Jaccard %.6f" % (len(rows), worst))
    return {"pass": bool(ok), "condition":
            "Jaccard == 1.0 for all 16 routes at every eps > 0 with k=1",
            "n_cells": len(rows), "min_jaccard": float(worst),
            "failures": [r for r in rows if r["jaccard"] != 1.0]}


def control_negative(geom, say):
    """eps=0 selects nothing, so the manifold path must BE the chord.

    Three conditions, because two of them can pass on a broken path: a path
    with no knots trivially reports n_centroids=0, and `detour_ratio` is
    arc/chord which is 1.0 for any curve of chord length. The pointwise
    comparison over 200 alphas is the one that cannot be satisfied by accident.
    """
    alphas = np.linspace(0.0, 1.0, N_ALPHA_NEGATIVE)
    rows, ok = [], True
    for route, _kind, P0, P1, _A, _B in geom.routes:
        chord = LinearPath(P0, P1).at_alpha(alphas)
        for chunk in CHUNKS:
            p = PersonaPath(P0, P1, geom.C, 0.0, k=PROD_K, lam=FIXED["lam"],
                            mode=FIXED["mode"], param=FIXED["param"], chunk=chunk)
            dev = float(np.linalg.norm(p.at_alpha(alphas) - chord, axis=1).max())
            det = abs(p.detour_ratio - 1.0)
            good = (p.n_centroids == 0 and det < CHORD_COLLAPSE_TOL
                    and dev < CHORD_COLLAPSE_TOL)
            ok &= bool(good)
            rows.append({"route": route, "chunk": chunk,
                         "n_centroids": int(p.n_centroids),
                         "detour_dev": float(det), "max_chord_dev": dev,
                         "pass": bool(good)})
    say("  negative (eps=0): %d/%d cells collapse onto the chord "
        "(max deviation %.3e)"
        % (sum(r["pass"] for r in rows), len(rows),
           max([r["max_chord_dev"] for r in rows], default=float("nan"))))
    return {"pass": bool(ok), "condition":
            "n_centroids == 0 and |detour_ratio - 1| < %g and "
            "max||path(alpha) - chord(alpha)|| < %g over %d alphas, both modes"
            % (CHORD_COLLAPSE_TOL, CHORD_COLLAPSE_TOL, N_ALPHA_NEGATIVE),
            "n_cells": len(rows), "cells": rows}


def control_interpolation(grid_rows, say):
    """lam=0 must mean the spline passes through every chosen knot exactly.

    Measured on the WHOLE grid, both modes, because this is the property that
    makes "the route visits these personas" true; a single cell where it fails
    is a route through points nobody picked. `PersonaPath.knot_error` measures
    against the centroids themselves, not against the fit's own `y_hat`.

    THIS ONE CANNOT RUN BEFORE THE GRID. The plan asks for it "across the entire
    grid", so the grid must be fitted first. It is still a gate: the grid is
    held in memory and NOTHING downstream of it -- no CSV, no Jaccard, no m
    ablation -- is produced if this fails.
    """
    errs = [r["knot_error"] for r in grid_rows]
    worst = max(errs) if errs else 0.0
    bad = [{"route": r["route"], "eps": r["eps"], "k": r["k"],
            "chunk": r["chunk"], "knot_error": r["knot_error"]}
           for r in grid_rows if r["knot_error"] >= KNOT_ERROR_MAX]
    say("  interpolation gate: max knot_error over %d cells = %.3e (limit %g)"
        % (len(grid_rows), worst, KNOT_ERROR_MAX))
    return {"pass": bool(worst < KNOT_ERROR_MAX), "condition":
            "max(knot_error) < %g across the entire grid, both modes" % KNOT_ERROR_MAX,
            "n_cells": len(grid_rows), "max_knot_error": float(worst),
            "failures": bad[:50]}


# --------------------------------------------------------------------------
# Steps 2-5
# --------------------------------------------------------------------------

def build_grid(geom, eps_grid, k_grid, say):
    """Step 2: one row per (route, eps, k, chunk)."""
    rows = []
    for route, kind, P0, P1, A, B in geom.routes:
        for eps in eps_grid:
            for k in k_grid:
                for chunk in CHUNKS:
                    rows.append(path_row(geom, route, kind, A, B, P0, P1,
                                         eps, k, chunk))
    say("grid: %d cells (%d routes x %d eps x %d k x %d modes)"
        % (len(rows), len(geom.routes), len(eps_grid), len(k_grid), len(CHUNKS)))
    return rows


def build_jaccard(grid_rows, say):
    """Step 3: the deciding metric, one row per (route, eps, k).

    `same_count` carries confound 1 from the plan: equal-width spans go empty
    and return fewer than k knots while equal-count bins essentially never do,
    so a Jaccard below 1 can come from COUNT rather than from CHOICE. The
    restricted median below is the mitigation, not a second hypothesis.
    """
    by_cell = {}
    for r in grid_rows:
        by_cell.setdefault((r["route"], r["eps"], r["k"]), {})[r["chunk"]] = r
    rows = []
    for (route, eps, k), got in sorted(by_cell.items()):
        if set(got) != set(CHUNKS):
            raise ValueError("cell %r has modes %r, expected %r"
                             % ((route, eps, k), sorted(got), list(CHUNKS)))
        sd, sn = got["distance"]["_idx"], got["density"]["_idx"]
        rows.append({"route": route, "eps": float(eps), "k": int(k),
                     "jaccard": float(jaccard(sd, sn)),
                     "n_dist": len(sd), "n_dens": len(sn),
                     "n_intersect": len(sd & sn),
                     "same_count": bool(len(sd) == len(sn))})
    say("jaccard: %d cells" % len(rows))
    return rows


JACCARD_COLUMNS = ["route", "eps", "k", "jaccard", "n_dist", "n_dens",
                   "n_intersect", "same_count"]


def summarise_jaccard(jrows, say):
    """The grid median + route-clustered CI, whole and restricted to same_count."""
    whole, restricted = {}, {}
    for r in jrows:
        whole.setdefault(r["route"], []).append(r["jaccard"])
        if r["same_count"]:
            restricted.setdefault(r["route"], []).append(r["jaccard"])
    out = {
        "grid_median": route_clustered_median_ci(whole),
        "grid_median_same_count_only": route_clustered_median_ci(restricted),
        "falsifier_threshold": 0.95,
        "would_matter_threshold": 0.70,
        "n_cells_total": len(jrows),
        "n_cells_same_count": sum(1 for r in jrows if r["same_count"]),
        "note": "the eps x k grid includes eps=0, where both modes select "
                "nothing and the plan's empty-set convention scores 1.0; those "
                "cells are part of the grid the plan defines and are not "
                "removed here.",
    }
    g, s = out["grid_median"], out["grid_median_same_count_only"]
    say("")
    say("SUMMARY -- deciding metric (knot-set Jaccard, distance vs density)")
    say("  grid median            %.4f  [95%% CI %.4f, %.4f]  "
        "(%d cells, %d routes, %d resamples, seed %d)"
        % (g["median"], g["ci_lo"], g["ci_hi"], g["n_cells"], g["n_routes"],
           g["n_boot"], g["seed"]))
    say("  same_count rows only   %.4f  [95%% CI %.4f, %.4f]  (%d cells)"
        % (s["median"], s["ci_lo"], s["ci_hi"], s["n_cells"]))
    say("  falsifier >= 0.95: %s   |   'would matter' <= 0.70: %s"
        % ("MET (refinement dead)" if g["median"] >= 0.95 else "not met",
           "yes" if g["median"] <= 0.70 else "no"))
    say("  NO WINNER IS DECLARED between the modes: the plan's path-quality "
        "decider is open (Questions, item 4).")
    say("")
    return out


def build_m_grid(geom, m_grid, m_eps_grid, say):
    """Step 4: density mode, bin count = ceil(n_candidates / m). k is INERT here.

    The `k` column is still written, carrying the value actually passed, so the
    CSV says what the call was rather than leaving a reader to assume. `n_bins`
    is the number that mattered.
    """
    rows = []
    for route, kind, P0, P1, A, B in geom.routes:
        for eps in m_eps_grid:
            n_cand = len(select_cylinder(geom.C, P0, P1, eps, FIXED["mode"])[0])
            for m in m_grid:
                row = path_row(geom, route, kind, A, B, P0, P1, eps, PROD_K,
                               "density", m=m)
                row["m"] = int(m)
                # mirrors manifold_paths._chunk_candidates, which floors at one
                # bin so an empty tube still has a partition to be empty in.
                row["n_bins"] = int(max(1, math.ceil(n_cand / float(m))))
                rows.append(row)
    say("m grid: %d cells (%d routes x %d eps x %d m, density only)"
        % (len(rows), len(geom.routes), len(m_eps_grid), len(m_grid)))
    return rows


FIG01_COLUMNS = ["route", "chunk", "bin", "u_lo", "u_hi", "cand_idx",
                 "cand_role", "cand_u", "cand_r", "cand_cos", "is_picked",
                 "picked_rank_r", "picked_rank_cos", "tube_radius"]


def pick_representative_route(geom, eps, say):
    """The route with the MEDIAN candidate count at eps, for figure 1.

    Chosen by candidate SUPPLY, never by any metric under test -- picking the
    route with the most interesting Jaccard would make the figure an
    illustration of the answer rather than of the method.

    16 is even, so there is no single middle route. The rule, fixed here: sort
    by (n_candidates, route name) ascending and take index (n-1)//2, the LOWER
    of the two middle routes, ties broken by name. Deterministic, and recorded
    in controls.json with the whole sorted list so the choice is auditable.
    """
    supply = []
    for route, _kind, P0, P1, _A, _B in geom.routes:
        n = len(select_cylinder(geom.C, P0, P1, eps, FIXED["mode"])[0])
        supply.append({"route": route, "n_candidates": int(n)})
    ordered = sorted(supply, key=lambda d: (d["n_candidates"], d["route"]))
    chosen = ordered[(len(ordered) - 1) // 2]
    say("fig01 route: %s (%d candidates at eps=%g; median of %d routes)"
        % (chosen["route"], chosen["n_candidates"], eps, len(ordered)))
    return chosen["route"], {
        "route": chosen["route"], "eps": float(eps),
        "n_candidates": chosen["n_candidates"],
        "rule": "the route with the median number of candidates in the "
                "eps-cylinder at eps=%g; sorted by (n_candidates, route) "
                "ascending, index (n-1)//2 = the lower of the two middle "
                "routes since n=%d is even. Chosen by candidate supply, not by "
                "any metric under test." % (eps, len(ordered)),
        "supply_sorted": ordered,
    }


def build_fig01(geom, route_name, eps, k, say):
    """Step 5: every candidate in every chunk, both modes, one route.

    `chunk_diagnostics` returns one record per NON-EMPTY chunk and keys it by
    its index in the partition, so a gap in `bin` under "distance" is an empty
    span -- the thing worth seeing. This flattens records to one row per
    CANDIDATE; `picked_rank_r` and `picked_rank_cos` are per-chunk and repeat
    down the chunk's rows.
    """
    sel = [r for r in geom.routes if r[0] == route_name]
    if not sel:
        raise ValueError("route %r not found" % route_name)
    _, _kind, P0, P1, _A, _B = sel[0]
    rows = []
    for chunk in CHUNKS:
        records, radius = chunk_diagnostics(geom.C, P0, P1, eps, k=k,
                                            mode=FIXED["mode"], chunk=chunk)
        for rec in records:
            cand = np.asarray(rec["cand_idx"], dtype=int)
            for j, ci in enumerate(cand):
                rows.append({
                    "route": route_name, "chunk": chunk, "bin": int(rec["bin"]),
                    "u_lo": float(rec["u_lo"]), "u_hi": float(rec["u_hi"]),
                    "cand_idx": int(ci), "cand_role": geom.names[int(ci)],
                    "cand_u": float(rec["cand_u"][j]),
                    "cand_r": float(rec["cand_r"][j]),
                    "cand_cos": float(rec["cand_cos"][j]),
                    "is_picked": bool(int(ci) == int(rec["picked"])),
                    "picked_rank_r": int(rec["picked_rank_r"]),
                    "picked_rank_cos": int(rec["picked_rank_cos"]),
                    "tube_radius": float(radius),
                })
    say("fig01 data: %d candidate rows for %s at eps=%g k=%d (both modes)"
        % (len(rows), route_name, eps, k))
    return rows


# --------------------------------------------------------------------------
# Step 6 -- centroid-jitter bootstrap (SECONDARY)
# --------------------------------------------------------------------------

JITTER_COLUMNS = ["b", "seed", "route", "eps", "k", "chunk", "jaccard",
                  "n_ref", "n_boot"]


def jitter_stability(geom, eps_grid, k_grid, say):
    """Secondary: how stable is a knot set under resampling the response rows?

    The plan asks for 20 resamples, `seed = 137 + b`, resampling each role's
    response rows with replacement, then knot-set stability WITHIN each mode.
    Needs the raw 331k-row cloud; the geometry cache holds only the 275
    centroids, so this is skipped -- never faked -- without --cloud.

    TWO CHOICES THE PLAN DOES NOT MAKE, recorded here and in controls.json
    rather than buried:
      1. "stability" is the Jaccard between the knot set under resample b and
         the knot set on the UNJITTERED centroids (the point estimate), which
         is the standard reading of bootstrap stability for a set-valued
         estimator and reuses the plan's own Jaccard, empty-set convention
         included. The alternative -- mean PAIRWISE Jaccard across resamples --
         is a different number.
      2. It is evaluated over the whole eps x k grid, both modes.
      3. The ENDPOINTS are held fixed at their unjittered values, including the
         axis case's, which is derived from the centroid mean. Jittering them
         too would move the chord, and then a changed knot set would not say
         whether the knots or the route moved -- which is the question.
    All three are flagged `definition_unratified_by_plan` in the output.
    """
    if geom.extra.get("geom") is None:
        raise ValueError("jitter needs the cloud-loaded geometry")
    # Same-package privates on purpose: `_load_cloud` caches the 2.7 GB array
    # the geometry load already paid for, and `_load_labels` carries the
    # row-alignment guard that makes a label array mean what it says. A second
    # copy of that guard here is exactly the failure it exists to prevent.
    from steering.geometry import _load_cloud, _load_labels

    cloud = geom.source["cloud"]
    X, meta, _man = _load_cloud(cloud, LAYER_INDEX)
    roles_arr = meta["role"].values
    labels = _load_labels(geom.source["labels"], X.shape[0], roles_arr)
    is_fully = labels == "fully"

    row_idx = {}
    for name in geom.names:
        sel = np.nonzero((roles_arr == name) & is_fully)[0]
        if len(sel) == 0:
            raise ValueError("role %r has no `fully` rows; the centroid it "
                             "contributes cannot be resampled" % name)
        row_idx[name] = sel
    say("jitter: %d roles, %d fully rows total"
        % (len(row_idx), int(sum(len(v) for v in row_idx.values()))))

    ref = {}
    for route, _kind, P0, P1, _A, _B in geom.routes:
        for eps in eps_grid:
            for k in k_grid:
                for chunk in CHUNKS:
                    _, _, idx = pick_centroids(geom.C, P0, P1, eps, k=k,
                                               mode=FIXED["mode"], chunk=chunk)
                    ref[(route, eps, k, chunk)] = frozenset(int(i) for i in idx)

    rows = []
    for b in range(N_JITTER):
        seed = BOOT_SEED + b
        rng = np.random.default_rng(seed)
        Cb = np.empty_like(geom.C)
        for i, name in enumerate(geom.names):
            sel = row_idx[name]
            draw = sel[rng.integers(0, len(sel), len(sel))]
            Cb[i] = X[draw].mean(0, dtype=np.float64)
        Cb = _require_finite(Cb, "jittered centroids (b=%d)" % b)
        for route, _kind, P0, P1, _A, _B in geom.routes:
            for eps in eps_grid:
                for k in k_grid:
                    for chunk in CHUNKS:
                        _, _, idx = pick_centroids(Cb, P0, P1, eps, k=k,
                                                   mode=FIXED["mode"], chunk=chunk)
                        got = frozenset(int(i) for i in idx)
                        want = ref[(route, eps, k, chunk)]
                        rows.append({"b": b, "seed": seed, "route": route,
                                     "eps": float(eps), "k": int(k),
                                     "chunk": chunk,
                                     "jaccard": float(jaccard(want, got)),
                                     "n_ref": len(want), "n_boot": len(got)})
        say("  jitter b=%d (seed %d) done" % (b, seed))

    summary = {"status": "run", "n_resamples": N_JITTER,
               "seed_rule": "137 + b for b = 0..%d" % (N_JITTER - 1),
               "definition_unratified_by_plan": True,
               "definition": "Jaccard between the knot set on jittered "
                             "centroids and the knot set on the unjittered "
                             "centroids, at matched (route, eps, k, chunk); "
                             "evaluated over the whole eps x k grid, both "
                             "modes, with the route endpoints held fixed.",
               "by_chunk": {}}
    for chunk in CHUNKS:
        vals = {}
        for r in rows:
            if r["chunk"] == chunk:
                vals.setdefault(r["route"], []).append(r["jaccard"])
        summary["by_chunk"][chunk] = route_clustered_median_ci(vals)
        m = summary["by_chunk"][chunk]
        say("  jitter stability, chunk=%s: median Jaccard %.4f [%.4f, %.4f]"
            % (chunk, m["median"], m["ci_lo"], m["ci_hi"]))
    return rows, summary


# --------------------------------------------------------------------------
# --show-example
# --------------------------------------------------------------------------

def show_example(cloud, labels, say):
    """One verbatim row from each input file. The plan's Data section requires it.

    Printed, not summarised: the report pastes what the files actually contain,
    so a reader can see the schema this run read rather than a description of it.
    """
    if not cloud or not labels:
        raise SystemExit("--show-example needs --cloud and --labels: it prints "
                         "verbatim rows from metadata.parquet and "
                         "role_labels.parquet, and there is nothing to print "
                         "from the geometry cache.")
    mp = Path(cloud) / "metadata.parquet"
    say("")
    say("--show-example: %s" % mp)
    md = pd.read_parquet(mp)
    say("  columns: %s" % list(md.columns))
    say("  shape:   %r" % (md.shape,))
    say("  row 0 (verbatim):")
    for kk, vv in md.iloc[0].items():
        say("    %s = %r" % (kk, vv))
    say("")
    say("--show-example: %s" % labels)
    lb = pd.read_parquet(labels)
    say("  columns: %s" % list(lb.columns))
    say("  shape:   %r" % (lb.shape,))
    say("  row 0 (verbatim):")
    for kk, vv in lb.iloc[0].items():
        say("    %s = %r" % (kk, vv))
    say("")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="run dir. EVERYTHING this script writes goes inside "
                         "it; nothing is written anywhere else, and never "
                         "under the read-only data/ trees.")
    ap.add_argument("--geom-cache", default=None,
                    help="npz holding C/names/axis_proj/axis_unit/kinds/P0/P1/"
                         "A/B. Used by default when it exists, because the "
                         "real load is 2.7 GB over NFS (~267 s).")
    ap.add_argument("--cloud", default=None,
                    help="resp_dir: the response-token activation cloud. The "
                         "slow, authoritative load. Required for --show-example "
                         "and for the centroid-jitter bootstrap.")
    ap.add_argument("--labels", default=None,
                    help="role_labels.parquet. Without it load_geometry falls "
                         "back to unfiltered centroids and `fully_only` refuses.")
    ap.add_argument("--layer", type=int, default=19,
                    help="layer LABEL, recorded in the manifest. The cloud is "
                         "pre-thinned, so the array index is always 0.")
    ap.add_argument("--show-example", action="store_true",
                    help="print one verbatim metadata.parquet row and one "
                         "role_labels.parquet row, for the report.")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid for smoke-testing. Keeps eps=12 and k=8 so "
                         "the baseline control still has its production point. "
                         "NOT a result: the manifest records quick=true.")
    args = ap.parse_args()

    out = OutDir(args.out)
    say = Log(out.path("logs/exp01.log"))
    started = datetime.datetime.now().isoformat(timespec="seconds")
    say("=" * 74)
    say("exp01 chunk-selection ablation   started %s" % started)
    say("plan: %s" % PLAN)
    say("out:  %s" % out.root)

    eps_grid = [0.0, 12.0] if args.quick else EPS_GRID
    k_grid = [1, 8] if args.quick else K_GRID
    m_grid = [1, 4] if args.quick else M_GRID
    m_eps_grid = [12.0] if args.quick else M_EPS_GRID
    if args.quick:
        say("QUICK MODE: eps=%s k=%s m=%s m_eps=%s -- smoke test, not a result"
            % (eps_grid, k_grid, m_grid, m_eps_grid))

    manifest = build_manifest(out.root, PLAN, extra={
        "experiment": "exp01-chunk-selection-ablation",
        "started": started,
        "quick": bool(args.quick),
        "layer_label": int(args.layer),
        "layer_index": LAYER_INDEX,
        "grids": {"eps": eps_grid, "k": k_grid, "m": m_grid,
                  "m_eps": m_eps_grid, "chunk": list(CHUNKS)},
        "fixed": dict(FIXED),
        "production_point": {"eps": PROD_EPS, "k": PROD_K},
        "seeds": {"bootstrap": BOOT_SEED, "n_boot": N_BOOT,
                  "centroid_jitter": "%d + b, b = 0..%d"
                                     % (BOOT_SEED, N_JITTER - 1),
                  "n_jitter": N_JITTER},
        "inputs": {"geom_cache": args.geom_cache, "cloud": args.cloud,
                   "labels": args.labels},
        "status": "running",
    })
    write_manifest(out.root, manifest)
    say("git %s%s" % (manifest["git_sha"][:12],
                      " (DIRTY)" if manifest["git_dirty"] else ""))

    if args.show_example:
        show_example(args.cloud, args.labels, say)

    geom = resolve_geometry(args, say)
    manifest["geometry_source"] = geom.source
    manifest["n_role_centroids"] = int(geom.C.shape[0])
    manifest["n_routes"] = len(geom.routes)
    say("centroids: %d roles x %d dims; routes: %d"
        % (geom.C.shape[0], geom.C.shape[1], len(geom.routes)))
    say("routes: %s" % ", ".join(r[0] for r in geom.routes))

    # ---------------- step 1: controls, and they run FIRST ----------------
    say("")
    say("STEP 1  controls")
    controls = {"plan": PLAN, "started": started,
                "geometry_source": geom.source,
                "grids": manifest["grids"], "fixed": dict(FIXED),
                "quick": bool(args.quick)}
    controls["baseline"] = control_baseline(geom, say)
    controls["positive"] = control_positive(geom, eps_grid, say)
    controls["negative"] = control_negative(geom, say)

    def _fail(stage):
        controls["all_pass"] = False
        controls["failed_at"] = stage
        controls["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        out.write_json("data/controls.json", controls)
        say("")
        say("!" * 74)
        say("CONTROL FAILURE at %s. Plan: 'if a control fails, nothing "
            "downstream is interpreted and the run reports the control failure "
            "only.' No grid output written. Exiting non-zero." % stage)
        say("!" * 74)
        manifest["status"] = "control-failed"
        manifest["failed_at"] = stage
        manifest["finished"] = controls["finished"]
        write_manifest(out.root, manifest)
        say.close()
        sys.exit(1)

    for stage in ("baseline", "positive", "negative"):
        if not controls[stage]["pass"]:
            _fail(stage)

    # ---------------- step 2: the main grid ----------------
    say("")
    say("STEP 2  main grid")
    grid_rows = build_grid(geom, eps_grid, k_grid, say)
    controls["interpolation_gate"] = control_interpolation(grid_rows, say)
    if not controls["interpolation_gate"]["pass"]:
        _fail("interpolation_gate")
    controls["all_pass"] = True
    say("  ALL FOUR CONTROLS PASS")

    p, n = out.write_csv("data/grid_eps_k.csv", grid_rows, GRID_COLUMNS)
    say("wrote %s (%d rows)" % (p, n))

    # ---------------- step 3: Jaccard ----------------
    say("")
    say("STEP 3  knot-set Jaccard")
    jrows = build_jaccard(grid_rows, say)
    p, n = out.write_csv("data/jaccard_summary.csv", jrows, JACCARD_COLUMNS)
    say("wrote %s (%d rows)" % (p, n))
    controls["summary"] = summarise_jaccard(jrows, say)

    # ---------------- step 4: the m ablation ----------------
    say("STEP 4  m ablation (density only)")
    mrows = build_m_grid(geom, m_grid, m_eps_grid, say)
    p, n = out.write_csv("data/grid_m.csv", mrows, M_COLUMNS)
    say("wrote %s (%d rows)" % (p, n))

    # ---------------- step 5: figure-1 data ----------------
    say("")
    say("STEP 5  figure-1 candidate dump")
    route_name, rep_info = pick_representative_route(geom, PROD_EPS, say)
    controls["fig01_route"] = rep_info
    frows = build_fig01(geom, route_name, PROD_EPS, PROD_K, say)
    p, n = out.write_csv("data/fig01_candidates.csv", frows, FIG01_COLUMNS)
    say("wrote %s (%d rows)" % (p, n))

    # ---------------- step 6: centroid jitter (secondary) ----------------
    say("")
    say("STEP 6  centroid-jitter bootstrap (secondary)")
    if geom.source["kind"] != "cloud":
        reason = ("needs the raw 331,200-row response cloud to resample each "
                  "role's rows; the geometry cache holds only the %d role "
                  "centroids. Re-run with --cloud/--labels to produce it."
                  % geom.C.shape[0])
        controls["centroid_jitter"] = {"status": "skipped", "reason": reason,
                                       "n_resamples_planned": N_JITTER,
                                       "seed_rule": "137 + b"}
        say("  SKIPPED: %s" % reason)
        say("  This is a SECONDARY metric. It is recorded as skipped in "
            "controls.json, not omitted, and nothing here stands in for it.")
    else:
        jit_rows, jit_summary = jitter_stability(geom, eps_grid, k_grid, say)
        controls["centroid_jitter"] = jit_summary
        p, n = out.write_csv("data/jitter_stability.csv", jit_rows, JITTER_COLUMNS)
        say("wrote %s (%d rows)" % (p, n))

    # ---------------- close ----------------
    finished = datetime.datetime.now().isoformat(timespec="seconds")
    controls["finished"] = finished
    p = out.write_json("data/controls.json", controls)
    say("wrote %s" % p)

    manifest["status"] = "ok"
    manifest["finished"] = finished
    manifest["outputs"] = sorted(
        str(q.relative_to(out.root)) for q in out.path("data").glob("*")) + \
        ["logs/exp01.log"]
    manifest["summary"] = controls["summary"]
    manifest["controls_all_pass"] = True
    write_manifest(out.root, manifest)
    say("wrote %s" % out.path("manifest.json"))
    say("finished %s" % finished)
    say("=" * 74)
    say.close()


if __name__ == "__main__":
    main()
