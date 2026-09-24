"""Shared constants and loaders for the follow-up run.

Plan: plans/2026-09-24-steering-followups.md.

Everything that more than one follow-up script needs to AGREE on lives here:
the four routes, the question list, the seed rule, the geometry default, the
collapse threshold and the unsteered-start (h0) loaders. A second copy of any of
these is how a figure ends up describing a grid that was never generated -- the
calibration would compute alpha_pos for one h0 while the driver steered from
another, and both would look internally consistent.

torch is pulled in transitively (run_steering -> interventions) for the
question list only; `.venv` carries CPU torch, so the CPU experiments (exp1,
exp3), the judge and the figures all still run on a login node.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from steering.run_steering import INTROSPECTIVE_QUESTIONS

# ---------------------------------------------------------------------------
# The grid's fixed vocabulary
# ---------------------------------------------------------------------------

PLAN = "plans/2026-09-24-steering-followups.md"
_REPO = Path(__file__).resolve().parents[2]

# Order is load-bearing: it is the first key of the seed order (route, arm,
# alpha, question), so reordering this list reseeds every generation.
ROUTES = ["validator>vampire", "validator>bard", "assistant>vampire", "assistant>bard"]

# The plan's own geometry: 275 role centroids at hidden state 19, response-token
# mean-pooled. A documented default, overridable everywhere with --geom.
GEOM_DEFAULT = _REPO / "output/steering-manifold/2026-09-17T01-01-chunk-ablation/data/geom_cache.npz"

# ONE output root for the whole follow-up run, fixed by the user (2026-09-24),
# not a minute-stamped runmeta dir: exps 1-6 are one experiment and every
# stage, shard and figure must find the others by a path known in advance.
# Under the repo's gitignored output/, on /home, so the Condor execute nodes
# see it at the same path as MP_ROOT. Override with STEERING_FIX_OUT.
OUT_ROOT = Path(os.environ.get("STEERING_FIX_OUT", _REPO / "output" / "steering-fix-24_09"))
SUBDIRS = ("exp1_calibration", "exp2_direction_check", "exp3_knots", "studyA", "studyB",
           "judge", "figures", "logs")

HIDDEN_STATE = 19               # hidden_states[19] = output of model.layers[18]
K_DEFAULT = 8                   # knots for every manifold arm in exps 4-5
EPS_ALL = 1e9                   # "every centroid is a candidate" -- the value
                                # jobs_condor/capture_steered_activations.py
                                # used to build manifold_v1/v2 for the demo
                                # routes; the knot rule then does the choosing

CANARY = "What is the capital of France?"
# (kind, text). The 5 identity questions are App. D.1.2 verbatim, imported
# rather than retyped so they cannot drift from run_steering.py.
QUESTIONS = [("identity", q) for q in INTROSPECTIVE_QUESTIONS] + [("canary", CANARY)]
N_SAMPLES = 3                   # per (cell, question)

SEED_BASE = 20260924
SEED_STRIDE = 100               # samples per seed unit; 3 used, 97 spare


def seed_for(unit_index: int, sample_index: int) -> int:
    """`20260924 + unit_index*100 + sample_index` (plan, Seeds).

    `unit_index` enumerates (route, arm, alpha, question) in that nesting
    order -- the plan's "cell_index ... route, then arm, then alpha, then
    question". See jobs_condor/followups_steer.py::build_cells for the one
    enumeration that assigns it.
    """
    if not 0 <= sample_index < SEED_STRIDE:
        raise ValueError("sample_index must be in [0, %d), got %d" % (SEED_STRIDE, sample_index))
    return SEED_BASE + int(unit_index) * SEED_STRIDE + int(sample_index)


# A response whose most frequent 4-gram repeats more than this many times is
# its own judge class ("collapsed", plan: Exclusions), never dropped and never
# sent to the judge. 10 because the budget is 256 tokens here; dose_readout's
# REP4_MAX = 4 was set for 96-token greedy text.
REP4_COLLAPSED = 10


def degeneracy(text: str, n: int = 4) -> dict:
    """Type/token ratio and max 4-gram repeat count.

    The same two measures as jobs_condor/dose_escalation.py::degeneracy, which
    is untracked and so cannot be imported by tracked code. Copied rather than
    rewritten: `rep4` must mean the same thing here as in dose_L19.md, where
    the collapse threshold was first read off.
    """
    toks = re.findall(r"\S+", str(text))
    if not toks:
        return dict(n_tok=0, distinct=0.0, rep4=0)
    grams = [tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))]
    return dict(n_tok=len(toks), distinct=len(set(toks)) / len(toks),
                rep4=max(Counter(grams).values()) if grams else 0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Geom:
    """The geom_cache.npz, with name lookup.

    `axis_unit` is the Assistant Axis at this layer (unit norm), used by exp1
    as the third calibration direction.
    """

    def __init__(self, path=GEOM_DEFAULT):
        d = np.load(path, allow_pickle=True)
        self.path = str(path)
        self.C = np.asarray(d["C"], dtype=np.float64)
        self.names = [str(x) for x in d["names"]]
        self.ix = {n: i for i, n in enumerate(self.names)}
        self.axis_unit = (np.asarray(d["axis_unit"], dtype=np.float64)
                          if "axis_unit" in d.files else None)
        if not np.isfinite(self.C).all():
            raise ValueError("geometry %s has non-finite centroids" % path)

    def c(self, role: str) -> np.ndarray:
        if role not in self.ix:
            raise KeyError("role %r not in geometry %s" % (role, self.path))
        return self.C[self.ix[role]]

    def route(self, route: str):
        A, B = route.split(">")
        return A, B, self.c(A), self.c(B)


def route_slug(route: str) -> str:
    return route.replace(">", "-")


# ---------------------------------------------------------------------------
# h0: where the model actually is, unsteered
# ---------------------------------------------------------------------------

# The only unsteered footprints that exist before this run, both with the
# validator system prompt (index 0) at hidden state 19. They are valid for BOTH
# validator routes: the unsteered state depends on the source prompt and the
# question, not on the target.
DOSE_DIR = Path("/data/project/eeg_foundation/data/manifold_persona/steer_demo")
H0_SOURCES = {
    # greedy, 96 tokens, 2 identity questions -- the file the plan quotes
    "dose_L19": (DOSE_DIR / "dose_L19_text_acts.npy", DOSE_DIR / "dose_L19.csv"),
    # temp 0.7 / top_p 0.9, 5 identity questions x 3 samples x 3 strategies at
    # alpha 0 -- the same decoding and questions as exp4
    "located_L19_temp07": (DOSE_DIR / "located_L19_temp07_acts.npy",
                           DOSE_DIR / "located_L19_temp07.csv"),
}


def load_h0(npy, csv=None) -> tuple[np.ndarray, int]:
    """Mean unsteered footprint from an activation file.

    `npy` 1-D: it IS h0. 2-D with a `csv` alongside (one row per activation):
    the rows with alpha == 0 and, when the csv has a `kind` column, kind ==
    'identity' -- the canary is not a persona question and its footprint sits
    ~12 units further out on the dose run. Returns (h0, n_rows_averaged).
    """
    X = np.load(npy)
    if X.ndim == 1:
        return X.astype(np.float64), 1
    if csv is None:
        return X.astype(np.float64).mean(0), len(X)
    import pandas as pd
    df = pd.read_csv(csv)
    if len(df) != len(X):
        raise ValueError("%s has %d rows but %s has %d" % (csv, len(df), npy, len(X)))
    m = (df["alpha"].astype(float) == 0.0).to_numpy().copy()
    if "kind" in df.columns:
        m &= (df["kind"] == "identity").to_numpy()
    if not m.any():
        raise ValueError("no alpha==0 identity rows in %s" % csv)
    return X[m].astype(np.float64).mean(0), int(m.sum())
