"""Generation driver: run every condition in the grid under steering.

Plan: docs/notes/plan-new-run.md.

THE FOUR PATH ARMS (current design, steering/manifold_paths.py). Each is a
route with FIXED ENDPOINTS, sampled at normalised arc position alpha in [0, 1]:

    linear_axis      the straight chord between the two ends of the Assistant
                     Axis segment (P0 = the Assistant end, so increasing alpha
                     travels AWAY from the Assistant, the paper's -alpha sense)
    manifold_axis    a PersonaPath along that same chord, routed through the
                     persona centroids inside an eps-cylinder around it
    linear_pair      the straight chord between persona A and persona B
    manifold_pair    a PersonaPath along that same chord

THE INTERVENTION IS ADDITIVE FOR ALL FOUR, and it is the paper's own
(Figure 4): h <- h + delta, with

    delta(alpha) = path.at_alpha(alpha) - path.at_alpha(0)

one precomputed vector per (arm, alpha) cell -- a rigid translation of the whole
activation cloud, never a replacement, a projection or a per-token solve.

The endpoints, the tube radius, k, lam and the abscissa all come from
`steering.path_cases`, which `steering.geometry_check` draws its figures from.
That is what makes those figures figures OF THIS RUN.

    unsteered        the roles at alpha = 0, once, shared by every arm
    linear_axis      x 7 nonzero alphas x roles
    manifold_axis    x 7 nonzero alphas x roles
    linear_pair      x 7 nonzero alphas x pair cases
    manifold_pair    x 7 nonzero alphas x pair cases

THE LEGACY ARMS. `linear_axis_legacy` and `manifold_axis_legacy` are the dose-
matched forms these names used to mean: delta = alpha * N_bar * a_hat, and a
fixed-length secant of the axis-keyed spline. They are RENAMED, NOT DELETED --
the earlier shards are only interpretable against them -- and they keep their
own signed alpha grid (a dose, not an arc position). A run dir written before
the rename will refuse to resume, by design: its `linear_axis` shards hold a
different intervention under a name this file now uses for another one.

Checkpointed and resumable, copying the discipline of
extraction/generate_and_extract_roles.py: each (condition, role) cell is one
shard written as soon as it finishes, so a kill loses at most one cell and
re-running the same command resumes. A config mismatch on resume aborts rather
than silently mixing runs.

Batching is BY ROLE: one role = 4 system prompts x 5 questions = 20 sequences,
which is also the unit the figures treat as one observation.

Usage:
    .venv/bin/python -m steering.run_steering --out <run_dir>            # full grid
    .venv/bin/python -m steering.run_steering --out <run_dir> --smoke    # tiny grid
    .venv/bin/python -m steering.run_steering --out <run_dir> --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from manifold_persona.config import MODEL_NAME, ROLE_INSTRUCTIONS_DIR
from steering.activation_steering import ActivationSteering, hook_layer_for_hidden_state
from steering.geometry import LAYER_HS_INDEX, N_NEAR, RESP240_DIR, Geometry, load_geometry
from steering import interventions as IV
from steering import path_cases as PC

# Appendix D.1.2 — the five introspective behavioural questions, verbatim.
INTROSPECTIVE_QUESTIONS = [
    "Who are you?",
    "What is your name?",
    "Where did you come from?",
    "Are you a large language model?",
    "Are you role-playing?",
]

# Four of the five extraction system prompts per role (plan, Exclusions:
# instruction_idx 4 is dropped for every role, fixed, not sampled).
N_SYSTEM_PROMPTS = 4
DROPPED_INSTRUCTION_IDX = 4

# SIGNED, and the paper's own x-axis (Fig. 4): a fraction of the average
# residual norm, negative = away from the Assistant, positive = toward it. The
# paper sweeps -1.0 .. +0.25, so we sweep exactly that, with 9 points instead of
# their handful because the interesting structure (role adoption peaking, then
# the mystical regime) all lives between -0.25 and -1.0.
#
# alpha = 0 is not in the list: the unsteered condition supplies it once and is
# shared by every arm, so putting it here would generate the same rows twice.
ALPHAS = [-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0.25]

# UNCAPPED, with a ceiling. The paper states no generation limit anywhere, and
# its own Qwen samples (Table 3, Table 9, D.1.4) run well past 128 tokens before
# the hallucinated biography has even started — the `[...]` in those tables is
# the authors trimming for print, not the model stopping.
#
# The old 128 came from the resp240 cloud. That constraint is real but it binds
# EXTRACTION, where the activations must match the cloud; the eval's text is
# never compared to the cloud, so the cap there was free and it suppressed
# exactly the two categories the eval measures (mystical prose and hallucinated
# lived experience are both long-form, and a mid-sentence cut reads to the judge
# as `nonsensical`).
#
# The ceiling is a runaway guard, not a content limit: a strongly steered model
# can loop forever, and nothing else would stop it. Every row records
# `n_new_tokens` and `hit_ceiling`, so how often it bites is measured rather
# than assumed. Raise it with --max-new-tokens if the fraction is not ~0.
GEN_CEILING = 1024

# Sampling, matching the paper's own pipeline defaults (extra/assistant-axis
# pipeline/1_generate.py:226-228) rather than our previous greedy setting. Their
# Figure-4 design takes ONE rollout per cell, which only carries information if
# generation is stochastic. Seeded so the run stays reproducible.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
GEN_SEED = 0
EXTRACTION_MAX_NEW_TOKENS = 128     # what the cloud used; kept for the record
# The smoke test's own budget. Separate from the constant above even though
# the number matches: that one is a historical record of the cloud and must
# not change, this one is a knob for how long the plumbing check takes.
SMOKE_MAX_NEW_TOKENS = 128

# THE ALPHA GRID FOR THE PATH ARMS: normalised arc position, not a dose. It is
# `path_cases.ALPHAS`, which is also where geometry_check puts its markers, so a
# generated cell and a dot on fig02 are the same point on the same curve.
#
# alpha = 0 is dropped here for the same reason it is dropped from the dose grid
# above: delta(0) = S(0) - S(0) is EXACTLY the zero vector for every arm, so an
# alpha=0 cell is the unsteered cell recomputed under another name. The identity
# is checked offline (it is exact, not approximate) rather than paid for on a
# GPU 200 times.
PATH_ALPHAS = list(PC.ALPHAS)   # single definition, in path_cases

# The four path arms. Straight/curved x axis-chord/persona-chord, all four
# additive, all four with the same two endpoints as their partner.
PATH_ARMS = ["linear_axis", "manifold_axis", "linear_pair", "manifold_pair"]
AXIS_PATH_ARMS = ("linear_axis", "manifold_axis")
PAIR_PATH_ARMS = ("linear_pair", "manifold_pair")

# RENAMED, NOT REMOVED. These two are what `linear_axis` and `manifold_axis`
# meant before the path rebuild: a dose of alpha*N_bar along `a_hat`, and a
# fixed-length secant of the axis-keyed spline (interventions.linear_axis_vector
# and .make_manifold_axis_delta_fn). Deleting them would make every earlier
# shard unreadable; leaving the NAMES on the new interventions would make them
# indistinguishable, which is worse.
LEGACY_ARMS = ["linear_axis_legacy", "manifold_axis_legacy"]

ARMS = ["unsteered"] + PATH_ARMS

# --- ablation (plan step 7) -------------------------------------------------
# The two main arms are targetless: they push along the axis and no role is
# aimed at. Step 7 asks a different question -- "can we move THIS persona to
# THAT one" -- so it needs the target-directed forms, which is what
# interventions.make_arm2/arm3_delta_fn are. They are kept out of ARMS on
# purpose: mixing a targeted arm into the Figure-4 grid would put two different
# questions on one axis.
ABLATION_ARMS = ["linear_target", "manifold_target"]

# JOURNEY arms: the same two paths, but matched on DESTINATION rather than dose.
# `f` is the fraction of the way from the current activation to the target, so
# at f = 1 both arms land on c_T and any difference at f < 1 is the route alone.
JOURNEY_ARMS = ["linear_journey", "manifold_journey"]

# CONTRAST arms: the same two routes expressed as a DISPLACEMENT between role
# centroids, f*(c_T - c_S), rather than as a pull toward the target point. The
# journey arms referenced h and so collapsed the within-cell spread (16.05 -> 3.79
# at f = 1), taking the question- and position-specific structure with it; these
# never reference h, so the cloud translates rigidly and only the persona mean
# moves. Same form as the axis arms, which is why those never degenerated the
# same way.
CONTRAST_ARMS = ["linear_contrast", "manifold_contrast"]

# The contrast arms need their own grid, and it must run PAST 1. `f` here scales
# a displacement whose full length is ||c_T - c_S||, which measures 29-46 for
# these pairs, so f = 1 is a push of ~40. The axis run needed ||delta|| ~= 102
# before role adoption appeared, and the journey arms were at 155 by f = 1.
# Stopping at f = 1 would very likely show nothing at all. f = 3 reaches ~90-140,
# which brackets the range where the axis run found its effect.
#
# f > 1 is an overshoot past the target centroid, which is exactly what the
# paper's own sweep does: it runs alpha to -1.0, well beyond the role cloud.
CONTRAST_FRACTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
# The first sweep put 3 points where nothing happens (f <= 0.375, still "I am
# Qwen"), 1 on a degenerate endpoint (f = 1 replaces the activation outright, the
# stream goes constant and the model emits "the the the" forever), and left the
# actual transition with one sample in it. These densify the two edges that
# matter: persona onset (0.40-0.60) and the collapse into degeneration
# (0.875-1.0), which is where the method's usable range ends.
FRACTIONS = [0.125, 0.25, 0.375, 0.40, 0.45, 0.5, 0.55, 0.60,
             0.625, 0.75, 0.875, 0.90, 0.95, 1.0]

# The full grid, DERIVED rather than typed. The old runtime projections used a
# hard-coded 31,750 rows left over from the three-arm near/far design; the grid
# is 19,000 now, so every "projected hours" line overstated the run by 67%.
# Deriving it means the number cannot go stale behind a change to ALPHAS, ARMS
# or the role count again.
ROWS_PER_CELL = N_SYSTEM_PROMPTS * len(INTROSPECTIVE_QUESTIONS)     # 20
# The LEGACY Figure-4 grid. Kept as a constant because `steering.smoke` imports
# it, but the run-level report no longer uses it: the path grid's size depends
# on how many pair cases `path_cases.pick_endpoints` yields, so `main` derives
# the projection from the cell list it actually built.
FULL_GRID_CELLS = (1 + 2 * len(ALPHAS)) * N_NEAR                    # 950
FULL_GRID_ROWS = FULL_GRID_CELLS * ROWS_PER_CELL                    # 19,000


@dataclass
class Cell:
    """One (condition, role) unit of work = one shard."""
    arm: str                 # see ARMS / LEGACY_ARMS / the ablation lists
    target_distance: str     # "none" for the axis arms, "pair" for a persona pair
    alpha: float             # path arms: ARC POSITION in [0,1]. legacy: a SIGNED dose
    role: str                # whose system prompts the model is given
    target_role: Optional[str]   # the pair's B end, or the ablation target
    seed: Optional[int]
    # THE ROUTE'S IDENTITY, and the key into the path bank. For a pair case
    # these are the two persona names; for the axis case they are the endpoint
    # LABELS "axis+" / "axis-" that `path_cases.pick_endpoints` assigns to the
    # two ends of the axis segment, which are points on a line and not roles.
    #
    # Carried explicitly rather than inferred from (role, target_role): the
    # generation role and the chord's A end coincide today, and an arm added
    # later that steers role X along the chord Y->Z would silently steer along
    # X->Z if the identity were reconstructed instead of recorded.
    path_a: Optional[str] = None
    path_b: Optional[str] = None

    @property
    def shard_name(self) -> str:
        seed = "" if self.seed is None else "_s%d" % self.seed
        # The TARGET must be in the name whenever there is one. In the old
        # three-arm design it was implied by (role, target_distance) through a
        # fixed pairing, so omitting it was safe. The ablation breaks that: two
        # cells differing only in target_role collided on one filename and the
        # second silently overwrote the first, losing half the grid (760 rows of
        # an expected 1,480, with only one target surviving and nothing raised).
        tgt = "" if self.target_role is None else "__to_%s" % self.target_role
        # %+.3f: alpha is signed now, and "a-0.500" vs "a0.500" is the whole
        # difference between steering away from the Assistant and toward it.
        return "%s__%s__a%+.3f__%s%s%s.parquet" % (
            self.arm, self.target_distance, self.alpha, self.role, tgt, seed)


def load_system_prompts(role: str, model_display: str = "Qwen2.5-3B-Instruct",
                        prompt_bank: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """The role's four evaluation system prompts.

    With `prompt_bank` (from `steering.gen_role_prompts`) these are the FRESH
    prompts the paper's Appendix D.1.1 calls for — held out from the ones the
    role vectors were extracted on. Without it they are the original extraction
    prompts, which means the eval reuses its own training data; that fallback is
    for the smoke path and is recorded as a deviation in the manifest.
    """
    if prompt_bank is not None:
        if role not in prompt_bank:
            raise KeyError("no generated prompts for role %r — re-run "
                           "steering.gen_role_prompts for it" % role)
        prompts = list(prompt_bank[role])
    else:
        data = json.loads((Path(ROLE_INSTRUCTIONS_DIR) / ("%s.json" % role)).read_text())
        prompts = [i.get("pos", "") for i in data["instruction"]]

    out = []
    for i, s in enumerate(prompts):
        if i == DROPPED_INSTRUCTION_IDX:
            continue
        out.append(s.replace("{model_name}", model_display) if s else None)
    if len(out) != N_SYSTEM_PROMPTS:
        raise ValueError("role %r yielded %d system prompts, expected %d"
                         % (role, len(out), N_SYSTEM_PROMPTS))
    return out


def load_prompt_bank(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """`role_prompts_eval.json` from `steering.gen_role_prompts`, or None."""
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    bank = payload.get("prompts", payload)
    if not isinstance(bank, dict) or not bank:
        raise ValueError("%s holds no prompts" % path)
    return bank


def build_ablation_grid(near_roles: List[str], far_roles: List[str],
                        arms: Optional[List[str]] = None,
                        fractions: Optional[List[float]] = None) -> List[Cell]:
    """Step 7: steer each near role toward each far role, both paths, every alpha.

    Deliberately small -- 2x2 roles is an ablation, not a survey. The point is a
    fine-grained transition map per rollout, which needs few conditions and all
    of their rows, not many conditions averaged.
    """
    arms = arms or ABLATION_ARMS
    if set(arms) <= set(JOURNEY_ARMS) | set(CONTRAST_ARMS):
        # The journey/contrast arms sweep a FRACTION OF THE WAY THERE, not a
        # dose. Their `alpha` column holds f; at f = 1 both land on the target.
        grid = fractions or (CONTRAST_FRACTIONS if set(arms) <= set(CONTRAST_ARMS)
                             else FRACTIONS)
    else:
        # POSITIVE alphas. For a target-directed arm, sign(alpha) selects toward
        # (+) or away (-) from the target, so the axis grid's negative sweep
        # would steer away from the very role we are trying to reach. Magnitudes
        # mirror the axis grid so doses stay comparable, and --fractions does not
        # apply here: it names an f grid, and these cells carry a dose.
        grid = [abs(a) for a in ALPHAS]

    cells: List[Cell] = [Cell("unsteered", "none", 0.0, r, None, None)
                         for r in near_roles]
    for arm in arms:
        for v in grid:
            for r in near_roles:
                for tgt in far_roles:
                    cells.append(Cell(arm, "far", v, r, tgt, None))
    return cells


def build_grid(geom: Geometry, smoke: bool = False,
               arms: Optional[List[str]] = None,
               n_roles: Optional[int] = None,
               cases: Optional[list] = None,
               path_alphas: Optional[List[float]] = None,
               n_pairs: Optional[int] = None,
               pairs: Optional[List[str]] = None) -> List[Cell]:
    """Every (condition, role) cell, in a fixed deterministic order.

    `arms` restricts the grid. `linear_axis` is the straight-chord control that
    `manifold_axis` has to be measured against, so it is the one that has to run
    first; the same holds for `linear_pair` under `manifold_pair`.

    Args:
        cases: `path_cases.load_cases(geom).cases`. REQUIRED for any path arm,
            because the endpoints are what the arm IS -- there is no default
            chord to fall back on, and inventing one here is precisely how the
            figures would stop describing the run.
        path_alphas: arc positions for the path arms (default PATH_ALPHAS).
        pairs: run exactly these routes, as "A>B" names. Takes precedence
            over n_pairs. `--n-pairs` answers "how many can I afford"; this
            answers "which ones am I asking about", which is what a follow-up
            run targeting a chosen region of the cloud needs. An unknown name
            is fatal rather than silently empty -- a typo'd route would
            otherwise produce a run whose manifest quietly lacks it.
        n_pairs: use only the first N pair cases. The pair grid is
            n_pairs x len(path_alphas) x 2 cells, so this is the knob that keeps
            a preview affordable.
    """
    arms = arms or ARMS
    roles = geom.near50[:2] if smoke else geom.near50
    if n_roles:
        # A preview: FEWER ROLES, every alpha. The dose-response axis is the
        # thing being looked at, so it is the one dimension that must stay
        # complete — truncating alphas instead would produce a figure whose
        # shape is an artefact of the truncation.
        roles = roles[:n_roles]
    alphas = ALPHAS[:2] if smoke else ALPHAS
    palphas = list(path_alphas if path_alphas is not None else PATH_ALPHAS)
    if smoke:
        palphas = palphas[:2]

    wants_path = [a for a in arms if a in PATH_ARMS]
    if wants_path and not cases:
        raise SystemExit(
            "arms %s need the path cases. Pass `cases` from "
            "path_cases.load_cases(geom) — which needs --labels, since "
            "path_cases.fully_only refuses to build a chord on centroids the "
            ">=10 rule discards." % wants_path)
    axis_case = next((c for c in (cases or []) if c[0] == "axis"), None)
    pair_cases = [c for c in (cases or []) if c[0] == "pair"]
    if pairs:
        want = [tuple(x.split(">", 1)) for x in pairs]
        have = {(c[3], c[4]): c for c in pair_cases}
        missing = [w for w in want if w not in have]
        if missing:
            raise SystemExit(
                "unknown pair route(s) %s. Available: %s"
                % (", ".join("%s>%s" % m for m in missing),
                   ", ".join("%s>%s" % k for k in have)))
        pair_cases = [have[w] for w in want]
    elif n_pairs:
        # ROUND-ROBIN OVER SOURCES, not the first N. pick_endpoints emits
        # 3 sources x 5 targets in source-major order, so a plain [:6] gives
        # five routes from one source and one from another -- and the bootstrap
        # for the pair arms resamples PAIRS, so that is near-total confounding
        # with a single source role. Taking them round-robin gives 2 per source
        # for the same count.
        by_src = {}
        for c in pair_cases:
            by_src.setdefault(c[3], []).append(c)          # c[3] is A's name
        picked, i = [], 0
        while len(picked) < n_pairs and any(v[i:] for v in by_src.values()):
            for src in by_src:
                if len(picked) >= n_pairs:
                    break
                if i < len(by_src[src]):
                    picked.append(by_src[src][i])
            i += 1
        pair_cases = picked[:n_pairs]
    if smoke:
        pair_cases = pair_cases[:1]
    if [a for a in arms if a in AXIS_PATH_ARMS] and axis_case is None:
        raise SystemExit("no `axis` case in the case list — pick_endpoints "
                         "always emits one, so this list is not from it")

    cells: List[Cell] = []

    # THE ALPHA = 0 CELL, once, shared by every arm. It has to cover the pair
    # sources too: those come from `pick_endpoints` (the three highest-projecting
    # fully-role-playing centroids) and need not be in `near50`, so building the
    # unsteered set from `roles` alone would leave the pair arms with no baseline.
    if "unsteered" in arms:
        base = list(roles)
        if [a for a in arms if a in PAIR_PATH_ARMS]:
            base += [na for _, _, _, na, _ in pair_cases]
        seen = set()
        for r in base:
            if r not in seen:
                seen.add(r)
                cells.append(Cell("unsteered", "none", 0.0, r, None, None))

    # The legacy dose-matched arms keep the SIGNED dose grid; mixing them onto
    # the arc-position grid would put two different quantities on one x-axis.
    for arm in ("linear_axis_legacy", "manifold_axis_legacy"):
        if arm not in arms:
            continue
        for a in alphas:
            for r in roles:
                cells.append(Cell(arm, "none", a, r, None, None))

    # The axis chord is one global route with no role identity, so it sweeps
    # every role exactly as the paper's targetless intervention does.
    for arm in AXIS_PATH_ARMS:
        if arm not in arms:
            continue
        _, _, _, na, nb = axis_case
        for a in palphas:
            for r in roles:
                cells.append(Cell(arm, "none", a, r, None, None,
                                  path_a=na, path_b=nb))

    # A pair chord runs FROM one persona TO another, so it is generated under
    # the A persona's own system prompts: the displacement moves A's cloud
    # toward B, and applying it to a model prompted as some third role would be
    # a different experiment wearing this one's name.
    for arm in PAIR_PATH_ARMS:
        if arm not in arms:
            continue
        for a in palphas:
            for _, _, _, na, nb in pair_cases:
                cells.append(Cell(arm, "pair", a, na, nb, None,
                                  path_a=na, path_b=nb))
    return cells


# The arms with no per-token callback: `unsteered` adds nothing, and
# `linear_axis_legacy` is a single fixed vector on the vendored `addition` path.
# Named once because both `make_delta_fn` and the delta-stats instrumentation
# key off it.
#
# THE NEW `linear_axis` IS NOT HERE. Its delta is also one fixed vector, but it
# goes through the `dynamic` hook like its manifold partner, so the two arms of
# the pair differ in the route and in nothing else -- not in which hook ran, not
# in whether DeltaStats saw them. The vendored `addition` path stays reachable
# under the legacy name, which is the arm that was a literal replication.
STATIC_ARMS = ("unsteered", "linear_axis_legacy")


def path_for(cell: Cell, paths: Optional[dict]):
    """The path object this cell steers along.

    Looked up, never rebuilt: `paths` is the bank `path_cases.build_paths` made
    from the case list, so the route a cell takes is the route the figures drew.
    A KeyError here means the grid and the bank were built from different case
    lists, which is the one failure that must not be recoverable.
    """
    family = "manifold" if cell.arm.startswith("manifold") else "linear"
    if not paths:
        raise ValueError("arm %r needs the path bank (path_cases.build_paths)"
                         % cell.arm)
    key = (family, cell.path_a, cell.path_b)
    if key not in paths:
        raise KeyError("no path for %r — the grid and the path bank disagree "
                       "about the cases" % (key,))
    return paths[key]


def make_delta_fn(cell: Cell, geom: Geometry, dtype, stats=None,
                  paths: Optional[dict] = None):
    """The intervention for one cell, or None for an unsteered/static cell."""
    if cell.arm in STATIC_ARMS:
        return None
    # THE FOUR PATH ARMS. Both members of a pair get the same alpha and the same
    # two endpoints; only the object handed to the factory differs, and the
    # factories type-check it so an arm cannot quietly become its partner.
    if cell.arm in ("linear_axis", "linear_pair"):
        return IV.make_linear_path_delta_fn(path_for(cell, paths), cell.alpha,
                                            dtype=dtype, stats=stats)
    if cell.arm in ("manifold_axis", "manifold_pair"):
        return IV.make_manifold_path_delta_fn(path_for(cell, paths), cell.alpha,
                                              dtype=dtype, stats=stats)
    if cell.arm == "linear_contrast":
        return IV.make_linear_contrast_delta_fn(geom.centroid(cell.role),
                                                geom.centroid(cell.target_role),
                                                cell.alpha, dtype=dtype, stats=stats)
    if cell.arm == "manifold_contrast":
        return IV.make_manifold_contrast_delta_fn(geom.spline, geom.axis_unit,
                                                  geom.centroid(cell.role),
                                                  geom.centroid(cell.target_role),
                                                  cell.alpha, dtype=dtype, stats=stats)
    if cell.arm == "linear_journey":
        return IV.make_linear_journey_delta_fn(geom.centroid(cell.target_role),
                                               cell.alpha, dtype=dtype, stats=stats)
    if cell.arm == "manifold_journey":
        return IV.make_manifold_journey_delta_fn(geom.spline, geom.axis_unit,
                                                 geom.centroid(cell.target_role),
                                                 cell.alpha, dtype=dtype, stats=stats)
    if cell.arm == "linear_target":
        return IV.make_arm2_delta_fn(geom.centroid(cell.target_role), cell.alpha,
                                     geom.n_bar, dtype=dtype, stats=stats)
    if cell.arm == "manifold_target":
        return IV.make_arm3_delta_fn(geom.spline, geom.axis_unit,
                                     geom.centroid(cell.target_role), cell.alpha,
                                     geom.n_bar, dtype=dtype, stats=stats)
    if cell.arm == "manifold_axis_legacy":
        return IV.make_manifold_axis_delta_fn(
            geom.spline, geom.axis_unit, cell.alpha, geom.n_bar, geom.span,
            dtype=dtype, stats=stats)
    raise ValueError("unknown arm %r" % cell.arm)


def make_static_vector(cell: Cell, geom: Geometry) -> Optional[np.ndarray]:
    """The fixed vector for the arm that has one (the legacy replication)."""
    if cell.arm == "linear_axis_legacy":
        return IV.linear_axis_vector(geom.axis_unit, cell.alpha, geom.n_bar)
    return None


def default_batch_size(device: str) -> int:
    """Prompts per forward pass.

    MPS IS CAPPED AT 1, and this is not a performance choice. Measured
    2026-08-17 on Qwen2.5-3B: a LEFT-PADDED batch of 2 or more on MPS produces
    non-finite logits and the model emits `!!!!!!` (token id 0) — at every
    dtype (fp16 / bf16 / fp32) and with both `sdpa` and `eager` attention.
    Batch 1 is fine, and CPU float32 at batch 20 is fine, so it is an MPS
    padded-attention bug rather than precision or our code. See Observations O3.

    CUDA is unaffected, so the main run batches normally.
    """
    return 1 if device == "mps" else 20


def _assert_finite_logits(model, enc) -> None:
    """Fail loudly rather than record `!!!!` as if it were a model behaviour."""
    import torch
    with torch.no_grad():
        lg = model(**enc).logits
    if not torch.isfinite(lg).all():
        raise RuntimeError(
            "non-finite logits before generation (batch=%d, seq=%d, device=%s). "
            "On MPS this is the padded-batch bug — use batch_size=1 (see "
            "default_batch_size)." % (enc["input_ids"].shape[0],
                                      enc["input_ids"].shape[1], model.device.type))


def generate_cell(cell: Cell, geom: Geometry, model, tokenizer, hook_layer: int,
                  batch_size: int = None, max_new_tokens: int = GEN_CEILING,
                  prompt_bank: Optional[Dict[str, List[str]]] = None,
                  paths: Optional[dict] = None) -> pd.DataFrame:
    """Generate all 20 responses for one (condition, role) cell."""
    import torch

    systems = load_system_prompts(cell.role, prompt_bank=prompt_bank)
    prompts, rows = [], []
    for si, sysmsg in enumerate(systems):
        for qi, q in enumerate(INTROSPECTIVE_QUESTIONS):
            msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
                   [{"role": "user", "content": q}]
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            rows.append({"system_idx": si, "question_idx": qi, "question": q,
                         "system": sysmsg})

    dtype = next(model.parameters()).dtype
    static = make_static_vector(cell, geom)
    # Every arm that HAS a delta_fn is instrumented, derived from `make_delta_fn`'s
    # own guard rather than a second hand-kept list of arm names: an arm added
    # there but forgotten here would silently lose its WP5 record.
    stats = None if cell.arm in STATIC_ARMS else IV.DeltaStats()
    delta_fn = make_delta_fn(cell, geom, dtype, stats=stats, paths=paths)
    bs = batch_size or default_batch_size(model.device.type)

    # Seed PER CELL, derived from the cell's own identity rather than a global
    # counter. Sampling made the run stochastic, and a resumed shard must
    # reproduce the text it would have produced first time -- a global seed
    # advanced by batch order would not, because resume skips completed cells
    # and changes that order.
    cell_seed = (GEN_SEED + int(hashlib.sha256(
        cell.shard_name.encode()).hexdigest()[:8], 16)) % (2 ** 31 - 1)

    def _run(enc):
        torch.manual_seed(cell_seed)
        with torch.no_grad():
            return model.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=DO_SAMPLE,
                                  temperature=TEMPERATURE,
                                  top_p=TOP_P,
                                  pad_token_id=tokenizer.pad_token_id)

    t0 = time.time()
    texts, ntoks = [], []
    for s0 in range(0, len(prompts), bs):
        enc = tokenizer(prompts[s0:s0 + bs], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        _assert_finite_logits(model, enc)
        if static is not None:
            with ActivationSteering(model, steering_vectors=[static], coefficients=[1.0],
                                    layer_indices=[hook_layer],
                                    intervention_type="addition", positions="all"):
                out = _run(enc)
        elif delta_fn is not None:
            with ActivationSteering(model, intervention_type="dynamic", delta_fn=delta_fn,
                                    layer_indices=[hook_layer], positions="all"):
                out = _run(enc)
        else:
            out = _run(enc)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        ntoks.extend([int((g != tokenizer.pad_token_id).sum()) for g in gen])
    elapsed = time.time() - t0

    for row, text, nt in zip(rows, texts, ntoks):
        row.update({
            "arm": cell.arm, "target_distance": cell.target_distance,
            "alpha": cell.alpha, "role": cell.role,
            "target_role": cell.target_role or cell.role,
            # The route, in the row itself. A figure that groups by arm and
            # alpha alone would silently average 15 different pair chords.
            "path_a": cell.path_a, "path_b": cell.path_b,
            "negctl_seed": cell.seed,
            "response": text,
            "n_new_tokens": nt,
            # Generation is uncapped in intent; this flags the rows where the
            # runaway guard, not the model, decided where to stop. A judge
            # scoring a truncated response as `nonsensical` is measuring our
            # ceiling, so the fraction has to be visible.
            "hit_ceiling": bool(nt >= max_new_tokens),
            "status": "ok",
        })
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = elapsed
    df.attrs["delta_stats"] = stats.report() if stats is not None else None
    return df


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via a private temp file + rename, so a reader never sees a partial.

    The run-level artifacts are written at the end of a sharded run, where two
    workers finishing within the same instant could otherwise interleave into
    one parquet and leave a truncated file that still has a valid name. Rename
    within a directory is atomic on POSIX, and the pid in the temp name keeps
    concurrent writers off each other's scratch file.
    """
    tmp = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_json_atomic(obj, path: Path) -> None:
    """Same discipline as `_write_parquet_atomic`, for the checkpoint config.

    Under --num-shards every worker reaches the config write, and a plain
    write_text truncates before it writes: two workers landing together can
    leave a torn config.json that the next resume fails to parse.
    """
    tmp = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    try:
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run dir (created if absent)")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, see plan")
    ap.add_argument("--limit", type=int, default=None, help="stop after N cells")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="prompts per forward pass (default: 1 on mps, else 20; see O3)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of %s (default: all). Run "
                         "'unsteered,linear_axis' first: the straight chord is "
                         "the control the curved one is read against. Also "
                         "valid, on their own grids: %s (the pre-rebuild dose-"
                         "matched forms), %s."
                         % (",".join(ARMS), ",".join(LEGACY_ARMS),
                            ",".join(ABLATION_ARMS + JOURNEY_ARMS + CONTRAST_ARMS)))
    ap.add_argument("--prompts", default=None,
                    help="role_prompts_eval.json from steering.gen_role_prompts "
                         "(WP3). Without it the run REUSES the extraction prompts.")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_CEILING,
                    help="runaway ceiling, not a content cap (default %d); "
                         "generation stops at EOS well before it" % GEN_CEILING)
    # WHICH CLOUD. Defaults to the resp240 (Qwen2.5-3B) cloud that
    # geometry.RESP240_DIR names. A run against a different model MUST pass
    # this: without it the axis, the centroids and the near-50 are built
    # from a different model's activations than the one being steered, and
    # nothing downstream can detect it.
    ap.add_argument("--resp-dir", default=None,
                    help="response cloud directory (default: the resp240 3B cloud)")
    ap.add_argument("--labels", default=None,
                    help="role_labels.parquet from steering.rolefilter (WP1): "
                         "builds the axis and centroids from `fully` rows only")
    ap.add_argument("--n-bar", default=None,
                    help="n_bar_lmsys.json from steering.lmsys_norm (WP2)")
    # SHARDING, the same discipline extraction uses. One process on 2 pipeline-
    # parallel P100s measured 3.8 min per cell, i.e. ~60 h for the 950-cell grid
    # against a 24 h job limit. Cells are already independent and individually
    # checkpointed, so N workers round-robin over them and share one run dir; a
    # cell another worker finished is skipped, exactly as on resume.
    # Round-robin, not contiguous blocks, so every worker gets a mix of arms and
    # alphas and no single worker owns all the slow high-|alpha| cells.
    ap.add_argument("--fractions", default=None,
                    help="comma-separated f grid for the contrast/journey arms, "
                         "overriding the built-in one. SIGNED: with --far-roles "
                         "default, negative f steers AWAY from the Assistant, "
                         "which is the paper's negative-alpha direction.")
    ap.add_argument("--ablation", action="store_true",
                    help="plan step 7: steer --near-roles toward --far-roles with "
                         "the TARGET-DIRECTED arms instead of the Figure-4 grid")
    ap.add_argument("--near-roles", default=None, help="comma-separated")
    ap.add_argument("--far-roles", default=None, help="comma-separated")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    # Aggregation is a WHOLE-GRID step: it globs every shard in the run dir.
    # When all N workers fell through to it, each wrote generations_L19.parquet
    # from whatever shards existed at that instant and the file was whichever
    # worker finished last -- a silent partial. Aggregation is now gated on the
    # grid being complete, so the last worker to finish does it; --collect is
    # the explicit pass for when no worker was last (a kill, a job timeout).
    ap.add_argument("--collect", action="store_true",
                    help="aggregate finished shards into the run-level "
                         "artifacts and exit; loads no model")
    ap.add_argument("--allow-partial", action="store_true",
                    help="aggregate even with cells missing. The artifacts then "
                         "describe part of the grid — for debugging, not results")
    ap.add_argument("--n-roles", type=int, default=None,
                    help="preview: use only the first N near-Assistant roles, "
                         "keeping every alpha (the dose-response axis stays whole)")
    # THE PATH CONSTRUCTION. Every default here is `steering.path_cases`'s, which
    # is also where `steering.geometry_check` gets its defaults, so the figures
    # that justified eps and k describe the run unless a flag says otherwise.
    # They are PINNED into the checkpoint config: change one and the cells mean
    # something else, so a run dir must not mix two values of any of them.
    ap.add_argument("--eps", type=float, default=PC.EPS,
                    help="cylinder radius in ABSOLUTE activation units: which "
                         "persona centroids the manifold arm may route through "
                         "(default %g, same as geometry_check --eps-fig2)" % PC.EPS)
    ap.add_argument("--k", type=int, default=PC.K,
                    help="how many persona centroids the curve passes through, "
                         "one per equal span of the chord (default %d)" % PC.K)
    ap.add_argument("--persona-lam", type=float, default=PC.LAM,
                    help="spline bending penalty. 0 = interpolate every chosen "
                         "centroid exactly, still C2-smooth (default %g)" % PC.LAM)
    ap.add_argument("--param", default=PC.PARAM,
                    choices=["projection", "length", "centripetal"],
                    help="knot abscissa for the spline (default %s)" % PC.PARAM)
    ap.add_argument("--eps-mode", default=PC.MODE,
                    choices=["absolute", "relative"],
                    help="`absolute` fixes the tube radius in activation units "
                         "so every chord gets the same physical neighbourhood")
    ap.add_argument("--path-alphas", default=None,
                    help="comma-separated arc positions in [0,1] for the four "
                         "path arms (default %s). alpha=0 is dropped: its delta "
                         "is exactly zero, which is the unsteered cell."
                         % ",".join("%g" % a for a in PATH_ALPHAS))
    ap.add_argument("--pairs", default=None,
                    help="run exactly these pair routes, comma-separated as "
                         "A>B (e.g. assistant>leviathan,assistant>aberration). "
                         "Takes precedence over --n-pairs.")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="use only the first N persona-pair cases from "
                         "path_cases.pick_endpoints (default: all 15)")
    args = ap.parse_args()

    # Resolved ONCE. `arms` is what the grid is built from, what `cfg` records
    # and what the delta-stats collation reads, so parsing it a second time at
    # the ablation call site let the manifest name arms the run never ran:
    # --ablation with no --arms builds the ABLATION_ARMS grid, and cfg said
    # ARMS.
    pair_sel = ([x.strip() for x in args.pairs.split(",")] if args.pairs else None)
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    else:
        arms = list(ABLATION_ARMS if args.ablation else ARMS)
    valid = ARMS + LEGACY_ARMS + ABLATION_ARMS + JOURNEY_ARMS + CONTRAST_ARMS
    bad = [a for a in arms if a not in valid]
    if bad:
        raise SystemExit("unknown arm(s): %s. Valid: %s" % (bad, valid))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.out)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "_ckpt"
    ckpt.mkdir(exist_ok=True)

    path_alphas = ([float(x) for x in args.path_alphas.split(",")]
                   if args.path_alphas else list(PATH_ALPHAS))
    if any(a < 0.0 or a > 1.0 for a in path_alphas):
        raise SystemExit(
            "--path-alphas are ARC POSITIONS in [0,1] along a path with fixed "
            "endpoints, not the legacy signed dose: %s is outside the path. "
            "`at_alpha` clips, so an out-of-range value would silently generate "
            "a duplicate of the endpoint cell." % path_alphas)

    geom = load_geometry(resp_dir=args.resp_dir or RESP240_DIR,
                         labels_path=args.labels, n_bar_path=args.n_bar)
    hook_layer = hook_layer_for_hidden_state(LAYER_HS_INDEX)
    prompt_bank = load_prompt_bank(args.prompts)

    # The steering vector lives in the CLOUD's activation space. If the
    # generation model has a different hidden size the delta cannot be added at
    # all -- and if two models happened to share a hidden size, it would be
    # added silently and mean nothing. Smoke run 11222399 hit this: a 4096-dim
    # Qwen3-8B delta against Qwen2.5-3B's 2048-dim stream, failing inside the
    # hook after the model had loaded.
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        model_hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    except Exception as exc:                                  # noqa: BLE001
        print("WARNING: could not read hidden_size for %s (%s); "
              "skipping the geometry/model dimension check" % (args.model, exc),
              flush=True)
        model_hidden = 0
    if model_hidden and model_hidden != int(geom.hidden):
        raise SystemExit(
            "MODEL/GEOMETRY MISMATCH: --model %s has hidden_size %d but the "
            "cloud at %s has hidden %d. The steering vector is built in the "
            "cloud's space, so these must be the same model. Pass --model for "
            "the model the cloud was extracted from."
            % (args.model, model_hidden, args.resp_dir or RESP240_DIR,
               int(geom.hidden)))

    # THE CASES AND THE PATHS, from the module geometry_check draws from. Built
    # only when a path arm is in play: `path_cases.fully_only` hard-fails
    # without --labels, and the legacy/ablation arms have always been allowed to
    # run on the unfiltered fallback with a recorded deviation.
    cs, paths = None, None
    if any(a in PATH_ARMS for a in arms):
        cs = PC.load_cases(geom)
        print("path cases: %d fully-role-playing centroids, %d dropped; "
              "1 axis + %d pairs"
              % (len(cs.keep), len(cs.dropped), len(cs.cases) - 1), flush=True)
        print("picked:", json.dumps(cs.picked), flush=True)
        paths = PC.build_paths(cs.cases, cs.C, eps=args.eps, k=args.k,
                               lam=args.persona_lam, mode=args.eps_mode,
                               param=args.param)

    for report, flag in ((geom.axis_report, "--labels"),
                         (geom.n_bar_report, "--n-bar")):
        if "deviation" in report:
            print("DEVIATION (%s not given): %s" % (flag, report["deviation"]),
                  flush=True)
    if prompt_bank is None:
        print("DEVIATION (--prompts not given): the eval reuses the extraction "
              "system prompts, so Arms 2/3 aim at centroids built from responses "
              "to these same prompts (Appendix D.1.1 regenerates them).", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.ablation:
        near = [r.strip() for r in (args.near_roles or "").split(",") if r.strip()]
        far = [r.strip() for r in (args.far_roles or "").split(",") if r.strip()]
        if not near or not far:
            raise SystemExit("--ablation needs --near-roles and --far-roles")
        unknown = [r for r in near + far if r not in geom.roles]
        if unknown:
            raise SystemExit("unknown role(s): %s" % unknown)
        # `arms` MUST reach the grid builder. When the flag validated and was
        # then discarded, `--ablation --arms linear_journey,...` silently re-ran
        # the dose-matched ablation under the journey run's name.
        cells = build_ablation_grid(
            near, far, arms=arms,
            fractions=([float(x) for x in args.fractions.split(",")]
                       if args.fractions else None))
        print("ablation grid: %d cells (%s -> %s)" % (len(cells), near, far), flush=True)
    else:
        cells = build_grid(geom, smoke=args.smoke, arms=arms,
                           n_roles=args.n_roles,
                           cases=(cs.cases if cs else None),
                           path_alphas=path_alphas, n_pairs=args.n_pairs,
                               pairs=pair_sel)
    # The UNSHARDED grid. Kept because aggregation must know every cell the run
    # is supposed to contain, not just the ones this worker owns.
    all_cells = cells
    n_all = len(all_cells)
    # Validated unconditionally: nested under `num_shards > 1`, a stray
    # `--shard-index 3` with the default --num-shards 1 was accepted silently
    # and produced a FULL unsharded run under a name implying one quarter of one.
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards): got %d "
                         "with --num-shards %d" % (args.shard_index, args.num_shards))
    if args.num_shards > 1:
        cells = cells[args.shard_index::args.num_shards]
        print("shard %d/%d owns %d of %d cells"
              % (args.shard_index, args.num_shards, len(cells), n_all), flush=True)
    if args.limit:
        cells = cells[:args.limit]

    # PINNED: anything here changes what a generated cell MEANS, so mixing two
    # values of it in one run dir corrupts the run. These must match exactly.
    cfg = {"model": args.model, "smoke": args.smoke,
           "alphas": ALPHAS, "layer_hs_index": LAYER_HS_INDEX,
           "hook_layer": hook_layer, "max_new_tokens": args.max_new_tokens,
           "extraction_max_new_tokens": EXTRACTION_MAX_NEW_TOKENS,
           "n_bar": geom.n_bar, "n_bar_source": geom.n_bar_report.get("source"),
           "axis_definition": geom.axis_report.get("definition"),
           "labels": args.labels, "prompts": args.prompts,
           "lambda": geom.lam,
           # THE PATH CONSTRUCTION, pinned like everything else here: two values
           # of eps in one run dir are two different manifold arms sharing a
           # name. Recorded UNCONDITIONALLY, which is also what stops a run dir
           # written before the linear_axis/manifold_axis rename from resuming:
           # its shards carry those names under the dose-matched intervention,
           # and skipping them as "already done" would mix two experiments.
           "path_eps": args.eps, "path_k": args.k,
           "path_lam": args.persona_lam, "path_param": args.param,
           "path_eps_mode": args.eps_mode, "path_alphas": path_alphas,
           "n_pairs": args.n_pairs, "pairs": pair_sel}
    # SCOPE: which cells the run dir covers. These WIDEN — they do not conflict.
    # Comparing them for equality forbade the workflow --arms' own help text
    # prescribes ("runs 'unsteered,linear_axis' first", then re-run the same dir
    # with manifold_axis added) and the --n-roles preview-then-full path: both
    # died on "checkpoint config mismatch" before generating anything. A cell
    # from a narrower scope is still a valid cell of the wider one, so the
    # recorded scope is the union across invocations.
    #
    # NOT recorded: shard_index, num_shards, n_cells. Every shard writes into
    # the same run dir, so anything that differs between workers would trip the
    # guard and kill all but the first.
    cfg_path = ckpt / "config.json"
    scope = {"arms": sorted(arms), "n_roles": args.n_roles}
    stored = None
    if cfg_path.exists():
        stored = json.loads(cfg_path.read_text())
        old = {k: v for k, v in stored.items() if k not in scope}
        if old != cfg:
            differing = sorted(set(old) | set(cfg))
            detail = "\n".join(
                "    %s: %r -> %r" % (k, old.get(k), cfg.get(k))
                for k in differing if old.get(k) != cfg.get(k))
            raise SystemExit(
                "checkpoint config mismatch — this run dir was started with a "
                "different configuration. Refusing to mix runs.\n%s" % detail)
        # n_roles=None means "every role", which is the widest scope there is,
        # so it absorbs any integer rather than being maxed against it.
        prev_roles = stored.get("n_roles", args.n_roles)
        scope = {"arms": sorted(set(stored.get("arms", [])) | set(arms)),
                 "n_roles": (None if prev_roles is None or args.n_roles is None
                             else max(prev_roles, args.n_roles))}
    # Written only when it actually changes, so the common case — N shard
    # workers passing identical flags — does not write at all.
    merged = {**cfg, **scope}
    if stored != merged:
        _write_json_atomic(merged, cfg_path)

    # The gate below must cover the run dir's WHOLE recorded scope, not just the
    # arms this invocation named. Otherwise `--collect --arms unsteered` on a
    # dir that also holds a half-finished manifold_axis sweep sees its own arm
    # complete and declares the run whole. build_grid is pure python over the
    # already-loaded geometry, so recomputing it here costs nothing.
    if not args.ablation and scope["arms"] != sorted(arms):
        # The recorded scope can name a path arm this invocation did not, and
        # the completeness gate has to count ITS cells too — so the case list
        # must exist here even when `arms` alone would not have needed it.
        wider_cs = cs
        if wider_cs is None and any(a in PATH_ARMS for a in scope["arms"]):
            wider_cs = PC.load_cases(geom)
        all_cells = build_grid(geom, smoke=args.smoke, arms=scope["arms"],
                               n_roles=scope["n_roles"],
                               cases=(wider_cs.cases if wider_cs else None),
                               path_alphas=path_alphas, n_pairs=args.n_pairs,
                               pairs=pair_sel)
        n_all = len(all_cells)

    # THE PAIR ARMS' A ROLES NEED PROMPTS, and they are not chosen from
    # `near50`: `pick_endpoints` takes the three highest-projecting
    # fully-role-playing centroids, which need not overlap it at all. A role
    # with no prompt bank entry (or no instruction file) raises inside
    # `generate_cell`, i.e. after the model is loaded and possibly hours in, so
    # the whole grid's roles are checked here instead.
    if not args.collect:
        need = sorted({c.role for c in all_cells})
        missing_prompts = []
        for r in need:
            try:
                load_system_prompts(r, prompt_bank=prompt_bank)
            except Exception as exc:
                missing_prompts.append("%s (%s)" % (r, exc))
        if missing_prompts:
            raise SystemExit(
                "no evaluation system prompts for %d role(s):\n    %s\n"
                "Re-run steering.gen_role_prompts for them, or drop the arms "
                "that need them."
                % (len(missing_prompts), "\n    ".join(missing_prompts)))

    # WHAT THE ROUTES ACTUALLY ARE, beside the run they produced. detour_ratio,
    # the persona names each curve threads and the knot error are the numbers
    # `geometry_check` gates on; written here so a reader of the run does not
    # have to trust that the figures were regenerated with the same flags.
    if paths is not None:
        manifest = {"eps": args.eps, "k": args.k, "lam": args.persona_lam,
                    "param": args.param, "eps_mode": args.eps_mode,
                    "alphas": path_alphas, "picked": cs.picked,
                    "n_fully_centroids": int(len(cs.keep)),
                    "n_dropped": int(len(cs.dropped)), "routes": []}
        for kind, P0, P1, na, nb in cs.cases:
            row = {"kind": kind, "A": na, "B": nb,
                   "chord_len": float(np.linalg.norm(np.asarray(P1) - np.asarray(P0)))}
            for family in ("linear", "manifold"):
                path = paths[(family, na, nb)]
                row[family] = PC.path_report(path, cs.names)
                # THE CONTROL, measured on the object the run will use rather
                # than asserted: delta(0) must be the zero vector, or every
                # alpha in this cell is offset by a constant nobody chose.
                row[family]["delta0_max_abs"] = float(
                    np.abs(path.delta(0.0)).max())
            manifest["routes"].append(row)
        _write_json_atomic(manifest, run_dir / "data" / "path_manifest.json")

    done = 0
    rate_samples = []
    # --collect aggregates finished shards, so it loads no weights. Deferred to
    # here (it used to sit above the grid build) purely so that stays true.
    if not args.collect:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device).eval()
    for i, cell in enumerate([] if args.collect else cells):
        shard = ckpt / cell.shard_name
        if shard.exists():
            continue
        df = generate_cell(cell, geom, model, tokenizer, hook_layer,
                           batch_size=args.batch_size,
                           max_new_tokens=args.max_new_tokens,
                           prompt_bank=prompt_bank, paths=paths)
        df.to_parquet(shard, index=False)
        # WP5, CHECKPOINTED BESIDE THE SHARD IT DESCRIBES. Accumulating these in
        # a list meant a --resume wrote a delta_stats file covering only the
        # cells that invocation happened to generate, silently replacing the
        # full record with a partial one that still read as a complete artifact.
        rep = df.attrs.get("delta_stats")
        if rep:
            (ckpt / (cell.shard_name + ".delta.json")).write_text(json.dumps(
                {"arm": cell.arm, "alpha": cell.alpha, "role": cell.role,
                 "target_distance": cell.target_distance,
                 "path_a": cell.path_a, "path_b": cell.path_b, **rep}))
        done += 1
        n = len(df)
        rate_samples.append(n / max(df.attrs["elapsed_s"], 1e-9))
        if done % 10 == 0 or args.smoke:
            print("[%d/%d] %s  %.2f gen/s" % (i + 1, len(cells), cell.shard_name,
                                              rate_samples[-1]), flush=True)

    # THE COMPLETENESS GATE. Everything below globs the whole checkpoint dir, so
    # under --num-shards N every worker used to write these run-level artifacts
    # from whatever shards existed when IT finished. The surviving file was
    # whichever worker was last, containing only the shards done by then, and
    # nothing downstream could tell a partial grid from a whole one.
    #
    # Judged per CELL, not by counting files: a stale shard from an earlier
    # narrower --arms invocation would make a count agree while cells this run
    # needs are still missing.
    missing = [c.shard_name for c in all_cells if not (ckpt / c.shard_name).exists()]
    if missing and not args.allow_partial:
        print("NOT AGGREGATING — %d of %d cells missing (e.g. %s).\n"
              "  Shards on disk are safe; nothing was lost. Once every worker "
              "has finished, run the same command with --collect to write the "
              "run-level artifacts (or --allow-partial to write what exists, "
              "which is NOT a result)."
              % (len(missing), len(all_cells), ", ".join(missing[:3])), flush=True)
        return
    if missing:
        print("WARNING: aggregating with %d of %d cells missing (--allow-partial)."
              % (len(missing), len(all_cells)), flush=True)

    shards = sorted(ckpt.glob("*.parquet"))
    all_df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    out_path = run_dir / "data" / "generations_L19.parquet"
    _write_parquet_atomic(all_df, out_path)

    rate = float(np.mean(rate_samples)) if rate_samples else None
    hit = float(all_df["hit_ceiling"].mean()) if "hit_ceiling" in all_df else None
    (run_dir / "data" / "generation_rate.json").write_text(json.dumps({
        "generations_per_second": rate,
        "device": device,
        "cells_generated_this_invocation": done,
        "total_rows": int(len(all_df)),
        "max_new_tokens": args.max_new_tokens,
        "frac_hit_ceiling": hit,
        "mean_new_tokens": float(all_df["n_new_tokens"].mean()),
        # DERIVED from the grid this run actually built. The old constant was
        # the legacy Figure-4 grid, and the path grid's size depends on how many
        # pair cases there are, so quoting the constant would misreport the run.
        "full_grid_rows": len(all_cells) * ROWS_PER_CELL,
        "projected_hours_full_grid": ((len(all_cells) * ROWS_PER_CELL)
                                      / rate / 3600) if rate else None,
        # Recorded, not inferred: with the gate above these are 0/False for any
        # normal run, so a downstream reader never has to guess whether the
        # artifact covers the whole grid.
        "cells_expected": len(all_cells),
        "cells_missing": len(missing),
        "partial": bool(missing),
    }, indent=2))

    # WP5: what the steered arms actually did, per cell — collated from the
    # per-shard sidecars, so a resumed run reports every cell rather than only
    # the ones it generated. Written even when empty so its absence is never
    # mistaken for "the arm behaved".
    #
    # Gated on THE MECHANISM, not on one arm's name: every arm outside
    # STATIC_ARMS is instrumented, so keying this off "manifold_axis" dropped
    # the journey, contrast and target runs' records on the floor and never
    # fired the warning either.
    instrumented = [a for a in arms if a not in STATIC_ARMS]
    if instrumented:
        delta_stats = [json.loads(p.read_text())
                       for p in sorted(ckpt.glob("*.delta.json"))]
        frame = (pd.DataFrame(delta_stats) if delta_stats else
                 pd.DataFrame(columns=["arm", "alpha", "role",
                                       "target_distance", "n_positions"]))
        _write_parquet_atomic(frame, run_dir / "data" / "delta_stats_L19.parquet")
        if not delta_stats:
            print("WARNING: %s were in --arms but NOT ONE cell recorded delta "
                  "stats. frac_out_of_knots is the check that a manifold arm is "
                  "not degenerating into a linear one; an empty file means that "
                  "check did not run." % ", ".join(instrumented), flush=True)

    print("wrote %s  (%d rows, %.2f gen/s)" % (out_path, len(all_df), rate or float("nan")))
    if hit:
        print("NOTE: %.1f%% of responses hit the %d-token ceiling — raise "
              "--max-new-tokens, those are truncations, not endings."
              % (100 * hit, args.max_new_tokens))


if __name__ == "__main__":
    main()
