"""Everything the ACE study's scripts must agree on: routes, layers, alphas,
prompts, seeds, the geometry and cloud behind each layer, and the output root.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from steering_core.run_steering import INTROSPECTIVE_QUESTIONS

_REPO = Path(__file__).resolve().parents[2]
_DATA = Path("/data/project/eeg_foundation/data/manifold_persona")

OUT_ROOT = Path(os.environ.get("ACE_OUT", _REPO / "output" / "ace-steering-29_09"))

# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

QUESTIONS = list(INTROSPECTIVE_QUESTIONS)           # App. D.1.2, the five identity questions
N_PROMPTS = 5                                       # system prompts per source persona, as the
                                                    # Assistant Axis paper (2601.10387 s2.1.1)
N_SAMPLES = {1: 3, 2: 1, 3: 1, 5: 1}                # answers per (prompt, question), by step
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.9, 20            # top_k 20 = Qwen3's generation_config,
                                                    # which every earlier sampled run used
MAX_NEW_TOKENS = 512

K = 8                                               # interior knots -> 10 route points
KNOT_RULE = "distance"                              # manifold_v1

SIGNED_ALPHAS = tuple(round(x, 2) for x in np.arange(-1.5, 1.5001, 0.25))   # linear, ACE
MANIFOLD_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)                               # arc position

STEP1_LAYER = 19
STEP1_ROUTES = ["assistant>bard", "assistant>vampire", "assistant>leviathan"]
SWEEP_LAYERS = (15, 19, 23, 25, 27, 31, 32)

# A 4-gram repeated more than this many times marks the answer `collapsed`: its own
# class, never sent to the judge. The follow-ups used 10 for 256 tokens; scaled
# with the budget.
REP4_COLLAPSED = 20

SEED_BASE = 20260929


def seed_for(route: str, layer: int, sample: int) -> int:
    """One torch seed per (route, layer, sample) BATCH, shared by every arm and alpha.

    Common random numbers: the 25 answers of sample s see the same sampling
    stream in every cell of a route, so a rollout (prompt, question, sample) is
    comparable across alphas and arms -- they differ only through the steering.
    Keyed by the route's NAME (crc32), not its position in a list, so step 2's
    sample 0 at layer 19 is the same draw as step 1's sample 0 and can be reused.
    """
    import zlib
    return (SEED_BASE + (zlib.crc32(route.encode()) % 100000) * 1000
            + 10 * int(layer) + int(sample))


# ---------------------------------------------------------------------------
# Where each layer's geometry and cloud live
# ---------------------------------------------------------------------------

_LABELS_FULL = _DATA / "steering/2026-09-10T15-46-fig4/data/role_labels.parquet"
_SUBSAMPLED = _DATA / "steering/2026-09-01-qwen3-8b/layers"

# (resp_dir, layer index inside that cloud, labels) for load_geometry / the PCA clouds
CLOUDS = {
    19: (_DATA / "embeddings_roles_qwen3_8b", 0, _LABELS_FULL),
    25: (_DATA / "embeddings_roles_qwen3_8b_L25L32", 0, _LABELS_FULL),
    32: (_DATA / "embeddings_roles_qwen3_8b_L25L32", 1, _LABELS_FULL),
}
for _L in (15, 23, 27, 31):
    CLOUDS[_L] = (_SUBSAMPLED / ("layer%d" % _L), 0, _SUBSAMPLED / ("layer%d" % _L) / "role_labels.parquet")

GEOM_DIR = OUT_ROOT / "geom"


def geom_path(layer: int) -> Path:
    """The centroid cache steered through at this layer.

    19, 25 and 32 are the caches the earlier runs used, so their routes are the
    routes those runs had. The other four are built by jobs_condor/ace_geom.py
    from the subsampled clouds (80 rows per role x label) with the same code.
    """
    fixed = {19: _REPO / "output/steering-manifold/2026-09-17T01-01-chunk-ablation/data/geom_cache.npz",
             25: _DATA / "geom_l25l32/geom_cache_L25.npz",
             32: _DATA / "geom_l25l32/geom_cache_L32.npz"}
    return fixed.get(int(layer), GEOM_DIR / ("geom_cache_L%d.npz" % int(layer)))


# Step 3 (follow-up A, 2026-10-02): other sources near the assistant, hard and easy targets.
STEP3_ROUTES = ["%s>%s" % (s, t) for s in ("validator", "summarizer")
                for t in ("vampire", "criminal", "void", "bard", "leviathan", "tree")]
STEP3_LAYERS = (19, 25)
# Step 5 (follow-up D): activation capping as a floor, applied at these layers at once.
STEP5_LAYERS = (19, 23, 25, 27)
STEP5_PERCENTILES = (1, 25, 50, 75)


def sweep_routes() -> list:
    """Step 2's ten routes, fixed by select_routes.py and confirmed by the user."""
    p = OUT_ROOT / "routes_step2.json"
    if not p.exists():
        raise FileNotFoundError("%s missing: run src/ace_steering/select_routes.py" % p)
    return json.loads(p.read_text())["routes"]


def routes_for(step: int) -> list:
    step = int(step)
    if step == 1:
        return STEP1_ROUTES
    if step == 3:
        return STEP3_ROUTES
    if step == 5:
        return [r for r in sweep_routes() if not r.endswith(">assistant") and r.split(">")[0] in
                ("assistant", "grader", "examiner", "scheduler")]
    return sweep_routes()


def layers_for(step: int) -> tuple:
    return {1: (STEP1_LAYER,), 3: STEP3_LAYERS, 5: (STEP5_LAYERS[0],)}.get(int(step), SWEEP_LAYERS)


def route_slug(route: str) -> str:
    return route.replace(">", "-")
