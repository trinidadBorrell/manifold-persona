"""Central configuration and paths for manifold-persona."""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo layout ----------------------------------------------------------
# This file lives at: <repo>/src/manifold_persona/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Sibling persona_vectors repo (source of trait artifacts + recycled code).
PERSONA_VECTORS_DIR = Path(
    os.environ.get("PERSONA_VECTORS_DIR", REPO_ROOT.parent / "persona_vectors")
)
TRAIT_ARTIFACTS_DIR = PERSONA_VECTORS_DIR / "data_generation" / "trait_data_extract"

# --- Data / outputs -------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"                 # persona-vectors study
ROLE_EMBEDDINGS_DIR = DATA_DIR / "embeddings_roles"      # assistant-axis roles study (prompt tokens)
RESP_ROLE_EMBEDDINGS_DIR = DATA_DIR / "embeddings_roles_resp"  # roles study, RESPONSE tokens (paper-matched)
AA_TRAIT_EMBEDDINGS_DIR = DATA_DIR / "embeddings_aa_traits"  # assistant-axis traits study
# Per-study figures live at exploratory/<study>/figures/<stamp>/ (see
# manifold_persona.common.FIGURES_DIR). This repo-level default named a path
# that has never existed and had no importer.
FIGURES_DIR = REPO_ROOT / "exploratory"

# --- assistant-axis source --------------------------------------------------
# Roles + extraction questions are VENDORED under refs/assistant-axis/ (see its
# PROVENANCE.md), so extraction runs with no sibling checkout. Setting
# ASSISTANT_AXIS_DIR overrides to a live checkout, which uses the upstream
# `data/` layout. The vendored copy drops that `data/` segment: the repo's
# unanchored `.gitignore` `data/` rule would otherwise untrack it.
VENDORED_AXIS_DIR = REPO_ROOT / "refs" / "assistant-axis"
_axis_override = os.environ.get("ASSISTANT_AXIS_DIR")

if _axis_override:
    ASSISTANT_AXIS_DIR = Path(_axis_override)
    ROLE_INSTRUCTIONS_DIR = ASSISTANT_AXIS_DIR / "data" / "roles" / "instructions"
    ROLE_QUESTIONS_FILE = ASSISTANT_AXIS_DIR / "data" / "extraction_questions.jsonl"
    # 240 behavioural traits (adjectives; bipolar pos/neg + 40 questions each).
    # Not vendored — only the deleted trait-extraction scripts used these.
    AA_TRAIT_INSTRUCTIONS_DIR = ASSISTANT_AXIS_DIR / "data" / "traits" / "instructions"
else:
    ASSISTANT_AXIS_DIR = VENDORED_AXIS_DIR
    ROLE_INSTRUCTIONS_DIR = VENDORED_AXIS_DIR / "roles" / "instructions"
    ROLE_QUESTIONS_FILE = VENDORED_AXIS_DIR / "extraction_questions.jsonl"
    AA_TRAIT_INSTRUCTIONS_DIR = VENDORED_AXIS_DIR / "traits" / "instructions"  # not vendored

# --- Model ----------------------------------------------------------------
MODEL_NAME = os.environ.get("MP_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

# The 7 persona-vector traits (extraction set). Order is fixed for reproducibility.
TRAITS = [
    "evil",
    "sycophantic",
    "hallucinating",
    "apathetic",
    "optimistic",
    "humorous",
    "impolite",
]

# --- Hugging Face ---------------------------------------------------------
HF_TOKEN_PATH = REPO_ROOT / "token" / "huggingface.txt"
HF_REPO_NAME = "manifold-persona"  # dataset repo id becomes <username>/manifold-persona


# --- Analysis depths --------------------------------------------------------
# Depths are FRACTIONS of decoder depth, not raw layer indices: "layer 19"
# means different depths in a 0.5B (24 blocks) vs a 3B (36 blocks) model,
# while "0.5 depth" transfers. Fractions live in BLOCK space (assistant-axis
# convention: the activation is the output of ``model.model.layers[k]``). Use
# :func:`depth_hidden_state` to index HF ``output_hidden_states``.
#
# For Qwen2.5-3B-Instruct (36 blocks) these give hidden states 10 / 19 / 26.
# "mid" and "late" reproduce the two existing clouds (resp40q-L19,
# prompt-L26); "late" ~ Persona Vectors' layer 20/28 depth.
DEPTH_FRACTIONS = {"early": 0.25, "mid": 0.5, "late": 0.70}


def depth_block(fraction: float, num_hidden_layers: int) -> int:
    """Decoder-block index at ``fraction`` of the model's depth."""
    return max(0, min(round(fraction * num_hidden_layers),
                      num_hidden_layers - 1))


def depth_hidden_state(fraction: float, num_hidden_layers: int) -> int:
    """``output_hidden_states`` index at ``fraction`` of the model's depth.

    ``hidden_states[i]`` is the output of block ``i - 1`` (index 0 is the
    embedding), so this is :func:`depth_block` + 1.
    """
    return depth_block(fraction, num_hidden_layers) + 1


def primary_layer(num_hidden_layers: int) -> int:
    """Paper-equivalent analysis layer.

    Persona Vectors analyses/steers Qwen2.5-7B (28 layers) at layer 20
    (~0.71 depth). We scale that ratio to the current model's depth. Hidden
    states are indexed 0..num_hidden_layers (0 = embeddings), so we clamp
    into that range.
    """
    ratio = 20 / 28
    layer = round(ratio * num_hidden_layers)
    return max(0, min(layer, num_hidden_layers))


def half_depth_layer(num_hidden_layers: int) -> int:
    """~0.5-depth **decoder-block** index, matching the Assistant Axis paper.

    The paper analyses at ~half depth (Gemma-2-27B L22/46, Qwen3-32B L32/64,
    Llama-3.3-70B L40/80 -> all ≈0.5).

    NOTE: this returns a *block* index, in assistant-axis's convention, where
    the activation is the output of ``model.model.layers[k]``. It is NOT a
    ``hidden_states`` index -- see :func:`half_depth_hidden_state`. Using this
    value to index ``output_hidden_states`` reads one block too shallow.
    """
    return depth_block(DEPTH_FRACTIONS["mid"], num_hidden_layers)


def half_depth_hidden_state(num_hidden_layers: int) -> int:
    """~0.5-depth layer as an index into HF ``output_hidden_states``.

    ``output_hidden_states`` returns ``num_hidden_layers + 1`` tensors where
    ``hidden_states[0]`` is the embedding output and ``hidden_states[i]`` is the
    output of decoder block ``i-1``. assistant-axis hooks block ``k`` directly
    (``pipeline/2_activations.py``), so:

        assistant-axis block k  ==  hidden_states[k + 1]

    For Qwen2.5-3B-Instruct (36 blocks): block 18 -> hidden_states[19].
    """
    return min(half_depth_layer(num_hidden_layers) + 1, num_hidden_layers)
