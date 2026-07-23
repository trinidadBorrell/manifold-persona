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

# --- assistant-axis source (sibling repo) ----------------------------------
ASSISTANT_AXIS_DIR = Path(
    os.environ.get("ASSISTANT_AXIS_DIR", REPO_ROOT.parent / "assistant-axis")
)
# 276 character-role personas (identities; pos-only)
ROLE_INSTRUCTIONS_DIR = ASSISTANT_AXIS_DIR / "data" / "roles" / "instructions"
ROLE_QUESTIONS_FILE = ASSISTANT_AXIS_DIR / "data" / "extraction_questions.jsonl"
# 240 behavioural traits (adjectives; bipolar pos/neg + 40 questions each)
AA_TRAIT_INSTRUCTIONS_DIR = ASSISTANT_AXIS_DIR / "data" / "traits" / "instructions"

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
    return max(0, min(round(0.5 * num_hidden_layers), num_hidden_layers))


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
