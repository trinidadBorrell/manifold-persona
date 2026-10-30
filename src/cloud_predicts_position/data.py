"""Loaders for the full-design clouds (plan 2026-10-29, section 1).

276 roles x 5 system prompts x 240 questions = 331,200 answers of Qwen3-8B, response-token mean
(`prompt_avg.npy`) and last response token (`prompt_last.npy`), fp16, shape (331200, n_layers, 4096),
128-token cap, greedy. Layer 19 is in one dataset folder, layers 25 and 32 in another; both have the
same row order (checked 2026-10-29).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import json
import os
import random

ROOT = Path("/data/project/eeg_foundation/data/manifold_persona")
PM = ROOT / "personamanifold"
OUT = PM / "CPP"
LAYERS = {19: (ROOT / "embeddings_roles_qwen3_8b", 0),
          25: (ROOT / "embeddings_roles_qwen3_8b_L25L32", 0),
          32: (ROOT / "embeddings_roles_qwen3_8b_L25L32", 1)}
VIEWS = {"avg": "prompt_avg.npy", "last": "prompt_last.npy"}
DEFAULT = "default"
N_PROMPTS, N_QUESTIONS = 5, 240
Q_SUBSET = None          # question_idx values kept (None = all 240)

# Other sources (Part M, plan section 5.4), set through the environment before import:
#   CPP_CLOUD_DIR  a cloud folder written by jobs_condor/shard_extract.py (manifest `source_layers` gives the layer index)
#   CPP_QSUBSET    "20" = the seed-0 sample of 20 of the 240 questions that `--n_questions 20` draws
#                  (manifold_persona.prompts_roles.build_role_records), or an explicit comma list of question_idx
#   CPP_OUT        output root (default $PM/CPP)
if os.environ.get("CPP_CLOUD_DIR"):
    _d = Path(os.environ["CPP_CLOUD_DIR"])
    _man = json.loads((_d / "manifest.json").read_text())
    LAYERS = {int(L): (_d, i) for i, L in enumerate(_man["source_layers"])}
    N_QUESTIONS = int(_man.get("n_questions", N_QUESTIONS))
if os.environ.get("CPP_QSUBSET"):
    _q = os.environ["CPP_QSUBSET"]
    Q_SUBSET = sorted(random.Random(0).sample(range(240), int(_q))) if _q.isdigit() else sorted(int(x) for x in _q.split(","))
    N_QUESTIONS = len(Q_SUBSET)
if os.environ.get("CPP_OUT"):
    OUT = Path(os.environ["CPP_OUT"])


def _filter(m):
    """Keep the question subset; the index stays the row index of the arrays."""
    if Q_SUBSET is not None:
        m = m[m.question_idx.isin(Q_SUBSET)]
    return m


def metadata(layer=25):
    d, _ = LAYERS[layer]
    return _filter(pd.read_csv(d / "metadata.csv", usecols=["role", "instruction_idx", "question_idx"]))


def response_lengths(layer=25):
    d, _ = LAYERS[layer]
    m = _filter(pd.read_csv(d / "metadata.csv", usecols=["role", "instruction_idx", "question_idx", "response"]))
    m["chars"] = m.response.fillna("").str.len()
    return m.groupby("role").chars.mean()


def open_layer(layer, view="avg"):
    """Memory-mapped (331200, 4096) fp16 for one layer and view."""
    d, idx = LAYERS[layer]
    A = np.load(d / VIEWS[view], mmap_mode="r")
    return A[:, idx, :]


def grid_rows(meta, role):
    """Row indices of one role, ordered by (instruction_idx, question_idx) -> (5*240,) and the two factors."""
    m = meta[meta.role == role].sort_values(["instruction_idx", "question_idx"])
    assert len(m) == N_PROMPTS * N_QUESTIONS, (role, len(m))
    return m.index.to_numpy(), m.instruction_idx.to_numpy(), m.question_idx.to_numpy()


def cloud(A, meta, role):
    """(1200, 4096) float32 cloud of one role, grid-ordered, with instr and quest factors."""
    rows, instr, quest = grid_rows(meta, role)
    order = np.argsort(rows)                              # mmap fancy indexing wants sorted rows; restore the grid order after
    X = np.asarray(A[rows[order]], dtype=np.float32)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return X[inv], instr, quest


def roles(meta):
    return sorted(meta.role.unique())
