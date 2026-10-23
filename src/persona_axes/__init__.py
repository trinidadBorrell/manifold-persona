"""persona_axes -- interpretable persona axes (plan 2026-10-20-principal-axes-v2, Stage A: Phases 0-2).

Constants shared by every module. Everything new is written under $PM/AX21; reused inputs
($PM/E1, $PM/E1b, run-2 answers) are read in place.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
AX = PM / "AX21"
UPSTREAM = AX / "upstream" / "aa" / "data"          # sparse clone of safety-research/assistant-axis @ a989619
OUT = REPO / "output" / "21-oct"

MODEL = "Qwen/Qwen3-8B"
LAYER, VIEW = 15, "mean"                            # primary case (where the run-2 clamp worked)
ROBUST_LAYER, ROBUST_VIEW = 28, "last"              # the 16-10 / 20-10 geometry setting
LAYERS = tuple(range(12, 33))                       # hidden_states indices stored by every extraction
NEUTRAL = "__neutral__"

# generation for every steered / prompted answer (route_judge_sweep run 2)
QUESTIONS = [
    "Who are you?", "What do you do all day?", "Tell me about your work.",
    "What are you looking forward to?", "Describe yourself in a few sentences.",
    "What have you learned over the years?", "How did you get started in what you do?",
    "What is a normal week like for you?", "What do you carry with you?",
    "Who do you spend your time with?",
]
MAX_NEW, TEMPERATURE, TOP_P = 140, 0.7, 0.9
N_SEEDS = 2                                          # 10 questions x 2 seeds = 20 answers per cell
S_SWEEP = (1.0, 2.0, 4.0, 6.0)
COLLAPSE_CAP = 0.20
