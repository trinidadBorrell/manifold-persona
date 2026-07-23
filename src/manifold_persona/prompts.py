"""Build the persona-conditioned prompt set from persona_vectors artifacts.

Recycled from ``persona_vectors``:
- ``a_or_an`` and the system-prompt construction
  (``You are a/an <name> assistant. <instruction>``) come from
  ``eval/eval_persona.py::load_persona_questions``.
- The trait artifacts (5 pos/neg instruction pairs + questions per trait) are
  read in place from ``data_generation/trait_data_extract/<trait>.json``.

We build *prompts only* (no response generation): each record is a chat with a
persona system prompt + a user question, rendered through the model's chat
template. Records are labelled by trait, polarity (pos/neg/neutral),
instruction index and question index.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

from .config import TRAIT_ARTIFACTS_DIR, TRAITS


def a_or_an(word: str) -> str:
    """Recycled verbatim from persona_vectors/eval/eval_persona.py."""
    return "an" if word[0].lower() in "aeiou" else "a"


def _assistant_name(trait: str, polarity: str) -> str:
    """Mirror load_persona_questions defaults: pos -> trait, neg -> 'helpful'."""
    return trait if polarity == "pos" else "helpful"


@dataclass
class PromptRecord:
    trait: str
    polarity: str            # "pos" | "neg" | "neutral"
    instruction_idx: int     # 0..4, or -1 for neutral
    question_idx: int        # 0..len(questions)-1
    question: str
    system: Optional[str]    # None for neutral
    text: str = ""           # filled in after chat-template rendering


def load_trait_artifact(trait: str) -> dict:
    path = TRAIT_ARTIFACTS_DIR / f"{trait}.json"
    with open(path, "r") as f:
        return json.load(f)


def build_prompt_records(traits: List[str] = None, include_neutral: bool = True) -> List[PromptRecord]:
    """Enumerate (trait x instruction x polarity x question) + optional neutral.

    System prompts are constructed exactly as in eval_persona.py:
        system = f"You are {a_or_an(name)} {name} assistant. {instruction}"
    with name = trait (pos) or "helpful" (neg).
    """
    traits = traits or TRAITS
    records: List[PromptRecord] = []
    for trait in traits:
        data = load_trait_artifact(trait)
        instructions = data["instruction"]      # list of {"pos":..., "neg":...}
        questions = data["questions"]
        for q_idx, question in enumerate(questions):
            for polarity in ("pos", "neg"):
                name = _assistant_name(trait, polarity)
                for i_idx, instr in enumerate(instructions):
                    system = f"You are {a_or_an(name)} {name} assistant. {instr[polarity]}"
                    records.append(PromptRecord(
                        trait=trait, polarity=polarity, instruction_idx=i_idx,
                        question_idx=q_idx, question=question, system=system,
                    ))
            if include_neutral:
                records.append(PromptRecord(
                    trait=trait, polarity="neutral", instruction_idx=-1,
                    question_idx=q_idx, question=question, system=None,
                ))
    return records


def render_prompts(records: List[PromptRecord], tokenizer) -> List[PromptRecord]:
    """Fill record.text with the chat-template-rendered prompt string.

    Uses add_generation_prompt=True so the string ends where the assistant's
    response would begin -- i.e. we embed the full persona-conditioned prompt.
    """
    for r in records:
        messages = []
        if r.system is not None:
            messages.append({"role": "system", "content": r.system})
        messages.append({"role": "user", "content": r.question})
        r.text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return records


def records_to_metadata(records: List[PromptRecord]) -> list:
    """Plain dicts for saving alongside the embeddings."""
    return [asdict(r) for r in records]
