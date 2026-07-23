"""Build the prompt set for the assistant-axis TRAITS study.

Same bipolar recipe as the persona-vectors trait study (`prompts.py`), applied
to the 240 behavioural traits in the assistant-axis repo
(`data/traits/instructions/<trait>.json`; 5 {pos, neg} pairs + 40 questions
each). Because there are 240 traits, we *sample* a few instructions and
questions per trait (seeded) to keep the cloud tractable.

System prompt convention matches the persona-vectors study:
    "You are a/an <name> assistant. <instruction>"
with name = trait (pos) or "helpful" (neg). Each trait's own questions are
sampled (so neutral prompts differ across traits).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional

from .config import AA_TRAIT_INSTRUCTIONS_DIR


def a_or_an(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _name(trait: str, polarity: str) -> str:
    return trait if polarity == "pos" else "helpful"


@dataclass
class AATraitRecord:
    trait: str
    polarity: str            # "pos" | "neg" | "neutral"
    instruction_idx: int     # -1 for neutral
    question_idx: int
    question: str
    system: Optional[str]
    text: str = ""


def list_traits() -> List[str]:
    return sorted(p.stem for p in AA_TRAIT_INSTRUCTIONS_DIR.glob("*.json"))


def build_aa_trait_records(n_instructions: int = 2, n_questions: int = 3,
                           seed: int = 0, traits: List[str] = None,
                           include_neutral: bool = True) -> List[AATraitRecord]:
    traits = traits or list_traits()
    rng = random.Random(seed)
    records: List[AATraitRecord] = []
    for trait in traits:
        data = json.load(open(AA_TRAIT_INSTRUCTIONS_DIR / f"{trait}.json"))
        instructions = data["instruction"]           # 5 × {pos, neg}
        questions = data["questions"]                 # 40
        i_idx = sorted(rng.sample(range(len(instructions)),
                                  k=min(n_instructions, len(instructions))))
        q_idx = sorted(rng.sample(range(len(questions)),
                                  k=min(n_questions, len(questions))))
        for qi in q_idx:
            question = questions[qi]
            for polarity in ("pos", "neg"):
                name = _name(trait, polarity)
                for ii in i_idx:
                    instr = instructions[ii][polarity]
                    system = f"You are {a_or_an(name)} {name} assistant. {instr}"
                    records.append(AATraitRecord(
                        trait=trait, polarity=polarity, instruction_idx=ii,
                        question_idx=qi, question=question, system=system))
            if include_neutral:
                records.append(AATraitRecord(
                    trait=trait, polarity="neutral", instruction_idx=-1,
                    question_idx=qi, question=question, system=None))
    return records


def render_prompts(records: List[AATraitRecord], tokenizer) -> List[AATraitRecord]:
    for r in records:
        messages = []
        if r.system is not None:
            messages.append({"role": "system", "content": r.system})
        messages.append({"role": "user", "content": r.question})
        r.text = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    return records


def records_to_metadata(records: List[AATraitRecord]) -> list:
    return [asdict(r) for r in records]
