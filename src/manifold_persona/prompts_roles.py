"""Build the character-role prompt set for the assistant-axis study.

Same recipe as ``prompts.py`` (persona-vectors), but the personas are the 276
character-role archetypes from the assistant-axis repo
(``data/roles/instructions/<role>.json``). Each role has 5 positive
instructions (no negatives -- these are role *embodiments*, so there is no
polarity axis). Questions are sampled from ``data/extraction_questions.jsonl``.

Each record is a chat: role system prompt + a user question, rendered through
the model's chat template with add_generation_prompt=True, then embedded as a
prompt activation. Records are labelled by role, instruction index and question
index. The ``default`` role is the neutral Assistant baseline.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional

from .config import ROLE_INSTRUCTIONS_DIR, ROLE_QUESTIONS_FILE


@dataclass
class RoleRecord:
    role: str
    is_default: bool
    instruction_idx: int
    question_idx: int
    question: str
    system: Optional[str]     # None only if the instruction string is empty
    text: str = ""


def list_roles() -> List[str]:
    return sorted(p.stem for p in ROLE_INSTRUCTIONS_DIR.glob("*.json"))


def load_questions() -> List[dict]:
    qs = []
    with open(ROLE_QUESTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def build_role_records(n_questions: int = 5, seed: int = 0,
                       roles: List[str] = None,
                       model_display: str = "Qwen2.5-3B-Instruct") -> List[RoleRecord]:
    """Enumerate (role x instruction x sampled-question).

    A fixed random subset of ``n_questions`` questions is shared across all
    roles (seeded) so every role answers the same questions -- keeping role the
    only varying factor per question.
    """
    roles = roles or list_roles()
    questions = load_questions()
    rng = random.Random(seed)
    q_idx = sorted(rng.sample(range(len(questions)), k=min(n_questions, len(questions))))
    sampled = [(i, questions[i]["question"]) for i in q_idx]

    records: List[RoleRecord] = []
    for role in roles:
        data = json.load(open(ROLE_INSTRUCTIONS_DIR / f"{role}.json"))
        instructions = data["instruction"]              # list of {"pos": ...}
        is_default = role == "default"
        for i_idx, instr in enumerate(instructions):
            system = instr.get("pos", "")
            system = system.replace("{model_name}", model_display) if system else None
            for q_idx_i, question in sampled:
                records.append(RoleRecord(
                    role=role, is_default=is_default, instruction_idx=i_idx,
                    question_idx=q_idx_i, question=question, system=system,
                ))
    return records


def render_prompts(records: List[RoleRecord], tokenizer) -> List[RoleRecord]:
    for r in records:
        messages = []
        if r.system is not None:
            messages.append({"role": "system", "content": r.system})
        messages.append({"role": "user", "content": r.question})
        r.text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return records


def records_to_metadata(records: List[RoleRecord]) -> list:
    return [asdict(r) for r in records]
