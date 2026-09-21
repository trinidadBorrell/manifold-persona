"""Personas, question split, and teacher-response generation.

Sequences are built by hand rather than via apply_chat_template so that teacher and
student share a **token-identical** suffix. Both get a system block; only its content
differs (persona vs empty), so everything after it aligns exactly.

    <|im_start|>system\n{sys}<|im_end|>\n          <- prefix (differs)
    <|im_start|>user\n{q}<|im_end|>\n             <- suffix (identical)
    <|im_start|>assistant\n{resp}<|im_end|>
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from manifold_persona.config import ROLE_INSTRUCTIONS_DIR, ROLE_QUESTIONS_FILE


def load_persona_prompt(role: str, instruction_idx: int = 0) -> str:
    """Roles in MP_EXTRA_ROLES win over the vendored set, which stays untouched."""
    if role == "default":
        return ""
    dirs = []
    if os.environ.get("MP_EXTRA_ROLES"):
        dirs.append(Path(os.environ["MP_EXTRA_ROLES"]))
    dirs.append(Path(ROLE_INSTRUCTIONS_DIR))
    for d in dirs:
        f = d / f"{role}.json"
        if f.exists():
            return json.loads(f.read_text())["instruction"][instruction_idx]["pos"]
    raise FileNotFoundError(f"no prompt for role {role!r} in {[str(d) for d in dirs]}")


def question_split(n_heldout: int = 48, seed: int = 0):
    with open(ROLE_QUESTIONS_FILE) as f:
        qs = [json.loads(l)["question"] for l in f if l.strip()]
    idx = list(range(len(qs)))
    random.Random(seed).shuffle(idx)
    held = sorted(idx[:n_heldout])
    train = sorted(idx[n_heldout:])
    return [qs[i] for i in train], [qs[i] for i in held]


def sys_block(sys_text: str) -> str:
    return f"<|im_start|>system\n{sys_text}<|im_end|>\n"


def suffix_text(question: str, response: str) -> str:
    return (f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>")


def gen_prompt_text(sys_text: str, question: str) -> str:
    return (sys_block(sys_text) + f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def encode_pair(tokenizer, sys_text: str, question: str, response: str):
    """Returns (prefix_ids, suffix_ids, response_start) — indices into suffix_ids."""
    prefix = tokenizer(sys_block(sys_text), add_special_tokens=False)["input_ids"]
    head = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(f"{response}<|im_end|>", add_special_tokens=False)["input_ids"]
    return prefix, head_ids + resp_ids, len(head_ids)


@torch.no_grad()
def generate_targets(model, tokenizer, personas, questions, device,
                     max_new_tokens=256, temperature=0.3, top_p=0.9, seed=0,
                     n_phrasings=1):
    """One teacher response per (persona, question). Yields dicts.

    With n_phrasings > 1 the role's alternative system prompts are cycled across
    questions, so a persona mean averages over phrasing instead of inheriting one.
    """
    torch.manual_seed(seed)
    for role in personas:
        for qi, q in enumerate(questions):
            ii = qi % n_phrasings
            sys_text = load_persona_prompt(role, ii)
            ids = tokenizer(gen_prompt_text(sys_text, q),
                            return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=top_p,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, ids["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            yield {"persona": role, "question_idx": qi, "question": q,
                   "instruction_idx": ii, "response": text.strip()}


def load_targets(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
