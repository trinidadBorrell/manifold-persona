"""Render (role, request, response) items into the exact D.1.3 judge prompts.

Any field other than id/role/request/response is dropped, so ground-truth labels
in a self-test file can never leak into what the judge sees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .judge_protocol import (JUDGE_SYSTEM, JUDGE_TRUNCATE_TOKENS, JUDGE_USER)


def truncate(text: str, tokenizer, n_tokens: int = JUDGE_TRUNCATE_TOKENS) -> str:
    if tokenizer is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= n_tokens:
        return text
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="jsonl: id, role, request, response")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="HF id used only to apply the 512-token truncation")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = [json.loads(l) for l in open(args.items) if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            user = JUDGE_USER.format(request=r["request"],
                                     response=truncate(r["response"], tok),
                                     role=r["role"])
            f.write(json.dumps({"id": r["id"], "system": JUDGE_SYSTEM,
                                "user": user}) + "\n")
    print(f"rendered {len(rows)} judge prompts -> {out}")


if __name__ == "__main__":
    main()
