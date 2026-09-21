"""Generate the teacher responses once. Every sweep cell trains on these exact targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import generate_targets, question_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--n_heldout", type=int, default=48)
    ap.add_argument("--limit_questions", type=int, default=None)
    ap.add_argument("--n_phrasings", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if args.dtype != "auto":
        dtype = fixed[args.dtype]
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    personas = json.loads(Path(args.personas).read_text())
    personas = personas[args.shard::args.nshards]
    train_q, held_q = question_split(args.n_heldout, args.seed)
    # held-out first so question_idx < n_heldout is the eval split, by construction
    questions = held_q + train_q
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    print(f"{len(personas)} personas x {len(questions)} questions "
          f"({args.n_heldout} held out) on {device}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for rec in generate_targets(model, tokenizer, personas, questions, device,
                                    args.max_new_tokens, args.temperature,
                                    args.top_p, args.seed, args.n_phrasings):
            f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n} generated", flush=True)
    print(f"wrote {n} targets -> {out}", flush=True)


if __name__ == "__main__":
    main()
