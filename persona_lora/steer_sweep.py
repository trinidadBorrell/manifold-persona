"""Does steering make the model stay in character? Role adoption vs strength.

Replicates the Assistant Axis §3.2 setup: the persona system prompt IS in context,
and we steer on top of it, sweeping the coefficient. Their Figure 4 is the fraction
of responses exhibiting each role type as a function of steering strength.

Two directions:
  persona  -- +alpha * v_p, v_p = mean(prompted - unprompted) per persona
  axis     -- -alpha * v_axis, v_axis = mean over personas of v_p (generic
              "away from Assistant"), which is closer to their global axis

Outputs generations only; judging is a separate pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import (gen_prompt_text, load_persona_prompt, load_targets, sys_block)
from .judge_protocol import IDENTITY_QUESTIONS
from .steering import Steerer, estimate_vectors
from .teacher import build_prefix_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--direction", default="persona", choices=["persona", "axis"])
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=110)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n_heldout", type=int, default=48)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    personas = json.loads(Path(args.personas).read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = True

    rows = load_targets(args.targets)
    held = set(sorted({r["question_idx"] for r in rows})[:args.n_heldout])
    train_by = {p: [r for r in rows if r["persona"] == p and r["response"]
                    and r["question_idx"] not in held] for p in personas}

    prefixes = {}
    with torch.no_grad():
        for role in personas:
            text = load_persona_prompt(role)
            ids = tokenizer(sys_block(text), return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
            cache, plen = build_prefix_cache(model, ids)
            prefixes[role] = {"text": text, "cache": cache, "len": plen}
        empty_ids = tokenizer(sys_block(""), return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
        empty_cache, empty_len = build_prefix_cache(model, empty_ids)

    steer = Steerer(len(personas), model.config.hidden_size, args.layers,
                    device, dtype)
    print(f"estimating vectors at layers {args.layers} ...", flush=True)
    model.config.use_cache = False
    estimate_vectors(model, tokenizer, personas, prefixes, empty_cache, empty_len,
                     train_by, device, steer, seed=args.seed)
    model.config.use_cache = True

    if args.direction == "axis":
        # one generic direction for every persona: mean of the persona vectors
        mean_v = steer.v.mean(dim=0, keepdim=True)
        steer.v = mean_v.expand_as(steer.v).contiguous()

    for slot, L in enumerate(args.layers):
        model.model.layers[max(0, L - 1)].register_forward_hook(steer.make_hook(slot))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    uid = 0
    with open(out, "w") as f, torch.no_grad():
        for alpha in args.alphas:
            steer.alpha.data.fill_(alpha)
            steer.enabled = alpha != 0.0
            for pi, role in enumerate(personas):
                steer.index = pi
                for q in IDENTITY_QUESTIONS:
                    for s in range(args.n_samples):
                        text = gen_prompt_text(prefixes[role]["text"], q)
                        ids = tokenizer(text, return_tensors="pt",
                                        add_special_tokens=False).to(device)
                        o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                           do_sample=True, temperature=args.temperature,
                                           top_p=0.95,
                                           pad_token_id=tokenizer.eos_token_id)
                        resp = tokenizer.decode(o[0, ids["input_ids"].shape[1]:],
                                                skip_special_tokens=True).strip()
                        uid += 1
                        f.write(json.dumps({
                            "id": uid, "role": role, "request": q, "response": resp,
                            "alpha": alpha, "layers": args.layers,
                            "direction": args.direction, "sample": s}) + "\n")
            print(f"  alpha={alpha} done ({uid} generations)", flush=True)
    print(f"wrote {uid} generations -> {out}", flush=True)


if __name__ == "__main__":
    main()
