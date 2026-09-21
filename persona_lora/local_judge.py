"""Stateless judge: one independent forward pass per item.

The papers score one response per API call -- the D.1.3 prompt says "one [request]
and one [response]", and assistant_axis/judge.py calls the model once per response.
Batching many items into one context (as a chat agent does) lets an early decision
propagate to every later item, which is how we got a 34-vs-0 swing on the same
content. Here each item gets a fresh conversation: system + user, nothing else.

Temperature 1.0 matches their judge call. Malformed JSON is retried; persistent
failures are recorded rather than silently dropped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .judge_protocol import CATEGORIES

JSON_RE = re.compile(r"\{.*\}", re.S)


def parse(text: str):
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    score = obj.get("score")
    if score not in CATEGORIES:
        return None
    return {"analysis": str(obj.get("analysis", ""))[:1200], "score": score}


@torch.no_grad()
def judge_one(model, tok, system, user, device, max_new_tokens, temperature, tries=3):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    for attempt in range(tries):
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        got = parse(raw)
        if got:
            got["attempt"] = attempt + 1
            return got
    return {"analysis": "", "score": None, "attempt": tries, "raw": raw[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--batch", required=True, help="jsonl with id/system/user")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None)
    model.eval()
    print(f"judge={args.model} device={device}", flush=True)

    rows = [json.loads(l) for l in open(args.batch) if l.strip()]
    rows = rows[args.shard::args.nshards]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            res = judge_one(model, tok, r["system"], r["user"], device,
                            args.max_new_tokens, args.temperature)
            if res["score"] is None:
                n_fail += 1
            f.write(json.dumps({"id": r["id"], **res}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)}  parse_failures={n_fail}", flush=True)
    print(f"done {len(rows)} items, {n_fail} unparseable -> {out}", flush=True)


if __name__ == "__main__":
    main()
