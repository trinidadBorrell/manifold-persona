"""One activation point per response, for the persona-as-distribution question.

The steering baseline averages (prompted - unprompted) over all responses into a
single direction per persona. Here we keep the per-response points, so a persona is
a cloud rather than an arrow.

Pooling is over RESPONSE tokens only. Position 0 is never included: its hidden state
is identical across records, so any pooling that divides by sequence length injects a
1/T term.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import encode_pair, load_persona_prompt, load_targets


@torch.no_grad()
def pooled(model, ids, resp_start, device, max_len):
    """Mean of each layer's hidden states over response positions."""
    ids = ids[:max_len]
    if resp_start >= len(ids) - 1:
        return None
    t = torch.tensor([ids], device=device)
    out = model(t, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0]          # [L+1, T, d]
    sel = hs[:, resp_start:, :]
    assert resp_start > 0, "response must start after the prompt"
    return sel.mean(1).float().cpu().numpy()               # [L+1, d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(device)
    model.eval()

    rows = load_targets(args.targets)[args.shard::args.nshards]
    P, U, meta = [], [], []
    for i, r in enumerate(rows):
        # targets use persona/question; generated eval files use role/request
        role = r.get("persona", r.get("role"))
        question = r.get("question", r.get("request"))
        r = {**r, "persona": role, "question": question,
             "question_idx": r.get("question_idx", r.get("id", i))}
        sys_text = load_persona_prompt(r["persona"], r.get("instruction_idx", 0))
        pre_p, suf, start = encode_pair(tok, sys_text, r["question"], r["response"])
        pre_u, suf_u, start_u = encode_pair(tok, "", r["question"], r["response"])
        assert suf == suf_u and start == start_u, "suffix must be token-identical"
        a = pooled(model, pre_p + suf, len(pre_p) + start, device, args.max_len)
        b = pooled(model, pre_u + suf, len(pre_u) + start, device, args.max_len)
        if a is None or b is None:
            continue
        P.append(a.astype(np.float16))
        U.append(b.astype(np.float16))
        meta.append({"persona": r["persona"], "question_idx": r["question_idx"]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, prompted=np.stack(P), unprompted=np.stack(U),
                        persona=np.array([m["persona"] for m in meta]),
                        question_idx=np.array([str(m["question_idx"]) for m in meta]))
    print(f"wrote {len(P)} points x {P[0].shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
