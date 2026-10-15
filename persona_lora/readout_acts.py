"""Read activations of generated answers under an empty system prompt.

Each row of the generations file (cell, alpha, qi, question, response) gives one point:
the mean hidden state over the response tokens at every layer, with nothing steered,
so the point can be scored against the persona clouds measured the same way.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import encode_pair


@torch.no_grad()
def pooled(model, ids, resp_start, device, max_len):
    ids = ids[:max_len]
    if resp_start >= len(ids) - 1:
        return None
    out = model(torch.tensor([ids], device=device), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0]
    return hs[:, resp_start:, :].mean(1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--layers", default="", help="comma list of layers to keep; empty keeps all 37")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(device).eval()

    keep = [int(x) for x in args.layers.split(",")] if args.layers else None
    rows = [json.loads(l) for g in args.gen for l in open(g) if l.strip()]
    i, n = (int(x) for x in args.shard.split("/"))
    rows = rows[i::n]
    X, meta = [], []
    for k, r in enumerate(rows):
        pre, suf, start = encode_pair(tok, "", r["question"], r["response"])
        a = pooled(model, pre + suf, len(pre) + start, device, args.max_len)
        if a is None:
            continue
        X.append((a[keep] if keep else a).astype(np.float16))
        meta.append((r["cell"], float(r["alpha"]), int(r["qi"])))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(rows)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, acts=np.stack(X), layers=np.array(keep if keep else list(range(len(X[0])))),
                        cell=np.array([m[0] for m in meta]),
                        alpha=np.array([m[1] for m in meta], dtype=np.float32),
                        qi=np.array([m[2] for m in meta], dtype=np.int32))
    print(f"wrote {len(X)} points x {X[0].shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
