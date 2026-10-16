"""Per-persona token statistics at one layer, from answers re-encoded with no prompt.

For every cell in the generations file, accumulates over the response tokens the count,
the sum and the sum of squares of the hidden state at the chosen layer, so a mean and a
per-dimension variance can be formed per persona, and pooled over all of them.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import encode_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cell_prefix", default="prompt|")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(device).eval()

    rows = [json.loads(l) for g in args.gen for l in open(g) if l.strip()]
    rows = [r for r in rows if r["cell"].startswith(args.cell_prefix)]
    i, n = (int(x) for x in args.shard.split("/"))
    rows = rows[i::n]
    cnt, s1, s2 = defaultdict(int), defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)
    with torch.no_grad():
        for k, r in enumerate(rows):
            pre, suf, start = encode_pair(tok, "", r["question"], r["response"])
            ids = (pre + suf)[:args.max_len]
            rs = len(pre) + start
            if rs >= len(ids) - 1:
                continue
            out = model(torch.tensor([ids], device=device), output_hidden_states=True)
            h = out.hidden_states[args.layer][0, rs:, :].float().cpu().numpy().astype(np.float64)
            c = r["cell"]
            cnt[c] += len(h)
            s1[c] = s1[c] + h.sum(0)
            s2[c] = s2[c] + (h * h).sum(0)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(rows)}", flush=True)
    cells = sorted(cnt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cells=np.array(cells), layer=args.layer,
                        count=np.array([cnt[c] for c in cells]),
                        sum1=np.stack([s1[c] for c in cells]).astype(np.float32),
                        sum2=np.stack([s2[c] for c in cells]).astype(np.float32))
    print(f"wrote {len(cells)} cells, {sum(cnt.values())} tokens -> {out}", flush=True)


if __name__ == "__main__":
    main()
