"""Generate with several orthonormal directions pinned per layer, empty system prompt.

Cells are [name, [distances]]. For each layer the vectors file holds
  {name}|L{L}    U, (k, hidden) orthonormal rows
  {name}|L{L}|a  assistant mean projected on U, (k,)
  {name}|L{L}|t  target displacement projected on U, (k,)
  {name}|n       norm of the target displacement at the last steered layer
The multiplier is s = min(distance / n, cap) and the goal is a + s * t.
A cell named prompt|{role} is generated with the role's system prompt and no steering.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import gen_prompt_text, load_persona_prompt


def pin_hook(U, goal):
    def hook(module, args):
        hs = args[0]
        h = hs.float()
        h = h + (goal - h @ U.T) @ U
        h = h.to(hs.dtype)
        return (h,) + tuple(args[1:])
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="26,27,28,29,30")
    ap.add_argument("--cap", type=float, default=6.0)
    ap.add_argument("--max_new_tokens", type=int, default=140)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "refusing to run on CPU"
    dev = "cuda"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(dev).eval()
    Z = np.load(args.vectors)
    layers = [int(x) for x in args.layers.split(",")]
    questions = json.load(open(args.questions))
    jobs = [(c, d) for c, ds in json.load(open(args.cells)) for d in ds]
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} cells x {len(questions)} questions", flush=True)

    t = lambda k: torch.tensor(np.atleast_1d(Z[k]), device=dev, dtype=torch.float32)
    with open(args.out, "w") as f:
        for ci, (cell, dist) in enumerate(jobs):
            hooks, sys_text, s = [], "", 0.0
            if cell.startswith("prompt|"):
                sys_text = load_persona_prompt(cell.split("|")[1])
            else:
                s = min(dist / float(Z[f"{cell}|n"]), args.cap)
                for L in layers:
                    k = f"{cell}|L{L}"
                    U = t(k).reshape(-1, model.config.hidden_size)
                    hk = pin_hook(U, t(k + "|a") + s * t(k + "|t"))
                    hooks.append(model.model.layers[L].register_forward_pre_hook(hk))
            try:
                for qi, q in enumerate(questions):
                    ids = tok(gen_prompt_text(sys_text, q), return_tensors="pt",
                              add_special_tokens=False).to(dev)
                    torch.manual_seed(qi)
                    o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=True, temperature=0.7, top_p=0.9,
                                       pad_token_id=tok.eos_token_id)
                    txt = tok.decode(o[0, ids["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
                    f.write(json.dumps({"cell": cell, "alpha": dist, "s": round(s, 3),
                                        "qi": qi, "question": q, "response": txt}) + "\n")
            finally:
                for h in hooks:
                    h.remove()
            print(f"  [{ci+1}/{len(jobs)}] {cell} D={dist} s={s:.2f}", flush=True)


if __name__ == "__main__":
    main()
