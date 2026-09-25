"""Generate under additive or pinned steering, empty system prompt.

Cells are [name, [strengths]]; the name's prefix picks the mode.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import gen_prompt_text

QUESTIONS = [
    "Who are you?", "What do you do all day?", "Tell me about your work.",
    "What are you looking forward to?", "Describe yourself in a few sentences.",
    "What have you learned over the years?", "How did you get started in what you do?",
    "What is a normal week like for you?", "What do you carry with you?",
    "Who do you spend your time with?",
]


def add_hook(vec, alpha):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        h = (hs.float() + alpha * vec).to(hs.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def pin_hook(u, a_proj, d, s):
    goal = a_proj + s * d

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        h = hs.float()
        h = h + (goal - h @ u).unsqueeze(-1) * u
        h = h.to(hs.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="26,27,28,29,30")
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
    jobs = [(c, s) for c, ss in json.load(open(args.cells)) for s in ss]
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} cells x {len(QUESTIONS)} questions", flush=True)

    t = lambda k: torch.tensor(Z[k], device=dev, dtype=torch.float32)
    with open(args.out, "w") as f:
        for ci, (cell, s) in enumerate(jobs):
            mode = cell.split("|")[0]
            hooks = []
            for L in layers:
                k = f"{cell}|L{L}"
                if mode == "pin":
                    hk = pin_hook(t(k), float(Z[k + "|a"]), float(Z[k + "|d"]), s)
                else:
                    hk = add_hook(t(k), s)
                hooks.append(model.model.layers[L].register_forward_hook(hk))
            try:
                for qi, q in enumerate(QUESTIONS):
                    ids = tok(gen_prompt_text("", q), return_tensors="pt",
                              add_special_tokens=False).to(dev)
                    torch.manual_seed(qi)
                    o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=True, temperature=0.7, top_p=0.9,
                                       pad_token_id=tok.eos_token_id)
                    txt = tok.decode(o[0, ids["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
                    f.write(json.dumps({"cell": cell, "alpha": s, "qi": qi,
                                        "question": q, "response": txt}) + "\n")
            finally:
                for h in hooks:
                    h.remove()
            print(f"  [{ci+1}/{len(jobs)}] {cell} s={s}", flush=True)


if __name__ == "__main__":
    main()
