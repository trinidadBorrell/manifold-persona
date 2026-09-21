"""Generate under a full-space steering vector, empty system prompt.

Tests whether a *predicted* attribute vector produces the persona it predicts:
the model is never told who to be, the character comes only from the residual edit.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import gen_prompt_text, load_persona_prompt

QUESTIONS = [
    "Who are you?",
    "What do you do all day?",
    "Tell me about your work.",
    "What are you looking forward to?",
    "Describe yourself in a few sentences.",
    "What have you learned over the years?",
    "How did you get started in what you do?",
    "What is a normal week like for you?",
    "What do you carry with you?",
    "Who do you spend your time with?",
]


def hook_for(vec, alpha):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        h = hs.float() + alpha * vec
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
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--pred_alphas", default="1.0,0.6")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=140)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "refusing to run this on CPU"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(dev)
    model.eval()

    Z = np.load(args.vectors)
    layers = [int(x) for x in args.layers.split(",")]
    cells = json.load(open(args.cells))
    questions = json.load(open(args.questions)) if args.questions else QUESTIONS

    jobs = []
    for c in cells:
        al = args.pred_alphas if "|pred|" in c else args.alphas
        for a in [float(x) for x in al.split(",")]:
            jobs.append((c, a))
    jobs = jobs[args.shard::args.nshards]
    print(f"{len(jobs)} cells x {len(questions)} questions on {dev}", flush=True)

    n = 0
    with open(args.out, "w") as f:
        for ci, (cell, alpha) in enumerate(jobs):
            arm = cell.split("|")[1]
            sys_text = ""
            if arm == "prompt":
                sys_text = load_persona_prompt(cell.split("|")[0])
                hooks = []
            else:
                vecs = {L: torch.tensor(Z[f"{cell}|L{L}"], device=dev, dtype=torch.float32)
                        for L in layers}
                hooks = [model.model.layers[L].register_forward_hook(hook_for(vecs[L], alpha))
                         for L in layers]
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
                    f.write(json.dumps({"cell": cell, "alpha": alpha, "qi": qi,
                                        "question": q, "response": txt}) + "\n")
                    n += 1
            finally:
                for h in hooks:
                    h.remove()
            print(f"  [{ci+1}/{len(jobs)}] {cell} a={alpha}", flush=True)
    print(f"wrote {n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
