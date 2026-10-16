"""Giovanni's clamp (persona-definition steer_pin.py) on Qwen3-8B, as close to his run as possible.

His recipe: u = (M_t - M_default) / norm, a = M_default . u, d = |M_t - M_default|, M = mean answer-token
state; pre-hook on model.model.layers[L] (hidden_states[L]), every position: h <- h + (a + s d - h.u) u;
empty system block; his 10 questions; T 0.7, top-p 0.9, 140 new tokens.
Here: M from our E1b extraction of Qwen3-8B (40 questions per persona, prompted, greedy; default = no
system prompt) instead of re-generating his 120 answers per persona; Qwen3 template with an empty system
block and thinking off; the 10 answers of a cell in one batch, seed 0 + cell (his seeds per question).

    python -m steering.gio_repro.qwen3_pin --layer 23 --target vampire --out $PM/GIO8B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
QUESTIONS = [
    "Who are you?", "What do you do all day?", "Tell me about your work.",
    "What are you looking forward to?", "Describe yourself in a few sentences.",
    "What have you learned over the years?", "How did you get started in what you do?",
    "What is a normal week like for you?", "What do you carry with you?",
    "Who do you spend your time with?",
]
STRENGTHS = [0.0, 1.0, 3.0, 4.0, 6.0, 8.0]          # 0 = unsteered baseline (pin at the default's own value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "jobs_condor")]
    import ace_steer as AS
    import torch
    L, t = a.layer, a.target
    meta = pd.read_csv(PM / "E1b" / "meta.csv", usecols=["role"])
    H = np.load(PM / "E1b" / ("mean_L%d.npy" % L), mmap_mode="r")
    M = lambda r: np.asarray(H[(meta.role == r).to_numpy()], dtype=np.float64).mean(0)   # noqa: E731
    Md, Mt = M("__neutral__"), M(t)
    v = Mt - Md
    d = float(np.linalg.norm(v))
    u = v / d
    a0 = float(Md @ u)
    tok, model = AS.load_model(AS.MODEL)
    texts = [tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in QUESTIONS]
    dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    P = enc["input_ids"].shape[1]
    ut = torch.tensor(u, dtype=torch.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, s in enumerate(STRENGTHS):
        goal = a0 + s * d

        def hook(module, args):
            hs = args[0]
            uu = ut.to(hs.device)
            h = hs.float()
            h = h + (goal - h @ uu).unsqueeze(-1) * uu
            return (h.to(hs.dtype),) + tuple(args[1:])
        hk = model.model.layers[L].register_forward_pre_hook(hook) if s > 0 else None
        torch.manual_seed(ci)
        torch.cuda.manual_seed_all(ci)
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=140, do_sample=True, temperature=0.7, top_p=0.9,
                                   pad_token_id=tok.pad_token_id)
        finally:
            if hk:
                hk.remove()
        for qi in range(len(QUESTIONS)):
            txt = tok.decode(o[qi, P:], skip_special_tokens=True).strip()
            rows.append(dict(model="qwen3-8b", layer=L, target=t, s=s, qi=qi, question=QUESTIONS[qi], response=txt))
        print("L%d %s s=%g | %r" % (L, t, s, rows[-10]["response"][:90]), flush=True)
    with open(out / ("gen_L%d_%s.jsonl" % (L, t)), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / ("vec_L%d_%s.json" % (L, t))).write_text(json.dumps(dict(d=d, a=a0)))


if __name__ == "__main__":
    main()
