"""Our routes with Giovanni's clamp, in his setup (Qwen3-8B): start = the empty-prompt model, his 10 questions,
140 tokens, T 0.7 / top-p 0.9, empty system block, thinking off.

Route = the pair's route personas with the start replaced by the empty-prompt model: [default] + route[1:],
through their mean answer-token states (E1b), LIN straight, PMG / MAN4 / FL the natural cubic at equal arc
length (geometry.smooth_path). Goal G_S(s) = M_0 + S (m(s) - M_0); clamp the route's span U (orthonormal
basis of m(s) - M_0 over the steps): h <- h + U U^T (G - h), every position, hidden_states[L].

    python -m steering.gio_repro.qwen3_route --layer 23 --pair assistant.R.vampire --strength 4 --out $PM/GIO8B_R
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.gio_repro.qwen3_pin import PM, QUESTIONS

ROUTES = ["LIN", "PMG", "MAN4", "FL"]
STEPS = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--strength", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "jobs_condor")]
    import ace_steer as AS
    import torch
    from steering.personamanifold import geometry as G
    L, S = a.layer, a.strength
    spec = json.loads((PM / "ST15" / "cells.json").read_text())
    p = spec["pairs"]["%s|L15" % a.pair]
    routes = p["routes"] if isinstance(p["routes"], dict) else ast.literal_eval(p["routes"])
    meta = pd.read_csv(PM / "E1b" / "meta.csv", usecols=["role"])
    H = np.load(PM / "E1b" / ("mean_L%d.npy" % L), mmap_mode="r")
    cache = {}

    def M(r):
        if r not in cache:
            cache[r] = np.asarray(H[(meta.role == r).to_numpy()], dtype=np.float64).mean(0)
        return cache[r]
    M0 = M("__neutral__")
    tok, model = AS.load_model(AS.MODEL)
    texts = [tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in QUESTIONS]
    dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    P = enc["input_ids"].shape[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for route in ROUTES:
        nodes = routes[route][1:]                                         # drop the pair's start persona
        Mn = np.stack([M0] + [M(r) for r in nodes])
        ts = [0.0] + STEPS
        pts = np.stack([(1 - t) * Mn[0] + t * Mn[1] for t in ts]) if len(Mn) == 2 else G.smooth_path(Mn, ts)
        Q, R = np.linalg.qr((pts[1:] - M0).T)
        k = int((np.abs(np.diag(R)) > 1e-6 * np.abs(R).max()).sum())
        U = torch.tensor(Q[:, :k], dtype=torch.float32)
        for j, s in enumerate(STEPS):
            goal = M0 + S * (pts[j + 1] - M0)
            tc = torch.tensor(Q[:, :k].T @ goal, dtype=torch.float32)

            def hook(module, args):
                hs = args[0]
                uu, cc = U.to(hs.device), tc.to(hs.device)
                h = hs.float()
                h = h + (cc - h @ uu) @ uu.T
                return (h.to(hs.dtype),) + tuple(args[1:])
            hk = model.model.layers[L].register_forward_pre_hook(hook)
            torch.manual_seed(j)
            torch.cuda.manual_seed_all(j)
            try:
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=140, do_sample=True, temperature=0.7, top_p=0.9,
                                       pad_token_id=tok.pad_token_id)
            finally:
                hk.remove()
            for qi in range(len(QUESTIONS)):
                rows.append(dict(model="qwen3-8b", layer=L, pair=a.pair, target=p["target"], route=route, k=k, S=S, s=s,
                                 qi=qi, question=QUESTIONS[qi], response=tok.decode(o[qi, P:], skip_special_tokens=True).strip()))
            print("L%d %s %s S%g s=%g | %r" % (L, a.pair, route, S, s, rows[-10]["response"][:80]), flush=True)
    with open(out / ("gen_L%d_%s_S%g.jsonl" % (L, a.pair.replace(".R.", "-"), S)), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
