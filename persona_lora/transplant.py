"""Steer persona A toward persona B by moving the residual stream between their clouds.

Two maps, both applied at a band of layers while generating under persona A's prompt:

  shift      h += (mu_B - mu_A)                       the mean only -- the arrow view
  transport  within the persona subspace, map A's Gaussian onto B's, mean and shape
             both; the orthogonal complement is left untouched so response content
             survives

If a persona is only a direction, the two behave the same. If it is a distribution,
transport should land more cleanly inside B.

The subspace is the between-persona scatter's top directions (at most n_personas-1),
fitted on the cached difference vectors.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import gen_prompt_text, load_persona_prompt
from .judge_protocol import IDENTITY_QUESTIONS


def sqrtm_psd(S, eps=1e-8):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def fit(acts_dir, layer, n_sub, shrink=0.3):
    """Gaussians over the RAW prompted activation, which is what the hook sees.

    Fitting on (prompted - unprompted) and then applying an affine map to a raw
    residual stream mixes two frames: the mean subtracted is a difference-frame
    mean while the point is a raw activation. A pure translation survives that
    (the baseline cancels); a transport does not.

    Covariances are shrunk toward a scaled identity before inversion, or
    Sigma^-1/2 explodes along the near-singular directions of the subspace.
    """
    files = sorted(glob.glob(f"{acts_dir}/shard*.npz"))
    P = np.concatenate([np.load(f)["prompted"] for f in files]).astype(np.float64)
    per = np.concatenate([np.load(f)["persona"] for f in files])
    keep = per != "default"
    X = P[keep, layer, :]
    per = per[keep]
    names = sorted(set(per.tolist()))
    mus = np.stack([X[per == n].mean(0) for n in names])
    # persona subspace = top directions of the between-persona scatter
    Sb = np.cov((mus - mus.mean(0)).T)
    w, V = np.linalg.eigh(Sb)
    Q = V[:, np.argsort(w)[::-1][:n_sub]]                    # [d, k]
    stats = {}
    for n in names:
        Z = X[per == n] @ Q
        S = np.cov(Z.T)
        S = (1 - shrink) * S + shrink * np.trace(S) / n_sub * np.eye(n_sub)
        stats[n] = (Z.mean(0), S)
    return names, Q, stats


class Transplant:
    def __init__(self, Q, mu_a, S_a, mu_b, S_b, mode, device, dtype):
        self.Q = torch.tensor(Q, device=device, dtype=torch.float32)
        self.mode = mode
        self.enabled = False
        self.dtype = dtype
        if mode == "shift":
            d = (mu_b - mu_a) @ Q.T
            self.delta = torch.tensor(d, device=device, dtype=torch.float32)
        else:
            Ra, Ra_inv = sqrtm_psd(S_a)
            Rb, _ = sqrtm_psd(S_b)
            self.T = torch.tensor(Rb @ Ra_inv, device=device, dtype=torch.float32)
            self.mu_a = torch.tensor(mu_a, device=device, dtype=torch.float32)
            self.mu_b = torch.tensor(mu_b, device=device, dtype=torch.float32)

    def make_hook(self):
        def hook(module, args, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            h = hs.float()
            if self.mode == "shift":
                h = h + self.delta
            else:
                z = h @ self.Q                                  # [.., k]
                z2 = self.mu_b + (z - self.mu_a) @ self.T.T
                h = h + (z2 - z) @ self.Q.T                     # only the subspace moves
            hs = h.to(hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["shift", "transport"], required=True)
    ap.add_argument("--layers", default="15,16,17,18,19")
    ap.add_argument("--fit_layer", type=int, default=17)
    ap.add_argument("--n_sub", type=int, default=6)
    ap.add_argument("--only", default=None,
                    help="comma list of personas to use as source and target")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    names, Q, stats = fit(args.acts, args.fit_layer, args.n_sub)
    if args.only:
        sel = args.only.split(",")
        names = [n for n in names if n in sel]
    print(f"personas {names}  subspace {Q.shape}  layers {layers}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    uid = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float32).to(device)
            model.eval()
            tp = Transplant(Q, *stats[a], *stats[b], args.mode, device, torch.float32)
            hs = [model.model.layers[l].register_forward_hook(tp.make_hook())
                  for l in layers]
            sys_text = load_persona_prompt(a)
            tp.enabled = True
            for q in IDENTITY_QUESTIONS:
                for s in range(args.n_samples):
                    ids = tok(gen_prompt_text(sys_text, q), return_tensors="pt",
                              add_special_tokens=False).to(device)
                    o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=True, temperature=1.0, top_p=0.95,
                                       pad_token_id=tok.eos_token_id)
                    uid += 1
                    rows.append({"id": uid, "mode": args.mode, "src": a, "role": b,
                                 "request": q, "sample": s,
                                 "response": tok.decode(o[0, ids["input_ids"].shape[1]:],
                                                        skip_special_tokens=True).strip()})
            for h in hs:
                h.remove()
            del model
            torch.cuda.empty_cache()
            print(f"  {a} -> {b}: {len(rows)} rows", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
