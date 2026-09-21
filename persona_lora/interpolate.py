"""Walk from the assistant distribution to a persona, and look at the middle.

Both arms are the same family, differing only in whether the covariance travels:

  linear     x + t * (mu_B - mu_A)                  the paper's vector, scaled
  geodesic   x + t * (T(x) - x),  T the Bures map   Wasserstein-2 displacement

At t=1 linear reaches N(mu_B, Sigma_A) and geodesic reaches N(mu_B, Sigma_B); the
pushforward of the geodesic at intermediate t is exactly N(mu_t, Sigma_t), the W2
shortest path. t=0 is the untouched assistant and t=1 is the endpoint-steering
comparison, so one sweep answers both questions.

Generation uses an EMPTY system prompt: we are steering the assistant into a
persona, not decorating a persona prompt.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import gen_prompt_text
from .judge_protocol import IDENTITY_QUESTIONS


def sqrtm(S, eps=1e-12):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(np.sqrt(np.clip(w, eps, None))) @ V.T


def bures_map(SA, SB):
    """W2-optimal map between centred Gaussians: symmetric PSD, minimal displacement."""
    RA = sqrtm(SA)
    RAi = np.linalg.inv(RA)
    return RAi @ sqrtm(RA @ SB @ RA) @ RAi


def fit(acts_dir, layers, n_sub, shrink=0.2):
    """Per-layer subspace and Gaussians.

    The activation scale grows steeply with depth, so one layer's map applied across a
    band under-corrects at the shallow end and over-corrects at the deep end. The
    paper likewise estimates its steering vector separately at each layer.
    """
    files = sorted(glob.glob(f"{acts_dir}/shard*.npz"))
    P = np.concatenate([np.load(f)["prompted"] for f in files]).astype(np.float64)
    per = np.concatenate([np.load(f)["persona"] for f in files])
    names = sorted(n for n in set(per.tolist()) if n != "default")
    out = {}
    for L in layers:
        X = P[:, L, :]
        mus = np.stack([X[per == n].mean(0) for n in names])
        w, V = np.linalg.eigh(np.cov((mus - mus.mean(0)).T))
        Q = V[:, np.argsort(w)[::-1][:n_sub]]

        def g(mask):
            Z = X[mask] @ Q
            S = np.cov(Z.T)
            return Z.mean(0), (1 - shrink) * S + shrink * np.trace(S) / n_sub * np.eye(n_sub)

        out[L] = (Q, {n: g(per == n) for n in names}, g(per == "default"))
    return names, out


class Walker:
    """Applies x -> x + t*(map(x) - x) inside the persona subspace only."""

    def __init__(self, Q, mu_a, S_a, mu_b, S_b, mode, device):
        self.Q = torch.tensor(Q, device=device, dtype=torch.float32)
        self.mode = mode
        self.t = 0.0
        self.enabled = False
        self.mu_a = torch.tensor(mu_a, device=device, dtype=torch.float32)
        self.delta = torch.tensor(mu_b - mu_a, device=device, dtype=torch.float32)
        if mode == "geodesic":
            M = bures_map(S_a, S_b)
            self.M_I = torch.tensor(M - np.eye(len(mu_a)), device=device,
                                    dtype=torch.float32)
            self.dev_ratio = float(np.linalg.norm(M - np.eye(len(mu_a)))
                                   / np.linalg.norm(np.eye(len(mu_a))))

    def make_hook(self):
        def hook(module, args, output):
            if not self.enabled or self.t == 0.0:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            h = hs.float()
            z = h @ self.Q
            step = self.delta.expand_as(z)
            if self.mode == "geodesic":
                step = step + (z - self.mu_a) @ self.M_I.T
            h = h + self.t * (step @ self.Q.T)
            hs = h.to(hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["linear", "geodesic"], required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--layers", default="26,27,28,29,30")
    ap.add_argument("--fit_layer", type=int, default=28)
    ap.add_argument("--n_sub", type=int, default=16)
    ap.add_argument("--personas", default="ghost,economist,musician")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    names, bylayer = fit(args.acts, layers, args.n_sub)
    want = [p for p in args.personas.split(",") if p in names]
    print(f"device={device} mode={args.mode} t={args.t} layers={layers} "
          f"n_sub={args.n_sub} personas={want}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.eval()

    rows, uid = [], 0
    for b in want:
        ws, hs = [], []
        for L in layers:
            Q, stats, asst = bylayer[L]
            w = Walker(Q, asst[0], asst[1], *stats[b], args.mode, device)
            w.t, w.enabled = args.t, True
            ws.append(w)
            hs.append(model.model.layers[L].register_forward_hook(w.make_hook()))
        if args.mode == "geodesic":
            print(f"  {b}: ||M-I||/||I|| per layer = "
                  f"{[round(x.dev_ratio, 3) for x in ws]}", flush=True)
        for q in IDENTITY_QUESTIONS:
            for s in range(args.n_samples):
                ids = tok(gen_prompt_text("", q), return_tensors="pt",
                          add_special_tokens=False).to(device)
                with torch.no_grad():
                    o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=True, temperature=1.0, top_p=0.95,
                                       pad_token_id=tok.eos_token_id)
                uid += 1
                rows.append({"id": uid, "mode": args.mode, "t": args.t, "role": b,
                             "request": q, "sample": s,
                             "response": tok.decode(o[0, ids["input_ids"].shape[1]:],
                                                    skip_special_tokens=True).strip()})
        for h in hs:
            h.remove()
        print(f"  {b}: {len(rows)} rows", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
