"""Probing the steering manifold: how far, how wide, and toward what.

Four things one script can answer, none of which need a judge to detect failure:

  alpha    push past alpha=1 and find where fluency breaks
  noise    add isotropic noise to the step -- if the manifold were a thin line any
           sideways move would break it; if it has volume, it survives
  diag     rescale each subspace dimension by sigma_B/sigma_A instead of applying the
           full Bures matrix. The full matrix destroyed the text; the diagonal is the
           gentlest way to let the covariance inform the step
  centroid steer toward the mean of several personas at once -- is the average of
           three characters still a character?
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


def fit(acts_dir, layers, n_sub, shrink=0.2, extra=()):
    """extra: npz files for personas outside the training set.

    The subspace Q and the assistant distribution always come from acts_dir -- the
    training personas -- so a held-out persona is projected into a basis fitted
    without it. Fitting Q on the held-out persona too would leak.
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
            S = (1 - shrink) * S + shrink * np.trace(S) / n_sub * np.eye(n_sub)
            return Z.mean(0), S

        stats = {n: g(per == n) for n in names}
        for f in extra:
            d = np.load(f)
            Z = d["prompted"].astype(np.float64)[:, L, :] @ Q
            S = np.cov(Z.T)
            S = (1 - shrink) * S + shrink * np.trace(S) / n_sub * np.eye(n_sub)
            stats[Path(f).stem] = (Z.mean(0), S)
        out[L] = (Q, stats, g(per == "default"))
    return sorted(set(names) | {Path(f).stem for f in extra}), out


class Probe:
    def __init__(self, Q, mu_a, S_a, mu_b, S_b, mode, alpha, noise, seed, device):
        self.Q = torch.tensor(Q, device=device, dtype=torch.float32)
        self.mode, self.alpha, self.noise = mode, alpha, noise
        self.enabled = False
        self.mu_a = torch.tensor(mu_a, device=device, dtype=torch.float32)
        self.delta = torch.tensor(mu_b - mu_a, device=device, dtype=torch.float32)
        # diagonal transport: one scalar per dimension, sigma_B / sigma_A
        r = np.sqrt(np.diag(S_b) / np.diag(S_a))
        self.gain = torch.tensor(r - 1.0, device=device, dtype=torch.float32)
        self.scale = float(np.linalg.norm(mu_b - mu_a))
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def make_hook(self):
        def hook(module, args, output):
            if not self.enabled or self.alpha == 0.0:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            h = hs.float()
            z = h @ self.Q
            step = self.delta.expand_as(z)
            if self.mode == "diag":
                step = step + (z - self.mu_a) * self.gain
            if self.noise > 0:
                e = torch.randn(z.shape, generator=self.g, device=self.device)
                e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                step = step + self.noise * self.scale * e
            h = h + self.alpha * (step @ self.Q.T)
            return ((h.to(hs.dtype),) + output[1:] if isinstance(output, tuple)
                    else h.to(hs.dtype))
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--acts", default=None)
    ap.add_argument("--stats", default=None,
                    help="precomputed fitted_stats npz; avoids every job re-reading "
                         "and decompressing the full activation set from NFS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["linear", "diag"], default="linear")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="isotropic noise added to the step, as a fraction of its norm")
    ap.add_argument("--step3", default=None,
                    help="steering step given in the top-3 subspace directions, "
                         "'x,y,z'. Sampled in the 3D view, applied in full dimension "
                         "as Q[:, :3] @ (x,y,z). Overrides --targets.")
    ap.add_argument("--step16", default=None,
                    help="steering step as 16 numbers in the subspace basis; the "
                         "caller decides the geometry, so a 3D view built from the "
                         "persona directions themselves stays exact")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cells", default=None,
                    help="json [[tag,[x,y,z]],...]; run many cells on one model load")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n_questions", type=int, default=0,
                    help="use only the first N identity questions (0 = all)")
    ap.add_argument("--acts_extra", nargs="*", default=[],
                    help="npz files for personas not in --acts")
    ap.add_argument("--targets", default="ghost",
                    help="comma list; more than one steers toward their centroid")
    ap.add_argument("--layers", default="26,27,28,29,30")
    ap.add_argument("--n_sub", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    if args.stats:
        d = np.load(args.stats)
        names = sorted({k.split("_", 2)[2] for k in d.files if k.startswith("mu_")})
        bylayer = {L: (d[f"Q_{L}"].astype(np.float64),
                       {n: (d[f"mu_{L}_{n}"].astype(np.float64),
                            d[f"S_{L}_{n}"].astype(np.float64)) for n in names},
                       (d[f"asst_mu_{L}"].astype(np.float64),
                        d[f"asst_S_{L}"].astype(np.float64))) for L in layers}
    else:
        names, bylayer = fit(args.acts, layers, args.n_sub, extra=args.acts_extra)
    tgts = [t for t in args.targets.split(",") if t in names]
    label = "+".join(tgts)
    print(f"device={device} mode={args.mode} alpha={args.alpha} noise={args.noise} "
          f"targets={tgts}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.eval()

    QS = IDENTITY_QUESTIONS[:args.n_questions] if args.n_questions else IDENTITY_QUESTIONS

    def run(spec, tag):
        """spec = (persona, alpha, lateral_magnitude, seed).

        The persona vector is recomputed AT EACH LAYER. A single vector fitted at
        one layer and reused across the band over-steers the shallow end and
        under-steers the deep end -- the step norm runs 6.99 at L26 to 17.05 at L30.
        """
        persona, alpha, lat, seed = spec
        hooks = []
        for L in layers:
            Q, stats, asst = bylayer[L]
            d = stats[persona][0] - asst[0]
            nrm = float(np.linalg.norm(d))
            step = alpha * d
            if lat > 0:
                r = np.random.default_rng(seed + L)
                e = r.normal(size=len(d))
                u = d / nrm
                e -= (e @ u) * u
                e /= np.linalg.norm(e)
                step = step + lat * nrm * e
            mu_b = asst[0] + step
            pr = Probe(Q, asst[0], asst[1], mu_b, asst[1], args.mode, args.alpha,
                       args.noise, args.seed + L, device)
            pr.enabled = True
            hooks.append(model.model.layers[L].register_forward_hook(pr.make_hook()))
        out_rows = []
        for qi, q in enumerate(QS):
            ids = tok(gen_prompt_text("", q), return_tensors="pt",
                      add_special_tokens=False).to(device)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                   do_sample=True, temperature=1.0, top_p=0.95,
                                   pad_token_id=tok.eos_token_id)
            out_rows.append({"tag": tag, "spec": list(map(str, spec)),
                             "request": q, "qi": qi,
                             "response": tok.decode(o[0, ids["input_ids"].shape[1]:],
                                                    skip_special_tokens=True).strip()})
        for h in hooks:
            h.remove()
        return out_rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # append after every cell: a run that dies partway keeps what it produced, and
    # progress is visible from the output file rather than from buffered stdout
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["tag"])
            except Exception:
                pass
        if done:
            print(f"resuming: {len(done)} cells already in {out}", flush=True)

    n = 0
    with open(out, "a", buffering=1) as f:
        if args.cells:
            cells = json.load(open(args.cells))[args.shard::args.nshards]
            for i, (tag, spec) in enumerate(cells):
                if tag in done:
                    continue
                for r in run((spec[0], float(spec[1]), float(spec[2]), int(spec[3])), tag):
                    f.write(json.dumps(r) + "\n")
                    n += 1
                f.flush()
                print(f"  [{i+1}/{len(cells)}] {tag}", flush=True)
        else:
            s3 = np.array([float(x) for x in args.step3.split(",")], dtype=np.float64)
            for r in run(s3, args.tag or args.step3):
                f.write(json.dumps(r) + "\n")
                n += 1
    rows = [None] * n
    print(f"wrote {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
