"""Affine steering vectors from per-persona token statistics in the no-prompt frame.

For each persona: the token mean, the pooled token mean over all personas, and the
per-dimension ratio of standard deviations, clipped, so that the hook at s = 1 matches
the persona's mean and spread. Also writes a mean-only variant (scale = 1).
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stats", nargs="+", default=sorted(glob.glob("tokstats_out/stats_*.npz")))
ap.add_argument("--personas", default="pirate,musician,spy,toddler,philosopher,hacker,witch,vampire,jester,detective,ghost,scholar")
ap.add_argument("--strengths", default="0.5,1,1.5,2")
ap.add_argument("--clip", type=float, default=3.0)
ap.add_argument("--out", default="l23all/aff12_vectors.npz")
ap.add_argument("--cells", default="l23all/cells_aff12.json")
args = ap.parse_args()

cnt, s1, s2 = defaultdict(float), defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)
L = None
for f in args.stats:
    z = np.load(f, allow_pickle=True)
    L = int(z["layer"])
    for c, n, a, b in zip(z["cells"].astype(str), z["count"], z["sum1"], z["sum2"]):
        t = c.split("|")[1]
        cnt[t] += float(n)
        s1[t] = s1[t] + a.astype(np.float64)
        s2[t] = s2[t] + b.astype(np.float64)
N = sum(cnt.values())
mu_p = sum(s1.values()) / N
var_p = sum(s2.values()) / N - mu_p ** 2
sd_p = np.sqrt(np.maximum(var_p, 1e-6))
print(f"{len(cnt)} personas, {int(N)} tokens, layer {L}, pooled sd median {np.median(sd_p):.3f}")

strengths = [float(x) for x in args.strengths.split(",")]
out, cells = {}, []
for t in args.personas.split(","):
    mu_t = s1[t] / cnt[t]
    var_t = s2[t] / cnt[t] - mu_t ** 2
    sd_t = np.sqrt(np.maximum(var_t, 1e-6))
    scale = np.clip(sd_t / sd_p, 1.0 / args.clip, args.clip)
    for mode, sc in (("aff", scale), ("affmean", np.ones_like(scale))):
        k = f"aff|{mode}_{t}|L{L}"
        out[k + "|mu_t"] = mu_t.astype(np.float32)
        out[k + "|mu_p"] = mu_p.astype(np.float32)
        out[k + "|scale"] = sc.astype(np.float32)
        cells.append([f"aff|{mode}_{t}", strengths])
    print(f"{t:>12} tokens {int(cnt[t]):>6} |mu_t - mu_p| {np.linalg.norm(mu_t - mu_p):.1f} "
          f"scale median {np.median(scale):.2f} [{scale.min():.2f}, {scale.max():.2f}]")
np.savez_compressed(args.out, **out)
json.dump(cells, open(args.cells, "w"))
print(f"{len(cells)} cells x {len(strengths)} strengths -> {args.out}, {args.cells}")
