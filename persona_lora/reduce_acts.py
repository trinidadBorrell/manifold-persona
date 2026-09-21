"""Compact an activation shard set: per-persona means at every layer, plus the
raw points at the layers we actually plot. Keeps the download small."""
import argparse
import glob

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="26,28,30")
a = ap.parse_args()
L = [int(x) for x in a.layers.split(",")]

P, N = [], []
for f in sorted(glob.glob(a.acts)):
    z = np.load(f, allow_pickle=True)
    P.append(z["prompted"].astype(np.float32))
    N.append(z["persona"].astype(str))
X = np.concatenate(P, 0)
who = np.concatenate(N)
personas = sorted(set(who))

out = {"personas": np.array(personas), "layers": np.array(L)}
out["means"] = np.stack([X[who == p].mean(0) for p in personas]).astype(np.float32)
for li in L:
    out[f"pts_{li}"] = np.stack([X[who == p][:, li, :] for p in personas]).astype(np.float16)
np.savez_compressed(a.out, **out)
print(f"{len(personas)} personas, means {out['means'].shape}, "
      f"pts {out[f'pts_{L[0]}'].shape} -> {a.out}", flush=True)
