"""Steer with reconstructions of increasing size, against the persona's own vector.

All arms are pinned to the true vector's displacement at each layer, so only the
direction differs.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS

LAYERS = (26, 27, 28, 29, 30)
BEST = {"ghost": 5.0, "pirate": 2.0, "vampire": 3.0, "musician": 4.0, "spy": 5.0}
KS = [2, 8, 32]

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}
sweep = json.load(open("attr_analysis/recon_sweep.json"))

out, cells = {}, []
for t in TARGETS:
    for L in LAYERS:
        a = M[idx["default"], L].astype(np.float32)
        true = M[idx[t], L].astype(np.float32) - a
        n_true = float(np.linalg.norm(true))
        arms = {"true": true}
        for k in KS:
            v = np.zeros_like(true)
            for name, w in sweep[f"{t}|ban8"]["recipes"][str(k)]:
                v += np.float32(w) * (M[idx[name], L].astype(np.float32) - a)
            arms[f"k{k}"] = v
        for tag, v in arms.items():
            u = v / np.linalg.norm(v)
            out[f"pin|{tag}_{t}|L{L}"] = u
            out[f"pin|{tag}_{t}|L{L}|a"] = np.array(float(a @ u))
            out[f"pin|{tag}_{t}|L{L}|d"] = np.array(n_true)
    for tag in ["true"] + [f"k{k}" for k in KS]:
        cells.append([f"pin|{tag}_{t}", [BEST[t]]])

np.savez_compressed("attr_analysis/ksweep_vectors.npz", **out)
json.dump(cells, open("attr_analysis/ksweep_cells.json", "w"), indent=1)
print(f"{len(cells)} cells, {sum(len(c[1]) for c in cells) * 10} generations")
