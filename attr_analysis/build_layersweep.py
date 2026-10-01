"""Is there one layer that steers a persona best, and is it the same for all of them?

Pins a single layer at a time, across the depth of the model. Each cell is run by a
separate job that registers its hook on one layer only.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS

LAYERS = [12, 16, 20, 23, 26, 28, 30, 32]
# one layer alone displaces far less than five, so push harder than the 5-layer best
SCALES = [4.0, 8.0]

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}

out, cells = {}, []
print(f"{'persona':<10}" + "".join(f"{f'L{L}':>8}" for L in LAYERS))
for t in TARGETS:
    norms = []
    for L in LAYERS:
        a = M[idx["default"], L].astype(np.float32)
        d = M[idx[t], L].astype(np.float32) - a
        n = float(np.linalg.norm(d))
        u = d / n
        out[f"pin|{t}|L{L}"] = u
        out[f"pin|{t}|L{L}|a"] = np.array(float(a @ u))
        out[f"pin|{t}|L{L}|d"] = np.array(n)
        norms.append(n)
    cells.append([f"pin|{t}", SCALES])
    print(f"{t:<10}" + "".join(f"{n:>8.1f}" for n in norms))

np.savez_compressed("attr_analysis/layersweep_vectors.npz", **out)
json.dump(cells, open("attr_analysis/layersweep_cells.json", "w"), indent=1)
json.dump(LAYERS, open("attr_analysis/layersweep_layers.json", "w"))
print(f"\n{len(cells)} personas x {len(SCALES)} strengths x {len(LAYERS)} layers x 10 q "
      f"= {len(cells) * len(SCALES) * len(LAYERS) * 10} generations")
