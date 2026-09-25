"""Steering vectors rebuilt from other personas, against the real thing.

Each arm is pinned to the same absolute displacement so the only difference is
the direction: the persona's own, the one rebuilt by the shared all-layer recipe,
or a random one. Strengths in the cells file are distances in activation units.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS

LAYERS = (26, 27, 28, 29, 30)
# the displacement that scored best for each persona under pinned steering
BEST = {"ghost": 105.0, "pirate": 91.0, "vampire": 82.0, "musician": 83.0, "spy": 86.0}
ALSO = 90.0

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}
rep = json.load(open("attr_analysis/shared_all.json"))["targets"]

out, cells = {}, []
rng = np.random.default_rng(0)
print(f"{'persona':<10}{'layer':>6}{'|true|':>8}{'|recon|':>9}{'cos':>7}")
for t in TARGETS:
    recipe = rep[t]["recipe"]
    for L in LAYERS:
        a = M[idx["default"], L].astype(np.float32)
        true = M[idx[t], L].astype(np.float32) - a
        recon = np.zeros_like(true)
        for name, w in recipe:
            recon += np.float32(w) * (M[idx[name], L].astype(np.float32) - a)
        rand = rng.normal(size=true.shape).astype(np.float32)

        for tag, v in (("true", true), ("recon", recon), ("rand", rand)):
            u = v / np.linalg.norm(v)
            out[f"{tag}|{t}|L{L}"] = u
            out[f"{tag}|{t}|L{L}|a"] = np.array(float(a @ u))
            out[f"{tag}|{t}|L{L}|d"] = np.array(1.0)  # cells carry absolute distance
        c = float(true @ recon / (np.linalg.norm(true) * np.linalg.norm(recon)))
        print(f"{t:<10}{L:>6}{np.linalg.norm(true):>8.1f}{np.linalg.norm(recon):>9.1f}{c:>7.3f}")

    ds = sorted({BEST[t], ALSO})
    cells += [[f"true|{t}", ds], [f"recon|{t}", ds], [f"rand|{t}", [BEST[t]]]]

np.savez_compressed("attr_analysis/recon_steer_vectors.npz", **out)
json.dump(cells, open("attr_analysis/recon_steer_cells.json", "w"), indent=1)
print(f"\n{len(cells)} cells, {sum(len(c[1]) for c in cells)} cell-strength pairs, "
      f"{sum(len(c[1]) for c in cells) * 10} generations")
