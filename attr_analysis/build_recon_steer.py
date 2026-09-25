"""Steering vectors rebuilt from other personas, against the real thing.

Every arm is pinned to the true vector's own displacement at each layer, so the
three arms differ only in direction: the persona's own, the one rebuilt by the
shared all-layer recipe, or a random one.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS

LAYERS = (26, 27, 28, 29, 30)
# the pin strength that scored best for each persona
BEST = {"ghost": 5.0, "pirate": 2.0, "vampire": 3.0, "musician": 4.0, "spy": 5.0}

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

        n_true = float(np.linalg.norm(true))
        for tag, v in (("true", true), ("recon", recon), ("rand", rand)):
            u = v / np.linalg.norm(v)
            # steer_pin picks the mode from the first segment, so it must be "pin"
            out[f"pin|{tag}_{t}|L{L}"] = u
            out[f"pin|{tag}_{t}|L{L}|a"] = np.array(float(a @ u))
            out[f"pin|{tag}_{t}|L{L}|d"] = np.array(n_true)
        c = float(true @ recon / (np.linalg.norm(true) * np.linalg.norm(recon)))
        print(f"{t:<10}{L:>6}{np.linalg.norm(true):>8.1f}{np.linalg.norm(recon):>9.1f}{c:>7.3f}")

    ds = sorted({BEST[t], round(BEST[t] * 0.75, 2)})
    cells += [[f"pin|true_{t}", ds], [f"pin|recon_{t}", ds], [f"pin|rand_{t}", [BEST[t]]]]

np.savez_compressed("attr_analysis/recon_steer_vectors.npz", **out)
json.dump(cells, open("attr_analysis/recon_steer_cells.json", "w"), indent=1)
print(f"\n{len(cells)} cells, {sum(len(c[1]) for c in cells)} cell-strength pairs, "
      f"{sum(len(c[1]) for c in cells) * 10} generations")
