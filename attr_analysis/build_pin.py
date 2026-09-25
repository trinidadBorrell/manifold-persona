"""Additive and pinned steering vectors for the same personas, at layers 26-30.

add: h += alpha * d_L, with d_L = persona mean - assistant mean at layer L.
pin: h += (goal_L - <h, u_L>) * u_L, with u_L = d_L / |d_L| and
     goal_L = <assistant mean, u_L> + s * |d_L|.
ref: the 3-layer, 22-unit additive setting that gave 83% in-role.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five

TARGETS = ["ghost", "pirate", "vampire", "musician", "spy"]
LAYERS = (26, 27, 28, 29, 30)
ADD_ALPHAS = [0.25, 0.5, 0.75, 1.0]
PIN_SCALES = [1.0, 1.5, 2.0, 3.0]

means, _p, _a = load_five()
out, cells = {}, []
for t in TARGETS:
    for L in LAYERS:
        a = means[L]["default"]
        d = means[L][t] - a
        n = float(np.linalg.norm(d))
        u = d / n
        out[f"add|{t}|L{L}"] = d
        out[f"pin|{t}|L{L}"] = u
        out[f"pin|{t}|L{L}|a"] = np.array(float(a @ u))
        out[f"pin|{t}|L{L}|d"] = np.array(n)
        out[f"ref|{t}|L{L}"] = (u * 22.0) if L in (26, 28, 30) else np.zeros_like(u)
    cells += [[f"add|{t}", ADD_ALPHAS], [f"pin|{t}", PIN_SCALES], [f"ref|{t}", [1.0]]]

np.savez_compressed("attr_analysis/pin_vectors.npz", **out)
json.dump(cells, open("attr_analysis/pin_cells.json", "w"), indent=1)
print(f"{len(cells)} cells, {sum(len(c[1]) for c in cells)} cell-alpha pairs")
for t in TARGETS:
    ns = [float(out[f'pin|{t}|L{L}|d']) for L in LAYERS]
    print(f"  {t:9} |d_L| " + " ".join(f"{x:5.1f}" for x in ns)
          + f"   additive sum {sum(ns):5.1f} vs pinned {ns[-1]:4.1f} at L30")
