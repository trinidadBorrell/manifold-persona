"""Single late layer, single mid layer, and five-layer bands, matched strength grid.

Each arm sees the same strengths so none is handicapped by the coherence wall, and
the winner is read off each arm's own best strength.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS

ARMS = {"one23": [23], "one20": [20], "one26": [26],
        "band19": [19, 20, 21, 22, 23], "band26": [26, 27, 28, 29, 30]}
STRENGTHS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}

out = {}
for t in TARGETS:
    for L in sorted({L for ls in ARMS.values() for L in ls}):
        a = M[idx["default"], L].astype(np.float32)
        d = M[idx[t], L].astype(np.float32) - a
        n = float(np.linalg.norm(d))
        u = d / n
        out[f"pin|{t}|L{L}"] = u
        out[f"pin|{t}|L{L}|a"] = np.array(float(a @ u))
        out[f"pin|{t}|L{L}|d"] = np.array(n)

cells = [[f"pin|{t}", STRENGTHS] for t in TARGETS]
np.savez_compressed("attr_analysis/bandcmp_vectors.npz", **out)
json.dump(cells, open("attr_analysis/bandcmp_cells.json", "w"), indent=1)
json.dump(ARMS, open("attr_analysis/bandcmp_arms.json", "w"), indent=1)
n = len(ARMS) * len(TARGETS) * len(STRENGTHS) * 10
print(f"{len(ARMS)} arms x {len(TARGETS)} personas x {len(STRENGTHS)} strengths x 10 q = {n}")
