"""Layer 23 across a wider persona set, with the basis rebuilt there.

Five arms: the persona's own vector at three depths, to test whether 23 holds
beyond the first five personas, and the shared all-layer reconstruction at 8 and
32 personas, to test whether the basis works at the better depth.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, omp

TARGETS = ["ghost", "pirate", "vampire", "musician", "spy", "toddler",
           "scholar", "witch", "detective", "jester", "philosopher", "hacker"]
STRENGTHS = [3.0, 4.0, 6.0, 8.0]
BAN = 8
KS = [8, 32]
ARMS = {"L20": [20], "L23": [23], "L26": [26], "k8": [23], "k32": [23]}
W = 2048

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}
pool = [p for p in names if p != "default"]

d = {p: np.concatenate([M[idx[p], L] - M[idx["default"], L] for L in range(37)])
     for p in pool}
sl = {L: slice(L * W, (L + 1) * W) for L in range(37)}

out, recipes = {}, {}
print(f"{'persona':<13}{'|d23|':>7}{'cos k8':>8}{'cos k32':>9}  banned")
for t in TARGETS:
    y = d[t]
    others = [p for p in pool if p != t]
    cos = {p: float(d[p] @ y / (np.linalg.norm(d[p]) * np.linalg.norm(y))) for p in others}
    banned = sorted(cos, key=cos.get, reverse=True)[:BAN]
    basis = [p for p in others if p not in banned]
    A = np.stack([d[p] for p in basis], 1)

    rec, cs = {}, []
    for k in KS:
        w, sup = omp(y, A, k)
        rec[k] = [[basis[j], float(wi)] for j, wi in zip(sup, w)]
        yh = A[:, sup] @ w
        q = yh[sl[23]]
        cs.append(float(q @ y[sl[23]] / (np.linalg.norm(q) * np.linalg.norm(y[sl[23]]))))
    recipes[t] = rec

    for arm, layers in ARMS.items():
        for L in layers:
            a = M[idx["default"], L].astype(np.float32)
            true = M[idx[t], L].astype(np.float32) - a
            n_true = float(np.linalg.norm(true))
            if arm.startswith("k"):
                v = np.zeros_like(true)
                for name, wi in rec[int(arm[1:])]:
                    v += np.float32(wi) * (M[idx[name], L].astype(np.float32) - a)
            else:
                v = true
            u = v / np.linalg.norm(v)
            out[f"pin|{arm}_{t}|L{L}"] = u
            out[f"pin|{arm}_{t}|L{L}|a"] = np.array(float(a @ u))
            out[f"pin|{arm}_{t}|L{L}|d"] = np.array(n_true)
    print(f"{t:<13}{np.linalg.norm(y[sl[23]]):>7.1f}{cs[0]:>8.3f}{cs[1]:>9.3f}  "
          + ", ".join(banned[:3]))

cells = [[f"pin|{arm}_{t}", STRENGTHS] for arm in ARMS for t in TARGETS]
np.savez_compressed("attr_analysis/l23_vectors.npz", **out)
json.dump(cells, open("attr_analysis/l23_cells.json", "w"), indent=1)
json.dump({"arms": ARMS, "targets": TARGETS, "recipes": recipes},
          open("attr_analysis/l23_meta.json", "w"), indent=1)
print(f"\n{len(cells)} cells, {len(cells) * len(STRENGTHS) * 10} generations")
