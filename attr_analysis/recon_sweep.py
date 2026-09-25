"""How close does a combination of other personas get, as a function of how many?

One weight vector shared across all 37 layers, fitted by matching pursuit. Run
with the target's nearest neighbours banned and again with nothing banned, so the
cost of forbidding a synonym is visible.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, TARGETS, omp, r2

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
BANS = [8, 0]
STEER = (26, 27, 28, 29, 30)
W = 2048

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}
pool = [p for p in names if p != "default"]

d = {}
for p in pool:
    d[p] = np.concatenate([M[idx[p], L] - M[idx["default"], L] for L in range(37)])
sl = {L: slice(L * W, (L + 1) * W) for L in range(37)}
steer_idx = np.concatenate([np.arange(sl[L].start, sl[L].stop) for L in STEER])

report = {}
for ban in BANS:
    print(f"\n===== nearest {ban} banned =====" if ban else
          "\n===== nothing banned (upper bound) =====")
    print(f"{'persona':<10}" + "".join(f"{f'k={k}':>7}" for k in KS))
    for t in TARGETS:
        y = d[t]
        others = [p for p in pool if p != t]
        cos = {p: float(d[p] @ y / (np.linalg.norm(d[p]) * np.linalg.norm(y)))
               for p in others}
        banned = sorted(cos, key=cos.get, reverse=True)[:ban]
        basis = [p for p in others if p not in banned]
        A = np.stack([d[p] for p in basis], 1)

        row, names_at = [], {}
        for k in KS:
            w, sup = omp(y, A, k)
            yh = A[:, sup] @ w
            c = float(yh[steer_idx] @ y[steer_idx] /
                      (np.linalg.norm(yh[steer_idx]) * np.linalg.norm(y[steer_idx])))
            row.append(c)
            names_at[k] = [[basis[j], float(wi)] for j, wi in zip(sup, w)]
        report[f"{t}|ban{ban}"] = {"ks": KS, "cos": row, "recipes": names_at,
                                   "banned": banned}
        print(f"{t:<10}" + "".join(f"{c:>7.3f}" for c in row))
    print(f"{'mean':<10}" + "".join(
        f"{np.mean([report[f'{t}|ban{ban}']['cos'][i] for t in TARGETS]):>7.3f}"
        for i in range(len(KS))))

json.dump(report, open("attr_analysis/recon_sweep.json", "w"), indent=1)
print("\n-> attr_analysis/recon_sweep.json")
