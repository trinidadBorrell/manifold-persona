"""Three ways to travel the age ladder, plus the rungs themselves.

  control    steer straight at each rung's centroid
  piecewise  follow the ladder itself, rung to rung
  spline     Catmull-Rom smooth curve through the rungs
  linear     straight line from youngest to oldest

beta = 0 is the youngest rung, 1 the oldest. All three paths are parameterised by
the SAME nominal age, so at a given beta every arm is aiming at the same target.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five as load_all
LAYERS = (26, 27, 28, 29, 30)   # all five, per-layer vectors

LADDER = ["infant", "toddler", "adolescent", "teenager", "graduate",
          "parent", "grandparent", "retiree", "elder"]
AGES = np.array([1, 3, 13, 16, 22, 38, 62, 66, 78], float)
BETAS = [round(b, 2) for b in np.linspace(0, 1, 11)]


def catmull_point(P, t):
    """t in [0, n-1] over control points P."""
    n = len(P)
    i = min(int(np.floor(t)), n - 2)
    u = t - i
    p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, n - 1)]
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


means, _p, _all = load_all()
out, meta = {}, {}
for L in LAYERS:
    a = means[L]["default"]
    D = np.stack([means[L][p] - a for p in LADDER])
    for i, p in enumerate(LADDER):
        out[f"control|{p}|L{L}"] = D[i]
    for b in BETAS:
        target_age = AGES[0] + b * (AGES[-1] - AGES[0])
        # position on the ladder in rung units, by nominal age
        t = float(np.interp(target_age, AGES, np.arange(len(LADDER))))
        i = min(int(np.floor(t)), len(LADDER) - 2)
        u = t - i
        out[f"piecewise|{b}|L{L}"] = D[i] * (1 - u) + D[i + 1] * u
        out[f"spline|{b}|L{L}"] = catmull_point(D, t)
        out[f"linear|{b}|L{L}"] = D[0] * (1 - b) + D[-1] * b
        if L == 28:
            meta[str(b)] = {"target_age": round(target_age, 1), "rung_t": round(t, 2)}

# Equal displacement for every target. Otherwise infant (a long vector) blows past
# the coherence wall at the same alpha that leaves the adult rungs barely moved.
TARGET_NORM = 22.0
for k, v in list(out.items()):
    meta.setdefault("raw_norms", {})[k] = round(float(np.linalg.norm(v)), 2)
    out[k] = v / np.linalg.norm(v) * TARGET_NORM
    meta.setdefault("norms", {})[k] = TARGET_NORM
np.savez_compressed("attr_analysis/age5_vectors.npz", **out)
cells = ([f"{p}|prompt" for p in LADDER]
         + [f"control|{p}" for p in LADDER]
         + [f"{arm}|{b}" for arm in ("piecewise", "spline", "linear") for b in BETAS])
json.dump(cells, open("attr_analysis/age5_cells.json", "w"), indent=1)
json.dump({"ladder": LADDER, "ages": AGES.tolist(), "betas": BETAS, **meta},
          open("attr_analysis/age5_meta.json", "w"), indent=1)
print(f"{len(out)} vectors, {len(cells)} cells")
print("\nbeta -> target age (all arms aim at the same one):")
print("  " + "  ".join(f"{b}:{meta[str(b)]['target_age']:.0f}y" for b in BETAS))
print("\nvector norm at L28 by arm (steering distance from the assistant):")
for arm in ("piecewise", "spline", "linear"):
    ns = [meta["norms"][f"{arm}|{b}|L28"] for b in BETAS]
    print(f"  {arm:10} min {min(ns):6.1f}  max {max(ns):6.1f}  median {np.median(ns):6.1f}")
