"""Pin many personas to the same absolute distance, instead of a multiple of their own.

If the sweet spot really is ~80-105 units regardless of persona, one distance should
work for all of them. s is just D / |d_L30|, so the existing pin machinery is unchanged.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five

DISTS = [70.0, 90.0, 110.0]
LAYERS = (26, 27, 28, 29, 30)
N_PICK = 16

means, _p, personas = load_five()
personas = [p for p in personas if p != "default"]
d30 = {p: float(np.linalg.norm(means[30][p] - means[30]["default"])) for p in personas}
# spread the picks evenly across the range of persona distances
order = sorted(personas, key=lambda p: d30[p])
pick = [order[int(round(i * (len(order) - 1) / (N_PICK - 1)))] for i in range(N_PICK)]
pick = list(dict.fromkeys(pick))

out, cells = {}, []
for t in pick:
    for L in LAYERS:
        a = means[L]["default"]
        d = means[L][t] - a
        n = float(np.linalg.norm(d))
        u = d / n
        out[f"pin|{t}|L{L}"] = u
        out[f"pin|{t}|L{L}|a"] = np.array(float(a @ u))
        out[f"pin|{t}|L{L}|d"] = np.array(n)
    cells.append([f"pin|{t}", [round(D / d30[t], 4) for D in DISTS]])

np.savez_compressed("attr_analysis/fixdist_vectors.npz", **out)
json.dump(cells, open("attr_analysis/fixdist_cells.json", "w"), indent=1)
json.dump({"dists": DISTS, "d30": {p: round(d30[p], 1) for p in pick}},
          open("attr_analysis/fixdist_meta.json", "w"), indent=1)
print(f"{len(pick)} personas, |d30| from {d30[pick[0]]:.1f} to {d30[pick[-1]]:.1f}")
print(f"{len(cells) * len(DISTS)} cells\n")
for t in pick:
    print(f"  {t:<20} |d30|={d30[t]:5.1f}   s for 70/90/110: "
          + " ".join(f"{D/d30[t]:4.1f}" for D in DISTS))
