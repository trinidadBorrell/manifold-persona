"""Pinned layer-23 steering along the judge classifier's optimal direction for every role.

The direction per persona comes from clf_direction.py. The pin reference is the read-out
centre projected on that direction, and the strength unit is the persona's own distance
from the assistant mean, so s is comparable with the own-vector runs.
"""
import json
from pathlib import Path

import numpy as np

L = 23
STRENGTHS = [4.0, 8.0, 12.0]

v = np.load("attr_analysis/clf_direction_vectors.npz", allow_pickle=True)
names = list(v["names"].astype(str))
dirs = v["dirs"].astype(np.float32)
centre = v["centre"].astype(np.float32)
a = v["a"].astype(np.float32)
z = np.load("attr_analysis/_alllayer_cache.npz", allow_pickle=True)
cnames = list(z["names"].astype(str))
M = z["M"][:, L, :].astype(np.float32)
idx = {p: i for i, p in enumerate(cnames)}

out, cells = {}, []
for t, u in zip(names, dirs):
    u = u / np.linalg.norm(u)
    out[f"pin|clf_{t}|L{L}"] = u
    out[f"pin|clf_{t}|L{L}|a"] = np.array(float(centre @ u))
    out[f"pin|clf_{t}|L{L}|d"] = np.array(float(np.linalg.norm(M[idx[t]] - a)))
    cells.append([f"pin|clf_{t}", STRENGTHS])
Path("l23all").mkdir(exist_ok=True)
np.savez_compressed("l23all/clfaim_vectors.npz", **out)
json.dump(cells, open("l23all/cells_clf.json", "w"))
print(f"{len(cells)} cells x {len(STRENGTHS)} strengths x 10 questions = {len(cells) * len(STRENGTHS) * 10} generations")
