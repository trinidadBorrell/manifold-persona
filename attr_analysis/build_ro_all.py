"""Pinned layer-23 steering in the no-prompt frame for all roles: own direction and rebuilds.

The direction of a persona is the mean read-out of its prompted answers, re-encoded with
no prompt, minus the centre of all personas' read-outs. Rebuilds fit that direction by
orthogonal matching pursuit over the other personas' directions, the eight nearest
neighbours banned. Writes the vectors, the cells for the full run and for the twelve
familiar personas, and the recipes.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import omp

L = 23
BAN = 8
KS = [8, 32]
OWN_STRENGTHS = [2.0, 3.0, 4.0, 6.0]
BASIS_STRENGTHS = [3.0, 4.0, 6.0]
SHOW = ["pirate", "musician", "spy", "toddler", "philosopher", "hacker", "witch",
        "vampire", "jester", "detective", "ghost", "scholar"]

X, cell = [], []
for f in sorted(glob.glob("l23all_out/acts_*.npz")):
    za = np.load(f, allow_pickle=True)
    li = list(za["layers"]).index(L)
    X.append(za["acts"][:, li, :].astype(np.float32))
    cell += list(za["cell"].astype(str))
X = np.concatenate(X)
ro = defaultdict(list)
for x, c in zip(X, cell):
    if c.startswith("prompt|"):
        ro[c.split("|")[1]].append(x)
ro = {t: np.mean(v, 0) for t, v in ro.items()}
targets = sorted(ro)
centre = np.mean([ro[t] for t in targets], 0)
d = {t: ro[t] - centre for t in targets}
print(f"{len(targets)} personas, centre from {sum(1 for c in cell if c.startswith('prompt|'))} prompted answers")

out, recipes, cells = {}, {}, {"own": [], "k8": [], "k32": []}
for t in targets:
    y = d[t]
    n = float(np.linalg.norm(y))
    u = (y / n).astype(np.float32)
    out[f"pin|ro_{t}|L{L}"] = u
    out[f"pin|ro_{t}|L{L}|a"] = np.array(float(centre @ u))
    out[f"pin|ro_{t}|L{L}|d"] = np.array(n)
    cells["own"].append([f"pin|ro_{t}", OWN_STRENGTHS])
    others = [p for p in targets if p != t]
    cos = {p: float(d[p] @ y / (np.linalg.norm(d[p]) * n)) for p in others}
    banned = sorted(cos, key=cos.get, reverse=True)[:BAN]
    basis = [p for p in others if p not in banned]
    A = np.stack([d[p] / np.linalg.norm(d[p]) for p in basis], 1)
    rec = {}
    for k in KS:
        w, sup = omp(y, A, k)
        rec[k] = [[basis[j], float(wi)] for j, wi in zip(sup, w)]
        v = A[:, sup] @ w
        uk = (v / np.linalg.norm(v)).astype(np.float32)
        out[f"pin|rk{k}_{t}|L{L}"] = uk
        out[f"pin|rk{k}_{t}|L{L}|a"] = np.array(float(centre @ uk))
        out[f"pin|rk{k}_{t}|L{L}|d"] = np.array(n)
        cells[f"k{k}"].append([f"pin|rk{k}_{t}", BASIS_STRENGTHS])
    recipes[t] = {"banned": banned, "k8": rec[8], "k32": rec[32], "norm": n,
                  "cos_k8": float(out[f"pin|rk8_{t}|L{L}"] @ u),
                  "cos_k32": float(out[f"pin|rk32_{t}|L{L}"] @ u)}

Path("l23all").mkdir(exist_ok=True)
np.savez_compressed("l23all/ro_all_vectors.npz", **out)
for arm, c in cells.items():
    json.dump(c, open(f"l23all/cells_ro_{arm}.json", "w"))
    json.dump([x for x in c if x[0].split("_", 1)[1] in SHOW], open(f"l23all/cells_ro12_{arm}.json", "w"))
json.dump({"layer": L, "targets": targets, "recipes": recipes},
          open("attr_analysis/ro_all_meta.json", "w"), indent=1)
n_gen = 10 * sum(len(s) for c in cells.values() for _, s in c)
print("cells: " + ", ".join(f"{k} {len(v)}" for k, v in cells.items()) + f"; {n_gen} generations")
cs8 = [r["cos_k8"] for r in recipes.values()]
cs32 = [r["cos_k32"] for r in recipes.values()]
print(f"cosine to the own direction: k8 median {np.median(cs8):.3f}, k32 median {np.median(cs32):.3f}")
print("twelve: " + ", ".join(f"{t} k8 {recipes[t]['cos_k8']:.2f} k32 {recipes[t]['cos_k32']:.2f}" for t in SHOW))
