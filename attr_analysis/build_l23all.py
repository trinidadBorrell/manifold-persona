"""Pinned layer-23 steering for all 276 roles: own vector, and rebuilds from 8 and 32 others.

Recipes are fitted once on the directions stacked over all 37 layers (orthogonal
matching pursuit, the target's eight nearest neighbours banned) and read at layer 23.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "attr_analysis")
from shared_all import CACHE, omp

ROLES = sorted(p.stem for p in Path("refs/assistant-axis/roles/instructions").glob("*.json"))
L = 23
BAN = 8
KS = [8, 32]
W = 2048
OWN_STRENGTHS = [3.0, 4.0, 6.0, 8.0]
BASIS_STRENGTHS = [6.0, 8.0]

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"]
idx = {p: i for i, p in enumerate(names)}
pool = [p for p in names if p != "default"]
targets = [r for r in ROLES if r in idx and r != "default"]
missing = [r for r in ROLES if r not in idx]
print(f"{len(targets)} targets, missing from the cache: {missing}")

d = {p: np.concatenate([M[idx[p], L2] - M[idx["default"], L2] for L2 in range(37)])
     for p in pool}
sl = slice(L * W, (L + 1) * W)
a = M[idx["default"], L].astype(np.float32)

out, recipes, cells = {}, {}, {"own": [], "prompt": [], "k8": [], "k32": []}
for t in targets:
    y = d[t]
    others = [p for p in pool if p != t]
    cos = {p: float(d[p] @ y / (np.linalg.norm(d[p]) * np.linalg.norm(y))) for p in others}
    banned = sorted(cos, key=cos.get, reverse=True)[:BAN]
    basis = [p for p in others if p not in banned]
    A = np.stack([d[p] for p in basis], 1)
    true = M[idx[t], L].astype(np.float32) - a
    n_true = float(np.linalg.norm(true))
    u = true / n_true
    out[f"pin|own_{t}|L{L}"] = u
    out[f"pin|own_{t}|L{L}|a"] = np.array(float(a @ u))
    out[f"pin|own_{t}|L{L}|d"] = np.array(n_true)
    cells["own"].append([f"pin|own_{t}", OWN_STRENGTHS])
    cells["prompt"].append([f"prompt|{t}", [0.0]])
    rec = {}
    for k in KS:
        w, sup = omp(y, A, k)
        rec[k] = [[basis[j], float(wi)] for j, wi in zip(sup, w)]
        v = np.zeros_like(true)
        for name, wi in rec[k]:
            v += np.float32(wi) * (M[idx[name], L].astype(np.float32) - a)
        uk = v / np.linalg.norm(v)
        out[f"pin|k{k}_{t}|L{L}"] = uk
        out[f"pin|k{k}_{t}|L{L}|a"] = np.array(float(a @ uk))
        out[f"pin|k{k}_{t}|L{L}|d"] = np.array(n_true)
        cells[f"k{k}"].append([f"pin|k{k}_{t}", BASIS_STRENGTHS])
    recipes[t] = {"banned": banned, "k8": rec[8], "k32": rec[32],
                  "cos_k8": float(out[f"pin|k8_{t}|L{L}"] @ u),
                  "cos_k32": float(out[f"pin|k32_{t}|L{L}"] @ u), "norm": n_true}

Path("l23all").mkdir(exist_ok=True)
np.savez_compressed("l23all/l23all_vectors.npz", **out)
for arm, c in cells.items():
    json.dump(c, open(f"l23all/cells_{arm}.json", "w"))
json.dump({"layer": L, "targets": targets, "recipes": recipes},
          open("attr_analysis/l23all_meta.json", "w"), indent=1)
n_gen = 10 * sum(len(s) for c in cells.values() for _, s in c)
print(f"cells: " + ", ".join(f"{k} {len(v)}" for k, v in cells.items()) + f"; {n_gen} generations")
cs8 = [r["cos_k8"] for r in recipes.values()]
cs32 = [r["cos_k32"] for r in recipes.values()]
print(f"cosine to the own direction at L{L}: k8 median {np.median(cs8):.3f}, "
      f"k32 median {np.median(cs32):.3f}")
