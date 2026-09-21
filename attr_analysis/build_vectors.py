"""Full-space steering vectors for the analogy test, fitted per layer.

Each cell is (tag, layer) -> a 2048-d vector added to the residual stream. Vectors
are recomputed at every layer: fitting at one layer and applying across a band
produced token soup in an earlier run.
"""
import glob
import json

import numpy as np

OCC = ["musician", "spy", "scholar", "soldier", "merchant"]
LAYERS = [26, 27, 28, 29, 30]
ATTRS = [("age", "young", "old"), ("era", "medieval", "futuristic")]

P, names = [], []
for f in sorted(glob.glob("attr_acts/*.npz")):
    z = np.load(f, allow_pickle=True)
    P.append(z["prompted"].astype(np.float32))
    names.append(z["persona"].astype(str))
X = np.concatenate(P, 0)
who = np.concatenate(names)

out, cells = {}, []
for L in LAYERS:
    M = {p: X[who == p, L, :].mean(0) for p in np.unique(who)}
    asst = M["default"]
    D = {p: M[p] - asst for p in M if p != "default"}
    for attr, lo, hi in ATTRS:
        off = {o: D[f"{hi}_{o}"] - D[f"{lo}_{o}"] for o in OCC}
        for o in OCC:
            base, real = f"{lo}_{o}", f"{hi}_{o}"
            loo = np.mean([off[k] for k in off if k != o], 0)   # leave-one-out
            out[f"{attr}|base|{o}|L{L}"] = D[base]
            out[f"{attr}|real|{o}|L{L}"] = D[real]
            out[f"{attr}|pred|{o}|L{L}"] = D[base] + loo
            if L == LAYERS[0]:
                cells += [f"{attr}|base|{o}", f"{attr}|real|{o}", f"{attr}|pred|{o}"]

np.savez_compressed("attr_analysis/analogy_vectors.npz", **out)
json.dump(sorted(set(cells)), open("attr_analysis/analogy_cells.json", "w"), indent=1)
print(f"{len(out)} vectors ({len(set(cells))} cells x {len(LAYERS)} layers)")
for t in ["age|base|musician", "age|real|musician", "age|pred|musician"]:
    print(f"  |{t}| = {np.linalg.norm(out[t + '|L28']):.1f}")
