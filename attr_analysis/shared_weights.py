"""One recipe for a persona that holds at every layer.

Per-layer fitting gives five different weight vectors and is therefore not a recipe
you can steer with. Here the weights are forced to explain all five layers at once:
each persona becomes a single 5x2048 column, and one w solves the stacked system.

Three variants, increasingly constrained:
  perlayer  independent fit at each layer      (5x the free parameters, upper bound)
  scaled    one shared direction + per-layer scale
  shared    one weight vector, nothing else    (what you steer with)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five

LAYERS = (26, 27, 28, 29, 30)
TARGETS = ["ghost", "pirate", "vampire", "musician", "spy", "philosopher"]
K = 8
BAN_NEAR = 8

means, _p, personas = load_five()
personas = [p for p in personas if p != "default"]
D = {L: {p: means[L][p] - means[L]["default"] for p in personas} for L in LAYERS}
print(f"{len(personas)} personas with full 5-layer data\n")

def stack(p):
    return np.concatenate([D[L][p] for L in LAYERS])

def omp(y, A, k):
    sel, r, w = [], y.copy(), None
    for _ in range(k):
        c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
             for i in range(A.shape[1]) if i not in sel]
        sel.append(max(c)[1])
        w, *_ = np.linalg.lstsq(A[:, sel], y, rcond=None)
        r = y - A[:, sel] @ w
    return sel, w, 1 - np.sum(r ** 2) / np.sum(y ** 2)

report = {}
print(f"{'target':<12}{'per-layer R2':>14}{'shared R2':>11}{'scaled R2':>11}{'cost':>8}")
for t in TARGETS:
    others = [q for q in personas if q != t]
    # ban near-synonyms, judged on the stacked vector so the ban is layer-consistent
    ys = stack(t)
    sim = sorted(others, key=lambda q: -(stack(q) @ ys /
                 (np.linalg.norm(stack(q)) * np.linalg.norm(ys))))
    banned = sim[:BAN_NEAR]
    far = [q for q in others if q not in banned]

    # 1. per-layer, independent  (upper bound)
    per_r2, per_terms = [], {}
    for L in LAYERS:
        A = np.stack([D[L][q] for q in far], 1)
        sel, w, r2 = omp(D[L][t], A, K)
        per_r2.append(r2)
        per_terms[L] = [[far[i], round(float(v), 2)] for i, v in
                        sorted(zip(sel, w), key=lambda z: -abs(z[1]))][:4]

    # 2. shared weights across all layers
    As = np.stack([stack(q) for q in far], 1)
    selS, wS, r2S = omp(ys, As, K)
    shared_terms = [[far[i], round(float(v), 3)] for i, v in
                    sorted(zip(selS, wS), key=lambda z: -abs(z[1]))]

    # 3. shared direction + per-layer scale
    sub = [far[i] for i in selS]
    scaled_num, scaled_den = 0.0, 0.0
    per_scale = {}
    for L in LAYERS:
        Ak = np.stack([D[L][q] for q in sub], 1)
        pred = Ak @ wS
        s = float(pred @ D[L][t] / (pred @ pred))
        per_scale[L] = round(s, 3)
        scaled_num += np.sum((D[L][t] - s * pred) ** 2)
        scaled_den += np.sum(D[L][t] ** 2)
    r2_scaled = 1 - scaled_num / scaled_den

    report[t] = {"per_layer_r2": [round(x, 3) for x in per_r2],
                 "per_layer_mean": round(float(np.mean(per_r2)), 3),
                 "shared_r2": round(float(r2S), 3),
                 "scaled_r2": round(float(r2_scaled), 3),
                 "per_scale": per_scale,
                 "shared_terms": shared_terms,
                 "banned": banned,
                 "per_layer_terms_L26": per_terms[26],
                 "per_layer_terms_L30": per_terms[30]}
    print(f"{t:<12}{np.mean(per_r2):>14.3f}{r2S:>11.3f}{r2_scaled:>11.3f}"
          f"{np.mean(per_r2) - r2S:>8.3f}")

print("\n=== the single recipe that holds at every layer ===")
for t in TARGETS:
    r = report[t]
    terms = " ".join(f"{w:+.2f}*{q}" for q, w in r["shared_terms"][:5])
    print(f"  {t:<12} R2={r['shared_r2']:.2f}  {terms}")
    print(f"  {'':12} per-layer scale needed: "
          + " ".join(f"L{L}:{s}" for L, s in r["per_scale"].items()))

print("\n=== do per-layer recipes even agree with each other? ===")
for t in TARGETS:
    a = {q for q, _ in report[t]["per_layer_terms_L26"]}
    b = {q for q, _ in report[t]["per_layer_terms_L30"]}
    print(f"  {t:<12} L26 top-4 {sorted(a)}")
    print(f"  {'':12} L30 top-4 {sorted(b)}   overlap {len(a & b)}/4")

json.dump(report, open("attr_analysis/shared_weights.json", "w"), indent=1)
print("\n-> attr_analysis/shared_weights.json")
