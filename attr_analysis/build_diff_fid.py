"""Two experiments.

A. age as a DIFFERENCE direction, not a destination. The rung vectors share a large
   'is a person' component that renders as childlike; a difference cancels it.

B. a fidelity ladder for reconstruction, with the control the critique demands:
   a random vector sharing the same cosine with the true vector. If that behaves
   like the persona too, then reconstruction adds nothing behaviourally.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all

YOUNG = ["toddler", "adolescent", "teenager"]
OLD = ["grandparent", "retiree", "elder"]
TARGETS = ["ghost", "pirate", "vampire"]
KS = [1, 2, 4, 8, 20]
NORM = 22.0
BETAS = [-1.0, -0.6, -0.3, 0.3, 0.6, 1.0]

means, _p, personas = load_all(need_pts=False)
personas = [p for p in personas if p != "default"]
rng = np.random.default_rng(0)
out, report = {}, {}

for L in LAYERS:
    a = means[L]["default"]
    D = {p: means[L][p] - a for p in personas}

    # ---- A. the age difference direction ----
    axis = np.mean([D[p] for p in OLD], 0) - np.mean([D[p] for p in YOUNG], 0)
    axis = axis / np.linalg.norm(axis)
    for b in BETAS:
        out[f"agediff|{b}|L{L}"] = axis * (NORM * b)
    # anchored variant: sit at a neutral adult and move along the axis
    anchor = D["graduate"] / np.linalg.norm(D["graduate"]) * NORM
    for b in BETAS:
        v = anchor + axis * (NORM * b)
        out[f"ageanch|{b}|L{L}"] = v / np.linalg.norm(v) * NORM

    # ---- B. fidelity ladder ----
    for t in TARGETS:
        y = D[t]
        others = [q for q in personas if q != t]
        A = np.stack([D[q] for q in others], 1)
        sel, r, w = [], y.copy(), None
        for k in range(1, max(KS) + 1):
            c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
                 for i in range(len(others)) if i not in sel]
            sel.append(max(c)[1])
            w, *_ = np.linalg.lstsq(A[:, sel], y, rcond=None)
            r = y - A[:, sel] @ w
            if k in KS:
                rec = A[:, sel] @ w
                cs = float(rec @ y / np.linalg.norm(rec) / np.linalg.norm(y))
                out[f"{t}|k{k}|L{L}"] = rec / np.linalg.norm(rec) * NORM
                if L == 28:
                    report.setdefault(t, {})[f"k{k}"] = {
                        "cos": round(cs, 4),
                        "r2": round(float(1 - np.sum(r**2)/np.sum(y**2)), 3),
                        "terms": [[others[i], round(float(v), 2)]
                                  for i, v in sorted(zip(sel, w), key=lambda z: -abs(z[1]))][:4]}
        out[f"{t}|true|L{L}"] = y / np.linalg.norm(y) * NORM
        # forced composition: drop the nearest neighbours so no synonym can carry it
        sim = sorted(others, key=lambda q: -(D[q] @ y /
                     (np.linalg.norm(D[q]) * np.linalg.norm(y))))
        banned = set(sim[:8])
        far = [q for q in others if q not in banned]
        Af = np.stack([D[q] for q in far], 1)
        selF, rF, wF = [], y.copy(), None
        for k in range(1, 13):
            c = [(abs(Af[:, i] @ rF) / np.linalg.norm(Af[:, i]), i)
                 for i in range(len(far)) if i not in selF]
            selF.append(max(c)[1])
            wF, *_ = np.linalg.lstsq(Af[:, selF], y, rcond=None)
            rF = y - Af[:, selF] @ wF
        recF = Af[:, selF] @ wF
        out[f"{t}|nosyn|L{L}"] = recF / np.linalg.norm(recF) * NORM
        if L == 28:
            report.setdefault(t, {})["nosyn"] = {
                "cos": round(float(recF @ y / np.linalg.norm(recF) / np.linalg.norm(y)), 4),
                "r2": round(float(1 - np.sum(rF**2)/np.sum(y**2)), 3),
                "banned": sim[:8],
                "terms": [[far[i], round(float(v), 2)]
                          for i, v in sorted(zip(selF, wF), key=lambda z: -abs(z[1]))][:5]}
        # random vectors matched to the cosine of the k=8 reconstruction
        rec8 = out[f"{t}|k8|L{L}"]
        target_cos = float(rec8 @ y / np.linalg.norm(rec8) / np.linalg.norm(y))
        u = y / np.linalg.norm(y)
        e = rng.normal(size=len(y)); e -= (e @ u) * u; e /= np.linalg.norm(e)
        tan = np.sqrt(max(1 - target_cos**2, 1e-12)) / target_cos
        v = u + tan * e
        out[f"{t}|cosmatch|L{L}"] = v / np.linalg.norm(v) * NORM
        e2 = rng.normal(size=len(y)); e2 /= np.linalg.norm(e2)
        out[f"{t}|rand|L{L}"] = e2 * NORM
        if L == 28:
            report[t]["cosmatch_target"] = round(target_cos, 4)

np.savez_compressed("attr_analysis/diff_fid_vectors.npz", **out)
cells = ([f"agediff|{b}" for b in BETAS] + [f"ageanch|{b}" for b in BETAS]
         + [f"{t}|{k}" for t in TARGETS
            for k in ["true"] + [f"k{k}" for k in KS] + ["nosyn", "cosmatch", "rand"]])
json.dump(cells, open("attr_analysis/diff_fid_cells.json", "w"), indent=1)
json.dump(report, open("attr_analysis/diff_fid_report.json", "w"), indent=1)
print(f"{len(out)} vectors, {len(cells)} cells\n")
print("fidelity ladder at L28 -- cosine with the true vector:")
for t in TARGETS:
    row = "  ".join(f"k{k}={report[t][f'k{k}']['cos']:.3f}" for k in KS)
    print(f"  {t:8} {row}   cosmatch random set to {report[t]['cosmatch_target']:.3f}")
print("\nforced composition (8 nearest neighbours banned):")
for t in TARGETS:
    n = report[t]["nosyn"]
    print(f"  {t:8} cos={n['cos']:.3f} R2={n['r2']:.2f}  banned: {', '.join(n['banned'][:4])}...")
    print(f"           = " + " ".join(f"{w:+.2f}*{q}" for q, w in n["terms"]))
print("\nk=1 and k=8 recipes:")
for t in TARGETS:
    for k in (1, 8):
        terms = " ".join(f"{w:+.2f}*{q}" for q, w in report[t][f"k{k}"]["terms"])
        print(f"  {t:8} k={k}: {terms}")
