"""Two coordinate systems for persona space, exported per layer.

  personas : greedily chosen persona directions, extended until 99% is reached
  concepts : named contrast axes, Gram-Schmidt orthogonalised

Both report, for every persona, its coordinates and how well they rebuild it.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all  # noqa: E402

LAYERS = list(LAYERS)
CONCEPTS = [
 ("character",   ["vampire","ghost","pirate","wind","nomad"],      ["accountant","consultant","economist"]),
 ("age",         ["grandparent","retiree","elder","ancient"],      ["infant","toddler","adolescent","teenager"]),
 ("nonhuman",    ["ghost","wind","robot","vampire"],               ["scholar","merchant","therapist","accountant"]),
 ("professional",["accountant","consultant","economist","scholar"],["pirate","vampire","nomad","orphan"]),
 ("era",         ["futuristic_musician","futuristic_spy","futuristic_scholar","futuristic_soldier"],
                 ["medieval_musician","medieval_spy","medieval_scholar","medieval_soldier"]),
 ("solitude",    ["orphan","nomad","wind","ghost"],                ["parent","grandparent","consultant","therapist"]),
 ("interiority", ["philosopher","idealist","therapist"],           ["soldier","merchant","spy"]),
 ("danger",      ["pirate","vampire","spy","soldier"],             ["therapist","scholar","idealist"]),
 ("commerce",    ["merchant","economist","accountant"],            ["musician","philosopher","orphan"]),
 ("wonder",      ["infant","toddler"],                             ["accountant","economist"]),
 ("authority",   ["soldier","consultant"],                         ["orphan","infant"]),
 ("craft",       ["musician"],                                     ["accountant","economist"]),
]

MEANS, _PTS, ALLP = load_all(need_pts=False)
personas = [p for p in ALLP if p != "default"]

out = {"layers": [str(L) for L in LAYERS], "personas": personas, "L": {}}
for L in LAYERS:
    a = MEANS[L]["default"]
    D = {p: MEANS[L][p] - a for p in personas}
    M = np.stack([D[p] for p in personas]); Mc = M - M.mean(0)
    tot = np.sum(Mc ** 2)
    sv = np.linalg.svd(Mc, compute_uv=False)
    pcavar = np.cumsum(sv ** 2) / np.sum(sv ** 2)

    # --- persona basis: greedily add whichever persona explains most remaining ---
    chosen, Q, curve = [], np.zeros((len(a), 0)), []
    R = Mc.copy()
    while True:
        best, bi = -1, None
        for i, p in enumerate(personas):
            if p in chosen:
                continue
            v = D[p].copy()
            if Q.shape[1]:
                v = v - Q @ (Q.T @ v)
            n = np.linalg.norm(v)
            if n < 1e-8:
                continue
            u = v / n
            g = float(np.sum((R @ u) ** 2))
            if g > best:
                best, bi, bu = g, p, u
        if bi is None:
            break
        chosen.append(bi); Q = np.hstack([Q, bu[:, None]])
        R = R - (R @ bu)[:, None] * bu[None, :]
        ev = 1 - np.sum(R ** 2) / tot
        curve.append(round(float(ev), 4))
        if ev >= 0.99 or len(chosen) >= 200:
            break

    # --- concept basis ---
    B, kept = [], []
    for name, pos, neg in CONCEPTS:
        pos = [p for p in pos if p in D]; neg = [p for p in neg if p in D]
        if not pos or not neg:
            continue
        v = np.mean([D[p] for p in pos], 0) - np.mean([D[p] for p in neg], 0)
        for b in B:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            B.append(v / n); kept.append(name)
    Bm = np.stack(B, 1)
    cev = 1 - np.sum((Mc - Mc @ Bm @ Bm.T) ** 2) / tot
    ccurve = []
    for k in range(1, len(kept) + 1):
        Bk = Bm[:, :k]
        ccurve.append(round(float(1 - np.sum((Mc - Mc @ Bk @ Bk.T) ** 2) / tot), 4))

    # leave-one-out: rebuild each persona from the basis WITHOUT itself, so a
    # persona in the basis cannot trivially explain itself
    looP, looFit = {}, {}
    for p in personas:
        idx = [i for i, b in enumerate(chosen) if b != p]
        Ai = np.stack([D[chosen[i]] for i in idx], 1)
        w, *_ = np.linalg.lstsq(Ai, D[p], rcond=None)
        r = D[p] - Ai @ w
        looP[p] = [[chosen[i], round(float(v), 3)] for i, v in zip(idx, w)]
        looFit[p] = round(float(1 - np.sum(r ** 2) / np.sum(D[p] ** 2)), 3)
    coordP = {p: np.round(Q.T @ D[p], 3).tolist() for p in personas}
    coordC = {p: np.round(Bm.T @ D[p], 3).tolist() for p in personas}
    fitP = {p: round(float(1 - np.sum((D[p] - Q @ (Q.T @ D[p])) ** 2) / np.sum(D[p] ** 2)), 3)
            for p in personas}
    fitC = {p: round(float(1 - np.sum((D[p] - Bm @ (Bm.T @ D[p])) ** 2) / np.sum(D[p] ** 2)), 3)
            for p in personas}
    out["L"][str(L)] = {
        "persona_basis": chosen, "persona_curve": curve,
        "concept_basis": kept, "concept_curve": ccurve,
        "concept_explained": round(float(cev), 3),
        "pca_curve": [round(float(x), 4) for x in pcavar[:200]],
        "pca_90": int(np.searchsorted(pcavar, .90) + 1),
        "pca_95": int(np.searchsorted(pcavar, .95) + 1),
        "pca_99": int(np.searchsorted(pcavar, .99) + 1),
        "n_personas": len(personas),
        "coordP": coordP, "coordC": coordC, "fitP": fitP, "fitC": fitC,
        "looP": looP, "looFit": looFit,
        "norms": {p: round(float(np.linalg.norm(D[p])), 2) for p in personas},
    }
    print(f"L{L}: {len(personas)} personas | persona basis {len(chosen)} -> {curve[-1]:.3f} | "
          f"{len(kept)} concepts -> {cev:.3f} | PCA 90/95/99% = "
          f"{int(np.searchsorted(pcavar,.90)+1)}/{int(np.searchsorted(pcavar,.95)+1)}/"
          f"{int(np.searchsorted(pcavar,.99)+1)}")

json.dump(out, open("attr_analysis/basis_data.json", "w"))
print("-> attr_analysis/basis_data.json")
