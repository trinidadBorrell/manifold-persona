"""Build an interpretable orthogonal basis and check it clears 95%.

Axes are defined as contrasts between groups of personas, then Gram-Schmidt
orthogonalised in the order given, so each axis is what remains after the ones
before it. Everything is recomputed per layer.
"""
import glob
import json

import numpy as np

LAYERS = [26, 28, 30]
AGE = ["infant", "toddler", "adolescent", "teenager", "graduate",
       "parent", "grandparent", "retiree", "elder", "ancient"]
OCC = ["musician", "spy", "scholar", "soldier", "merchant"]
EXTRA = ["vampire", "ghost", "pirate", "wind", "philosopher", "nomad", "robot",
         "therapist", "accountant", "economist", "consultant", "idealist", "orphan"]

# contrast definitions: (name, positive group, negative group)
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
 ("craft",       ["musician","artisan","composer","poet"],         ["accountant","economist","auditor"]),
 ("commerce",    ["merchant","economist","accountant"],            ["musician","philosopher","orphan"]),
 ("authority",   ["soldier","consultant","judge"],                 ["orphan","infant","toddler"]),
 ("wonder",      ["infant","toddler","dreamer"],                   ["accountant","auditor","economist"]),
]


def load():
    P, N = [], []
    for f in sorted(glob.glob("attr_acts/*.npz")) + sorted(glob.glob("attrx_acts/*.npz")):
        z = np.load(f, allow_pickle=True)
        P.append(z["prompted"].astype(np.float32)); N.append(z["persona"].astype(str))
    return np.concatenate(P, 0), np.concatenate(N)


X, who = load()
res = {}
for L in LAYERS:
    Xa = X[:, L, :]
    a = Xa[who == "default"].mean(0)
    D = {p: Xa[who == p].mean(0) - a for p in sorted(set(who)) if p != "default"}
    M = np.stack([D[p] for p in sorted(D)])
    Mc = M - M.mean(0)
    s = np.linalg.svd(Mc, compute_uv=False)
    var = np.cumsum(s ** 2) / np.sum(s ** 2)

    B, kept, raw = [], [], []
    for name, pos, neg in CONCEPTS:
        pos = [p for p in pos if p in D]; neg = [p for p in neg if p in D]
        if not pos or not neg:
            continue
        v = np.mean([D[p] for p in pos], 0) - np.mean([D[p] for p in neg], 0)
        raw.append((name, v / np.linalg.norm(v)))
        for b in B:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            B.append(v / n); kept.append(name)
    Bm = np.stack(B, 1)
    ev = 1 - np.sum((Mc - Mc @ Bm @ Bm.T) ** 2) / np.sum(Mc ** 2)
    # how independent were the concepts before orthogonalisation?
    cos = [abs(raw[i][1] @ raw[j][1]) for i in range(len(raw)) for j in range(i + 1, len(raw))]
    res[str(L)] = {"n_axes": len(kept), "axes": kept, "explained": round(float(ev), 3),
                   "pca_same_dim": round(float(var[len(kept) - 1]), 3),
                   "pca_90": int(np.searchsorted(var, .9) + 1),
                   "pca_95": int(np.searchsorted(var, .95) + 1),
                   "pca_99": int(np.searchsorted(var, .99) + 1),
                   "raw_cos_mean": round(float(np.mean(cos)), 3),
                   "raw_cos_max": round(float(np.max(cos)), 3)}
    print(f"L{L}: {len(kept)} concept axes explain {ev:.3f} "
          f"(PCA same dim {var[len(kept)-1]:.3f}) | PCA needs {res[str(L)]['pca_95']} for 95%, "
          f"{res[str(L)]['pca_99']} for 99%")
    print(f"     raw concept cosines: mean {np.mean(cos):.2f}, max {np.max(cos):.2f}")

json.dump(res, open("attr_analysis/concepts.json", "w"), indent=1)
print("\n-> attr_analysis/concepts.json")
