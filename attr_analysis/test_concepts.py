"""Do the 18 concept axes clear the 95% bar, and are they independent enough?

Reports, per layer: variance explained before and after orthogonalisation, the
raw cosines between concepts (the previous set had a pair at 0.98, which wasted
axes), and how many PCA dims would be needed for the same coverage.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all            # noqa: E402
from concept_defs import CONCEPTS              # noqa: E402

means, _pts, personas = load_all(need_pts=False)
personas = [p for p in personas if p != "default"]
res = {}
for L in LAYERS:
    a = means[L]["default"]
    D = {p: means[L][p] - a for p in personas}
    M = np.stack([D[p] for p in personas]); Mc = M - M.mean(0)
    tot = np.sum(Mc ** 2)
    sv = np.linalg.svd(Mc, compute_uv=False)
    pca = np.cumsum(sv ** 2) / np.sum(sv ** 2)

    raw, B, kept, dropped = [], [], [], []
    for name, pos, neg in CONCEPTS:
        pos = [p for p in pos if p in D]; neg = [p for p in neg if p in D]
        if len(pos) < 2 or len(neg) < 2:
            dropped.append((name, len(pos), len(neg))); continue
        v = np.mean([D[p] for p in pos], 0) - np.mean([D[p] for p in neg], 0)
        raw.append((name, v / np.linalg.norm(v)))
        for b in B:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            B.append(v / n); kept.append(name)
    Bm = np.stack(B, 1)
    ev = float(1 - np.sum((Mc - Mc @ Bm @ Bm.T) ** 2) / tot)
    curve = [float(1 - np.sum((Mc - Mc @ Bm[:, :k] @ Bm[:, :k].T) ** 2) / tot)
             for k in range(1, len(kept) + 1)]
    cos = {f"{raw[i][0]}~{raw[j][0]}": float(abs(raw[i][1] @ raw[j][1]))
           for i in range(len(raw)) for j in range(i + 1, len(raw))}
    worst = sorted(cos.items(), key=lambda z: -z[1])[:3]
    res[str(L)] = {"n_axes": len(kept), "axes": kept, "explained": round(ev, 3),
                   "curve": [round(c, 3) for c in curve],
                   "pca_same_dim": round(float(pca[len(kept) - 1]), 3),
                   "pca_95": int(np.searchsorted(pca, .95) + 1),
                   "pca_99": int(np.searchsorted(pca, .99) + 1),
                   "worst_pairs": worst, "dropped": dropped,
                   "mean_raw_cos": round(float(np.mean(list(cos.values()))), 3)}
    ok = "PASS" if ev >= 0.95 else "below bar"
    print(f"L{L}: {len(kept)} axes -> {ev:.3f}  [{ok}]  "
          f"(PCA same dim {pca[len(kept)-1]:.3f}; PCA needs {res[str(L)]['pca_95']} for 95%)")
    print(f"     mean raw cosine {res[str(L)]['mean_raw_cos']:.2f}; most redundant: " +
          ", ".join(f"{k} {v:.2f}" for k, v in worst))
    if dropped:
        print(f"     dropped for too few anchors: {[d[0] for d in dropped]}")

json.dump(res, open("attr_analysis/concepts18.json", "w"), indent=1)
print("\n-> attr_analysis/concepts18.json")
