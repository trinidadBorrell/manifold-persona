"""Four ways to become a persona, as steering vectors (plus a prompt arm).

  prompt   the persona's own system prompt, no steering at all
  true     steer with the persona's own mean direction
  recon    steer with a reconstruction from OTHER personas
  concept  steer with only the part that the 18 named concept axes can express
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all           # noqa: E402
from concept_defs import CONCEPTS             # noqa: E402

TARGETS = ["ghost", "pirate", "vampire", "musician", "spy", "philosopher"]
N_SPARSE = 8

means, _p, personas = load_all(need_pts=False)
personas = [p for p in personas if p != "default"]
out, report = {}, {}
for L in LAYERS + (27, 29):
    a = means[L]["default"] if L in means else None
    if a is None:
        continue
    D = {p: means[L][p] - a for p in personas}

    # concept basis at this layer
    B = []
    names = []
    for name, pos, neg in CONCEPTS:
        pos = [p for p in pos if p in D]; neg = [p for p in neg if p in D]
        if len(pos) < 2 or len(neg) < 2:
            continue
        v = np.mean([D[p] for p in pos], 0) - np.mean([D[p] for p in neg], 0)
        for b in B:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-9:
            B.append(v / n); names.append(name)
    Bm = np.stack(B, 1)

    for t in TARGETS:
        y = D[t]
        out[f"{t}|true|L{L}"] = y
        # reconstruction from other personas, greedy sparse
        others = [q for q in personas if q != t]
        A = np.stack([D[q] for q in others], 1)
        sel, r, w = [], y.copy(), None
        for _ in range(N_SPARSE):
            c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
                 for i in range(len(others)) if i not in sel]
            sel.append(max(c)[1])
            w, *_ = np.linalg.lstsq(A[:, sel], y, rcond=None)
            r = y - A[:, sel] @ w
        out[f"{t}|recon|L{L}"] = A[:, sel] @ w
        # the part expressible in named concepts
        cc = Bm.T @ y
        out[f"{t}|concept|L{L}"] = Bm @ cc
        if L == 28:
            report[t] = {
                "recon_r2": round(float(1 - np.sum(r**2)/np.sum(y**2)), 3),
                "recon_terms": [[others[i], round(float(v), 2)]
                                for i, v in sorted(zip(sel, w), key=lambda z: -abs(z[1]))],
                "concept_r2": round(float(1 - np.sum((y - Bm@cc)**2)/np.sum(y**2)), 3),
                "concept_terms": [[names[i], round(float(cc[i]), 2)]
                                  for i in np.argsort(-np.abs(cc))[:6]],
                "norm": round(float(np.linalg.norm(y)), 2)}

# equal displacement for every persona: alpha meant very different distances
# otherwise (||pirate|| = 38.8 against ||spy|| = 14.1)
TARGET_NORM = 20.0
for k in list(out):
    v = out[k]
    out[k] = v / np.linalg.norm(v) * TARGET_NORM
np.savez_compressed("attr_analysis/demo_vectors.npz", **out)
cells = [f"{t}|{arm}" for t in TARGETS for arm in ("prompt", "true", "recon", "concept")]
json.dump(cells, open("attr_analysis/demo_cells.json", "w"), indent=1)
json.dump(report, open("attr_analysis/demo_report.json", "w"), indent=1)
print(f"{len(out)} vectors, {len(cells)} cells\n")
for t, v in report.items():
    print(f"{t}  (|v|={v['norm']})")
    print(f"   recon   R2={v['recon_r2']}  " +
          " ".join(f"{w:+.2f}*{q}" for q, w in v["recon_terms"][:5]))
    print(f"   concept R2={v['concept_r2']}  " +
          " ".join(f"{w:+.1f}*{q}" for q, w in v["concept_terms"][:5]))
