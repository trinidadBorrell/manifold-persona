"""Steering vectors for the causal test: can a persona rebuilt from others act like it?

Three arms per target: the true vector, a 6-component reconstruction from the other
40 personas, and a random vector of matched length. Everything refit per layer.
"""
import glob
import json

import numpy as np

TARGETS = ["old_musician", "medieval_spy", "old_spy", "futuristic_scholar",
           "toddler", "elder"]
LAYERS = [26, 27, 28, 29, 30]

P, N = [], []
for f in sorted(glob.glob("attr_acts/*.npz")):
    z = np.load(f, allow_pickle=True)
    P.append(z["prompted"].astype(np.float32)); N.append(z["persona"].astype(str))
X = np.concatenate(P, 0); who = np.concatenate(N)
rng = np.random.default_rng(0)

out, report = {}, {}
for L in LAYERS:
    Xa = X[:, L, :]
    a = Xa[who == "default"].mean(0)
    D = {p: Xa[who == p].mean(0) - a for p in sorted(set(who)) if p != "default"}
    names = sorted(D)
    for t in TARGETS:
        others = [q for q in names if q != t]
        A = np.stack([D[q] for q in others], 1)
        y = D[t]
        sel, r, w = [], y.copy(), None
        for _ in range(6):
            c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
                 for i in range(len(others)) if i not in sel]
            sel.append(max(c)[1])
            w, *_ = np.linalg.lstsq(A[:, sel], y, rcond=None)
            r = y - A[:, sel] @ w
        rec = A[:, sel] @ w
        rv = rng.normal(size=y.shape); rv *= np.linalg.norm(y) / np.linalg.norm(rv)
        out[f"{t}|true|L{L}"] = y
        out[f"{t}|recon|L{L}"] = rec
        out[f"{t}|rand|L{L}"] = rv
        if L == 28:
            report[t] = {"r2": round(float(1 - np.sum(r**2)/np.sum(y**2)), 3),
                         "cos_true_recon": round(float(y @ rec /
                                                       (np.linalg.norm(y)*np.linalg.norm(rec))), 3),
                         "terms": [[others[i], round(float(v), 2)]
                                   for i, v in sorted(zip(sel, w), key=lambda z: -abs(z[1]))]}

np.savez_compressed("attr_analysis/recon_vectors.npz", **out)
json.dump([f"{t}|{arm}" for t in TARGETS for arm in ["true", "recon", "rand"]],
          open("attr_analysis/recon_cells.json", "w"), indent=1)
json.dump(report, open("attr_analysis/recon_report.json", "w"), indent=1)
print(f"{len(out)} vectors, {len(TARGETS)*3} cells x {len(LAYERS)} layers")
for t, v in report.items():
    print(f"  {t:20} R2={v['r2']:.2f} cos(true,recon)={v['cos_true_recon']:.3f}")
