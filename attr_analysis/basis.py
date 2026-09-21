"""Can composites be built from the *vendored* personas only, and what is the
smallest interpretable basis that spans persona space?

Vendored = prompts shipped with the paper's role set (we did not write them).
Ours     = the composite and era prompts written for this experiment.
"""
import glob
import itertools
import json

import numpy as np

AGE = ["infant", "toddler", "adolescent", "teenager", "graduate",
       "parent", "grandparent", "retiree", "elder", "ancient"]
OCC = ["musician", "spy", "scholar", "soldier", "merchant"]
EXTRA = ["vampire", "ghost", "pirate", "wind", "philosopher", "nomad", "robot",
         "therapist", "accountant", "economist", "consultant", "idealist", "orphan"]
VENDORED = AGE + OCC + EXTRA
LAYER = 28

P, N = [], []
for f in sorted(glob.glob("attr_acts/*.npz")) + sorted(glob.glob("attrx_acts/*.npz")):
    z = np.load(f, allow_pickle=True)
    P.append(z["prompted"].astype(np.float32)); N.append(z["persona"].astype(str))
X = np.concatenate(P, 0)[:, LAYER, :]
who = np.concatenate(N)
a = X[who == "default"].mean(0)
D = {p: X[who == p].mean(0) - a for p in sorted(set(who)) if p != "default"}
OURS = [p for p in D if p not in VENDORED]
print(f"{len(D)} personas: {len(VENDORED)} vendored, {len(OURS)} ours\n")


def omp(y, A, cols, k):
    sel, r = [], y.copy()
    for _ in range(k):
        c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
             for i in range(len(cols)) if i not in sel]
        sel.append(max(c)[1])
        w, *_ = np.linalg.lstsq(A[:, sel], y, rcond=None)
        r = y - A[:, sel] @ w
    return sel, w, 1 - np.sum(r ** 2) / np.sum(y ** 2)


# ---- 1. composites from vendored personas only -------------------------------
A = np.stack([D[q] for q in VENDORED], 1)
print("[1] our composites rebuilt from VENDORED personas only")
rows = []
for t in sorted(OURS):
    wf, *_ = np.linalg.lstsq(A, D[t], rcond=None)
    r2f = 1 - np.sum((D[t] - A @ wf) ** 2) / np.sum(D[t] ** 2)
    sel, w, r2s = omp(D[t], A, VENDORED, 4)
    rows.append((t, r2f, r2s, [(VENDORED[i], round(float(v), 2))
                               for i, v in sorted(zip(sel, w), key=lambda z: -abs(z[1]))]))
print(f"    full 28-vendored basis  R2 mean {np.mean([r[1] for r in rows]):.3f}")
print(f"    sparse 4 components     R2 mean {np.mean([r[2] for r in rows]):.3f}")
for t, r2f, r2s, terms in rows:
    if t in ("old_musician", "young_musician", "medieval_spy", "futuristic_scholar",
             "era_victorian", "old_soldier"):
        print(f"      {t:19} R2={r2s:.2f}  " +
              " ".join(f"{v:+.2f}*{q}" for q, v in terms))

# ---- 2. how many dimensions does persona space need? -------------------------
M = np.stack([D[p] for p in sorted(D)])
Mc = M - M.mean(0)
s = np.linalg.svd(Mc, compute_uv=False)
var = np.cumsum(s ** 2) / np.sum(s ** 2)
need = {f"{int(q*100)}%": int(np.searchsorted(var, q) + 1) for q in (.9, .95, .99)}
cos = [D[x] @ D[y] / np.linalg.norm(D[x]) / np.linalg.norm(D[y])
       for x, y in itertools.combinations(sorted(D), 2)]
print(f"\n[2] {len(D)} personas in 2048-d")
print(f"    components for 90/95/99% of the spread: {need}")
print(f"    mean pairwise cosine {np.mean(cos):+.3f}  (they do sit in one region)")
print(f"    participation ratio {float((s**2).sum()**2/(s**4).sum()):.1f} effective dims")

# ---- 3. an interpretable basis, orthogonalised -------------------------------
def grp(names):
    return np.mean([D[n] for n in names if n in D], 0)

CONCEPTS = {
    "character": -(a - np.mean([D[p] for p in D], 0)) * 0 + np.mean([D[p] for p in D], 0),
    "age": grp(["grandparent", "retiree", "elder", "ancient"]) - grp(["infant", "toddler", "adolescent", "teenager"]),
    "nonhuman": grp(["ghost", "wind", "robot", "vampire"]) - grp(["scholar", "merchant", "accountant", "therapist"]),
    "professional": grp(["accountant", "consultant", "economist", "scholar"]) - grp(["pirate", "vampire", "ghost", "nomad"]),
    "era": grp(["futuristic_musician", "futuristic_spy", "futuristic_scholar"]) - grp(["medieval_musician", "medieval_spy", "medieval_scholar"]),
    "solitude": grp(["orphan", "nomad", "hermit" if "hermit" in D else "wind"]) - grp(["parent", "grandparent", "consultant"]),
}
B, keep = [], []
for name, v in CONCEPTS.items():
    for b in B:
        v = v - (v @ b) * b
    n = np.linalg.norm(v)
    if n > 1e-6:
        B.append(v / n); keep.append(name)
B = np.stack(B, 1)
proj = Mc @ B
ev = 1 - np.sum((Mc - proj @ B.T) ** 2) / np.sum(Mc ** 2)
print(f"\n[3] interpretable orthogonal basis: {keep}")
print(f"    {len(keep)} concept axes explain {ev:.3f} of persona spread")
print(f"    PCA with the same {len(keep)} dims explains {var[len(keep)-1]:.3f}")
print(f"    cost of interpretability: {var[len(keep)-1]-ev:+.3f}")

json.dump({"need": need, "mean_cos": float(np.mean(cos)),
           "concept_axes": keep, "concept_var": float(ev),
           "pca_same_dim": float(var[len(keep) - 1]),
           "vendored_full_r2": float(np.mean([r[1] for r in rows])),
           "vendored_sparse4_r2": float(np.mean([r[2] for r in rows])),
           "recipes": {t: terms for t, _, _, terms in rows}},
          open("attr_analysis/basis.json", "w"), indent=1)
print("\n-> attr_analysis/basis.json")
