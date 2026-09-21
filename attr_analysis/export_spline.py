"""Export the attribute ladders as splines for the 3D viewer.

Three display bases are offered because only three directions fit on screen and
the two ladders do not share them. Everything is recomputed per layer.
"""
import glob
import json

import numpy as np

AGE = ["infant", "toddler", "adolescent", "teenager", "graduate",
       "parent", "grandparent", "retiree", "elder", "ancient"]
ERA = ["era_medieval", "era_victorian", "era_midcentury",
       "era_contemporary", "era_futuristic"]
OCC = ["musician", "spy", "scholar", "soldier", "merchant"]
LAYERS = [26, 28, 30]
VIEWS = {"age ladder": AGE, "era ladder": ERA, "occupations": OCC}

P, names = [], []
for f in sorted(glob.glob("attr_acts/*.npz")):
    z = np.load(f, allow_pickle=True)
    P.append(z["prompted"].astype(np.float32))
    names.append(z["persona"].astype(str))
X = np.concatenate(P, 0)
who = np.concatenate(names)


def basis(pts):
    """Top three directions of the spread of these persona means."""
    C = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    return vt[:3].T


def catmull(pts, n=24):
    """Smooth curve through every rung, so the drawn line is the ladder itself."""
    p = np.asarray(pts, float)
    ext = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0, 1, n, endpoint=(i == len(p) - 2)):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


out = {"views": {}, "layers": [str(L) for L in LAYERS],
       "ladders": {"age": AGE, "era": ERA, "occupations": OCC}}

for vname, vset in VIEWS.items():
    per_layer = {}
    for L in LAYERS:
        M = {p: X[who == p, L, :].mean(0) for p in np.unique(who)}
        asst = M["default"]
        E = basis(np.stack([M[p] - asst for p in vset]))

        def proj(p):
            return (X[who == p, L, :] - asst) @ E

        lay = {"groups": {}}
        pj = proj("default")
        lay["assistant"] = {"mu": pj.mean(0).tolist(),
                            "S": np.cov(pj.T).tolist()}
        for gname, members in [("age", AGE), ("era", ERA), ("occupations", OCC)]:
            nodes = []
            for p in members:
                q = proj(p)
                nodes.append({"name": p.replace("era_", ""),
                              "mu": [round(float(x), 3) for x in q.mean(0)],
                              "S": np.round(np.cov(q.T), 4).tolist()})
            g = {"nodes": nodes}
            if gname in ("age", "era"):
                g["spline"] = [[round(float(x), 3) for x in q]
                               for q in catmull([n["mu"] for n in nodes])]
            lay["groups"][gname] = g
        # how much of each ladder this basis actually keeps
        keep = {}
        for gname, members in [("age", AGE), ("era", ERA), ("occupations", OCC)]:
            A = np.stack([M[p] - asst for p in members])
            keep[gname] = round(float((np.linalg.norm(A @ E, axis=1) ** 2).sum()
                                      / (np.linalg.norm(A, axis=1) ** 2).sum()), 2)
        lay["keep"] = keep
        per_layer[str(L)] = lay
    out["views"][vname] = per_layer

json.dump(out, open("attr_analysis/spline_data.json", "w"))
v = out["views"]["age ladder"]["28"]
print(f"wrote spline_data.json | {len(VIEWS)} views x {len(LAYERS)} layers")
print("variance kept by the 'age ladder' basis at L28:", v["keep"])
print("age rungs:", [n["name"] for n in v["groups"]["age"]["nodes"]])
