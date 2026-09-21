"""Geometry for the age-steering artifact: the ladder, the two paths, per layer."""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five

LAD = ["infant", "toddler", "adolescent", "teenager", "graduate",
       "parent", "grandparent", "retiree", "elder"]
AGES = [1, 3, 13, 16, 22, 38, 62, 66, 78]
LAYERS = (26, 27, 28, 29, 30)
BETAS = [round(b, 2) for b in np.linspace(0, 1, 11)]


def catmull(P, t):
    n = len(P); i = min(int(np.floor(t)), n - 2); u = t - i
    p0, p1, p2, p3 = P[max(i-1, 0)], P[i], P[i+1], P[min(i+2, n-1)]
    u2, u3 = u*u, u*u*u
    return 0.5*((2*p1) + (-p0+p2)*u + (2*p0-5*p1+4*p2-p3)*u2 + (-p0+3*p1-3*p2+p3)*u3)


means, _p, _a = load_five()
out = {"ladder": LAD, "ages": AGES, "betas": BETAS, "layers": [str(L) for L in LAYERS], "L": {}}
for L in LAYERS:
    a = means[L]["default"]
    D = np.stack([means[L][p] - a for p in LAD])
    C = D - D.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    E = vt[:3].T
    proj = lambda v: (v @ E).round(3).tolist()

    # geometry facts worth showing
    steps = [float(np.linalg.norm(D[i+1]-D[i])) for i in range(len(LAD)-1)]
    turns = []
    for i in range(1, len(LAD)-1):
        u, v = D[i]-D[i-1], D[i+1]-D[i]
        turns.append(round(float(np.degrees(np.arccos(
            np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v), -1, 1)))), 1))
    line = D[-1] - D[0]; un = line/np.linalg.norm(line)
    along = [float((D[i]-D[0]) @ un) for i in range(len(LAD))]
    off = [float(np.linalg.norm((D[i]-D[0]) - along[i]*un)) for i in range(len(LAD))]

    dense = [proj(catmull(D, t)) for t in np.linspace(0, len(LAD)-1, 160)]
    pw, sp, ln = [], [], []
    for b in BETAS:
        tgt = AGES[0] + b*(AGES[-1]-AGES[0])
        t = float(np.interp(tgt, AGES, np.arange(len(LAD))))
        i = min(int(np.floor(t)), len(LAD)-2); u = t - i
        pw.append(proj(D[i]*(1-u) + D[i+1]*u))
        sp.append(proj(catmull(D, t)))
        ln.append(proj(D[0]*(1-b) + D[-1]*b))
    out["L"][str(L)] = {
        "rungs": [proj(D[i]) for i in range(len(LAD))],
        "spline_dense": dense, "piecewise": pw, "spline": sp, "linear": ln,
        "steps": [round(s, 1) for s in steps], "turns": turns,
        "along": [round(x, 1) for x in along], "off": [round(x, 1) for x in off],
        "path_len": round(float(sum(steps)), 1),
        "chord": round(float(np.linalg.norm(line)), 1),
    }
json.dump(out, open("attr_analysis/age_viz.json", "w"))
v = out["L"]["28"]
print(f"wrote age_viz.json | path {v['path_len']} vs chord {v['chord']} "
      f"= {v['path_len']/v['chord']:.1f}x longer")
print("turn angles L28:", v["turns"])
