"""Data for the "layer 19 steering, seen at layer 32" page.

Inputs: output/steering-l19-to-l32-25_09/{rows.csv, steer_L19.npy, steer_L32.npy}
        (jobs_condor/l32_readout.py) and the L19 / L32 centroid caches.
Each layer gets its OWN PCA-3, fitted on that layer's 275 centroids. At L19 the intended
paths are drawn (linear chord, v1 and v2 splines). At L32 nothing was steered, so the
reference drawn there is the chord between the L32 source/target centroids and a polyline
through the SAME knot personas' L32 centroids. Realised points: mean steered state over
the 5 identity questions x 3 samples, per (arm, alpha). Distances are full 4096-d.

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.l32_prep
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups import grid as GR
from steering.followups.common import Geom

RUN = Path("output/steering-l19-to-l32-25_09")
L32_GEOM = "/data/project/eeg_foundation/data/manifold_persona/geom_l25l32/geom_cache_L32.npz"
ARMS = {"linear": None, "manifold_v1": "distance", "manifold_v2": "density"}


def pca3(C):
    mu = C.mean(0)
    _, s, Vt = np.linalg.svd(C - mu, full_matrices=False)
    evr = (s ** 2 / (s ** 2).sum())[:3]
    return mu, Vt[:3], evr


def r3(x):
    return [round(float(v), 3) for v in x]


def main():
    G19, G32 = Geom(), Geom(L32_GEOM)
    rows = pd.read_csv(RUN / "rows.csv")
    S = {19: np.load(RUN / "steer_L19.npy"), 32: np.load(RUN / "steer_L32.npy")}
    iv = GR.Interventions(G19, k=8)
    P = {19: pca3(G19.C), 32: pca3(G32.C)}
    proj = lambda L, X: ((np.atleast_2d(X) - P[L][0]) @ P[L][1].T)  # noqa: E731
    out = dict(routes=[], evr={str(L): r3(P[L][2]) for L in P}, names=G19.names,
               cent={str(L): [r3(z) for z in proj(L, G.C)] for L, G in ((19, G19), (32, G32))})
    ts = np.linspace(0, 1, 41)
    for route in ["validator>vampire", "validator>bard", "assistant>vampire", "assistant>bard"]:
        A, B, _, _ = G19.route(route)
        R = dict(route=route, src=G19.ix[A], tgt=G19.ix[B], paths={"19": {}, "32": {}},
                 knots={}, real={"19": {}, "32": {}}, dist={"19": {}, "32": {}})
        for arm, rule in ARMS.items():
            cA, cB = G19.c(A), G19.c(B)
            if rule is None:
                pts19, kn = cA[None] + ts[:, None] * (cB - cA)[None], []
            else:
                p = iv.path(route, rule, cA, cB, "cA")
                pts19, kn = p.at_alpha(ts), [int(i) for i in p.centroid_idx]
            R["knots"][arm] = kn
            R["paths"]["19"][arm] = [r3(z) for z in proj(19, pts19)]
            # L32 reference: same personas, their L32 centroids, in chord order
            seq = [G32.ix[A]] + kn + [G32.ix[B]]
            R["paths"]["32"][arm] = [r3(z) for z in proj(32, G32.C[seq])]
            m = (rows.route == route) & (rows.arm == arm)
            for L, G in ((19, G19), (32, G32)):
                cAL, cBL, mu = G.c(A), G.c(B), G.C.mean(0)
                pts, dist = [], []
                for al in sorted(rows[m].alpha.unique()):
                    X = S[L][(m & (rows.alpha == al)).to_numpy()].astype(np.float64)
                    X = X[np.isfinite(X).all(1)]
                    h = X.mean(0)
                    pts.append([float(al)] + r3(proj(L, h)[0]))
                    Xc = X - mu
                    cos = lambda c: float(np.mean(Xc @ (c - mu) / (np.linalg.norm(Xc, axis=1)  # noqa: E731
                                                                   * np.linalg.norm(c - mu))))
                    dist.append(dict(a=float(al),
                                     d_tgt=float(np.linalg.norm(X - cBL, axis=1).mean()),
                                     d_src=float(np.linalg.norm(X - cAL, axis=1).mean()),
                                     cos_tgt=cos(cBL), cos_src=cos(cAL)))
                R["real"][str(L)][arm] = pts
                R["dist"][str(L)][arm] = dist
            # chord lengths for context
        R["chord"] = {str(L): float(np.linalg.norm(G.c(B) - G.c(A))) for L, G in ((19, G19), (32, G32))}
        out["routes"].append(R)
    (RUN / "artifact_data.json").write_text(json.dumps(out))
    print("wrote", RUN / "artifact_data.json", "%.0f KB" % ((RUN / "artifact_data.json").stat().st_size / 1e3))
    for R in out["routes"]:
        for L in ("19", "32"):
            d = R["dist"][L]["linear"]
            print(R["route"], "L" + L, "linear d_tgt:", " ".join("a%g=%.0f" % (x["a"], x["d_tgt"]) for x in d))


if __name__ == "__main__":
    main()
