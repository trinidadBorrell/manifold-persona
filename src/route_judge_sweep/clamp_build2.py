"""Clamp version 2 of the judge-sweep answers: the same routes, rebuilt in the clamp's own units.

Version 1 (clamp_build.py) pinned to h0 + delta(s), where delta is the E1 addition vector (last-token
persona vectors, alpha = 2): its goals sat ~5x further out than the prompted target's own mean state.
Here every route point is a mix of the route's personas' MEAN ANSWER-TOKEN states (E1b mean_L, prompted),
the space the clamp acts on and the one Giovanni's d is measured in:
  m(s)    the route through its personas' mean states M_i, in order: LIN the straight line M_A -> M_B;
          PMG / MAN4 / FL the natural cubic through the route's personas at equal arc length
          (geometry.smooth_path, the construction the PersonaManifold routes use), sampled at s
  goal    G_S(s) = M_A + S (m(s) - M_A)      S = 1: s = 1 lands on the prompted target's own mean
  U       orthonormal basis of span{m(s) - M_0} over the route's steps (M_0 = no system prompt)
  hook    h <- h + U U^T (G_S(s) - h), every position (his clamp, several directions)

    python -m route_judge_sweep.clamp_build2
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from route_judge_sweep.build import ALPHA, MODELS, PAIRS, PM, ROUTES
from pm_geometry import geometry as G

KMAX = 6
STRENGTHS = (1.0, 4.0)
E1B = {"8b": PM / "E1b", "14b": PM / "qwen3_14b" / "E1b"}


def main():
    report = {}
    for name, (d, L) in MODELS.items():
        spec = json.loads((d / "cells.json").read_text())
        D = np.load(d / "deltas.npy").astype(np.float64)
        pz = np.load(d / ("personas_L%d.npz" % L), allow_pickle=True)
        roles, Xl = list(pz["roles"].astype(str)), pz["X_abs_L"].astype(np.float64)
        meta = pd.read_csv(E1B[name] / "meta.csv", usecols=["role"])
        H = np.load(E1B[name] / ("mean_L%d.npy" % L), mmap_mode="r")
        cache = {}

        def M(r):
            if r not in cache:
                cache[r] = np.asarray(H[(meta.role == r).to_numpy()], dtype=np.float64).mean(0)
            return cache[r]
        M0 = M("__neutral__")
        cells = pd.DataFrame(spec["cells"])
        out = d.parent / d.name.replace("ST", "STD")
        out.mkdir(exist_ok=True)
        keep, Us, Ts, ks, res = [], [], [], [], []
        for pair in PAIRS:
            p = spec["pairs"]["%s|L%d" % (pair, L)]
            routes = p["routes"] if isinstance(p["routes"], dict) else ast.literal_eval(p["routes"])
            cp = cells[(cells.pair == pair) & (cells.alpha == ALPHA) & (cells.cond == "none")]
            start = cp[cp.route == "start"].iloc[0]
            for route in ROUTES + ["start"]:
                nodes = routes[route] if route != "start" else [p["start"]]
                Mn = np.stack([M(r) for r in nodes])                          # (n, D) mean answer-token states
                cr = cp[cp.route == route].sort_values("s") if route != "start" else cp[cp.route == "start"]
                ss = [0.0] + list(cr.s) if route != "start" else [0.0]
                idxs = [start.idx] + list(cr.idx) if route != "start" else [start.idx]
                if len(nodes) == 1:
                    pts = Mn[:1]
                elif len(nodes) == 2:
                    pts = np.stack([(1 - t) * Mn[0] + t * Mn[1] for t in ss])
                else:
                    pts = G.smooth_path(Mn, ss)
                Q, R = np.linalg.qr((pts - M0).T)
                rank = int((np.abs(np.diag(R)) > 1e-6 * np.abs(R).max()).sum())
                U = Q[:, :rank]
                MA = M(p["start"])
                for j, c in enumerate(cr.itertuples()):
                    mj = pts[ss.index(c.s)] if route != "start" else pts[0]
                    for S in STRENGTHS:
                        goal = MA + S * (mj - MA)
                        Up, t = np.zeros((D.shape[1], KMAX)), np.zeros(KMAX)
                        Up[:, :rank], t[:rank] = U, U.T @ goal
                        cid = "%s_pinS%d" % (c.cell_id, S)
                        keep.append(dict({k: v for k, v in c._asdict().items() if k != "Index"}, cell_id=cid,
                                         src_cell=c.cell_id, k=rank, strength=S, alpha=S, idx=len(keep)))
                        Us.append(Up.astype(np.float32))
                        Ts.append(t.astype(np.float32))
                        ks.append(rank)
        sp = dict(spec, cells=keep, clamp=dict(rule="h <- h + U U^T (M_A + S (m(s) - M_A) - h), mean answer-token space",
                                               strengths=list(STRENGTHS), kmax=KMAX))
        (out / "cells.json").write_text(json.dumps(sp, indent=0, default=float))
        np.savez(out / "pins.npz", U=np.stack(Us), t=np.stack(Ts), k=np.array(ks))
        # how far the S = 1, s = 1 goal sits from the no-prompt mean, against the prompted target's own mean
        report[name] = dict(cells=len(keep), k=sorted(set(ks)))
        print(name, out, report[name])
    (PM / "STD_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
