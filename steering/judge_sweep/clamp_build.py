"""Clamp version of the judge-sweep answers (option A): the E1 cells of the four far pairs, routes LIN /
PMG / MAN4 / FL, alpha = 2, with Giovanni's clamp in the span of the route instead of the addition.

For each route (pair, route) the deltas of E1 at s = 0, .2, ..., 1 (alpha = 2, eq. 6) span the route; U is
an orthonormal basis of that span (QR, rank-checked). A cell at position s pins every token's coordinates
in span(U) to those of T(s) = h0 + delta(s), h0 = the unsteered model's mean answer-token state at that
layer (E1 references with no system prompt, nothing added):  h <- h + U U^T (T(s) - h)  (his steer_pin2
with several directions; our make_subspace_pin_delta_fn). The start cell (s = 0) pins span(delta(0)).
Everything else, rows, questions, paraphrases, samples, seeds, sampling, 1024-token cap, is the E1 run.

    python -m steering.judge_sweep.clamp_build
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.judge_sweep.build import ALPHA, MODELS, PAIRS, PM, ROUTES

KMAX = 6


def h0_of(d):
    """mean answer-token state of the references with no system prompt and nothing added"""
    g = pd.read_csv(d / "generations.csv", low_memory=False, usecols=["cell_id", "sys_role"])
    cells = pd.DataFrame(json.loads((d / "cells.json").read_text())["cells"])
    ref_n = set(cells.cell_id[(cells.route == "ref") & (cells.cond == "ref_neutral")])
    m = (g.cell_id.isin(ref_n) & g.sys_role.isna()).to_numpy()
    H = np.load(d / "states.npz")["mean_L"]
    return H[m].astype(np.float64).mean(0), int(m.sum())


def main():
    for name, (d, L) in MODELS.items():
        spec = json.loads((d / "cells.json").read_text())
        D = np.load(d / "deltas.npy").astype(np.float64)
        h0, n0 = h0_of(d)
        cells = pd.DataFrame(spec["cells"])
        out = d.parent / (d.name.replace("ST", "STC"))
        out.mkdir(exist_ok=True)
        keep, Us, Ts, ks = [], [], [], []
        for pair in PAIRS:
            cp = cells[(cells.pair == pair) & (cells.alpha == ALPHA) & (cells.cond == "none")]
            start = cp[cp.route == "start"].iloc[0]
            for route in ROUTES + ["start"]:
                cr = cp[cp.route == route].sort_values("s")
                path = [start.idx] + list(cr.idx) if route != "start" else [start.idx]
                Q, R = np.linalg.qr(D[path].T)
                rank = int((np.abs(np.diag(R)) > 1e-6 * np.abs(R).max()).sum())
                U = Q[:, :rank]
                for c in cr.itertuples():
                    T = h0 + D[c.idx]
                    Up = np.zeros((D.shape[1], KMAX))
                    Up[:, :rank] = U
                    t = np.zeros(KMAX)
                    t[:rank] = U.T @ T
                    keep.append(dict(c._asdict(), cell_id=c.cell_id + "_pin", src_cell=c.cell_id, k=rank,
                                     idx=len(keep)))
                    Us.append(Up.astype(np.float32))
                    Ts.append(t.astype(np.float32))
                    ks.append(rank)
        for c in keep:
            c.pop("Index", None)
        sp = dict(spec, cells=keep, clamp=dict(h0_rows=n0, kmax=KMAX, rule="h <- h + U U^T (h0 + delta(s) - h)"))
        (out / "cells.json").write_text(json.dumps(sp, indent=0, default=float))
        np.savez(out / "pins.npz", U=np.stack(Us), t=np.stack(Ts), k=np.array(ks))
        print(name, out, len(keep), "cells; k per route:", sorted(set(ks)), "; h0 from", n0, "rows")


if __name__ == "__main__":
    main()
