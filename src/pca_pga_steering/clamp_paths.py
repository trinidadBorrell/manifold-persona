"""Component steering with Giovanni's clamp (persona-definition @ b860780, steer_pin.py) on the
PROPER axes: the PCA components and Fletcher's principal geodesic directions themselves.

Same cells, questions and sampling as the PC round (component_paths.py), Qwen3-8B, hidden state 28.
Per (method, component k):
  u_k   unit axis in activation space: PCA component k (pca.V[k]); PGA direction k = P["V"][:, k]
        mapped from PCA coordinates to activations (V_k @ pca.V), normalised
  a     the start persona's coordinate on u_k: its mean answer-token state (E1 mean_L28, prompted)
  d_k   2 sigma_k, sigma_k = std over the 276 personas' mean answer-token states along u_k
        (the PC round put each target at the start +/- 2 sigma_k along the component)
  goal  a + S * sign * step * d_k,  step in {0, .25, .5, .75, 1},  S in {1, 4}
  hook  h <- h + (goal - h.u) u at every position (his pin), hidden state 28
Step 0 pins the start's own coordinate on u_k (his s = 0), one cell per direction.

    python -m pca_pga_steering.clamp_paths --out $PM/PCC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pm_geometry import components as C
from pm_addition_round.component_paths import N_COMP, QUESTIONS, STEPS

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
STRENGTHS = (1.0, 4.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pc = json.loads((PM / "PC" / "cells.json").read_text())
    L = pc["layer"]
    roles, sets, _ = C.persona_sets(PM / "E1", "last", L)
    res, st = C.analyse(sets["all"], roles)
    pca, Z, P = st["pca"], st["Z"], st["P"]
    Vp = np.asarray(pca.V, dtype=np.float64)                         # (D', D) rows = components
    axes = {}
    for k in range(N_COMP):
        axes[("pca", k + 1)] = Vp[k] / np.linalg.norm(Vp[k])
        v = P["V"][:, k] @ Vp
        axes[("pga", k + 1)] = v / np.linalg.norm(v)
    # mean answer-token state per persona (prompted, not neutral-subtracted), E1
    meta = pd.read_csv(PM / "E1" / "meta.csv")
    H = np.load(PM / "E1" / ("mean_L%d.npy" % L), mmap_mode="r")
    M = {r: np.asarray(H[(meta.role == r).to_numpy()], dtype=np.float64).mean(0) for r in roles}
    Mall = np.stack([M[r] for r in roles])
    U, goals, cells = [], [], []
    for (m, k), u in sorted(axes.items()):
        U.append(u)
    keys = sorted(axes)
    info = {}
    for ai, (m, k) in enumerate(keys):
        u = axes[(m, k)]
        coords = Mall @ u
        info["%s%d" % (m, k)] = dict(sigma=float(coords.std()), d=float(2 * coords.std()),
                                     cos_pca_pga=None)
    for start in pc["starts"]:
        for ai, (m, k) in enumerate(keys):
            u = axes[(m, k)]
            a0 = float(M[start] @ u)
            d = info["%s%d" % (m, k)]["d"]
            for sign in (-1, 1):
                for s in (0.0,) + STEPS:
                    if s == 0.0 and sign == 1:
                        continue                                   # one step-0 cell per direction
                    cid = "pcc_L%d_%s_%s_c%d_%s_s%s" % (L, start, m, k, {-1: "m", 1: "p"}[sign] if s else "o",
                                                       str(s).replace(".", "p"))
                    cells.append(dict(cell_id=cid, start=start, method=m, component=k, sign=sign if s else 0,
                                      step=s, axis=ai, a=a0, d=d,
                                      goals={str(S): a0 + S * sign * s * d for S in STRENGTHS}))
    for i, c in enumerate(cells):
        c["idx"] = i
    cos = abs(float(axes[("pca", 1)] @ axes[("pga", 1)]))
    np.save(out / "axes.npy", np.stack([axes[k] for k in keys]).astype(np.float32))
    (out / "cells.json").write_text(json.dumps(dict(layer=L, starts=pc["starts"], axes=["%s%d" % k for k in keys],
                                                    axis_info=info, strengths=list(STRENGTHS), cells=cells,
                                                    questions=QUESTIONS, cos_pca1_pga1=cos), indent=1))
    print("layer %d, %d cells x %d strengths, |cos(pca1, pga1)| = %.2f" % (L, len(cells), len(STRENGTHS), cos))
    for k, v in info.items():
        print(k, "sigma %.2f" % v["sigma"])


if __name__ == "__main__":
    main()
