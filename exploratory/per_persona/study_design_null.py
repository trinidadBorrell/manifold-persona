"""DESIGN NULL — is a metric measuring the persona, or the shape of our experiment?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 1)

This is the most important control in the study. Each role's 200 points are a
complete 5 x 40 grid — 5 instruction phrasings x 40 questions, one point per
cell — and that grid forces structure into the data before the model contributes
anything. A complete two-factor grid has additive rank (n_i - 1) + (n_q - 1) =
43, so any intrinsic dimension measured on gridded points reports that rank
unless the interaction term carries real variance.

So we synthesise clouds with the same grid and NO persona in them: random
instruction effects a[i] and question effects b[j], drawn from the real data's
empirical effect covariances, with each point set to a[i] + b[j]. Interaction is
exactly zero by construction. Then we run the identical panel on them.

**If a metric's real median sits inside the null's IQR, that metric is measuring
our extraction design, not the persona.** This exact test is what killed the
earlier 25-point prompt cloud, where 99.4% of within-role variance was the grid.

Produces `design_null_L<L>.json` -> fig04, and the DESIGN-EXPLAINED flags the
ladder prints.

Usage:
    .venv/bin/python exploratory/per_persona/study_design_null.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (load_role_clouds, grid_shape, design_null_draws,
                    small_matrix_ops)
from metrics import PANEL_COLS, SEED, panel_metrics


def _band(s) -> dict:
    a = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if a.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-null", type=int, default=100)
    args = ap.parse_args()
    run_dir, L = Path(args.outdir), args.label_layer
    t0 = time.time()

    df = pd.read_csv(run_dir / "data" / f"per_role_panel_L{L}.csv")
    _roles, clouds, factors, _ = load_role_clouds(args.view, args.layer)
    n_i, n_q, add_rank = grid_shape(factors)

    print(f"DESIGN NULL — {args.n_null} persona-free clouds through the panel ...")
    null_rows = []
    with small_matrix_ops():
        for j, (Xn, instr, quest) in enumerate(
                design_null_draws(clouds, factors, n_draws=args.n_null, seed=SEED)):
            m, _, _ = panel_metrics(Xn, instr, quest)
            null_rows.append(m)
            if (j + 1) % 25 == 0:
                print(f"    {j+1}/{args.n_null} draws  ({time.time()-t0:.0f}s)")
    ndf = pd.DataFrame(null_rows)

    summary = {"exploratory": True,
               "_meta": {"n_draws": args.n_null, "seed": SEED,
                         "grid": [n_i, n_q], "additive_rank": add_rank},
               "design_null": {c: _band(ndf[c]) for c in PANEL_COLS if c in ndf},
               "real": {c: _band(df[c]) for c in PANEL_COLS if c in df},
               "design_explained": {}}
    for c in PANEL_COLS:
        if c not in ndf or c not in df:
            continue
        b, rb = summary["design_null"][c], summary["real"][c]
        if b["median"] is None or rb["median"] is None:
            summary["design_explained"][c] = None
            continue
        summary["design_explained"][c] = bool(b["q25"] <= rb["median"] <= b["q75"])
    json.dump(summary, open(run_dir / "data" / f"design_null_L{L}.json", "w"),
              indent=2, default=float)

    print("\n== real vs design null (median [IQR]) ==")
    for c in PANEL_COLS:
        if c not in summary["real"] or summary["real"][c]["median"] is None:
            continue
        r_, n_ = summary["real"][c], summary["design_null"][c]
        flag = "DESIGN-EXPLAINED" if summary["design_explained"].get(c) else ""
        print(f"  {c:26s} real {r_['median']:9.3f} [{r_['q25']:8.3f},{r_['q75']:8.3f}]  "
              f"null {n_['median']:9.3f} [{n_['q25']:8.3f},{n_['q75']:8.3f}]  {flag}")

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
