"""THE PANEL — every geometry metric for all 276 roles, plus the three predictors.

This is the data file the rest of the study reads. It runs
`metrics.panel_metrics` once per role and joins on the three readings of "how
Assistant-like is this role?". Every downstream script (ladder, regression,
families, design null, figures) then works from one table, and none of them
recompute geometry.

THE THREE PREDICTORS ARE NOT REDUNDANT
--------------------------------------
  axis_proj     = mean(role) . axis_unit        (the paper's, and `id_vs_axis`'s)
  mean_norm     = ||mean(role)||                 (magnitude alone)
  cos_axis      = axis_proj / mean_norm          (direction alone)

so that **axis_proj == mean_norm * cos_axis exactly**. A correlation with
axis_proj can therefore come from a role's mean vector being *longer* rather
than pointing more toward the Assistant. Only cos_axis separates the two.
`dist_default` is the fourth: distance from the Assistant in ANY direction,
not just along the one line. METHODS.md records that it and axis_proj agree
at r = 0.999 and are one finding stated twice. cos_axis is the one that can
disagree.

The persistence diagrams are saved per role so the Betti lifetime threshold can
be re-cut later without re-running ripser, which is the expensive part.

Produces `per_role_panel_L<L>.csv` and `data/persistence/*.npz` -> fig07.

Usage:
    .venv/bin/python exploratory/per_persona/study_panel.py --outdir <run>
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from manifold_persona.common import load_points, center, assistant_axis, project

from common import (load_role_clouds, grid_shape, resolve_run_dir,
                    small_matrix_ops, assert_finite)
from metrics import PANEL_COLS, panel_metrics


def add_predictors(df: pd.DataFrame, view: str, layer) -> pd.DataFrame:
    """Join the four assistant-closeness readings onto the per-role panel.

    Computed from the ROLE-MEAN cloud (one point per role, mean-centred across
    the 276). That is the between-role object the assistant-axis study works
    in. It is deliberately a different space from the within-role clouds the
    panel metrics come from.
    """
    X_roles, meta_roles, _ = load_points(view=view, layer=layer, aggregate="role")
    Xc = center(X_roles)
    ax = assistant_axis(Xc, meta_roles)
    proj = project(Xc, ax)
    idx = {r: i for i, r in enumerate(meta_roles["role"].values)}
    norms = np.linalg.norm(Xc, axis=1)
    d0 = Xc[idx["default"]]
    df["axis_proj"] = df["role"].map(lambda r: float(proj[idx[r]]))
    df["mean_norm"] = df["role"].map(lambda r: float(norms[idx[r]]))
    df["cos_axis"] = df["axis_proj"] / df["mean_norm"].replace(0, np.nan)
    df["dist_default"] = df["role"].map(
        lambda r: float(np.linalg.norm(Xc[idx[r]] - d0)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--label-layer", type=int, default=19,
                    help="layer number used in OUTPUT filenames. The resp_40q "
                         "cloud stores all 37 layers (manifest "
                         "primary_layer=19); this flag only names the depth.")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.outdir)
    (run_dir / "data" / "persistence").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    L = args.label_layer
    t0 = time.time()

    roles, clouds, factors, _ = load_role_clouds(args.view, args.layer)
    n_i, n_q, add_rank = grid_shape(factors)
    n_per = len(next(iter(clouds.values())))
    print(f"view={args.view} label_layer={L} roles={len(roles)} points/role={n_per} "
          f"grid={n_i}x{n_q} additive_rank={add_rank} "
          f"ambient={clouds[roles[0]].shape[1]}")

    print(f"\nPANEL — {len(roles)} roles x {len(PANEL_COLS)} metrics ...")
    rows = []
    with small_matrix_ops():
        for i, r in enumerate(roles):
            instr, quest = factors[r]
            m, dgms = panel_metrics(clouds[r], instr, quest, keep_diagrams=True)
            m["role"] = r
            rows.append(m)
            np.savez_compressed(run_dir / "data" / "persistence" / f"{r}.npz",
                                H0=dgms[0], H1=dgms[1])
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(roles)} roles  ({time.time()-t0:.0f}s)")
    df = add_predictors(pd.DataFrame(rows), args.view, args.layer)

    assert_finite(df, "panel metrics")
    out = run_dir / "data" / f"per_role_panel_L{L}.csv"
    df.to_csv(out, index=False)
    print(f"    wrote {out.name}  ({len(df)} roles)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
