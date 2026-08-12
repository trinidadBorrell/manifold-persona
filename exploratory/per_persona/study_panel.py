"""THE PANEL — every geometry metric for all 276 roles, plus the three predictors.

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 2)

This is the data file the rest of the study reads. It runs `metrics.panel_metrics`
once per role and joins on the three readings of "how Assistant-like is this
role?", so that every downstream script (ladder, regression, families, design
null, figures) works from one table and none of them recompute geometry.

FOUR WAYS TO BE "CLOSE TO THE ASSISTANT"
----------------------------------------
The ladder correlates every metric against all four, because a result that
holds under one definition of closeness and not the others is a result about
the definition. `closeness.py` documents them; in short:

  axis_proj      position along the Assistant Axis      (the paper's)
  cos_centroid   angle between role and `default` centroids
  mknn_align     shared nearest neighbours over the 40 questions
  cka            same similarity structure over the 40 questions

The first two collapse a role to one point; the last two compare its whole
response cloud against `default`'s, so they can see a role that sits far away
yet answers the questions in the same relative order.

Two more columns are kept for the magnitude-vs-direction check (fig07) but are
no longer ladder predictors:

  mean_norm     = ||mean(role)||                 (magnitude alone)
  cos_axis      = axis_proj / mean_norm          (direction alone)

so that **axis_proj == mean_norm * cos_axis exactly**. `dist_default`
(Euclidean distance to `default`) is kept for the same reason — md/METHODS.md
records that it and axis_proj agree at r = 0.999, which is why it was dropped
as a predictor rather than reported as a second confirmation.

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

from closeness import MKNN_K, cloud_closeness, cos_to_reference
from common import (load_role_clouds, grid_shape, resolve_run_dir,
                    small_matrix_ops, assert_finite)
from metrics import PANEL_COLS, panel_metrics


def add_centroid_predictors(df: pd.DataFrame, view: str, layer) -> pd.DataFrame:
    """Join the centroid-level closeness readings onto the per-role panel.

    Computed from the ROLE-MEAN cloud (one point per role, mean-centred across
    the 276), which is the between-role object the assistant-axis study works
    in — deliberately a different space from the within-role clouds the panel
    metrics come from.
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

    # Cosine to `default`'s centroid, mean-centred and raw. See closeness.py for
    # why the mean-centred one is the predictor: the raw cosine is squeezed into
    # [0.873, 0.999] by the component every role shares.
    cos_c = cos_to_reference(Xc, d0)
    cos_raw = cos_to_reference(X_roles, X_roles[idx["default"]])
    df["cos_centroid"] = df["role"].map(lambda r: float(cos_c[idx[r]]))
    df["cos_centroid_raw"] = df["role"].map(lambda r: float(cos_raw[idx[r]]))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--label-layer", type=int, default=19,
                    help="layer number used in OUTPUT filenames. The resp40 "
                         "manifest stores primary_layer=0 because it holds a "
                         "single extracted layer; the real depth is 19.")
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
    rows, pointwise = [], {}
    with small_matrix_ops():
        for i, r in enumerate(roles):
            instr, quest = factors[r]
            m, dgms, pw = panel_metrics(clouds[r], instr, quest, keep_diagrams=True)
            m["role"] = r
            rows.append(m)
            np.savez_compressed(run_dir / "data" / "persistence" / f"{r}.npz",
                                **{f"H{k}": dg for k, dg in enumerate(dgms)})
            for name, arr in pw.items():
                pointwise.setdefault(name, {})[r] = np.asarray(arr, float)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(roles)} roles  ({time.time()-t0:.0f}s)")

    # Per-point and per-edge values, one file per metric keyed by role. The
    # panel only carries their summaries; the family distribution figures need
    # what those summaries threw away. Edge arrays differ in length per role,
    # which is why this is a keyed npz and not a rectangular table.
    pw_dir = run_dir / "data" / "pointwise"
    pw_dir.mkdir(parents=True, exist_ok=True)
    for name, by_role in pointwise.items():
        np.savez_compressed(pw_dir / f"{name}.npz", **by_role)
    print(f"    wrote {len(pointwise)} pointwise arrays -> data/pointwise/")
    df = add_centroid_predictors(pd.DataFrame(rows), args.view, args.layer)

    # Cloud-level closeness: the role's 40 question-means against `default`'s.
    print(f"\nCLOUD CLOSENESS — mKNN (k={MKNN_K}) and CKA vs `default` ...")
    with small_matrix_ops():
        cc = cloud_closeness(clouds, factors, roles)
    for col in ("mknn_align", "cka", "cka_cosine"):
        df[col] = df["role"].map(lambda r, c=col: cc[r][c])
    d_ = df[df.role != "default"]
    for col in ("cos_centroid", "mknn_align", "cka", "cka_cosine"):
        print(f"    {col:16s} {d_[col].min():+.3f} .. {d_[col].max():+.3f}  "
              f"(median {d_[col].median():+.3f})")

    assert_finite(df, "panel metrics")
    out = run_dir / "data" / f"per_role_panel_L{L}.csv"
    df.to_csv(out, index=False)
    print(f"    wrote {out.name}  ({len(df)} roles)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
