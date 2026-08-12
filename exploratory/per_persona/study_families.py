"""ROLE FAMILIES on this cloud — does the pattern hold for GROUPS of roles?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 5)

WHY THE FAMILIES ARE RECOMPUTED
-------------------------------
`exploratory/assistant_axis/04_role_families.py` already produced 15 Ward
families, but from the PROMPT cloud at layer 26. Reusing those labels here would
be a silent cross-cloud transfer: the grouping would come from one representation
and the metrics from another, and any family-level structure found would be
partly an artefact of that mismatch. So the dendrogram is rebuilt on this
cloud's own role means, with the same linkage and the same k.

The test is Kruskal-Wallis (not ANOVA) across families per metric: the panel is
non-normal, several metrics are integer-valued, and family sizes are unequal.

Produces `families_L<L>.json` and `role_families_L<L>.csv` -> fig08.

Usage:
    .venv/bin/python exploratory/per_persona/study_families.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from manifold_persona.common import load_points, center, ward_families
from common import small_matrix_ops
from metrics import geometry_columns
from study_ladder import PREDICTORS

N_FAMILIES = 15          # matches exploratory/assistant_axis/04_role_families.py


def families_experiment(run_dir: Path, L: int, view: str, layer):
    df = pd.read_csv(run_dir / "data" / f"per_role_panel_L{L}.csv")
    panel = geometry_columns(df)

    X, meta, _ = load_points(view=view, layer=layer, aggregate="role")
    roles = list(meta["role"].values)
    with small_matrix_ops():
        lab, _linkage = ward_families(center(X), k=N_FAMILIES)
    d = df.merge(pd.DataFrame({"role": roles, "family": lab}), on="role", how="left")

    # Family ordering along the axis, so "families ordered by assistant-likeness"
    # is a property of this cloud rather than an inherited label. The ordering
    # uses axis_proj; the other predictors are carried alongside so fig08 can be
    # redrawn against each of them without re-clustering.
    order = d.groupby("family")["axis_proj"].mean().sort_values(ascending=False)
    fam_summary = []
    for f in order.index:
        g = d[d.family == f]
        fam_summary.append({
            "family": int(f), "n_roles": int(len(g)),
            **{f"mean_{p}": float(g[p].mean()) for p in PREDICTORS},
            "median_MLE": float(g["MLE"].median()),
            "median_orc_mean": float(g["orc_mean"].median()),
            "example_roles": sorted(g["role"].tolist())[:6]})

    kw = {}
    for c in panel:
        groups = [g[c].dropna().to_numpy() for _, g in d.groupby("family")
                  if g[c].notna().sum() >= 3]
        if len(groups) >= 3:
            H, p = stats.kruskal(*groups)
            kw[c] = {"H": float(H), "p": float(p), "n_families": len(groups)}

    # Does family closeness-rank track family geometry? Asked once per
    # predictor: if the family-level trend only holds for one definition of
    # closeness, that is the same caveat fig02 carries, seen at group level.
    fs = pd.DataFrame(fam_summary)
    spear = {}
    for p_ in PREDICTORS:
        spear[p_] = {}
        for tgt in ("median_MLE", "median_orc_mean"):
            rho_, pv = stats.spearmanr(fs[f"mean_{p_}"], fs[tgt])
            spear[p_][tgt] = {"spearman_rho": float(rho_), "p": float(pv)}

    out = {"exploratory": True, "n_families": N_FAMILIES,
           "recomputed_on": "this response cloud (NOT the layer-26 prompt families)",
           "predictors": PREDICTORS,
           "families": fam_summary,
           "kruskal_wallis_across_families": kw,
           "family_rank_vs_geometry": spear,
           # Kept at the top level because the report prose quotes them.
           "family_axis_vs_median_MLE": spear["axis_proj"]["median_MLE"],
           "family_axis_vs_median_curvature":
               spear["axis_proj"]["median_orc_mean"]}
    json.dump(out, open(run_dir / "data" / f"families_L{L}.json", "w"),
              indent=2, default=float)
    d[["role", "family"]].to_csv(run_dir / "data" / f"role_families_L{L}.csv",
                                 index=False)

    print(f"\n== families (k={N_FAMILIES}, recomputed on THIS cloud) ==")
    print(f"  {'fam':>4s} {'n':>4s} {'axis':>8s} {'MLE':>7s} {'orc':>7s}  examples")
    for r in fam_summary:
        print(f"  {r['family']:4d} {r['n_roles']:4d} {r['mean_axis_proj']:8.3f} "
              f"{r['median_MLE']:7.2f} {r['median_orc_mean']:7.3f}  "
              + ", ".join(r["example_roles"][:4]))
    sig = sum(1 for v in kw.values() if v["p"] < 0.05)
    print(f"\n  Kruskal-Wallis p<0.05 for {sig}/{len(kw)} metrics across families")
    print(f"\n  {'family rank by':16s} {'vs median MLE':>16s} {'vs median orc':>16s}")
    for p_ in PREDICTORS:
        s = spear[p_]
        print(f"  {p_:16s} {s['median_MLE']['spearman_rho']:16.3f} "
              f"{s['median_orc_mean']['spearman_rho']:16.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    families_experiment(Path(args.outdir), args.label_layer, args.view, args.layer)


if __name__ == "__main__":
    main()
