"""Intrinsic dimension of each role's OWN manifold — one estimate per role.

Mirrors exploratory/assistant_axis/01_intrinsic_dimension.py, but instead of one
ID for the 276-point between-role cloud it produces one ID per role from that
role's within-role points, plus the two nulls defined in common.py.

Estimators come from ``manifold.idim.id_estimates`` and NOT from the
assistant-axis 01 script: that one hardcodes ``skdim.id.MLE(K=20)``, which
skdim 0.3.6 ignores under the default ``neighborhood_based=True`` — the fit then
falls back to 20 neighbours anyway, which on a 25-point cloud uses 20 of the 24
available neighbours and makes a "local" estimator global. ``manifold.idim``
passes the neighbourhood where skdim reads it, ``fit(n_neighbors=min(10, n-2))``,
for exactly this small-n case.

Usage:
    .venv/bin/python exploratory/per_persona/id_per_role.py
    .venv/bin/python exploratory/per_persona/id_per_role.py --n_null 100
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from manifold.idim import id_estimates, ESTIMATORS
from manifold_persona.common import assistant_axis, center, project, load_points
from stats_utils import fmt_p, linfit
from common import (load_role_clouds, design_fractions, pca_stats, resolve_run_dir,
                    savefig, design_null_draws, gaussian_null_draws, band, small_matrix_ops, assert_finite,
                    grid_shape, C_REAL, C_DESIGN, C_GAUSS, C_INSTR, C_QUEST, C_INTER)

# The scalar columns we summarise per role. PR and d90 are cheap PCA readouts
# that do not depend on a neighbour graph, so they stay meaningful at n=25 where
# the neighbour-based estimators are strained.
ID_COLS = list(ESTIMATORS) + ["PCA_participation_ratio", "PCA_dim_90pct"]


def per_cloud(Xr, instr=None, quest=None) -> dict:
    """All ID readouts for one role's cloud (+ design split when labelled)."""
    out = {**id_estimates(Xr), **pca_stats(Xr)}
    if instr is not None:
        out.update(design_fractions(Xr, instr, quest))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n_null", type=int, default=100, help="draws per null model")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    t0 = time.time()
    roles, clouds, factors, manifest = load_role_clouds(args.view, args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    n_per = len(next(iter(clouds.values())))
    n_i, n_q, add_rank = grid_shape(factors)
    print(f"view={args.view} layer={layer} roles={len(roles)} points/role={n_per} "
          f"grid={n_i}x{n_q} additive_rank={add_rank} ambient={clouds[roles[0]].shape[1]}")

    # --- one estimate per role -------------------------------------------------
    rows = []
    with small_matrix_ops():
        for i, r in enumerate(roles):
            instr, quest = factors[r]
            rows.append({"role": r, **per_cloud(clouds[r], instr, quest)})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(roles)} roles  ({time.time()-t0:.1f}s)")
    df = assert_finite(pd.DataFrame(rows), "per-role results")

    # Where does each role sit on the assistant axis? Lets us ask whether ID
    # varies with how far a persona is from the default Assistant.
    Xr_mean, meta_role, _ = load_points(view=args.view, layer=args.layer)
    Xc = center(Xr_mean)
    proj = project(Xc, assistant_axis(Xc, meta_role))
    df["axis_proj"] = pd.Series(proj, index=meta_role["role"].values).reindex(df["role"]).values

    # Does per-role ID track how far the persona sits from the default Assistant?
    # `default` is a huge outlier on both axes (it is the one non-character role),
    # so it alone can manufacture a correlation — report the fit with and without.
    _y = df["PCA_participation_ratio"]
    _ok = df["axis_proj"].notna() & _y.notna()
    _ok_nd = _ok & (df["role"] != "default")
    axis_fit = {
        "pearson_r_all": float(np.corrcoef(df.loc[_ok, "axis_proj"], _y[_ok])[0, 1]),
        "pearson_r_excl_default": float(np.corrcoef(df.loc[_ok_nd, "axis_proj"], _y[_ok_nd])[0, 1]),
        "default_participation_ratio": float(_y[df["role"] == "default"].iloc[0])
        if (df["role"] == "default").any() else None,
    }

    # --- nulls --------------------------------------------------------------
    print(f"nulls: {args.n_null} draws each ...")
    # Materialise the draws first: building them does one eigh of a 2048x2048
    # covariance, which is the one operation here that genuinely wants threads.
    d_draws = list(design_null_draws(clouds, factors, args.n_null))
    g_draws = list(gaussian_null_draws(clouds, args.n_null))
    with small_matrix_ops():
        dn = [per_cloud(Xn, i_, q_) for Xn, i_, q_ in d_draws]
        gn = [per_cloud(Xn) for Xn in g_draws]
    df_design, df_gauss = pd.DataFrame(dn), pd.DataFrame(gn)

    nulls = {"design": {c: band(df_design[c]) for c in ID_COLS},
             "gaussian": {c: band(df_gauss[c]) for c in ID_COLS}}

    # --- the headline comparison -------------------------------------------
    # For each estimator: is the real spread of per-role ID distinguishable from
    # the design null? If the real band sits inside the design band, the
    # per-persona manifold is the design grid.
    verdict = {}
    for c in ID_COLS:
        real, dnull = band(df[c]), nulls["design"][c]
        if real["median"] is None or dnull["median"] is None:
            verdict[c] = None
            continue
        inside = dnull["q25"] <= real["median"] <= dnull["q75"]
        verdict[c] = {"real": real, "design_null": dnull,
                      "real_median_inside_design_iqr": bool(inside),
                      "delta_median": round(real["median"] - dnull["median"], 3)}

    results = {"_meta": {"view": args.view, "layer": layer, "n_roles": len(roles),
                         "points_per_role": int(n_per),
                         "ambient": int(clouds[roles[0]].shape[1]),
                         "grid": [n_i, n_q], "additive_design_rank": add_rank,
                         "n_null": args.n_null,
                         "runtime_s": round(time.time() - t0, 1)},
               "design_variance": {k: band(df[k]) for k in
                                   ("instr_frac", "quest_frac", "interaction_frac")},
               "per_role_summary": {c: band(df[c]) for c in ID_COLS},
               "nulls": nulls, "verdict": verdict, "axis_vs_id": axis_fit}

    df.to_csv(run_dir / f"01_per_role_id_{args.view}_L{layer}.csv", index=False)
    json.dump(results, open(run_dir / f"01_per_persona_id_{args.view}_L{layer}.json", "w"),
              indent=2, default=float)

    print(f"\n== per-role ID ({len(roles)} roles), real vs design null ==")
    for c in ID_COLS:
        v = verdict[c]
        if v is None:
            print(f"  {c:26s} n/a")
            continue
        print(f"  {c:26s} real {v['real']['median']:6.2f} "
              f"[{v['real']['q25']:.2f},{v['real']['q75']:.2f}]   "
              f"design-null {v['design_null']['median']:6.2f} "
              f"[{v['design_null']['q25']:.2f},{v['design_null']['q75']:.2f}]   "
              f"{'INSIDE null' if v['real_median_inside_design_iqr'] else 'outside'}")
    dv = results["design_variance"]
    print(f"\n  variance split: instruction {dv['instr_frac']['median']:.1%} | "
          f"question {dv['quest_frac']['median']:.1%} | "
          f"interaction {dv['interaction_frac']['median']:.1%}")

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # A: per-role ID spread per estimator, against both null bands.
    ax = axes[0]
    names = [c for c in ID_COLS if df[c].notna().any()]
    rng = np.random.default_rng(0)
    for j, c in enumerate(names):
        vals = df[c].dropna().values
        ax.scatter(np.full(len(vals), j) + rng.uniform(-0.13, 0.13, len(vals)), vals,
                   s=7, alpha=0.35, color=C_REAL, linewidths=0, zorder=3)
        for key, col, off in (("design", C_DESIGN, 0.30), ("gaussian", C_GAUSS, 0.44)):
            b = nulls[key][c]
            if b["median"] is None:
                continue
            ax.vlines(j + off, b["q25"], b["q75"], color=col, lw=4, alpha=0.85, zorder=2)
            ax.plot([j + off - 0.07, j + off + 0.07], [b["median"]] * 2, color=col, lw=2, zorder=4)
    ax.axhline(add_rank, color="#333333", ls="--", lw=1, zorder=1)
    ax.annotate(f"additive {n_i}x{n_q} design rank = {add_rank}",
                (len(names) - 0.5, add_rank), fontsize=8,
                ha="right", va="bottom", color="#333333")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("PCA_", "").replace("_", " ") for n in names],
                       rotation=30, ha="right")
    ax.set_ylabel("estimated dimension")
    ax.set_title(f"Per-role intrinsic dimension\n{len(roles)} roles, {n_per} points each, layer {layer}")
    for lbl, col in ((f"{len(roles)} real roles", C_REAL), ("design null (IQR)", C_DESIGN),
                     ("Gaussian null (IQR)", C_GAUSS)):
        ax.plot([], [], "o" if col == C_REAL else "|", color=col, ms=6, label=lbl)
    ax.legend(fontsize=8, loc="upper left")

    # B: where the within-role variance actually lives.
    ax = axes[1]
    for c, col, lbl in ((("instr_frac"), C_INSTR, "instruction phrasing"),
                        ("quest_frac", C_QUEST, "question"),
                        ("interaction_frac", C_INTER, "interaction (role-specific)")):
        ax.hist(df[c], bins=40, color=col, alpha=0.75, label=lbl)
    ax.set_xlabel("fraction of within-role variance")
    ax.set_ylabel("roles")
    ax.set_xlim(0, 1)
    ax.set_title("Within-role variance decomposition\n(only interaction is not forced by the grid)")
    ax.legend(fontsize=8)

    # C: does ID depend on how far the persona is from the default Assistant?
    ax = axes[2]
    y = df["PCA_participation_ratio"]
    ax.scatter(df["axis_proj"], y, s=10, alpha=0.5, color=C_REAL, linewidths=0)
    rho, rho_nd = axis_fit["pearson_r_all"], axis_fit["pearson_r_excl_default"]
    ok_nd = df["axis_proj"].notna() & y.notna() & (df["role"] != "default")
    _f = linfit(df.loc[ok_nd, "axis_proj"], y[ok_nd])
    m, b0 = _f["slope"], _f["intercept"]
    xs = np.linspace(df["axis_proj"].min(), df["axis_proj"].max(), 50)
    ax.plot(xs, m * xs + b0, color=C_DESIGN, lw=2,
            label=f"r = {rho_nd:.2f} (excl. default)\nr = {rho:.2f} (all)")
    d_row = df[df["role"] == "default"]
    if len(d_row):
        ax.scatter(d_row["axis_proj"], d_row["PCA_participation_ratio"], s=70,
                   facecolor="none", edgecolor="#111111", lw=1.6, zorder=5)
        ax.annotate("default", (float(d_row["axis_proj"].iloc[0]),
                                float(d_row["PCA_participation_ratio"].iloc[0])),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("assistant-axis projection of the role  (high = Assistant-like)")
    ax.set_ylabel("participation ratio (per-role ID)")
    ax.set_title("Per-role ID vs distance from the default Assistant")
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, f"01_per_persona_id_{args.view}_L{layer}.png", run_dir)
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
