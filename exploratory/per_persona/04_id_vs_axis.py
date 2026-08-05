"""Does a role's intrinsic dimension depend on where it sits on the Assistant Axis?

01 reports this for ONE estimator (participation ratio) as a single r. This runs
it for every ID measure we compute, against two different readings of "close to
the Assistant", with the controls that make a correlation of ~0.3 on n=276
interpretable rather than decorative.

TWO WAYS TO BE "CLOSE TO THE ASSISTANT" — they are not the same question:

  axis_proj   signed projection on assistant_axis = mean(default) - mean(roles).
              High = Assistant-like. This is the paper's axis (arXiv:2601.10387)
              and the quantity 01 already uses.
  dist_default  Euclidean distance from the `default` role mean in the full
              space. Captures "unlike the Assistant in ANY direction", not just
              along the one axis. A role can sit at mid-axis yet be far from
              default off-axis; only this sees that.

THE CONFOUNDS THIS CONTROLS FOR — two, reported as two partials:

  log_var     a role whose cloud is simply BIGGER will tend to score a higher ID
              from every estimator, and spread also moves a role along the axis.
              So a raw ID-vs-axis correlation can be entirely a scale effect.
  mean_text_len  the mean character length of the text each point embeds. Longer
              texts are more varied, so length alone can raise ID, and in-role
              personas answer at a different length than the Assistant does.

We report the partial holding log_var fixed (as before) AND the partial holding
both fixed. Where the raw and partial correlations disagree, the raw one was the
confound talking.

Six estimators are tested at once, so p-values get Benjamini-Hochberg FDR
correction; Spearman is reported alongside Pearson because several of the ID
distributions are skewed.

Usage:
    .venv/bin/python exploratory/per_persona/04_id_vs_axis.py --rundir <figures/STAMP>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from common import (load_role_clouds, resolve_run_dir, savefig, small_matrix_ops,
                    grid_shape, role_text_lengths, C_REAL, C_DESIGN, C_QUEST)

ID_COLS = ["TwoNN", "MLE", "lPCA", "PCA_participation_ratio",
           "PCA_dim_90pct", "PCA_dim_95pct"]
PREDICTORS = ["axis_proj", "dist_default"]


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        adj[i] = prev
    return adj


def partial_corr(x, y, z):
    """Pearson r between x and y with the control(s) z linearly removed from both.

    ``z`` is one control column [n] or a stack of k of them [n, k]."""
    Z = np.column_stack([np.ones(len(x)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, k = len(x), Z.shape[1] - 1
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - k - 2) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df=n - k - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--rundir", required=True,
                    help="run folder holding 01_per_role_id_<view>_L<layer>.csv")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir or args.rundir)

    roles, clouds, factors, manifest = load_role_clouds(args.view, args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    n_i, n_q, add_rank = grid_shape(factors)
    df = pd.read_csv(f"{args.rundir}/01_per_role_id_{args.view}_L{layer}.csv")

    # Scale control + the off-axis distance measure, both from the raw clouds.
    with small_matrix_ops():
        means = {r: clouds[r].mean(0) for r in roles}
        tot_var = {r: float(((clouds[r] - means[r]) ** 2).sum()) for r in roles}
    d0 = means["default"]
    df["dist_default"] = df["role"].map(lambda r: float(np.linalg.norm(means[r] - d0)))
    df["log_var"] = df["role"].map(lambda r: np.log(tot_var[r]))

    # Second control: how long the embedded text is. On a response-token cloud
    # that is the model's response length; on a prompt-token cloud it is the
    # prompt length. Hence the basis-neutral column name.
    text_len, text_col = role_text_lengths()
    df["mean_text_len"] = df["role"].map(text_len)

    # `default` is distance 0 from itself and anchors the axis; it cannot inform
    # a trend about distance FROM it, so every fit here excludes it. It is still
    # PLOTTED, and its percentile against the other roles recorded: whether the
    # Assistant itself obeys the trend its own axis defines is the whole point.
    d = df[df["role"] != "default"].dropna(
        subset=ID_COLS + PREDICTORS + ["log_var", "mean_text_len"])
    dflt = df[df["role"] == "default"]
    dflt = dflt.iloc[0] if len(dflt) else None
    print(f"view={args.view} layer={layer} grid={n_i}x{n_q} n={len(d)} roles (default excluded from fits)")

    default_pos = {}
    if dflt is not None:
        default_pos["axis_proj"] = {
            "value": float(dflt["axis_proj"]),
            "pct_of_other_roles_below": float((d["axis_proj"] < dflt["axis_proj"]).mean() * 100),
            "others_min": float(d["axis_proj"].min()), "others_max": float(d["axis_proj"].max())}
        for c in ID_COLS:
            default_pos[c] = {
                "value": float(dflt[c]),
                "pct_of_other_roles_below": float((d[c] < dflt[c]).mean() * 100),
                "others_median": float(d[c].median())}

    rows = []
    for pred in PREDICTORS:
        for c in ID_COLS:
            x, y, z = d[pred].to_numpy(float), d[c].to_numpy(float), d["log_var"].to_numpy(float)
            z2 = d[["log_var", "mean_text_len"]].to_numpy(float)
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            r_par, p_par = partial_corr(x, y, z)
            r_par2, p_par2 = partial_corr(x, y, z2)
            rows.append({"predictor": pred, "estimator": c, "n": len(d),
                         "pearson_r": r_p, "pearson_p": p_p,
                         "spearman_r": r_s, "spearman_p": p_s,
                         "partial_r_ctrl_logvar": r_par, "partial_p": p_par,
                         "partial_r_ctrl_logvar_textlen": r_par2,
                         "partial_p_ctrl_logvar_textlen": p_par2})
    res = pd.DataFrame(rows)
    for col, out in (("pearson_p", "pearson_q"), ("partial_p", "partial_q"),
                     ("partial_p_ctrl_logvar_textlen", "partial_q_ctrl_logvar_textlen")):
        res[out] = np.concatenate([bh_fdr(g[col].values) for _, g in res.groupby("predictor")])

    # How much of the ID<->axis link is just cloud scale, or just text length?
    r_scale = {c: float(stats.pearsonr(d["log_var"], d[c])[0]) for c in ID_COLS}
    r_len = {c: float(stats.pearsonr(d["mean_text_len"], d[c])[0]) for c in ID_COLS}

    res.to_csv(run_dir / f"04_id_vs_axis_{args.view}_L{layer}.csv", index=False)
    json.dump({"_meta": {"view": args.view, "layer": layer, "grid": [n_i, n_q],
                         "n_roles": int(len(d)),
                         "default_excluded_from_fits": True, "default_plotted": True,
                         "text_len_column": text_col},
               "id_vs_log_variance_pearson_r": r_scale,
               "id_vs_mean_text_len_pearson_r": r_len,
               "default_position": default_pos,
               "results": res.to_dict("records")},
              open(run_dir / f"04_id_vs_axis_{args.view}_L{layer}.json", "w"),
              indent=2, default=float)

    for pred in PREDICTORS:
        print(f"\n== ID vs {pred} ==")
        print(f"  {'estimator':26s} {'pearson':>9s} {'q':>8s} {'spearman':>9s} "
              f"{'part|var':>9s} {'q':>8s} {'part|var+len':>13s} {'q':>8s}")
        for _, row in res[res.predictor == pred].iterrows():
            star = "*" if row["partial_q"] < 0.05 else " "
            star2 = "*" if row["partial_q_ctrl_logvar_textlen"] < 0.05 else " "
            print(f"  {row['estimator']:26s} {row['pearson_r']:9.3f} {row['pearson_q']:8.3g} "
                  f"{row['spearman_r']:9.3f} {row['partial_r_ctrl_logvar']:9.3f} "
                  f"{row['partial_q']:8.3g}{star} "
                  f"{row['partial_r_ctrl_logvar_textlen']:12.3f} "
                  f"{row['partial_q_ctrl_logvar_textlen']:8.3g}{star2}")
    print(f"\n  the confounds themselves (r with ID), {text_col} length in chars:")
    print(f"    {'estimator':26s} {'log_var':>9s} {'mean_text_len':>14s}")
    for c in ID_COLS:
        print(f"    {c:26s} {r_scale[c]:9.3f} {r_len[c]:14.3f}")

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(2, len(ID_COLS), figsize=(3.1 * len(ID_COLS), 6.6),
                             sharex="row")
    for i, pred in enumerate(PREDICTORS):
        for j, c in enumerate(ID_COLS):
            ax = axes[i, j]
            row = res[(res.predictor == pred) & (res.estimator == c)].iloc[0]
            ax.scatter(d[pred], d[c], s=7, alpha=0.4, color=C_REAL, linewidths=0)
            m, b = np.polyfit(d[pred], d[c], 1)
            xs = np.linspace(d[pred].min(), d[pred].max(), 50)
            sig = row["partial_q"] < 0.05
            ax.plot(xs, m * xs + b, lw=2, color=C_DESIGN if sig else C_QUEST)
            # The Assistant itself: excluded from the fit, but shown, because
            # "does default obey the trend its own axis defines?" is the point.
            # The fit line is extended to reach it when it sits outside the
            # character roles' range, so the extrapolation is visible as such.
            if dflt is not None:
                dx, dy = float(dflt[pred]), float(dflt[c])
                if not (xs[0] <= dx <= xs[-1]):
                    ex = np.linspace(min(xs[0], dx), max(xs[-1], dx), 50)
                    ax.plot(ex, m * ex + b, lw=1, ls=":", color=C_DESIGN if sig else C_QUEST)
                ax.scatter([dx], [dy], s=70, marker="*", color="#111111", zorder=6)
            ax.set_title(f"{c.replace('PCA_','').replace('_',' ')}\n"
                         f"r={row['pearson_r']:.2f}  partial={row['partial_r_ctrl_logvar']:.2f}"
                         f"{' *' if sig else ''}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{pred}\n\nestimated dimension" if i == 0
                              else f"{pred}\n\nestimated dimension", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[i, 0].set_xlabel("")
    for j in range(len(ID_COLS)):
        axes[1, j].set_xlabel("distance from default", fontsize=8)
        axes[0, j].set_xlabel("assistant-axis projection", fontsize=8)
    fig.suptitle(f"Per-role intrinsic dimension vs closeness to the Assistant "
                 f"({n_i}x{n_q} grid, layer {layer}, n={len(d)} character roles)\n"
                 f"orange = significant after controlling for cloud scale (BH q<0.05); "
                 f"blue = not.   ★ = `default`, the Assistant itself "
                 f"(excluded from every fit; dotted = fit extrapolated to reach it)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, f"04_id_vs_axis_{args.view}_L{layer}.png", run_dir)


if __name__ == "__main__":
    main()
