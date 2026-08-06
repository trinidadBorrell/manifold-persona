"""THE CORRELATION LADDER: every panel metric vs every closeness measure.

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiments 1, 3)

WHAT A "LADDER" IS AND WHY IT REPLACES A SINGLE CORRELATION
-----------------------------------------------------------
`id_vs_axis.py` reports two numbers per estimator: a raw r and a partial r
controlling cloud scale. That is already better than a raw correlation, but it
answers only one confound. This reports THREE correlations per (metric,
predictor) pair, each adding a control, so that erosion is visible as a
trajectory rather than asserted at one point:

  rung 1  raw Pearson r                      (what `id_vs_axis` headlines)
  rung 2  partial | log_var                  (its own control, its own code)
  rung 3  partial | log_var, mean_norm       (cloud size AND mean-vector length)

A metric whose r survives rung 3 is saying something about geometry. A metric
whose r collapses between rung 1 and rung 3 was a confound wearing geometry's
clothes, and which rung it dies at names the confound.

Rung 2 calls the SAME ``partial_corr`` that produced the published `id_vs_axis`
numbers — now a normal import from ``stats_utils`` rather than a copy — so the
regression check below is meaningful.

RESPONSE LENGTH IS NOT A RUNG (removed 2026-08-03)
--------------------------------------------------
Earlier revisions carried a `| mean_tokens` rung and a fourth control set
including `trunc_rate`. Both are gone. These activations are MEAN-pooled over
the generated tokens, so the token count is divided out before the role's vector
is formed; length is not a mechanism by which a longer answer moves a role along
the axis. The run's own numbers agreed — no metric was ever flagged
LENGTH-EXPLAINED, and the raw length-vs-axis correlation was +0.258.

THE THREE PREDICTORS ARE NOT REDUNDANT
--------------------------------------
  axis_proj     = mean(role) . axis_unit        (the paper's)
  mean_norm     = ||mean(role)||                 (magnitude alone)
  cos_axis      = axis_proj / mean_norm          (direction alone)

so that **axis_proj == mean_norm * cos_axis exactly**. A correlation with
axis_proj can therefore come from a role's mean vector being *longer* rather
than pointing more toward the Assistant, and only cos_axis separates the two.
md/METHODS.md already records that axis_proj and dist_default agree at r = 0.999
and are one finding stated twice; cos_axis is the predictor that can disagree.

Produces `ladder_L<L>.csv` -> fig02, fig02b, fig03.

Usage:
    .venv/bin/python exploratory/per_persona/study_ladder.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metrics import geometry_columns
from stats_utils import bh_fdr, boot_ci, partial_corr, partial_corr_multi

PREDICTORS = ["axis_proj", "cos_axis", "dist_default"]
CTRL_LOGVAR = ["log_var"]
# Cloud size and mean-vector length: the two ways a role can score high on
# axis_proj without its geometry differing at all.
CTRL_ALL = ["log_var", "mean_norm"]
RUNGS = [("raw", []), ("ctrl_logvar", CTRL_LOGVAR), ("ctrl_all", CTRL_ALL)]

N_BOOT = 2000
N_SHUFFLE = 1000
SEED = 0
EFFECT_BAR = 0.30      # |r| in the fully-controlled rung that would "matter"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--prior", default="exploratory/per_persona/figures/resp40/"
                                       "04_id_vs_axis_prompt_avg_L0.csv",
                    help="the published id_vs_axis result, for the regression check")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    args = ap.parse_args()
    run_dir = Path(args.outdir)
    L = args.label_layer
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(run_dir / "data" / f"per_role_panel_L{L}.csv")
    null = json.load(open(run_dir / "data" / f"design_null_L{L}.json"))
    panel = geometry_columns(df)

    # `default` defines the axis and is distance 0 from itself, so it cannot
    # inform a trend about distance from it. `id_vs_axis`'s rule, inherited.
    d = df[df["role"] != "default"].copy()
    d = d.dropna(subset=[c for c in set(PREDICTORS) | set(CTRL_ALL)
                         if c in d.columns])
    print(f"n={len(d)} roles (default excluded), {len(panel)} panel metrics, "
          f"{len(PREDICTORS)} predictors, {len(RUNGS)} rungs")

    # --- consistency check: does the generalised partial reduce to the one-
    # control form that produced the published numbers? ----------------------
    _x = d["axis_proj"].to_numpy(float)
    _y = d["MLE"].to_numpy(float)
    _z = d["log_var"].to_numpy(float)
    r_ref = partial_corr(_x, _y, _z)[0]
    r_new = partial_corr_multi(_x, _y, _z[:, None])[0]
    assert abs(r_ref - r_new) < 1e-10, (r_ref, r_new)
    print(f"partial_corr_multi reduces to partial_corr: "
          f"{r_ref:.12f} vs {r_new:.12f}  OK")

    # ---------------- the ladder --------------------------------------------
    rows = []
    for pred in PREDICTORS:
        x = d[pred].to_numpy(float)
        for c in panel:
            y = d[c].to_numpy(float)
            if not np.isfinite(y).all() or np.nanstd(y) == 0:
                rows.append({"predictor": pred, "metric": c, "n": len(d),
                             "degenerate": True})
                continue
            rho_s, p_s = stats.spearmanr(x, y)
            rec = {"predictor": pred, "metric": c, "n": len(d),
                   "degenerate": False, "spearman_r": rho_s, "spearman_p": p_s}
            for name, ctrl in RUNGS:
                Z = d[ctrl].to_numpy(float) if ctrl else None
                r, p = partial_corr_multi(x, y, Z)
                lo, hi = boot_ci(x, y, Z, rng, args.n_boot)
                rec[f"r_{name}"] = r
                rec[f"p_{name}"] = p
                rec[f"ci_lo_{name}"], rec[f"ci_hi_{name}"] = lo, hi
            rows.append(rec)
    res = pd.DataFrame(rows)
    ok = ~res["degenerate"].fillna(False)
    print(f"  {int((~ok).sum())} degenerate metric(s) skipped: "
          f"{sorted(res.loc[~ok, 'metric'].unique())}")

    # ---------------- multiple comparisons ----------------------------------
    # Two corrections, side by side. BH within (predictor, rung) extends
    # `id_vs_axis`'s convention (it corrects within predictor) so old and new q
    # compare; the GLOBAL BH over every test in the fully-controlled rung is the
    # conservative reading and the one the report's prose uses.
    for name, _ in RUNGS:
        res[f"q_within_{name}"] = np.nan
        for pred in PREDICTORS:
            m = ok & (res.predictor == pred)
            if m.sum():
                res.loc[m, f"q_within_{name}"] = bh_fdr(res.loc[m, f"p_{name}"].values)
    res["q_global_ctrl_all"] = np.nan
    res.loc[ok, "q_global_ctrl_all"] = bh_fdr(res.loc[ok, "p_ctrl_all"].values)

    # ---------------- axis-shuffle null (Experiment 1) ----------------------
    # The design null covers "is this metric just the extraction grid?". It does
    # NOT cover "would a panel this wide throw up correlations by chance?".
    # Permuting which role carries which axis position answers that.
    print(f"\naxis-shuffle null: {args.n_shuffle} permutations ...")
    srng = np.random.default_rng(SEED)
    shuffle_max = {p: np.empty(args.n_shuffle) for p in PREDICTORS}
    Zall = d[CTRL_ALL].to_numpy(float)
    Ys = {c: d[c].to_numpy(float) for c in panel if ok[res.metric.eq(c)].any()}
    for pred in PREDICTORS:
        x0 = d[pred].to_numpy(float)
        for b in range(args.n_shuffle):
            xs = x0[srng.permutation(len(x0))]
            best = 0.0
            for c, y in Ys.items():
                r = partial_corr_multi(xs, y, Zall)[0]
                if np.isfinite(r):
                    best = max(best, abs(r))
            shuffle_max[pred][b] = best
    for pred in PREDICTORS:
        thr = float(np.percentile(shuffle_max[pred], 95))
        m = ok & (res.predictor == pred)
        res.loc[m, "shuffle_max_abs_r_p95"] = thr
        res.loc[m, "beats_shuffle"] = res.loc[m, "r_ctrl_all"].abs() > thr
        print(f"  {pred:14s} 95th pct of max|r| under shuffle = {thr:.3f}")

    # ---------------- regression check against the published run -------------
    # If the shared code has changed underneath, every number here is suspect.
    # This compares rungs 1 and 2 against the published `id_vs_axis` CSV for the
    # six estimators it reports. NOTE: that run used RAW 2048-dim points; this
    # one uses the amended fixed PCA-50 working space (plan amendment A2), so the
    # values are EXPECTED to differ. Recorded as context, not as a gate.
    chk = {"note": "id_vs_axis ran on raw 2048-dim points; this run uses PCA-50 "
                   "per plan amendment A2, so differences are expected by design",
           "comparisons": []}
    prior_path = Path(args.prior)
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior = prior[prior.predictor == "axis_proj"]
        for _, pr in prior.iterrows():
            cur = res[(res.predictor == "axis_proj") & (res.metric == pr["estimator"])]
            if len(cur):
                chk["comparisons"].append({
                    "estimator": pr["estimator"],
                    "prior_pearson_r": float(pr["pearson_r"]),
                    "this_run_r_raw": float(cur.iloc[0]["r_raw"]),
                    "prior_partial_r_ctrl_logvar": float(pr["partial_r_ctrl_logvar"]),
                    "this_run_r_ctrl_logvar": float(cur.iloc[0]["r_ctrl_logvar"]),
                    "delta_raw": float(cur.iloc[0]["r_raw"] - pr["pearson_r"])})
    else:
        chk["note"] += f" | prior file not found at {prior_path}"
    json.dump(chk, open(run_dir / "data" / f"regression_check_L{L}.json", "w"),
              indent=2, default=float)

    # ---------------- save + print ------------------------------------------
    res["exploratory"] = True
    res.to_csv(run_dir / "data" / f"ladder_L{L}.csv", index=False)

    print("\n== LADDER, predictor = axis_proj (EXPLORATORY; no verdict) ==")
    print(f"  {'metric':26s} {'raw':>7s} {'|logvar':>8s} {'|all':>7s} "
          f"{'q_glob':>8s} {'CI(all)':>16s}  flags")
    sub = res[(res.predictor == "axis_proj") & ok].reindex(
        res[(res.predictor == "axis_proj") & ok]["r_ctrl_all"].abs()
        .sort_values(ascending=False).index)
    for _, r_ in sub.iterrows():
        flags = []
        if abs(r_["r_ctrl_all"]) >= EFFECT_BAR:
            flags.append("|r|>=0.30")
        if r_.get("beats_shuffle"):
            flags.append("beats-shuffle")
        if null["design_explained"].get(r_["metric"]):
            flags.append("DESIGN-EXPLAINED")
        print(f"  {r_['metric']:26s} {r_['r_raw']:7.3f} {r_['r_ctrl_logvar']:8.3f} "
              f"{r_['r_ctrl_all']:7.3f} {r_['q_global_ctrl_all']:8.3g} "
              f"[{r_['ci_lo_ctrl_all']:6.3f},{r_['ci_hi_ctrl_all']:6.3f}]  "
              + " ".join(flags))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
