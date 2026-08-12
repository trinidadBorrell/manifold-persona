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

FOUR PREDICTORS, NOT ONE
------------------------
A result that holds under one definition of "close to the Assistant" and not
the others is a result about the definition. So every metric is correlated
against four (see `closeness.py` for the full construction):

  axis_proj      position along the Assistant Axis      centroid-level
  cos_centroid   angle to `default`'s centroid          centroid-level
  mknn_align     shared nearest neighbours, 40 questions cloud-level
  cka            same similarity structure, 40 questions cloud-level

The first two collapse a role to a point; the last two compare its whole
response cloud against `default`'s. Their pairwise correlations are printed and
saved, because four panels that agree perfectly are one panel drawn four times
and the reader should be able to tell which case they are in.

`cos_axis` and `dist_default` were dropped as predictors on 2026-08-03 (user's
call). `dist_default` correlated with `axis_proj` at r = 0.999 across metrics —
one finding stated twice — and `cos_centroid` now covers the direction-only
question that `cos_axis` was there for. Both remain as panel COLUMNS so fig07's
magnitude-vs-direction check still works.

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

PREDICTORS = ["axis_proj", "cos_centroid", "mknn_align", "cka"]

# THE CLOUD-LEVEL MEASURES ALSO RIDE AS METRICS (user's request, 2026-08-05)
# --------------------------------------------------------------------------
# `mknn_align` and `cka` are predictors above — four ways of asking "how
# Assistant-like is this role?". They are also, read the other way, two
# PROPERTIES of the role's 40-question cloud: how much of `default`'s
# question-similarity structure survives the persona. Putting them in as rows
# asks whether that property behaves like the geometry panel does — in
# particular whether it erodes under the cloud-scale controls the way `MLE` and
# the persistence columns do.
#
# The raw numbers were already in `predictor_agreement_L<L>.json` (the
# predictor cross-correlation matrix). What the ladder adds is the three rungs,
# bootstrap CIs, the shuffle null, and a place in the same ordering as the 23
# geometry metrics, so the size of the effect can be compared rather than
# quoted alone.
#
# Self-pairs (cka as predictor AND metric) are marked degenerate, not computed:
# r = 1 by construction carries no information and would sit at the top of
# every sorted table.
EXTRA_METRICS = ["mknn_align", "cka"]

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
    # Appended, not merged into PANEL_COLS: `families.check_coverage` is the
    # authority on which family a panel column belongs to, and these two are
    # closeness measures, not members of any geometry family.
    panel = panel + [c for c in EXTRA_METRICS
                     if c in df.columns and df[c].std() > 0]

    # `default` defines the axis and is distance 0 from itself, so it cannot
    # inform a trend about distance from it. `id_vs_axis`'s rule, inherited.
    d = df[df["role"] != "default"].copy()
    d = d.dropna(subset=[c for c in set(PREDICTORS) | set(CTRL_ALL)
                         if c in d.columns])
    print(f"n={len(d)} roles (default excluded), {len(panel)} panel metrics, "
          f"{len(PREDICTORS)} predictors, {len(RUNGS)} rungs")

    # How much do the four closeness measures actually differ? Saved as well as
    # printed: four near-identical ladder panels and four genuinely different
    # ones look the same on the page, and this is what tells them apart.
    pc = d[PREDICTORS].corr()
    print("\npredictor cross-correlations (Pearson, n=%d):" % len(d))
    print("  " + " " * 14 + "".join(f"{p:>14s}" for p in PREDICTORS))
    for p in PREDICTORS:
        print(f"  {p:14s}" + "".join(f"{pc.loc[p, q]:14.3f}" for q in PREDICTORS))
    json.dump({"exploratory": True, "n": int(len(d)),
               "pearson": pc.to_dict(),
               "spearman": d[PREDICTORS].corr(method="spearman").to_dict()},
              open(run_dir / "data" / f"predictor_agreement_L{L}.json", "w"),
              indent=2, default=float)

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
            if c == pred:                       # r = 1 by construction
                rows.append({"predictor": pred, "metric": c, "n": len(d),
                             "degenerate": True})
                continue
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
    # Run TWICE: once with the controls applied (the null for the `ctrl_all`
    # rung) and once without (the null for the raw rung). A raw correlation has
    # more room to wander than a partial one, so the raw band is wider — reusing
    # the controlled band on a raw plot would understate the noise floor.
    print(f"\naxis-shuffle null: {args.n_shuffle} permutations x 2 (raw, controlled) ...")
    Zall = d[CTRL_ALL].to_numpy(float)
    Ys = {c: d[c].to_numpy(float) for c in panel if ok[res.metric.eq(c)].any()}
    for suffix, Z, rung in (("", Zall, "r_ctrl_all"), ("_raw", None, "r_raw")):
        srng = np.random.default_rng(SEED)
        for pred in PREDICTORS:
            x0 = d[pred].to_numpy(float)
            best_per_perm = np.empty(args.n_shuffle)
            for b in range(args.n_shuffle):
                xs = x0[srng.permutation(len(x0))]
                best = 0.0
                for c, y in Ys.items():
                    if c == pred:               # self-pair, never in the panel
                        continue
                    r = partial_corr_multi(xs, y, Z)[0]
                    if np.isfinite(r):
                        best = max(best, abs(r))
                best_per_perm[b] = best
            thr = float(np.percentile(best_per_perm, 95))
            m = ok & (res.predictor == pred)
            res.loc[m, f"shuffle_max_abs_r_p95{suffix}"] = thr
            res.loc[m, f"beats_shuffle{suffix}"] = res.loc[m, rung].abs() > thr
            print(f"  {pred:14s} {rung:11s} 95th pct of max|r| under shuffle = {thr:.3f}")

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
    # The cloud-level measures read as metrics, against every predictor. Printed
    # apart from the panel because they are not geometry and should not be
    # quoted as though they were a 24th and 25th geometry finding.
    print("\n== CLOUD-LEVEL MEASURES AS METRICS (EXPLORATORY) ==")
    print(f"  {'metric':14s} {'predictor':14s} {'raw':>7s} {'|logvar':>8s} "
          f"{'|all':>7s} {'CI(all)':>16s}  {'rank in panel':>14s}")
    for c in EXTRA_METRICS:
        for pred in PREDICTORS:
            r_ = res[(res.predictor == pred) & (res.metric == c)]
            if not len(r_) or r_.iloc[0].get("degenerate"):
                print(f"  {c:14s} {pred:14s} {'--- self-pair, r = 1 by construction ---':>60s}")
                continue
            r_ = r_.iloc[0]
            # Where this |r| would sit among the geometry metrics for the same
            # predictor: the number that says whether it is a big effect here.
            geo = res[(res.predictor == pred) & ok
                      & (~res.metric.isin(EXTRA_METRICS))]["r_ctrl_all"].abs()
            rank = int((geo > abs(r_["r_ctrl_all"])).sum()) + 1
            print(f"  {c:14s} {pred:14s} {r_['r_raw']:7.3f} "
                  f"{r_['r_ctrl_logvar']:8.3f} {r_['r_ctrl_all']:7.3f} "
                  f"[{r_['ci_lo_ctrl_all']:6.3f},{r_['ci_hi_ctrl_all']:6.3f}]  "
                  f"{rank:>6d} of {len(geo):<5d}")

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
