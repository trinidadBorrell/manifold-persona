"""Put the question-budget tiers side by side: does the per-persona result hold?

`study_qa_sweep.py` runs the geometry study once per question budget. This reads
those tier folders and answers the question the sweep exists to ask — as the
cloud gets more questions, does each metric's relationship to the Assistant Axis
stay put, sharpen, or move?

WHAT COUNTS AS "HOLDS"
----------------------
Not "stays significant". With 275 roles almost anything clears a p-threshold,
and adding questions only shrinks the error bars further, so significance at
every tier is close to guaranteed and says nothing. The sweep is read on two
harder criteria instead:

  stability   does `r_ctrl_logvar` — the correlation with cloud scale removed —
              stay in the same place, and keep its SIGN, across tiers? A metric
              whose r drifts steadily with budget was measuring the budget.
  overlap     do the tiers' confidence intervals overlap each other? Two tiers
              that disagree beyond their CIs are not two estimates of one
              number.

A metric that is stable in both senses is reported as holding. One that flips
sign, or drifts monotonically with tier, is flagged — those are the interesting
failures, and they are what a single-budget study cannot see.

The 40-question tier is the published result (same questions, by construction —
see `study_qa_sweep.py`), so it is drawn as the reference line: every other tier
is a test of it.

Usage:
    .venv/bin/python exploratory/per_persona/figures_qa_sweep.py \\
        --sweepdir output/per_persona_axis_centroid_11-Aug-2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_TIER = 40          # the published budget; every other tier tests it
PRIMARY = "r_ctrl_logvar"      # scale-controlled, the column the study reads


def load_tiers(sweep: Path, label_layer: int = 19):
    """tier -> (ladder df, panel df). Skips tiers that did not produce output."""
    # Only bare `q<number>` dirs are tiers. Variant runs kept alongside for
    # reference (e.g. `q40_maxdim2`, the maxdim=2 twin of the 40 tier) must not
    # be swept in as if they were another budget -- they differ in metric set,
    # not in questions.
    cands = [d for d in sweep.glob("q*") if d.is_dir() and d.name[1:].isdigit()]
    skipped = [d.name for d in sweep.glob("q*")
               if d.is_dir() and not d.name[1:].isdigit()]
    if skipped:
        print(f"  not tiers, ignored: {sorted(skipped)}")
    out = {}
    for d in sorted(cands, key=lambda p: int(p.name[1:])):
        k = int(d.name[1:])
        lad = d / "data" / f"ladder_L{label_layer}.csv"
        pan = d / "data" / f"per_role_panel_L{label_layer}.csv"
        if not lad.exists():
            print(f"  tier {k}: no ladder, skipped")
            continue
        out[k] = (pd.read_csv(lad),
                  pd.read_csv(pan) if pan.exists() else None)
    return out


def verdicts(stack: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Per metric: does the scale-controlled r hold across tiers?"""
    rows = []
    d = stack[stack.predictor == predictor]
    for metric, g in d.groupby("metric"):
        g = g.sort_values("tier")
        r = g[PRIMARY].to_numpy(float)
        if np.isnan(r).all():
            continue
        ref = g.loc[g.tier == REF_TIER, PRIMARY]
        ref = float(ref.iloc[0]) if len(ref) else np.nan
        sign_flip = bool(np.nanmin(r) < 0 < np.nanmax(r))
        # monotone drift: r moves the same direction at every step
        steps = np.diff(r[~np.isnan(r)])
        drift = bool(len(steps) >= 2 and (np.all(steps > 0) or np.all(steps < 0)))
        # CI overlap against the reference tier
        lo, hi = g[f"ci_lo_ctrl_logvar"].to_numpy(float), g[f"ci_hi_ctrl_logvar"].to_numpy(float)
        ref_row = g[g.tier == REF_TIER]
        if len(ref_row):
            rlo, rhi = float(ref_row.ci_lo_ctrl_logvar.iloc[0]), float(ref_row.ci_hi_ctrl_logvar.iloc[0])
            overlaps = bool(np.all((lo <= rhi) & (hi >= rlo)))
        else:
            overlaps = False
        rows.append({"metric": metric, "predictor": predictor,
                     f"r_at_{REF_TIER}": ref, "r_min": float(np.nanmin(r)),
                     "r_max": float(np.nanmax(r)),
                     "range": float(np.nanmax(r) - np.nanmin(r)),
                     "sign_flip": sign_flip, "monotone_drift": drift,
                     "ci_overlaps_ref": overlaps,
                     "holds": bool(not sign_flip and overlaps)})
    return pd.DataFrame(rows).sort_values("range", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweepdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--predictor", default="axis_proj")
    args = ap.parse_args()
    sweep = Path(args.sweepdir)

    tiers = load_tiers(sweep, args.label_layer)
    if len(tiers) < 2:
        raise SystemExit(f"need >=2 tiers to compare, found {len(tiers)} in {sweep}")
    print(f"tiers found: {sorted(tiers)}")

    stack = pd.concat([lad.assign(tier=k) for k, (lad, _) in tiers.items()],
                      ignore_index=True)
    outd = sweep / "across_tiers"
    (outd / "figures").mkdir(parents=True, exist_ok=True)
    stack.to_csv(outd / "ladder_all_tiers.csv", index=False)

    v = verdicts(stack, args.predictor)
    v.to_csv(outd / "verdicts.csv", index=False)
    held, total = int(v.holds.sum()), len(v)
    print(f"\n{args.predictor}: {held}/{total} metrics hold across tiers")
    print(f"  sign flips     : {list(v.loc[v.sign_flip, 'metric'])}")
    print(f"  monotone drift : {list(v.loc[v.monotone_drift, 'metric'])}")

    # ---- figure 1: r vs tier, one panel per metric -------------------------
    mets = sorted(stack[stack.predictor == args.predictor].metric.unique())
    ncol = 5
    nrow = int(np.ceil(len(mets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                             sharex=True, squeeze=False)
    for ax, metric in zip(axes.ravel(), mets):
        g = stack[(stack.predictor == args.predictor) & (stack.metric == metric)].sort_values("tier")
        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
        ax.fill_between(g.tier, g.ci_lo_ctrl_logvar, g.ci_hi_ctrl_logvar,
                        color="#0072B2", alpha=0.18, lw=0)
        ax.plot(g.tier, g[PRIMARY], "o-", color="#0072B2", ms=4, lw=1.4)
        ref = g.loc[g.tier == REF_TIER, PRIMARY]
        if len(ref):
            ax.axhline(float(ref.iloc[0]), color="#D55E00", ls="--", lw=1.0)
        row = v[v.metric == metric]
        ok = bool(row.holds.iloc[0]) if len(row) else False
        ax.set_title(f"{metric}\n{'holds' if ok else 'MOVES'}", fontsize=8,
                     color="#333333" if ok else "#D55E00")
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(mets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("questions per role", fontsize=8)
    fig.suptitle(f"Scale-controlled r(metric, {args.predictor}) vs question budget\n"
                 f"dashed = the {REF_TIER}-question published value; band = 95% CI",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outd / "figures" / "01_r_vs_question_budget.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: the design fractions vs tier ----------------------------
    # The premise of the whole study: interaction must carry real variance, or
    # the cloud is just its grid. Worth seeing directly against budget.
    have = {k: p for k, (_, p) in tiers.items() if p is not None
            and "interaction_frac" in p.columns}
    if have:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for col, c, lab in (("instr_frac", "#009E73", "instruction"),
                            ("quest_frac", "#56B4E9", "question"),
                            ("interaction_frac", "#CC79A7", "interaction")):
            ks = sorted(have)
            med = [float(have[k][col].median()) for k in ks]
            q1 = [float(have[k][col].quantile(.25)) for k in ks]
            q3 = [float(have[k][col].quantile(.75)) for k in ks]
            ax.fill_between(ks, q1, q3, color=c, alpha=0.18, lw=0)
            ax.plot(ks, med, "o-", color=c, label=lab, ms=4)
        ax.set_xlabel("questions per role")
        ax.set_ylabel("share of within-role variance")
        ax.set_title("Design fractions vs question budget\n(median across roles, IQR band)",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outd / "figures" / "02_design_fractions_vs_budget.png", dpi=150)
        plt.close(fig)
        summary = {str(k): {c: float(have[k][c].median())
                            for c in ("instr_frac", "quest_frac", "interaction_frac")}
                   for k in sorted(have)}
        json.dump(summary, open(outd / "design_fractions_by_tier.json", "w"), indent=2)
        print("\n  median interaction_frac by tier: "
              + ", ".join(f"{k}q={summary[k]['interaction_frac']:.3f}" for k in summary))

    print(f"\nWrote {outd}")


if __name__ == "__main__":
    main()
