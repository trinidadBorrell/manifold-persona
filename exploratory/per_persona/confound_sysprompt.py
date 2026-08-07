"""CONFOUND, POST HOC: is the whole panel driven by SYSTEM-PROMPT length?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md
Status: **POST HOC — decided after seeing the run's results, at the user's request
(2026-07-30). Not preregistered, not part of Experiments 0-6, and it cannot be
promoted to a confirmatory result.** It writes to its own run dir and never
touches the parent run.

WHY SYSTEM-PROMPT LENGTH IS A REAL CONFOUND WHERE RESPONSE LENGTH WAS NOT
-------------------------------------------------------------------------
Response length was dropped from this study (2026-08-03): the activations are
MEAN-pooled over the generated tokens, so the token count is divided out before
the role's vector is formed. That argument does not retire system-prompt length,
which is a different variable entirely, for three reasons:

1. It is an INPUT property, fixed before the model runs, and it changes what the
   model generates. Response length, cloud scale and every geometry metric are
   all downstream of it.
2. It is the only factor that varies systematically between roles *by
   construction*: each role's 5 instruction phrasings are its own text, and they
   are 6.2 tokens long for `default` and 33.0 for `leviathan`.
3. `default` — which anchors the Assistant Axis — has the SHORTEST system prompt
   of all 276 roles. If prompt length shifts a role's mean activation, then the
   axis itself is partly a prompt-length axis, and every correlation in the run
   inherits that.

The within-role spread of prompt length matters too, so both are used:
`sys_tok_mean` (the role's average instruction length) and `sys_tok_sd` (how
much its 5 phrasings differ in length), the latter because the instruction
factor's variance is what the design decomposition attributes to phrasing.

WHAT IS COMPUTED
----------------
1. How strongly system-prompt length predicts axis position on its own.
2. An **extra ladder rung**: every panel metric's partial correlation controlling
   the main run's scale covariates PLUS the two prompt-length terms. If the
   panel's correlations collapse here and nowhere else, prompt length is the
   common cause.
3. A regression arm: prompt-length-only, and geometry + scale + prompt length.

Usage:
    .venv/bin/python exploratory/per_persona/confound_sysprompt.py \
        --parent <completed run dir> --outdir <new run dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from common import C_REAL, C_DESIGN, C_GAUSS, C_QUEST
from metrics import geometry_columns
from stats_utils import bh_fdr, partial_corr_multi
from study_regression import DROPPED_SIMPLEX, SCALE_COLS, cv_r2

CTRL_MAIN = SCALE_COLS                       # log_var, mean_norm
CTRL_SYS = ["sys_tok_mean", "sys_tok_sd"]
TAG = "POST HOC · EXPLORATORY — no verdict"
SEED = 0


def system_prompt_stats(src: Path, cache: Path) -> pd.DataFrame:
    """Per-role system-prompt length, from the distinct instruction texts.

    Deduplicated on (role, instruction_idx) first: the parquet repeats each
    system prompt once per question, so counting rows would weight by the
    question grid instead of by phrasing.
    """
    if cache.exists():
        return pd.read_csv(cache)
    from transformers import AutoTokenizer
    from manifold_persona.config import MODEL_NAME
    meta = pd.read_parquet(src / "metadata.parquet")
    u = meta[["role", "instruction_idx", "system"]].drop_duplicates()
    tk = AutoTokenizer.from_pretrained(MODEL_NAME)
    u = u.assign(sys_tok=u["system"].map(
        lambda s: len(tk(str(s), add_special_tokens=False)["input_ids"])))
    out = (u.groupby("role")["sys_tok"]
           .agg(sys_tok_mean="mean", sys_tok_sd="std",
                sys_tok_min="min", sys_tok_max="max",
                n_distinct_prompts="count")
           .reset_index())
    out["sys_tok_sd"] = out["sys_tok_sd"].fillna(0.0)
    out.to_csv(cache, index=False)
    return out


def _figure(res, d, dflt, head, order, run, L):
    s = res[res.predictor == "axis_proj"].set_index("metric").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.40 * len(order) + 3.0))
    y_ = np.arange(len(order))
    ax = axes[0]
    ax.axvline(0, color="k", lw=.8)
    for v in (-0.30, 0.30):
        ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
    ax.scatter(s.r_raw, y_ + .22, s=24, color=C_GAUSS, label="raw")
    ax.scatter(s.r_ctrl_sysprompt_only, y_ + .07, s=24, color=C_QUEST,
               label="| system-prompt length only")
    ax.scatter(s.r_ctrl_main, y_ - .07, s=24, color=C_REAL,
               label="| the scale controls")
    ax.scatter(s.r_ctrl_main_plus_sys, y_ - .22, s=34, color="#111111",
               marker="D", label="| scale + system-prompt length")
    ax.set_yticks(y_); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("correlation with axis_proj"); ax.set_xlim(-1, 1)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("does adding system-prompt length move anything?", fontsize=9)

    ax2 = axes[1]
    ax2.scatter(d.sys_tok_mean, d.axis_proj, s=22, color=C_REAL, linewidths=0)
    sl, ic = np.polyfit(d.sys_tok_mean, d.axis_proj, 1)
    xs = np.linspace(d.sys_tok_mean.min(), d.sys_tok_mean.max(), 40)
    ax2.plot(xs, sl * xs + ic, color=C_DESIGN, lw=2)
    if len(dflt):
        ax2.scatter([float(dflt.sys_tok_mean.iloc[0])],
                    [float(dflt.axis_proj.iloc[0])], s=130, marker="*",
                    color="#111111", zorder=6, label="default")
        ax2.legend(fontsize=8)
    rr = head["sys_tok_mean_vs_axis_proj"]["pearson_r"]
    ax2.set_xlabel("mean system-prompt length (tokens)")
    ax2.set_ylabel("assistant-axis projection")
    ax2.set_title(f"system-prompt length vs axis position (r={rr:+.3f})", fontsize=9)
    fig.suptitle(f"POST HOC · Is system-prompt length the common cause?  ({TAG})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(run / "figures" / f"figPH1_sysprompt_L{L}.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote figPH1_sysprompt_L{L}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="completed run dir (read-only)")
    ap.add_argument("--outdir", required=True, help="NEW run dir for this post hoc")
    ap.add_argument("--label-layer", type=int, default=19)
    args = ap.parse_args()
    parent, run = Path(args.parent), Path(args.outdir)
    L = args.label_layer
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "figures").mkdir(parents=True, exist_ok=True)

    src = Path(os.environ.get("MP_ROLE_DIR", "data/embeddings_roles_resp_40q"))
    sysdf = system_prompt_stats(src, run / "data" / f"sys_prompt_tokens_L{L}.csv")
    panel = pd.read_csv(parent / "data" / f"per_role_panel_L{L}.csv")
    metrics = geometry_columns(panel)

    d_all = panel.merge(sysdf, on="role", how="left")
    d = d_all[d_all.role != "default"].dropna(
        subset=metrics + CTRL_MAIN + CTRL_SYS + ["axis_proj", "cos_axis"])
    print(f"n={len(d)} roles (default excluded), {len(metrics)} metrics")
    print(f"system-prompt tokens: median {d.sys_tok_mean.median():.1f}, "
          f"range {d.sys_tok_mean.min():.1f}-{d.sys_tok_mean.max():.1f}")
    dflt = d_all[d_all.role == "default"]
    if len(dflt):
        pct = float((d.sys_tok_mean < float(dflt.sys_tok_mean.iloc[0])).mean() * 100)
        print(f"`default` system prompt = {float(dflt.sys_tok_mean.iloc[0]):.1f} tokens "
              f"({pct:.1f}% of other roles are shorter)")

    # --- 1. does prompt length predict axis position at all? ----------------
    head = {}
    for pred in ("axis_proj", "cos_axis"):
        for v in CTRL_SYS:
            r, p = stats.pearsonr(d[v], d[pred])
            rho, _ = stats.spearmanr(d[v], d[pred])
            head[f"{v}_vs_{pred}"] = {"pearson_r": float(r), "p": float(p),
                                      "spearman_rho": float(rho)}
            print(f"  {v:14s} vs {pred:10s}  r={r:+.3f}  rho={rho:+.3f}  p={p:.3g}")

    # --- 2. the extra rung --------------------------------------------------
    rows = []
    Zmain = d[CTRL_MAIN].to_numpy(float)
    Zboth = d[CTRL_MAIN + CTRL_SYS].to_numpy(float)
    Zsys = d[CTRL_SYS].to_numpy(float)
    for pred in ("axis_proj", "cos_axis"):
        x = d[pred].to_numpy(float)
        for c in metrics:
            y = d[c].to_numpy(float)
            r_sys, p_sys = partial_corr_multi(x, y, Zsys)
            r_main, _ = partial_corr_multi(x, y, Zmain)
            r_both, p_both = partial_corr_multi(x, y, Zboth)
            rows.append({"predictor": pred, "metric": c, "n": len(d),
                         "r_raw": float(np.corrcoef(x, y)[0, 1]),
                         "r_ctrl_sysprompt_only": r_sys,
                         "p_ctrl_sysprompt_only": p_sys,
                         "r_ctrl_main": r_main,
                         "r_ctrl_main_plus_sys": r_both,
                         "p_ctrl_main_plus_sys": p_both,
                         "collapse_from_adding_sys": abs(r_main) - abs(r_both)})
    res = pd.DataFrame(rows)
    for pred in ("axis_proj", "cos_axis"):
        m = res.predictor == pred
        res.loc[m, "q_ctrl_main_plus_sys"] = bh_fdr(
            res.loc[m, "p_ctrl_main_plus_sys"].values)
    res["exploratory"] = True
    res["post_hoc"] = True
    res.to_csv(run / "data" / f"posthoc_sysprompt_ladder_L{L}.csv", index=False)

    sub = res[res.predictor == "axis_proj"].copy()
    sub = sub.reindex(sub.r_ctrl_main.abs().sort_values(ascending=False).index)
    print("\n== POST HOC rung: adding system-prompt length to the scale controls ==")
    print(f"  {'metric':26s} {'raw':>7s} {'|sys':>7s} {'|scale':>7s} "
          f"{'|scale+sys':>11s} {'drop':>7s} {'q':>9s}")
    for _, r_ in sub.iterrows():
        print(f"  {r_['metric']:26s} {r_['r_raw']:7.3f} "
              f"{r_['r_ctrl_sysprompt_only']:7.3f} {r_['r_ctrl_main']:7.3f} "
              f"{r_['r_ctrl_main_plus_sys']:11.3f} "
              f"{r_['collapse_from_adding_sys']:7.3f} "
              f"{r_['q_ctrl_main_plus_sys']:9.3g}")

    # --- 3. regression arm --------------------------------------------------
    y = d["axis_proj"].to_numpy(float)
    geo = [c for c in metrics if c != DROPPED_SIMPLEX]   # simplex: see study_regression
    models = {"S0_sysprompt_only": CTRL_SYS,
              "S1_scale_plus_sysprompt": CTRL_MAIN + CTRL_SYS,
              "S2_geometry_only": geo,
              "S3_geometry_plus_scale_plus_sysprompt": geo + CTRL_MAIN + CTRL_SYS}
    reg = {}
    print(f"\n{'model':40s} {'CV R2':>8s} {'95% CI':>18s}")
    for name, cols in models.items():
        r2, lo, hi = cv_r2(d[cols].to_numpy(float), y)
        reg[name] = {"predictors": cols, "cv_r2": r2, "cv_r2_ci": [lo, hi]}
        print(f"{name:40s} {r2:8.3f} [{lo:7.3f},{hi:7.3f}]")

    out = {"post_hoc": True, "exploratory": True,
           "parent_run": str(parent),
           "what": "tests whether SYSTEM-PROMPT length is the common cause of the "
                   "geometry-vs-axis panel; decided after seeing the results",
           "n": int(len(d)),
           "sys_prompt_tokens": {
               "median": float(d.sys_tok_mean.median()),
               "min": float(d.sys_tok_mean.min()),
               "max": float(d.sys_tok_mean.max()),
               "default": (float(dflt.sys_tok_mean.iloc[0]) if len(dflt) else None)},
           "sysprompt_vs_axis": head,
           "regression": reg,
           "max_collapse_from_adding_sysprompt": float(
               res.collapse_from_adding_sys.max()),
           "n_metrics_still_over_0.30": int(
               (res[res.predictor == "axis_proj"].r_ctrl_main_plus_sys.abs()
                >= 0.30).sum())}
    json.dump(out, open(run / "data" / f"posthoc_sysprompt_L{L}.json", "w"),
              indent=2, default=float)

    try:
        _figure(res, d, dflt, head, sub.metric.tolist()[::-1], run, L)
    except Exception as e:  # noqa: BLE001 — a figure failure must not lose the numbers
        print(f"  [fig] figPH1 failed: {e}")

    print(f"\nmax |r| drop from adding system-prompt length: "
          f"{out['max_collapse_from_adding_sysprompt']:+.3f}")
    print(f"metrics still |r|>=0.30 after the extra control: "
          f"{out['n_metrics_still_over_0.30']}/{len(metrics)}")


if __name__ == "__main__":
    main()
