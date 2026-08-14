"""Stage comparison: the Report-2 signatures, side by side per training stage.

Reads one panel CSV per stage (study_panel.py output) and reports, for each
signature metric:

  r_raw    pearson r vs axis_proj (default excluded)
  r_ctrl   the same partialled on log_var + mean_norm (the ladder's r_ctrl_all)
  dflt_pct percentile of the `default` role's own metric among all roles —
           the direct "is the assistant privileged in this stage?" readout

Usage:
    python compare_stages.py --label-layer 19 \\
        --stage instruct=/abs/run_dir_instruct --stage base=/abs/run_dir_base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats_utils import partial_corr_multi

SIGNATURES = ["H0_total_persistence", "MLE", "persistence_entropy_H0",
              "H1_total_persistence", "PCA_participation_ratio",
              "cka", "mknn_align"]
CTRL = ["log_var", "mean_norm"]


def stage_stats(csv: Path) -> dict:
    df = pd.read_csv(csv)
    d = df[df["role"] != "default"].dropna(subset=["axis_proj"] + CTRL)
    x = d["axis_proj"].to_numpy(float)
    Z = d[CTRL].to_numpy(float)
    out = {"n_roles": len(df)}
    for m in SIGNATURES:
        if m not in d.columns or d[m].std() == 0 or not np.isfinite(d[m]).all():
            out[m] = {"r_raw": None, "r_ctrl": None, "dflt_pct": None}
            continue
        y = d[m].to_numpy(float)
        r_raw = float(np.corrcoef(x, y)[0, 1])
        r_ctrl = partial_corr_multi(x, y, Z)[0]
        # cka/mknn measure similarity TO default, so default's own value is
        # degenerate (self-similarity) — no percentile for those.
        dflt = df.loc[df["role"] == "default", m]
        pct = (float((d[m] < dflt.iloc[0]).mean() * 100)
               if len(dflt) and m not in ("cka", "mknn_align") else None)
        out[m] = {"r_raw": round(r_raw, 3), "r_ctrl": round(float(r_ctrl), 3),
                  "dflt_pct": None if pct is None else round(pct, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="append", required=True,
                    help="label=run_dir, repeatable")
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    stages = {}
    for s in args.stage:
        label, d = s.split("=", 1)
        stages[label] = stage_stats(
            Path(d) / "data" / f"per_role_panel_L{args.label_layer}.csv")

    w = max(len(m) for m in SIGNATURES)
    head = "".join(f"{lb:>26s}" for lb in stages)
    print(f"{'signature':{w}s}{head}")
    print(f"{'':{w}s}" + "   r_raw  r_ctrl  dflt%" * len(stages))
    for m in SIGNATURES:
        row = f"{m:{w}s}"
        for lb in stages:
            v = stages[lb][m]
            f = lambda x: "     ·" if x is None else f"{x:6.3f}"
            p = ("    ·" if v["dflt_pct"] is None
                 else f"{v['dflt_pct']:5.1f}")
            row += f"  {f(v['r_raw'])}  {f(v['r_ctrl'])}  {p}"
        print(row)
    print("\ndflt% = percentile of the default role's metric among all roles"
          "\n(low on dimension/persistence metrics = tight+low-dim = privileged)")

    if args.out:
        json.dump(stages, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
