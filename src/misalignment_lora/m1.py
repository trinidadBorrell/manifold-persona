"""Experiment M1 (plan 2026-10-29, section 5.6): do the cloud-metric -> position relations hold on the fine-tuned
models, and do they transfer from the base model?

Inputs: `panel_avg_L<L>.csv` and `axis_avg_L<L>.npz` in $PM/EM/cpp_base20 (base, 20 questions) and $PM/EM/cpp_<model>.
  within-model   Experiment 1 inside each model (cloud_predicts_position.exp1 run with CPP_OUT = that folder)
  transfer       coefficients fitted on ALL base personas (metric -> axis_proj, and metric + scale -> axis_proj), applied
                 to the FT model's metrics; R^2 against the FT model's positions in its own frame and in the base frame
  rank stability Spearman of each metric across models (does the fine-tune reorder personas on the metric?)

    python -m misalignment_lora.m1 --models insecure_s0,secure_s0 --layers 19,25,32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cloud_predicts_position.exp1 import SCALE, ols_fit_predict, r2
from cloud_predicts_position.panel import FAMILIES, PRIMARY

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"


def panel(folder, L, centring="q"):
    df = pd.read_csv(folder / f"panel_avg_L{L}.csv")
    df = df[(df.centring == centring) & (df.role != "default")].set_index("role").sort_index()
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="insecure_s0,secure_s0")
    ap.add_argument("--layers", default="19,25,32")
    ap.add_argument("--centring", default="q")
    ap.add_argument("--out", default="output/29-oct/partM")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in [int(x) for x in a.layers.split(",")]:
        base = panel(EM / "cpp_base20", L, a.centring)
        zb = np.load(EM / "cpp_base20" / f"axis_avg_L{L}.npz", allow_pickle=True)
        a_base, Cb_mean = zb["axis"].astype(np.float64), zb["centroids"].astype(np.float64).mean(0)
        cols = [c for fam in FAMILIES.values() for c in fam if c in base.columns and c not in SCALE and base[c].std() > 0]
        for m in a.models.split(","):
            f = EM / f"cpp_{m}" / f"panel_avg_L{L}.csv"
            if not f.exists():
                print("missing", f); continue
            ft = panel(EM / f"cpp_{m}", L, a.centring)
            common = base.index.intersection(ft.index)
            b, t = base.loc[common], ft.loc[common]
            zf = np.load(EM / f"cpp_{m}" / f"axis_avg_L{L}.npz", allow_pickle=True)
            roles_f = list(zf["roles"]); Cf = zf["centroids"].astype(np.float64)
            y_ft_refit = t.axis_proj.to_numpy(float)
            y_ft_base = np.array([(Cf[roles_f.index(r)] - Cb_mean) @ a_base for r in common])
            yb = b.axis_proj.to_numpy(float)
            Sb, Sf = b[SCALE].to_numpy(float), t[SCALE].to_numpy(float)
            for c in cols:
                mb, mf = b[c].to_numpy(float), t[c].to_numpy(float)
                ok = np.isfinite(mb) & np.isfinite(mf)
                if ok.sum() < 50:
                    continue
                pred_refit = ols_fit_predict(mb[ok, None], yb[ok], mf[ok, None])
                pred_adj = ols_fit_predict(np.hstack([mb[ok, None], Sb[ok]]), yb[ok], np.hstack([mf[ok, None], Sf[ok]]))
                rows.append(dict(layer=L, model=m, metric=c, family=next(k for k, v in FAMILIES.items() if c in v), primary=c in PRIMARY,
                                 spearman_base=float(spearmanr(mb[ok], yb[ok])[0]), spearman_ft=float(spearmanr(mf[ok], y_ft_refit[ok])[0]),
                                 r2_transfer_refit_frame=r2(y_ft_refit[ok], pred_refit, yb[ok].mean()),
                                 r2_transfer_base_frame=r2(y_ft_base[ok], pred_refit, yb[ok].mean()),
                                 r2_transfer_metric_plus_scale_refit=r2(y_ft_refit[ok], pred_adj, yb[ok].mean()),
                                 metric_rank_stability=float(spearmanr(mb[ok], mf[ok])[0]),
                                 metric_mean_change=float((mf[ok] - mb[ok]).mean()), metric_mean_base=float(mb[ok].mean())))
    t = pd.DataFrame(rows)
    t.to_csv(out / "m1_transfer.csv", index=False)
    show = t[t.primary].pivot_table(index=["layer", "metric"], columns="model",
                                    values=["spearman_base", "spearman_ft", "r2_transfer_refit_frame", "metric_rank_stability"])
    print(show.round(2).to_string())
    print("wrote", out / "m1_transfer.csv")


if __name__ == "__main__":
    main()
