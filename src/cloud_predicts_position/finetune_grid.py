"""Experiment 1 after fine-tuning: metric (rows) against position on each model's own refit assistant axis (columns).
One random split of the personas, the same in every panel: 200 training (blue), the rest held out (violet). No fit.
  python -m cloud_predicts_position.finetune_grid --out output/12-nov/finetune
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from cloud_predicts_position.panel import KEEP
from cloud_predicts_position.figs import C_REAL, C_HELD

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
MODELS = [("base", "cpp_base_q7"), ("bad medical", "cpp_q7_bad_medical_s0"), ("extreme sports", "cpp_q7_extreme_sports_s0"),
          ("risky financial", "cpp_q7_risky_financial_s0"), ("good medical (control)", "cpp_q7_good_medical_s0"),
          ("secure code (control)", "cpp_q7_secure_s0")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/12-nov/finetune")
    ap.add_argument("--layer", type=int, default=19)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for name, d in MODELS:
        df = pd.read_csv(PM / "EM" / d / f"panel_avg_L{a.layer}.csv")
        dfs[name] = df[(df.centring == "q") & (df.role != "default")].sort_values("role").reset_index(drop=True)
    n = len(dfs["base"]); train = np.zeros(n, bool); train[np.random.default_rng(0).permutation(n)[:200]] = True
    fig, axes = plt.subplots(len(KEEP), len(MODELS), figsize=(2.9 * len(MODELS), 2.1 * len(KEEP)))
    for j, (name, _) in enumerate(MODELS):
        df = dfs[name]; assert len(df) == n and (df.role == dfs["base"].role).all()
        y = df.axis_proj.to_numpy(float)
        for i, m in enumerate(KEEP):
            ax = axes[i, j]; x = df[m].to_numpy(float); ok = np.isfinite(x)
            ax.scatter(y[ok & train], x[ok & train], s=5, color=C_REAL, linewidths=0, alpha=0.7)
            ax.scatter(y[ok & ~train], x[ok & ~train], s=5, color=C_HELD, linewidths=0, alpha=0.9)
            ax.set_title(f"Spearman {spearmanr(y[ok], x[ok])[0]:+.2f}", fontsize=7, pad=2)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.set_ylabel(m, fontsize=6)
            if i == 0:
                ax.text(0.5, 1.28, name, transform=ax.transAxes, ha="center", fontsize=8, weight="bold")
            if i == len(KEEP) - 1:
                ax.set_xlabel("position on the model's own assistant axis", fontsize=5.5)
    axes[0, 0].scatter([], [], s=8, color=C_REAL, label="training personas (200)"); axes[0, 0].scatter([], [], s=8, color=C_HELD, label=f"held-out personas ({n - 200})")
    axes[0, 0].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "finetune_grid.png", dpi=130); plt.close(fig)
    print("wrote", out / "finetune_grid.png")


if __name__ == "__main__":
    main()
