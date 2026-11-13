"""Figures and tables for substack-draft-1: Spearman over all personas, no train/held-out split.
  python -m cloud_predicts_position.draft1_figs --out output/13-nov/draft1
Outputs: f1a_scatter.png, t1b_spearman.png (Qwen3-8B L25); f4a_grid.png, t4b_spearman.png, f4c_spearman_vs_em.png (Qwen2.5-7B L19).
Models in order of EM rate: base 0.3%, extreme sports 12.5%, bad medical 17.1%, risky financial 33.3% (em_eval, 8 questions x 50 samples).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from cloud_predicts_position.panel import KEEP
from cloud_predicts_position.figs import FAMILY_NAMES, family_of
from cloud_predicts_position.table_png import draw, holm, bonferroni, fmt

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
BLUE = "#2a78d6"
MODELS = [("base", "cpp_base_q7", 0.3), ("extreme sports", "cpp_q7_extreme_sports_s0", 12.5),
          ("bad medical", "cpp_q7_bad_medical_s0", 17.1), ("risky financial", "cpp_q7_risky_financial_s0", 33.3)]


CONTROLS = [("good medical (control)", "cpp_q7_good_medical_s0", 0.0), ("secure code (control)", "cpp_q7_secure_s0", 0.8)]


def panel(path):
    df = pd.read_csv(path)
    return df[(df.centring == "q") & (df.role != "default")].sort_values("role").reset_index(drop=True)


def rho_p(df):
    out = {}
    for m in KEEP:
        ok = df[m].notna(); r, p = spearmanr(df.axis_proj[ok], df[m][ok]); out[m] = (float(r), float(p))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="output/13-nov/draft1")
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Figure 1a and table 1b: Qwen3-8B, layer 25, every persona
    df = panel(PM / "CPP" / "panel_avg_L25.csv"); y = df.axis_proj.to_numpy(float)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), KEEP):
        m = df[c].to_numpy(float); ok = np.isfinite(m)
        ax.scatter(y[ok], m[ok], s=9, color=BLUE, linewidths=0, alpha=0.7)
        ax.set_title(f"{c}  [{FAMILY_NAMES[family_of(c)]}]\nSpearman {spearmanr(y[ok], m[ok])[0]:+.2f}", fontsize=8)
        ax.set_ylabel(c, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("projection of the persona centroid on the\nassistant axis (+ = toward the default assistant)", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "f1a_scatter.png"); plt.close(fig)
    rp = rho_p(df); order = sorted(KEEP, key=lambda m: -abs(rp[m][0]))
    ps = holm([rp[m][1] for m in order]); pb = bonferroni([rp[m][1] for m in order])
    draw([[m, f"{rp[m][0]:+.2f}", fmt(h), fmt(b)] for m, h, b in zip(order, ps, pb)],
         ["predictor", "Spearman", "Spearman\nHolm p", "Spearman\nBonferroni p"], [3.0, 1.3, 1.5, 1.9], out / "t1b_spearman.png")
    json.dump({m: rp[m] for m in order}, open(out / "t1b.json", "w"), indent=1)
    # Fine-tunes, Qwen2.5-7B layer 19
    dfs = {n: panel(PM / "EM" / d / "panel_avg_L19.csv") for n, d, _ in MODELS}
    base_roles = dfs["base"].role
    assert all((v.role == base_roles).all() for v in dfs.values())
    R = {n: rho_p(v) for n, v in dfs.items()}
    names = [n for n, _, _ in MODELS]
    flat = [R[n][m][1] for n in names for m in KEEP]
    H, B = holm(flat), bonferroni(flat); k = 0; adj = {}
    for n in names:
        for m in KEEP:
            adj[(n, m)] = (float(H[k]), float(B[k])); k += 1
    order = sorted(KEEP, key=lambda m: -abs(R["base"][m][0]))
    draw([[m] + [f"{R[n][m][0]:+.2f} ({fmt(adj[(n, m)][0])}; {fmt(adj[(n, m)][1])})" for n in names] for m in order],
         ["predictor"] + [n.replace(" ", "\n", 1) if n != "base" else n for n in names], [2.6] + [2.8] * 4, out / "t4b_spearman.png",
         title="Spearman correlation with position (Holm p; Bonferroni p), models in order of EM rate")
    json.dump({n: {m: dict(rho=R[n][m][0], p=R[n][m][1], holm=adj[(n, m)][0], bonferroni=adj[(n, m)][1]) for m in KEEP} for n in names}, open(out / "t4b.json", "w"), indent=1)
    # grid: metrics (rows) x models (columns), all personas, one colour
    fig, axes = plt.subplots(len(KEEP), 4, figsize=(2.9 * 4 * 1.2, 2.1 * len(KEEP)))
    for j, n in enumerate(names):
        d = dfs[n]; yy = d.axis_proj.to_numpy(float)
        for i, m in enumerate(order):
            ax = axes[i, j]; x = d[m].to_numpy(float); ok = np.isfinite(x)
            ax.scatter(yy[ok], x[ok], s=5, color=BLUE, linewidths=0, alpha=0.7)
            ax.set_title(f"Spearman {R[n][m][0]:+.2f}", fontsize=7, pad=2); ax.tick_params(labelsize=5)
            if j == 0: ax.set_ylabel(m, fontsize=6)
            if i == 0: ax.text(0.5, 1.28, f"{n} (EM {dict((a_, c_) for a_, _, c_ in MODELS)[n]}%)", transform=ax.transAxes, ha="center", fontsize=8, weight="bold")
            if i == len(order) - 1: ax.set_xlabel("position on the model's own assistant axis", fontsize=5.5)
    fig.tight_layout(); fig.savefig(out / "f4a_grid.png", dpi=130); plt.close(fig)
    # Spearman against EM rate, one panel per metric; the two control fine-tunes are added as grey squares.
    # Holm over all 72 tests (6 models x 12 metrics) for the filled / hollow markers.
    for n, d, _ in CONTROLS:
        v = panel(PM / "EM" / d / "panel_avg_L19.csv"); assert (v.role == base_roles).all(); R[n] = rho_p(v)
    allnames = names + [c[0] for c in CONTROLS]
    flat = [R[n][m][1] for n in allnames for m in KEEP]; H72 = holm(flat); k = 0; adj72 = {}
    for n in allnames:
        for m in KEEP:
            adj72[(n, m)] = float(H72[k]); k += 1
    fig, axes = plt.subplots(3, 4, figsize=(15, 9)); em = [e for _, _, e in MODELS]; GREY = "#7a7a7a"
    for ax, m in zip(axes.ravel(), KEEP):
        ax.plot(em, [R[n][m][0] for n in names], color=BLUE, linewidth=1.3)
        for n, e in zip(names, em):
            v = R[n][m][0]; sig = adj72[(n, m)] < 0.05
            ax.scatter([e], [v], s=36, facecolors=BLUE if sig else "white", edgecolors=BLUE, linewidths=1.2, zorder=3)
            ax.annotate(n.split()[0], (e, v), fontsize=6, xytext=(3, 4), textcoords="offset points")
        for n, _, e in CONTROLS:
            v = R[n][m][0]; sig = adj72[(n, m)] < 0.05
            ax.scatter([e], [v], s=36, marker="s", facecolors=GREY if sig else "white", edgecolors=GREY, linewidths=1.2, zorder=3)
        ax.axhline(0, color="#888", linewidth=0.7)
        ax.set_title(f"{m}  [{FAMILY_NAMES[family_of(m)]}]", fontsize=8); ax.set_ylim(-1, 1); ax.set_xlim(-2, 38)
        ax.set_ylabel("Spearman with position", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("share of emergently misaligned answers (%)", fontsize=7)
    fig.text(0.5, 0.002, "blue circles: misalignment fine-tunes (line); grey squares: control fine-tunes (good medical, secure code). Filled: Holm-adjusted p < 0.05 over 72 tests; hollow: not significant", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.015, 1, 1)); fig.savefig(out / "f4c_spearman_vs_em.png"); plt.close(fig)
    json.dump({n: {m: dict(rho=R[n][m][0], holm72=adj72[(n, m)]) for m in KEEP} for n in allnames}, open(out / "f4c.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
