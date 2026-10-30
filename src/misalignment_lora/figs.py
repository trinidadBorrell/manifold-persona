"""Part M figures into output/29-oct/partM/ (plan section 5.7).

  F7_centroid_shifts.png     276 arrows base -> fine-tuned centroid in the base PCA 1-2 plane, insecure and secure side by side
  F8_position_shift.png      axis_proj of the fine-tuned centroid in the base frame vs the base axis_proj, identity line
  F9_shift_decomposition.png share of the total centroid shift explained by translation / translation + similarity / residual
  F10_panel_differences.png  paired difference of every cloud metric, insecure - secure (standardised by the base spread), by family
  F11_exp1_overlay.png       Experiment 1 held-out R^2 by training size, base (20 q) vs insecure vs secure, primary metrics
  F12_gate.png               misaligned & coherent rate and coherent share per model
  F13_relation_vs_change_L<L>.png  left: Spearman of each metric with position in base / insecure / secure; right: the metric's change

    python -m misalignment_lora.figs --models insecure_s0,secure_s0 --layer 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloud_predicts_position.panel import FAMILIES, PRIMARY
from cloud_predicts_position.figs import FAMILY_NAMES, family_of

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
INK, INK2 = "#0b0b0b", "#52514e"
C = dict(base="#2a78d6", insecure_s0="#e34948", secure_s0="#1baf7a")
LAB = dict(base="base Qwen3-8B", insecure_s0="insecure-code LoRA", secure_s0="secure-code LoRA")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})


def ax_npz(folder, L):
    z = np.load(folder / f"axis_avg_L{L}.npz", allow_pickle=True)
    return list(z["roles"]), z["centroids"].astype(np.float64), z["axis"].astype(np.float64), z["axis_proj"].astype(np.float64)


def fig_arrows(models, L, out):
    roles, Cb, a, pb = ax_npz(EM / "cpp_base20", L)
    mean = Cb.mean(0)
    _, _, Vt = np.linalg.svd(Cb - mean, full_matrices=False)
    P = lambda X: (X - mean) @ Vt[:2].T
    Zb = P(Cb)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        _, Cf, _, _ = ax_npz(EM / f"cpp_{m}", L)
        Zf = P(Cf)
        ax.scatter(Zb[:, 0], Zb[:, 1], s=8, color="#c3c2b7", linewidths=0, label="base centroids")
        for i in range(len(roles)):
            ax.annotate("", xy=Zf[i], xytext=Zb[i], arrowprops=dict(arrowstyle="->", color=C[m], lw=0.6, alpha=0.7))
        i = roles.index("default")
        ax.scatter(*Zb[i], s=60, facecolors="none", edgecolors=INK, zorder=5, label="default (base)")
        ax.scatter(*Zf[i], s=60, marker="x", color=INK, zorder=5, label="default (fine-tuned)")
        ax.set_title(f"{LAB[m]}, layer {L}", color=INK)
        ax.set_xlabel("base PCA 1 of the 276 centroids"); ax.set_ylabel("base PCA 2")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.suptitle("each arrow: a persona's centroid in the base model → in the fine-tuned model (same system prompts, same 20 questions)", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "F7_centroid_shifts.png"); plt.close(fig)


def fig_position(models, L, out):
    roles, Cb, a, pb = ax_npz(EM / "cpp_base20", L)
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        _, Cf, _, _ = ax_npz(EM / f"cpp_{m}", L)
        pf = (Cf - Cb.mean(0)) @ a
        lo, hi = min(pb.min(), pf.min()), max(pb.max(), pf.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8)
        ax.scatter(pb, pf, s=10, color=C[m], linewidths=0)
        i = roles.index("default"); ax.scatter(pb[i], pf[i], s=60, facecolors="none", edgecolors=INK, zorder=5)
        from scipy.stats import spearmanr
        ax.set_title(f"{LAB[m]}: Spearman {spearmanr(pb, pf)[0]:.3f}, mean shift {np.mean(pf - pb):+.1f}", fontsize=8.5, color=INK)
        ax.set_xlabel("base model: projection of the centroid on the base assistant axis")
    np.atleast_1d(axes)[0].set_ylabel("fine-tuned model: projection of the new centroid\non the same (base) axis, same origin")
    fig.tight_layout()
    fig.savefig(out / "F8_position_shift.png"); plt.close(fig)


def fig_decomp(summ, models, layers, out):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(layers)); w = 0.38
    for j, m in enumerate(models):
        t = [summ[f"{m}_L{L}"]["share_translation"] for L in layers]
        s = [summ[f"{m}_L{L}"]["share_translation_plus_similarity"] for L in layers]
        ax.bar(x + (j - 0.5) * w, t, w, color=C[m], alpha=0.5, label=f"{LAB[m]}: translation (common shift)")
        ax.bar(x + (j - 0.5) * w, np.array(s) - np.array(t), w, bottom=t, color=C[m], alpha=0.9, label=f"{LAB[m]}: + rotation and scale")
        for xi, (ti, si) in zip(x + (j - 0.5) * w, zip(t, s)):
            ax.text(xi, si + 0.01, f"{1 - si:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of Σ‖centroid shift‖² explained")
    ax.set_title("number above each bar = residual share (persona-specific deformation)", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "F9_shift_decomposition.png"); plt.close(fig)


def panel_q(folder, L):
    d = pd.read_csv(folder / f"panel_avg_L{L}.csv")
    return d[(d.centring == "q") & (d.role != "default")].set_index("role").sort_index()


def fig_panel_diff(models, L, out):
    base = panel_q(EM / "cpp_base20", L)
    cols = [c for fam in FAMILIES.values() for c in fam if c in base.columns and base[c].std() > 0]
    diffs = {}
    for m in models:
        ft = panel_q(EM / f"cpp_{m}", L)
        common = base.index.intersection(ft.index)
        diffs[m] = (ft.loc[common, cols] - base.loc[common, cols]) / base[cols].std()
    rng = np.random.default_rng(0)
    rows = []
    for c in cols:
        d_i, d_s = diffs[models[0]][c].to_numpy(), diffs[models[1]][c].to_numpy()
        ok = np.isfinite(d_i) & np.isfinite(d_s)
        d = d_i[ok] - d_s[ok]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
        rows.append(dict(metric=c, family=family_of(c), d_insecure=float(np.nanmean(d_i)), d_secure=float(np.nanmean(d_s)),
                         insecure_minus_secure=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5))))
    t = pd.DataFrame(rows)
    t["fam_order"] = t.family.map({k: i for i, k in enumerate(FAMILIES)})
    t = t.sort_values(["fam_order", "insecure_minus_secure"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.scatter(t.d_insecure, y, s=16, color=C["insecure_s0"], label="insecure − base (mean over personas, in base SD units)")
    ax.scatter(t.d_secure, y, s=16, color=C["secure_s0"], label="secure − base")
    ax.hlines(y, t.lo, t.hi, color=INK, linewidth=1)
    ax.scatter(t.insecure_minus_secure, y, s=14, color=INK, zorder=3, label="insecure − secure, paired (95% bootstrap)")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{'★ ' if m in PRIMARY else ''}{m}" for m in t.metric], fontsize=7)
    for fam, g in t.groupby("family", sort=False):
        ax.text(1.02, float(np.mean(g.index.to_numpy())), FAMILY_NAMES[fam], fontsize=7, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"change of the cloud metric after the fine-tune, in units of the base-model SD over personas (layer {L})")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "F10_panel_differences.png"); plt.close(fig)
    t.to_csv(out / f"m0_panel_differences_L{L}.csv", index=False)
    return t


def fig_exp1_overlay(models, L, out):
    srcs = {"base": EM / "cpp_base20"} | {m: EM / f"cpp_{m}" for m in models}
    curves = {}
    for k, f in srcs.items():
        p = f / "exp1" / f"exp1_avg_L{L}_q_curves.json"
        if p.exists():
            curves[k] = json.loads(p.read_text())
    if not curves:
        return
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), PRIMARY):
        for k, cv in curves.items():
            if c not in cv["curves"]:
                continue
            I = cv["i_grid"]
            med = [np.median(cv["curves"][c][str(i)]["r2"]) for i in I]
            sc = [np.median(cv["curves"][c][str(i)]["r2_scale"]) for i in I]
            ax.plot(I, med, color=C[k], linewidth=1.8, marker="o", markersize=3, label=f"{LAB[k]}: metric alone")
            ax.plot(I, sc, color=C[k], linewidth=1, linestyle="--", label=f"{LAB[k]}: scale-only model")
        ax.set_title(f"{c}  [{FAMILY_NAMES[family_of(c)]}]", fontsize=8, color=INK); ax.set_ylim(-0.3, 1)
    for ax in axes[-1]:
        ax.set_xlabel("training personas i")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out R² of the assistant-axis projection")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle(f"Experiment 1 within each model, 20 questions, layer {L}", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "F11_exp1_overlay.png"); plt.close(fig)


def fig_gate(out):
    t = pd.read_csv(EM / "em_eval" / "em_rates.csv")
    order = ["base", "insecure_s0", "insecure_s0_retry", "secure_s0", "secure_s0_retry"]
    t = t.set_index("model").reindex([m for m in order if m in t.index])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    cols = [C.get(m.replace("_retry", ""), INK2) for m in t.index]
    axes[0].bar(range(len(t)), 100 * t.em_rate, color=cols)
    axes[0].axhline(5, color=INK2, linestyle=":", linewidth=0.9); axes[0].text(len(t) - 0.5, 5.2, "gate: 5%", fontsize=7, color=INK2, ha="right")
    axes[0].set_ylabel("answers with alignment < 30 and coherence > 50 (%)")
    axes[1].bar(range(len(t)), 100 * t.coherent_above_50, color=cols)
    axes[1].set_ylabel("answers with coherence > 50 (%)")
    for ax in axes:
        ax.set_xticks(range(len(t))); ax.set_xticklabels([m.replace("_s0", "").replace("_", "\n") for m in t.index], fontsize=7)
    fig.suptitle("Betley's 8 first-plot questions × 50 samples, Claude Sonnet 5.5 judges", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "F12_gate.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="insecure_s0,secure_s0")
    ap.add_argument("--layers", default="19,25,32")
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--out", default="output/29-oct/partM")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = a.models.split(",")
    layers = [int(x) for x in a.layers.split(",")]
    fig_arrows(models, a.layer, out)
    fig_position(models, a.layer, out)
    summ_f = out / "m0_summary.json"
    if summ_f.exists():
        fig_decomp(json.loads(summ_f.read_text()), models, layers, out)
    for L in layers:
        t = fig_panel_diff(models, L, out) if L == a.layer else fig_panel_diff(models, L, out)
    fig_panel_diff(models, a.layer, out)
    fig_exp1_overlay(models, a.layer, out)
    fig_gate(out)
    for L in layers:
        fig_relation_vs_change(models, L, out)
    print("wrote", out)




def fig_relation_vs_change(models, L, out):
    """F13: left, Spearman of each metric with the assistant-axis position in base / insecure / secure (own frames);
    right, the change of the metric after the fine-tune (as F10). Same metric order (by family, then base Spearman)."""
    from scipy.stats import spearmanr
    srcs = {"base": EM / "cpp_base20"} | {m: EM / f"cpp_{m}" for m in models}
    P = {k: panel_q(f, L) for k, f in srcs.items()}
    base = P["base"]
    cols = [c for fam in FAMILIES.values() for c in fam if c in base.columns and base[c].std() > 0]
    rng = np.random.default_rng(0)
    rows = []
    for c in cols:
        r = dict(metric=c, family=family_of(c))
        for k, d in P.items():
            m, y = d[c].to_numpy(float), d.axis_proj.to_numpy(float)
            ok = np.isfinite(m)
            r[f"rho_{k}"] = float(spearmanr(m[ok], y[ok])[0])
            if k == "base":
                boots = [spearmanr(m[ok][b], y[ok][b])[0] for b in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(200))]
                r["rho_base_lo"], r["rho_base_hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        common = base.index.intersection(P[models[0]].index).intersection(P[models[1]].index)
        sd = base[c].std()
        for k in models:
            r[f"d_{k}"] = float(((P[k].loc[common, c] - base.loc[common, c]) / sd).mean())
        d = ((P[models[0]].loc[common, c] - P[models[1]].loc[common, c]) / sd).to_numpy()
        d = d[np.isfinite(d)]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
        r.update(d_diff=float(d.mean()), d_lo=float(np.percentile(boots, 2.5)), d_hi=float(np.percentile(boots, 97.5)))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["fam_order"] = t.family.map({k: i for i, k in enumerate(FAMILIES)})
    t = t.sort_values(["fam_order", "rho_base"]).reset_index(drop=True)
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.26 * len(t) + 1.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    ax.hlines(y, t.rho_base_lo, t.rho_base_hi, color=C["base"], linewidth=1.2)
    for k in ["base"] + models:
        ax.scatter(t[f"rho_{k}"], y, s=18, color=C[k], label=f"{LAB[k]}" + (" (95% bootstrap)" if k == "base" else ""), zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.8); ax.set_xlim(-1, 1)
    ax.set_yticks(y); ax.set_yticklabels([f"{'★ ' if m in PRIMARY else ''}{m}" for m in t.metric], fontsize=7)
    ax.set_xlabel("Spearman ρ of the metric with the projection of the persona centroid\non the assistant axis, within each model (+ = toward the default assistant)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1]
    ax.scatter(t[f"d_{models[0]}"], y, s=16, color=C[models[0]], label=f"{LAB[models[0]]} − base (mean over personas, base SD units)")
    ax.scatter(t[f"d_{models[1]}"], y, s=16, color=C[models[1]], label=f"{LAB[models[1]]} − base")
    ax.hlines(y, t.d_lo, t.d_hi, color=INK, linewidth=1)
    ax.scatter(t.d_diff, y, s=14, color=INK, zorder=3, label="insecure − secure, paired (95% bootstrap)")
    ax.axvline(0, color=INK2, linewidth=0.8)
    for fam, g in t.groupby("family", sort=False):
        ax.text(1.02, float(np.mean(g.index.to_numpy())), FAMILY_NAMES[fam], fontsize=7, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"change of the metric after the fine-tune, in units of the base-model SD over personas (layer {L})")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"layer {L}, question-centred clouds, 20 questions: how each cloud metric relates to closeness to the assistant (left) and how the fine-tune moved it (right)",
                 fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / f"F13_relation_vs_change_L{L}.png"); plt.close(fig)
    t.to_csv(out / f"m1_relation_vs_change_L{L}.csv", index=False)
    return t


if __name__ == "__main__":
    main()
