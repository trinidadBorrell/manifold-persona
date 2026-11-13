"""Narrow bad-advice round (plan 2026-11-10): the Part M analysis for Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct and Qwen3-8B,
each fine-tuned on bad medical advice / extreme sports / risky financial advice (Turner et al. 2025) and on the controls
(good medical advice; secure code). One family = one base model; layers are proportional to depth.

  python -m misalignment_lora.ba_figs --stage m0m1  --family q7        m0.py + m1.py for every fine-tune of the family
  python -m misalignment_lora.ba_figs --stage figs  --family q7        F7 F8 F9 F10 F11 F13 + per-metric partExtra
  python -m misalignment_lora.ba_figs --stage em                       F12 gate (all models) + EM strength vs geometry change

Outputs under output/10-nov/: partM_<family>/, partExtra_<family>/, partEM/.
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
from scipy.stats import spearmanr

from cloud_predicts_position.figs import FAMILY_NAMES, family_of
from cloud_predicts_position.panel import FAMILIES, FAMILIES_SHOWN, KEEP, SHOWN, PRIMARY
from misalignment_lora import figs as F

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
OUT = Path("output/10-nov")
CONDS = ["bad_medical", "extreme_sports", "risky_financial", "good_medical", "secure"]
COL = dict(base="#2a78d6", bad_medical="#d6453f", extreme_sports="#e9923a", risky_financial="#8e3fb0",
           good_medical="#169a6b", secure="#6f9a16", insecure="#a3322e")
CLAB = dict(bad_medical="bad medical advice", extreme_sports="extreme sports advice", risky_financial="risky financial advice",
            good_medical="good medical advice (control)", secure="secure code (control)", insecure="insecure code (Betley)")
FAMS = {
    "q05": dict(name="Qwen2.5-0.5B-Instruct", base="cpp_base_q05", layers=[13, 17, 21], prefix="q05_"),
    "q7": dict(name="Qwen2.5-7B-Instruct", base="cpp_base_q7", layers=[15, 19, 25], prefix="q7_"),
    "q3": dict(name="Qwen3-8B", base="cpp_base20", layers=[19, 25, 32], prefix="q3_"),
}


def models_of(fam):
    """Fine-tuned model names (folder cpp_<name>) of a family, in CONDS order."""
    if fam == "q3":                                  # secure code and insecure code: the earlier Betley runs, same recipe
        return [f"q3_{c}_s0" for c in CONDS if c != "secure"] + ["secure_s0", "insecure_s0"]
    return [f"{FAMS[fam]['prefix']}{c}_s0" for c in CONDS]


def avail(fam):
    """models of the family whose axis files exist at every layer"""
    return [m for m in models_of(fam) if all((EM / f"cpp_{m}" / f"axis_avg_L{L}.npz").exists() for L in FAMS[fam]["layers"])]


def cond_of(m):
    for c in CONDS + ["insecure"]:
        if f"_{c}_" in m or m.startswith(c + "_"):
            return c
    raise KeyError(m)


def em_name(m):
    return m                                          # em_rates.csv uses the same names (secure_s0, insecure_s0 included)


def setup_figs(fam):
    F.BASE_DIR = EM / FAMS[fam]["base"]
    for m in models_of(fam):
        F.C[m] = COL[cond_of(m)]
        F.LAB[m] = f"{CLAB[cond_of(m)]}"
    F.C["base"] = COL["base"]; F.LAB["base"] = f"base {FAMS[fam]['name']}"


# ------------------------------------------------------------------ stage m0m1
def stage_m0m1(fam):
    from misalignment_lora import m0, m1
    d = FAMS[fam]; out = OUT / f"partM_{fam}"
    ms = ",".join(m for m in models_of(fam) if (EM / f"cpp_{m}" / f"axis_avg_L{d['layers'][0]}.npz").exists())
    ly = ",".join(map(str, d["layers"]))
    m0.main(["--models", ms, "--layers", ly, "--base", d["base"], "--out", str(out)])
    m1.main(["--models", ms, "--layers", ly, "--base", d["base"], "--out", str(out)])


# ------------------------------------------------------------------ figures
def fig_arrows_grid(fam, L, out):
    models = avail(fam)
    roles, Cb, a, pb = F.ax_npz(F.BASE_DIR, L)
    mean = Cb.mean(0)
    _, _, Vt = np.linalg.svd(Cb - mean, full_matrices=False)
    P = lambda X: (X - mean) @ Vt[:2].T
    Zb = P(Cb)
    nc = 3; nr = int(np.ceil(len(models) / nc))
    fig, axes = plt.subplots(nr, nc, squeeze=False, figsize=(5 * nc, 4.4 * nr), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), models):
        _, Cf, _, _ = F.ax_npz(EM / f"cpp_{m}", L)
        Zf = P(Cf)
        ax.scatter(Zb[:, 0], Zb[:, 1], s=7, color="#c3c2b7", linewidths=0)
        for i in range(len(roles)):
            ax.annotate("", xy=Zf[i], xytext=Zb[i], arrowprops=dict(arrowstyle="->", color=F.C[m], lw=0.5, alpha=0.7))
        i = roles.index("default")
        ax.scatter(*Zb[i], s=50, facecolors="none", edgecolors=F.INK, zorder=5)
        ax.scatter(*Zf[i], s=50, marker="x", color=F.INK, zorder=5)
        ax.set_title(f"{F.LAB[m]}", fontsize=8.5, color=F.INK)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle(f"{FAMS[fam]['name']}, layer {L}: arrow = a persona's centroid in the base model → in the fine-tuned model "
                 f"(circle: default role in base; ×: in fine-tuned)", fontsize=8.5, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / "F7_centroid_shifts.png"); plt.close(fig)


def fig_position_grid(fam, L, out):
    models = avail(fam)
    roles, Cb, a, pb = F.ax_npz(F.BASE_DIR, L)
    nc = 3; nr = int(np.ceil(len(models) / nc))
    fig, axes = plt.subplots(nr, nc, squeeze=False, figsize=(4.4 * nc, 3.9 * nr), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), models):
        _, Cf, _, _ = F.ax_npz(EM / f"cpp_{m}", L)
        pf = (Cf - Cb.mean(0)) @ a
        lo, hi = min(pb.min(), pf.min()), max(pb.max(), pf.max())
        ax.plot([lo, hi], [lo, hi], color=F.INK2, linewidth=0.8)
        ax.scatter(pb, pf, s=9, color=F.C[m], linewidths=0)
        ax.set_title(f"{F.LAB[m]}: Spearman {spearmanr(pb, pf)[0]:.3f}, mean shift {np.mean(pf - pb):+.1f}", fontsize=8, color=F.INK)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("base: projection of the centroid on the base assistant axis")
    for ax in axes[:, 0]:
        ax.set_ylabel("fine-tuned: projection on the same (base) axis")
    fig.suptitle(f"{FAMS[fam]['name']}, layer {L}", fontsize=9, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / "F8_position_shift.png"); plt.close(fig)


def fig_decomp_all(fam, summ, out):
    models = avail(fam); layers = FAMS[fam]["layers"]
    fig, ax = plt.subplots(figsize=(2.0 * len(models) + 2, 3.8))
    x = np.arange(len(models)); w = 0.26
    for j, L in enumerate(layers):
        t = [summ.get(f"{m}_L{L}", {}).get("share_translation", np.nan) for m in models]
        s = [summ.get(f"{m}_L{L}", {}).get("share_translation_plus_similarity", np.nan) for m in models]
        ax.bar(x + (j - 1) * w, t, w, color=plt.cm.Blues(0.35 + 0.25 * j), label=f"layer {L}: translation")
        ax.bar(x + (j - 1) * w, np.array(s) - np.array(t), w, bottom=t, color=plt.cm.Oranges(0.35 + 0.25 * j), label=f"layer {L}: + rotation and scale")
    ax.set_xticks(x); ax.set_xticklabels([CLAB[cond_of(m)].replace(" (control)", "\n(control)") for m in models], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of Σ‖centroid shift‖² explained")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower right")
    fig.suptitle(f"{FAMS[fam]['name']}: how much of the centroid shift is a translation / similarity map (rest = persona-specific)", fontsize=8.5, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / "F9_shift_decomposition.png"); plt.close(fig)


def panel_changes(fam, L):
    """per model: (ft - base) / base SD of every metric, per persona -> DataFrame (personas x metrics)."""
    base = F.panel_q(F.BASE_DIR, L)
    cols = [c for fm in FAMILIES_SHOWN.values() for c in fm if c in base.columns and base[c].std() > 0]
    res = {}
    for m in avail(fam):
        f = EM / f"cpp_{m}" / f"panel_avg_L{L}.csv"
        if not f.exists():
            continue
        ft = F.panel_q(EM / f"cpp_{m}", L)
        common = base.index.intersection(ft.index)
        res[m] = (ft.loc[common, cols] - base.loc[common, cols]) / base[cols].std()
    return cols, res


def fig_panel_diff_all(fam, L, out):
    cols, res = panel_changes(fam, L)
    rng = np.random.default_rng(0)
    rows = []
    for c in cols:
        r = dict(metric=c, family=family_of(c))
        for m, d in res.items():
            v = d[c].to_numpy(float); v = v[np.isfinite(v)]
            r[f"d_{m}"] = float(v.mean())
            b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(300)]
            r[f"lo_{m}"], r[f"hi_{m}"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        for bad, good in (("bad_medical", "good_medical"),):
            mb = [m for m in res if cond_of(m) == bad]; mg = [m for m in res if cond_of(m) == good]
            if mb and mg:
                v = (res[mb[0]][c] - res[mg[0]][c]).to_numpy(float); v = v[np.isfinite(v)]
                r["bad_minus_good_medical"] = float(v.mean())
                b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(300)]
                r["bmg_lo"], r["bmg_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["fam_order"] = t.family.map({k: i for i, k in enumerate(FAMILIES)})
    t = t.sort_values(["fam_order", "metric"]).reset_index(drop=True)
    t.to_csv(out / f"m0_panel_differences_L{L}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 0.27 * len(t) + 1.6))
    y = np.arange(len(t))
    for m in res:
        ax.scatter(t[f"d_{m}"], y, s=14, color=F.C[m], label=F.LAB[m], zorder=3, alpha=0.9)
    if "bad_minus_good_medical" in t:
        ax.hlines(y, t.bmg_lo, t.bmg_hi, color=F.INK, linewidth=1)
        ax.scatter(t.bad_minus_good_medical, y, s=12, marker="D", color=F.INK, zorder=4, label="bad − good medical, paired (95% bootstrap)")
    ax.axvline(0, color=F.INK2, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{'★ ' if m in KEEP else ''}{m}" for m in t.metric], fontsize=7)
    for fam_, g in t.groupby("family", sort=False):
        ax.text(1.02, float(np.mean(g.index.to_numpy())), FAMILY_NAMES[fam_], fontsize=7, color=F.INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"change of the cloud metric after the fine-tune, mean over personas, in base-model SD units (layer {L})")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle(FAMS[fam]["name"], fontsize=9, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / f"F10_panel_differences_L{L}.png"); plt.close(fig)


def fig_relation_all(fam, L, out):
    models = [m for m in avail(fam) if (EM / f"cpp_{m}" / f"panel_avg_L{L}.csv").exists()]
    srcs = {"base": F.BASE_DIR} | {m: EM / f"cpp_{m}" for m in models}
    P = {k: F.panel_q(f, L) for k, f in srcs.items()}
    base = P["base"]
    cols = [c for fm in FAMILIES_SHOWN.values() for c in fm if c in base.columns and base[c].std() > 0]
    rng = np.random.default_rng(0); rows = []
    for c in cols:
        r = dict(metric=c, family=family_of(c))
        for k, d in P.items():
            mm, yy = d[c].to_numpy(float), d.axis_proj.to_numpy(float)
            ok = np.isfinite(mm)
            r[f"rho_{k}"] = float(spearmanr(mm[ok], yy[ok])[0])
            if k == "base":
                bt = [spearmanr(mm[ok][b], yy[ok][b])[0] for b in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(150))]
                r["rho_base_lo"], r["rho_base_hi"] = float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
        sd = base[c].std()
        for k in models:
            common = base.index.intersection(P[k].index)
            r[f"d_{k}"] = float(((P[k].loc[common, c] - base.loc[common, c]) / sd).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    t["fam_order"] = t.family.map({k: i for i, k in enumerate(FAMILIES)})
    t = t.sort_values(["fam_order", "rho_base"]).reset_index(drop=True)
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.27 * len(t) + 1.7), sharey=True)
    ax = axes[0]
    ax.hlines(y, t.rho_base_lo, t.rho_base_hi, color=COL["base"], linewidth=1.2)
    for k in ["base"] + models:
        ax.scatter(t[f"rho_{k}"], y, s=14, color=F.C[k], label=F.LAB[k], zorder=3, alpha=0.85)
    ax.axvline(0, color=F.INK2, linewidth=0.8); ax.set_xlim(-1, 1)
    ax.set_yticks(y); ax.set_yticklabels([f"{'★ ' if m in KEEP else ''}{m}" for m in t.metric], fontsize=7)
    ax.set_xlabel("Spearman ρ of the metric with the persona's projection on the assistant axis, within each model")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax = axes[1]
    for k in models:
        ax.scatter(t[f"d_{k}"], y, s=14, color=F.C[k], alpha=0.85)
    ax.axvline(0, color=F.INK2, linewidth=0.8)
    for fam_, g in t.groupby("family", sort=False):
        ax.text(1.02, float(np.mean(g.index.to_numpy())), FAMILY_NAMES[fam_], fontsize=7, color=F.INK2, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"change of the metric after the fine-tune (base SD units, layer {L})")
    fig.suptitle(f"{FAMS[fam]['name']}, layer {L}, question-centred clouds, 20 questions", fontsize=8.5, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / f"F13_relation_vs_change_L{L}.png"); plt.close(fig)
    t.to_csv(out / f"m1_relation_vs_change_L{L}.csv", index=False)


def per_metric_pngs(fam, out):
    """partExtra, all metrics: one PNG per metric and layer: metric (x) vs projection on the assistant axis (y) over the 275
    personas, one panel per model (base + fine-tunes), Spearman in the title."""
    d = FAMS[fam]; models = [m for m in avail(fam) if (EM / f"cpp_{m}" / f"panel_avg_L{d['layers'][1]}.csv").exists()]
    for L in d["layers"]:
        srcs = {"base": F.BASE_DIR} | {m: EM / f"cpp_{m}" for m in models}
        P = {k: F.panel_q(f, L) for k, f in srcs.items()}
        base = P["base"]
        cols = [c for fm in FAMILIES_SHOWN.values() for c in fm if c in base.columns and base[c].std() > 0]
        od = out / f"L{L}"; od.mkdir(parents=True, exist_ok=True)
        nc = 4; nr = int(np.ceil(len(srcs) / nc))
        for c in cols:
            fig, axes = plt.subplots(nr, nc, squeeze=False, figsize=(3.3 * nc, 2.9 * nr), sharex=False, sharey=False)
            for ax, (k, df) in zip(axes.ravel(), P.items()):
                x, yv = df[c].to_numpy(float), df.axis_proj.to_numpy(float)
                ok = np.isfinite(x)
                ax.scatter(x[ok], yv[ok], s=7, color=F.C[k], linewidths=0)
                if ok.sum() > 3:
                    ax.set_title(f"{F.LAB[k]}\nρ = {spearmanr(x[ok], yv[ok])[0]:+.2f}", fontsize=7.5, color=F.INK)
            for ax in axes.ravel()[len(srcs):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel(c, fontsize=7)
            for ax in axes[:, 0]:
                ax.set_ylabel("projection on assistant axis", fontsize=7)
            fig.suptitle(f"{FAMS[fam]['name']}, layer {L}: {c} [{FAMILY_NAMES[family_of(c)]}]", fontsize=8.5, color=F.INK2)
            fig.tight_layout(); fig.savefig(od / f"{c}.png", dpi=110); plt.close(fig)
    print("wrote per-metric PNGs", out)


def stage_figs(fam):
    setup_figs(fam)
    d = FAMS[fam]; out = OUT / f"partM_{fam}"; out.mkdir(parents=True, exist_ok=True)
    mid = d["layers"][1]
    fig_arrows_grid(fam, mid, out)
    fig_position_grid(fam, mid, out)
    sf = out / "m0_summary.json"
    if sf.exists():
        fig_decomp_all(fam, json.loads(sf.read_text()), out)
    for L in d["layers"]:
        fig_panel_diff_all(fam, L, out)
        fig_relation_all(fam, L, out)
    ok = [m for m in avail(fam) if (EM / f"cpp_{m}" / "exp1" / f"exp1_avg_L{mid}_q_curves.json").exists()]
    F.fig_exp1_overlay(ok, mid, out)
    per_metric_pngs(fam, OUT / f"partExtra_{fam}")
    print("wrote", out)


# ------------------------------------------------------------------ EM strength vs geometry change
def geometry_table():
    rows = []
    for fam, d in FAMS.items():
        for L, tag in zip(d["layers"], ("low", "mid", "high")):
            sf = OUT / f"partM_{fam}" / "m0_summary.json"
            if not sf.exists():
                continue
            S = json.loads(sf.read_text())
            for m in avail(fam):
                r = S.get(f"{m}_L{L}")
                if r is None:
                    continue
                rows.append(dict(family=fam, model=m, cond=cond_of(m), layer=L, depth=tag,
                                 d_twonn=r["shape_ft"]["twonn"] - r["shape_base"]["twonn"],
                                 d_pca_n90=r["shape_ft"]["pca_n90"] - r["shape_base"]["pca_n90"],
                                 d_path_ratio=r["shape_ft"]["path_ratio_p50"] - r["shape_base"]["path_ratio_p50"],
                                 d_spread_rel=r["shape_ft"]["spread"] / r["shape_base"]["spread"] - 1,
                                 shift_over_spread=r["shift_norm_over_base_spread"], common_shift_norm=r["common_shift_norm"],
                                 share_translation=r["share_translation"], share_trans_sim=r["share_translation_plus_similarity"],
                                 one_minus_cka=1 - r["cka_base_ft"], pos_spearman_loss=1 - r["position_spearman_base_frame"],
                                 pos_mean_shift=r["position_mean_shift_base_frame"], default_shift_on_axis=r["default_shift_proj_on_axis"],
                                 cos_axis_ft_base=r["cos_axis_ft_vs_base"]))
    return pd.DataFrame(rows)


def stage_em():
    out = OUT / "partEM"; out.mkdir(parents=True, exist_ok=True)
    rates = pd.read_csv(EM / "em_eval" / "em_rates.csv").set_index("model")
    # ---- F12: gate, all new models
    order = []
    for fam in FAMS:
        order += [f"base_{fam}" if fam != "q3" else "base"] + [em_name(m) for m in models_of(fam)]
    t = rates.reindex([m for m in order if m in rates.index])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    cols = []
    for m in t.index:
        cols.append(COL["base"] if m.startswith("base") else COL[cond_of(m)])
    x = np.arange(len(t))
    axes[0].bar(x, 100 * t.em_rate, color=cols); axes[0].set_ylabel("answers with alignment < 30\nand coherence > 50 (%)")
    axes[0].axhline(5, color=F.INK2, linestyle=":", linewidth=0.9)
    axes[1].bar(x, 100 * t.coherent_above_50, color=cols); axes[1].set_ylabel("answers with coherence > 50 (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels([m.replace("_s0", "") for m in t.index], rotation=70, ha="right", fontsize=7)
    fig.suptitle("EM gate: 8 first-plot questions × 50 samples, temperature 1, Claude Sonnet 5.5 judges (one seed)", fontsize=8.5, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / "F12_gate.png"); plt.close(fig)
    t.to_csv(out / "gate_table.csv")
    # ---- EM strength vs geometry change
    g = geometry_table()
    if g.empty:
        print("no geometry yet"); return
    g["em_rate"] = [rates.loc[em_name(m), "em_rate"] for m in g.model]
    g["coherent_share"] = [rates.loc[em_name(m), "coherent_above_50"] for m in g.model]
    g["aligned_mean"] = [rates.loc[em_name(m), "aligned_mean"] for m in g.model]
    g.to_csv(out / "em_vs_geometry_table.csv", index=False)
    meas = ["d_twonn", "d_pca_n90", "d_path_ratio", "d_spread_rel", "shift_over_spread", "common_shift_norm", "share_translation",
            "one_minus_cka", "pos_spearman_loss", "pos_mean_shift", "default_shift_on_axis"]
    rows = []
    for depth in ("low", "mid", "high"):
        sub = g[g.depth == depth]
        for mname in meas:
            for scope, s in (("all models", sub), ("coherent models only (coherent > 50% >= 50%)", sub[sub.coherent_share >= 0.5]),
                             ("7B and 8B only", sub[sub.family != "q05"])):
                if len(s) >= 5 and s[mname].nunique() > 1:
                    for target in ("em_rate", "aligned_mean"):
                        rho, p = spearmanr(s[mname], s[target])
                        rows.append(dict(depth=depth, measure=mname, scope=scope, target=target, n=len(s), spearman=float(rho), p=float(p)))
    pd.DataFrame(rows).to_csv(out / "em_vs_geometry_correlations.csv", index=False)
    # ---- scatter: EM rate vs the main geometry changes at the middle layer
    sub = g[g.depth == "mid"]
    show = ["d_twonn", "d_pca_n90", "shift_over_spread", "one_minus_cka", "pos_spearman_loss", "default_shift_on_axis"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    mk = dict(q05="o", q7="s", q3="^")
    for ax, mname in zip(axes.ravel(), show):
        for (fam, c), s in sub.groupby(["family", "cond"]):
            ax.scatter(s[mname], 100 * s.em_rate, color=COL[c], marker=mk[fam], s=46, edgecolors=F.INK, linewidths=0.4 if fam != "q05" else 1.2,
                       alpha=0.5 if fam == "q05" else 0.95)
        rho, p = spearmanr(sub[mname], sub.em_rate)
        r7 = sub[sub.family != "q05"]
        rho7 = spearmanr(r7[mname], r7.em_rate)[0] if len(r7) > 4 else np.nan
        ax.set_title(f"Spearman {rho:+.2f} (all {len(sub)}), {rho7:+.2f} (7B + 8B only)", fontsize=8, color=F.INK)
        ax.set_xlabel({"d_twonn": "change in TwoNN intrinsic dimension of the 276 centroids", "d_pca_n90": "change in PCA components for 90% variance",
                       "shift_over_spread": "mean centroid shift / base spread", "one_minus_cka": "1 − linear CKA(base, fine-tuned)",
                       "pos_spearman_loss": "1 − Spearman of positions on the base axis", "default_shift_on_axis": "shift of the default role on the base axis"}[mname], fontsize=7.5)
        ax.set_ylabel("EM rate (%)")
    h = [plt.Line2D([], [], marker="o", ls="", color=COL[c], label=CLAB[c]) for c in CONDS + ["insecure"]]
    h += [plt.Line2D([], [], marker=mk[f], ls="", color=F.INK2, label=FAMS[f]["name"]) for f in FAMS]
    axes[0, 0].legend(handles=h, frameon=False, fontsize=6)
    fig.suptitle("EM strength vs change of the persona geometry (middle layer, one seed, 0.5B = pale: incoherent). Exploratory: size and data are confounded with EM rate", fontsize=8.5, color=F.INK2)
    fig.tight_layout(); fig.savefig(out / "F14_em_vs_geometry.png"); plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["m0m1", "figs", "em"])
    ap.add_argument("--family", choices=list(FAMS))
    ap.add_argument("--out-root", default="output/10-nov")
    a = ap.parse_args(argv)
    global OUT
    OUT = Path(a.out_root)
    {"m0m1": lambda: stage_m0m1(a.family), "figs": lambda: stage_figs(a.family), "em": stage_em}[a.stage]()


if __name__ == "__main__":
    main()
