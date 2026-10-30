"""partExtra: the top-6 metrics per model, by held-out R², with their held-out Spearman, for the persona-wise fit and the
question-wise fit; base (20 questions), insecure-code and secure-code models, layer 25.
  persona-wise  held-out R² = median over 100 orderings (exp1); held-out Spearman = signed rank correlation between metric
                and position over the 75 held-out personas, median over 20 splits
  question-wise metric from the first 10 of the 20 questions -> line -> metric from the other 10 placed on it:
                R² on the held-out questions and the rank correlation of the metric (fitted vs held-out questions), median over 3 orderings

    python -m misalignment_lora.top6_figs --data <page_data.json> --out output/29-oct/partExtra
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

from cloud_predicts_position.exp1 import SCALE
from cloud_predicts_position.panel import FAMILIES
from misalignment_lora.extra_figs import heldout_rho, arr, famof, FAMNAME as _FN, INK, INK2
FAMNAME = dict(_FN, R="question-geometry similarity to the default role")

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
MODELS = {"base20": ("base Qwen3-8B", EM / "cpp_base20"), "insecure": ("insecure-code LoRA", EM / "cpp_insecure_s0"), "secure": ("secure-code LoRA", EM / "cpp_secure_s0")}
C_R2, C_RHO = "#1f5f8b", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})


def question_scores(folder, i=10, seeds=(0, 1, 2)):
    z = np.load(folder / "axis_avg_L25.npz", allow_pickle=True); pos = pd.Series(z["axis_proj"].astype(float), index=z["roles"])
    acc = {}
    for s in seeds:
        f = folder / "exp2" / f"metrics_avg_L25_s{s}_i{i}.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f); g = g[(g.centring == "q") & (g.role != "default")]
        tr = g[g.split == "train"].set_index("role").sort_index(); te = g[g.split == "test"].set_index("role").sort_index()
        common = tr.index.intersection(te.index); y = pos.loc[common].to_numpy()
        for c in tr.columns:
            if c not in sum(FAMILIES.values(), []) or c in SCALE or c in FAMILIES["S"] or c in FAMILIES["G"]:
                continue
            a, b = tr.loc[common, c].to_numpy(float), te.loc[common, c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 50 or a[ok].std() == 0:
                continue
            f_ = np.polyfit(a[ok], y[ok], 1); ref = y[ok].mean()
            r2 = 1 - ((y[ok] - np.polyval(f_, b[ok])) ** 2).sum() / ((y[ok] - ref) ** 2).sum()
            acc.setdefault(c, []).append((r2, spearmanr(a[ok], b[ok])[0], spearmanr(b[ok], y[ok])[0]))
    return {c: dict(r2=float(np.median([v[0] for v in L])), rel=float(np.median([v[1] for v in L])), rho=float(np.median([v[2] for v in L])), n=len(L)) for c, L in acc.items()}


def panel(ax, rows, title, lab_r2, lab_rho):
    y = np.arange(len(rows))[::-1]; h = 0.38
    ax.barh(y + h / 2, [r["r2"] for r in rows], h, color=C_R2, label=lab_r2)
    ax.barh(y - h / 2, [abs(r["rho"]) for r in rows], h, color=C_RHO, label=lab_rho)
    for yi, r in zip(y, rows):
        ax.text(r["r2"] + 0.01, yi + h / 2, f"{r['r2']:.2f}", va="center", fontsize=7.5, color=INK)
        ax.text(abs(r["rho"]) + 0.01, yi - h / 2, f"{r['rho']:+.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['metric']}\n{FAMNAME[famof(r['metric'])]}" for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1.12); ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel("held-out R² (blue) and |held-out Spearman| (orange; sign printed)")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="output/29-oct/partExtra")
    a = ap.parse_args(argv)
    D = json.loads(Path(a.data).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    table = []
    # persona-wise
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (key, (lab, folder)) in zip(axes, MODELS.items()):
        rec = D["models"][key]["25"]
        rows = [dict(metric=m, r2=e["r2"], rho=heldout_rho(rec, m)) for m, e in rec["exp1"].items() if m not in SCALE and m not in FAMILIES["S"] and m not in FAMILIES["G"] and e["r2"] is not None]
        rows = sorted(rows, key=lambda r: -r["r2"])[:6]
        panel(ax, rows, f"{lab}, layer 25, 20 questions\npersona-wise fit: 200 fitted, 75 held out", "held-out R² of position from the metric (median, 100 orderings)", "|held-out Spearman| metric vs position (median, 20 splits)")
        table += [dict(fit="persona", model=key, rank=i + 1, **r) for i, r in enumerate(rows)]
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "top6_persona_fit.png"); plt.close(fig)
    # question-wise
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (key, (lab, folder)) in zip(axes, MODELS.items()):
        q = question_scores(folder)
        rows = sorted([dict(metric=m, r2=v["r2"], rho=v["rho"], rel=v["rel"], n=v["n"]) for m, v in q.items()], key=lambda r: -r["r2"])[:6]
        panel(ax, rows, f"{lab}, layer 25\nquestion-wise fit: first 10 questions fitted, 10 held out", "R² of position from the metric computed on the held-out questions", "|Spearman| of the held-out-question metric with position")
        table += [dict(fit="question", model=key, rank=i + 1, **r) for i, r in enumerate(rows)]
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "top6_question_fit.png"); plt.close(fig)
    pd.DataFrame(table).to_csv(out / "top6_L25.csv", index=False)
    t = pd.DataFrame(table)
    for fit in ("persona", "question"):
        print("==", fit); print(t[t.fit == fit].pivot(index="rank", columns="model", values="metric").to_string())
    grouped(out, D, 6)
    grouped(out, D, 10)
    grouped(out, D, 10, select=['base20'], tag='base')
    grouped(out, D, 10, select=['insecure', 'secure'], tag='insecure_secure')
    two_figures(out, D, 10)
    print("wrote", out)


def grouped(out, D, k=6, select=None, tag=None):
    """select: the models whose top-k metrics define the x-axis (default: all three); tag: file-name prefix."""
    select = select or list(MODELS); tag = tag or 'union'
    """Four grouped bar charts: x = union of the top-6 metrics of base / insecure / secure; bars = the three models;
    one figure per (fit: persona, question) x (score: held-out R², held-out Spearman)."""
    C = {"base20": "#2a78d6", "insecure": "#d6453f", "secure": "#169a6b"}
    scores = {}
    for key, (lab, folder) in MODELS.items():
        rec = D["models"][key]["25"]
        per = {m: dict(r2=e["r2"], rho=heldout_rho(rec, m)) for m, e in rec["exp1"].items() if m not in SCALE and m not in FAMILIES["S"] and m not in FAMILIES["G"] and e["r2"] is not None}
        que = {m: dict(r2=v["r2"], rho=v["rho"]) for m, v in question_scores(folder).items()}
        scores[key] = dict(persona=per, question=que)
    for fit, desc in (("persona", "persona-wise fit: 200 personas fitted, 75 held out (median over 100 orderings; Spearman over 20 splits)"),
                      ("question", "question-wise fit: first 10 of the 20 questions fitted, the other 10 held out (median over 3 orderings)")):
        union = []
        for key in select:
            top = sorted(scores[key][fit].items(), key=lambda kv: -kv[1]["r2"])[:k]
            union += [m for m, _ in top if m not in union]
        # order the union by the mean held-out R² over the three models
        union = sorted(union, key=lambda m: -np.nanmean([scores[kk][fit].get(m, {}).get("r2", np.nan) for kk in select]))
        for score, ylab, fname in (("r2", "held-out R² of the position predicted from the metric", f"{tag}_top{k}_{fit}_fit_R2.png"),
                                   ("rho", "held-out Spearman correlation between the metric and the position (signed)", f"{tag}_top{k}_{fit}_fit_spearman.png")):
            fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(union) + 2), 5.4))
            x = np.arange(len(union)); w = 0.26
            for j, (key, (lab, _)) in enumerate(MODELS.items()):
                vals = [scores[key][fit].get(m, {}).get(score, np.nan) for m in union]
                ax.bar(x + (j - 1) * w, vals, w, color=C[key], label=lab)
                for xi, v in zip(x + (j - 1) * w, vals):
                    if np.isfinite(v):
                        ax.text(xi, v + (0.015 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({FAMNAME[famof(m)]})" for m in union], fontsize=7 if len(union) > 9 else 7.5, rotation=30, ha="right")
            ax.set_ylabel(ylab, fontsize=8.5)
            allv = [scores[k][fit].get(m, {}).get(score, np.nan) for m in union for k in MODELS]
            lo = np.nanmin(allv)
            ax.set_ylim(min(-0.05, lo - 0.1) if score == "r2" else -1.1, 1.08)
            who = {'union': 'union of the three models', 'base': 'Qwen3-8B without fine-tune', 'insecure_secure': 'union of the insecure-code and secure-code models'}[tag]
            ax.set_title(f"Layer 25, 20 questions: x = the {k} metrics with the highest held-out R² in the {who}; bars = all three models\n{desc}", fontsize=9, color=INK)
            ax.legend(frameon=False, fontsize=8, loc="lower right" if score == "rho" else "upper right")
            fig.tight_layout(); fig.savefig(out / fname); plt.close(fig)
        print(tag, fit, k, "x-axis:", union)




def two_figures(out, D, k=10):
    """Figure A: the untuned model's top-k metrics (by held-out R²), persona-wise and question-wise panels, bars = R² and |Spearman|.
    Figure B: union of the insecure and secure models' top-k metrics, same panels, bars = R² and |Spearman| for each of the two."""
    C = {"base20": "#2a78d6", "insecure": "#d6453f", "secure": "#169a6b"}
    scores = {}
    for key, (lab, folder) in MODELS.items():
        rec = D["models"][key]["25"]
        per = {m: dict(r2=e["r2"], rho=heldout_rho(rec, m)) for m, e in rec["exp1"].items() if m not in SCALE and m not in FAMILIES["S"] and m not in FAMILIES["G"] and e["r2"] is not None}
        que = {m: dict(r2=v["r2"], rho=v["rho"]) for m, v in question_scores(folder).items()}
        scores[key] = dict(persona=per, question=que)
    DESC = dict(persona="persona-wise fit: 200 personas fitted, 75 held out", question="question-wise fit: first 10 of the 20 questions fitted, the other 10 held out")

    def draw(ax, union, keys, fit):
        x = np.arange(len(union)); nb = 2 * len(keys); w = 0.8 / nb
        for j, key in enumerate(keys):
            for q, (score, hatch, lab) in enumerate((("r2", None, "held-out R²"), ("rho", "//", "|held-out Spearman|"))):
                vals = [scores[key][fit].get(m, {}).get(score, np.nan) for m in union]
                shown = [abs(v) if score == "rho" else v for v in vals]
                off = (2 * j + q - (nb - 1) / 2) * w
                ax.bar(x + off, shown, w, color=C[key], alpha=1 if q == 0 else 0.45, hatch=hatch, edgecolor="white", linewidth=0.5,
                       label=f"{MODELS[key][0]}: {lab}")
                for xi, v, sv in zip(x + off, vals, shown):
                    if np.isfinite(v):
                        ax.text(xi, sv + 0.012, f"{v:+.2f}" if score == "rho" else f"{v:.2f}", ha="center", fontsize=5.5 if nb > 2 else 6.5, color=INK2, rotation=90 if nb > 2 else 0)
        ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({FAMNAME[famof(m)]})" for m in union], fontsize=7, rotation=30, ha="right")
        ax.axhline(0, color=INK2, linewidth=0.8); ax.set_ylim(min(-0.05, np.nanmin([scores[k][fit].get(m, {}).get("r2", 0) for m in union for k in keys]) - 0.1), 1.25)
        ax.set_ylabel("held-out R² (solid) · |held-out Spearman| (hatched, sign printed)", fontsize=8)
        ax.set_title(DESC[fit], fontsize=9, color=INK)

    for name, keys in (("base", ["base20"]), ("insecure_secure", ["insecure", "secure"])):
        fig, axes = plt.subplots(2, 1, figsize=(15, 11))
        for ax, fit in zip(axes, ("persona", "question")):
            union = []
            for key in keys:
                union += [m for m, _ in sorted(scores[key][fit].items(), key=lambda kv: -kv[1]["r2"])[:k] if m not in union]
            union = sorted(union, key=lambda m: -np.nanmean([scores[kk][fit].get(m, {}).get("r2", np.nan) for kk in keys]))
            draw(ax, union, keys, fit)
        axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
        who = "Qwen3-8B without fine-tune" if name == "base" else "insecure-code and secure-code LoRA (union of their top metrics)"
        fig.suptitle(f"Layer 25, 20 questions: the {k} metrics with the highest held-out R² — {who}", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / f"top{k}_{name}.png"); plt.close(fig)
    print("wrote", out / f"top{k}_base.png", out / f"top{k}_insecure_secure.png")


if __name__ == "__main__":
    main()
