"""partExtra: one PNG per metric with |held-out Spearman| >= 0.75 at layer 25, persona-wise fit (left) and
question-wise fit (right), for the base model (240 questions) and for the insecure-code model (20 questions);
plus a forest of the qualifying metrics per model and a CSV of the held-out Spearman values.
Held-out Spearman = signed rank correlation between the metric and the position over the 75 held-out personas,
median over 20 random 200/75 splits (the same definition as the results page).

    python -m misalignment_lora.extra_figs --data <page_data.json> --out output/29-oct/partExtra
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

from cloud_predicts_position.panel import FAMILIES

INK, INK2, HELD = "#15232e", "#4e5c68", "#6250d6"
COL = dict(base240="#2a78d6", insecure="#d6453f", secure="#169a6b", base20="#2a78d6")
LAB = dict(base240="base Qwen3-8B, 240 questions", insecure="insecure-code LoRA, 20 questions", secure="secure-code LoRA, 20 questions", base20="base Qwen3-8B, 20 questions")
FAMNAME = dict(S="scale (controls)", D="intrinsic dimension", P="spectral shape", N="density", T="topology", C="curvature", A="anisotropy to the assistant axis", G="design decomposition", R="alignment with the default cloud", I="consistency")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})
TOP = 0.75


def famof(m):
    return next((k for k, v in FAMILIES.items() if m in v), "S")


def arr(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def heldout_rho(rec, m, n_splits=20, n_train=200):
    y, v = arr(rec["values"]["axis_proj"]), arr(rec["values"][m])
    ok = np.flatnonzero(np.isfinite(y) & np.isfinite(v))
    vals = []
    for s in range(n_splits):
        idx = np.random.default_rng(100 + s).permutation(ok)
        te = idx[n_train:]
        vals.append(spearmanr(v[te], y[te])[0])
    return float(np.median(vals))


def persona_panel(ax, rec, m, color, n_train=200, seed=7):
    y, v = arr(rec["values"]["axis_proj"]), arr(rec["values"][m])
    ok = np.flatnonzero(np.isfinite(y) & np.isfinite(v))
    idx = np.random.default_rng(seed).permutation(ok); tr, te = idx[:n_train], idx[n_train:]
    b, a = np.polyfit(v[tr], y[tr], 1)                       # position from metric (what the experiment fits)
    r2 = 1 - ((y[te] - (a + b * v[te])) ** 2).sum() / ((y[te] - y[tr].mean()) ** 2).sum()
    g = np.polyfit(y[tr], v[tr], 1)                            # drawn as metric against position
    xs = np.array([np.nanmin(y), np.nanmax(y)])
    ax.plot(xs, np.polyval(g, xs), color=INK2, linewidth=1)
    ax.scatter(y[tr], v[tr], s=12, color=color, linewidths=0, label=f"fitted on {len(tr)} personas")
    ax.scatter(y[te], v[te], s=18, facecolors="none", edgecolors=HELD, linewidths=0.9, label=f"held out ({len(te)})")
    ax.set_title(f"persona-wise fit: held-out R² {r2:.2f}, held-out Spearman {spearmanr(v[te], y[te])[0]:+.2f}\n(this split; 200 fitted, 75 held out)", fontsize=8.5, color=INK)
    ax.set_xlabel("projection of the persona centroid on the assistant axis (+ = toward the default assistant)")
    ax.set_ylabel(m)
    ax.legend(frameon=False, fontsize=7.5)


def question_panel(ax, cell, m, i, color):
    y, a_, b_ = arr(cell["axis_proj"]), arr(cell["train"][m]), arr(cell["test"][m])
    ok = np.isfinite(y) & np.isfinite(a_) & np.isfinite(b_)
    y, a_, b_ = y[ok], a_[ok], b_[ok]
    g = np.polyfit(y, a_, 1); xs = np.array([y.min(), y.max()])
    f = np.polyfit(a_, y, 1); ref = y.mean()
    r2tr = 1 - ((y - np.polyval(f, a_)) ** 2).sum() / ((y - ref) ** 2).sum()
    r2te = 1 - ((y - np.polyval(f, b_)) ** 2).sum() / ((y - ref) ** 2).sum()
    rel = spearmanr(a_, b_)[0]
    ax.plot(xs, np.polyval(g, xs), color=INK2, linewidth=1)
    ax.scatter(y, a_, s=12, color=color, linewidths=0, label=f"metric from the first {i} questions")
    ax.scatter(y, b_, s=18, facecolors="none", edgecolors=HELD, linewidths=0.9, label=f"metric from {i} held-out questions")
    ax.set_title(f"question-wise fit, i = {i}: R² of position from the metric {r2tr:.2f} (fitted questions), {r2te:.2f} (held-out questions)\nrank correlation of the metric, fitted vs held-out questions: {rel:.2f}", fontsize=8.5, color=INK)
    ax.set_xlabel("projection of the persona centroid on the assistant axis (all questions)")
    ax.set_ylabel(m)
    ax.legend(frameon=False, fontsize=7.5)


def forest(out, rec, rows, key):
    rows = sorted(rows, key=lambda r: (list(FAMILIES).index(famof(r["metric"])), r["heldout_spearman"]))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.scatter([r["heldout_spearman"] for r in rows], y, color=COL[key], s=24, zorder=3, label="held-out Spearman, metric vs position (median of 20 splits)")
    ax.scatter([rec["forest"][r["metric"]]["rho"] for r in rows], y, facecolors="none", edgecolors=INK2, s=28, zorder=3, label="Spearman over all personas")
    ax.axvline(0, color=INK2, linewidth=0.8); ax.axvline(TOP, color=INK2, linewidth=0.6, linestyle=":"); ax.axvline(-TOP, color=INK2, linewidth=0.6, linestyle=":")
    ax.set_yticks(y); ax.set_yticklabels([r["metric"] for r in rows], fontsize=7.5)
    prev = None
    for i, r in enumerate(rows):
        f = famof(r["metric"])
        if f != prev:
            ax.text(1.02, i, FAMNAME[f], fontsize=7, color=INK2, va="center", transform=ax.get_yaxis_transform()); prev = f
    ax.set_xlim(-1, 1); ax.set_xlabel("Spearman correlation with the projection of the persona centroid on the assistant axis")
    ax.set_title(f"{LAB[key]}, layer 25: metrics with |held-out Spearman| ≥ {TOP}", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / f"forest_{key}.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output/29-oct/partExtra")
    ap.add_argument("--layer", default="25")
    a = ap.parse_args(argv)
    D = json.loads(Path(a.data).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    table = []
    for key, cells, i_show in (("base240", D["exp2"]["cells"], 40), ("insecure", D["exp2_ft"]["insecure"], 10)):
        rec = D["models"][key][a.layer]
        (out / key).mkdir(exist_ok=True)
        rows = []
        for m in rec["forest"]:
            rho = heldout_rho(rec, m)
            e = rec["exp1"].get(m, {})
            rows.append(dict(model=key, layer=a.layer, metric=m, family=FAMNAME[famof(m)], heldout_spearman=round(rho, 3),
                             spearman_all=rec["forest"][m]["rho"], heldout_r2=e.get("r2"), qualifies=abs(rho) >= TOP,
                             has_question_fit=m in cells[str(i_show)]["train"]))
        table += rows
        q = [r for r in rows if r["qualifies"]]
        forest(out, rec, q, key)
        for r in q:
            m = r["metric"]
            has_q = r["has_question_fit"]
            fig, axes = plt.subplots(1, 2 if has_q else 1, figsize=(13 if has_q else 6.5, 4.6))
            axes = np.atleast_1d(axes)
            persona_panel(axes[0], rec, m, COL[key])
            if has_q:
                question_panel(axes[1], cells[str(i_show)], m, i_show, COL[key])
            fig.suptitle(f"{LAB[key]}, layer {a.layer}: {m}  [{FAMNAME[famof(m)]}], held-out Spearman {r['heldout_spearman']:+.2f}", fontsize=10, color=INK)
            fig.tight_layout(); fig.savefig(out / key / f"{m}.png"); plt.close(fig)
    t = pd.DataFrame(table).sort_values(["model", "qualifies", "heldout_spearman"], ascending=[True, False, True])
    t.to_csv(out / "heldout_spearman_L25.csv", index=False)
    print(t[t.qualifies].groupby("model").metric.count().to_dict(), "wrote", out)


if __name__ == "__main__":
    main()
