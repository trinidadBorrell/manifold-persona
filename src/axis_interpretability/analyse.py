"""Figures and tables for Exp 1-3 from $PM/AXI_L28 into output/<date>/ (default output/20-oct).

    python -m axis_interpretability.analyse --dir $PM/AXI_L28 --out output/20-oct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

K = 3
AXES = ["PCA%d" % k for k in range(1, K + 1)] + ["PGA%d" % k for k in range(1, K + 1)]
COL = {"PCA": "#2a78d6", "PGA": "#eb6834"}
MARK = {"PCA": "o", "PGA": "s"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Assistant Axis (arXiv 2601.10387) Table 1, Qwen 3 32B column: top 5 roles per end of role PC1-3
AA_QWEN = {1: dict(label="ROLE-PLAYING ↔ ASSISTANT-LIKE", minus=["bohemian", "wanderer", "trickster", "exile", "actor"],
                   plus=["evaluator", "examiner", "reviewer", "analyst", "consultant"]),
           2: dict(label="COLLECTIVE ↔ INDIVIDUAL", minus=["swarm", "hive", "egregore", "crystalline", "purist"],
                   plus=["procrastinator", "amateur", "teenager", "improviser", "influencer"]),
           3: dict(label="EMPATHETIC ↔ BLUNT ?", minus=["caregiver", "counselor", "empath", "idealist", "healer"],
                   plus=["hacker", "spy", "saboteur", "smuggler", "robot"])}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ci(k, n):
    """Clopper-Pearson 95% interval for k successes in n"""
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def load(D, sub):
    return {f.stem: json.loads(f.read_text()) for f in sorted((D / sub).glob("*.json"))}


def exp1(D, items):
    rep = load(D, "exp1")
    out = {}
    for it in items:
        v = rep[it["id"]]["reply"]
        o = out.setdefault(it["axis"], dict(theme=[], contrast=[], themes=[]))
        o["theme"] += [v["score_A"], v["score_B"]]
        o["contrast"].append(v["score_contrast"])
        th = {it["A_is"]: v["theme_A"], ("minus" if it["A_is"] == "plus" else "plus"): v["theme_B"]}
        o["themes"].append(th)
    return out


def exp2(D, items):
    labs, sorts = load(D, "exp2_label"), load(D, "exp2_sort")
    out = {}
    for it in items:
        lab = labs[it["id"]]["reply"]
        other = "minus" if it["A_is"] == "plus" else "plus"
        to_end = {"A": it["A_is"], "B": other}
        o = out.setdefault(it["axis"], dict(held=[], seen=[], labels=[]))
        o["labels"].append({it["A_is"]: lab["pole_A"], other: lab["pole_B"], "axis": lab["axis"], "A_is": it["A_is"]})
        for half in ("held", "seen"):
            v = sorts["%s_%s" % (it["id"], half)]["reply"]
            o[half].append(sum(to_end[v[r]["pole"]] == it["truth"][r] for r in it[half]) / len(it[half]))
    return out


def fig_spearman(S, out):
    rho = np.array([[np.corrcoef(np.argsort(np.argsort(S["pca_score"][:, i])), np.argsort(np.argsort(S["pga_score"][:, j])))[0, 1]
                     for j in range(K)] for i in range(K)])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, "%.2f" % rho[i, j], ha="center", va="center", fontsize=10,
                    color="white" if abs(rho[i, j]) > 0.6 else INK)
    ax.set_xticks(range(K), ["PGA %d" % (j + 1) for j in range(K)])
    ax.set_yticks(range(K), ["PCA %d" % (i + 1) for i in range(K)])
    ax.set_title("Spearman ρ between role scores,\nPCA direction i vs PGA direction j (276 roles)", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "fig0_pca_vs_pga_spearman.png", dpi=200)
    plt.close(fig)
    return rho


def fig_exp1(r1, out):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, key, title in zip(axs, ("theme", "contrast"),
                              ("Theme score of each end (1–5)\nmean ± SD over 2 ends × 5 repeats",
                               "Contrast score between ends (1–5)\nmean ± SD over 5 repeats")):
        for d, m in ((-0.12, "PCA"), (0.12, "PGA")):
            y = [np.mean(r1["%s%d" % (m, k)][key]) for k in range(1, K + 1)]
            e = [np.std(r1["%s%d" % (m, k)][key]) for k in range(1, K + 1)]
            ax.errorbar(np.arange(1, K + 1) + d, y, yerr=e, fmt=MARK[m], color=COL[m], ms=8, capsize=3, lw=1.5, label=m)
        ax.set_xticks(range(1, K + 1), ["direction %d" % k for k in range(1, K + 1)])
        ax.set_ylim(0.8, 5.2)
        ax.set_title(title, fontsize=10, color=INK)
        style(ax)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Exp 1: LLM rating of the 10 roles at each end (Claude Sonnet 5.5, blind to method)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_exp1_ratings.png", dpi=200)
    plt.close(fig)


def fig_exp2(r2, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for d, m in ((-0.12, "PCA"), (0.12, "PGA")):
        for k in range(1, K + 1):
            o = r2["%s%d" % (m, k)]
            n = 20 * len(o["held"])
            c = round(np.mean(o["held"]) * n)
            lo, hi = ci(c, n)
            x = k + d
            ax.errorbar(x, c / n, yerr=[[c / n - lo], [hi - c / n]], fmt=MARK[m], color=COL[m], ms=8, capsize=3, lw=1.5,
                        label="%s, held-out roles" % m if k == 1 else None)
            ax.plot(x + 0.06, np.mean(o["seen"]), MARK[m], mfc="none", mec=COL[m], ms=7, mew=1.5,
                    label="%s, roles the labeller saw" % m if k == 1 else None)
            ax.scatter(np.full(len(o["held"]), x - 0.06), o["held"], s=10, color=COL[m], alpha=0.35, lw=0)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(3.45, 0.505, "chance", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(1, K + 1), ["direction %d" % k for k in range(1, K + 1)])
    ax.set_ylim(0.3, 1.03)
    ax.set_ylabel("share of roles put at their true end", color=INK2, fontsize=9)
    ax.set_title("Exp 2: a label written from 10+10 roles, used to sort 20 other roles\n"
                 "filled: held-out roles, 95% CI over 5 splits × 20 roles\nsmall dots: each split; hollow: the 20 roles the labeller saw (upper bound)",
                 fontsize=9, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig2_exp2_heldout_accuracy.png", dpi=200)
    plt.close(fig)


def positions(S, roles):
    res = {}
    for m in ("pca", "pga"):
        for k in range(K):
            x = S["%s_score" % m][:, k]
            lo, hi = x.min(), x.max()
            row = {r: float((x[roles.index(r)] - lo) / (hi - lo)) for r in ("assistant", "default")}
            if m == "pca" and "pca_noprompt" in S:
                row["no system prompt"] = float((S["pca_noprompt"][k] - lo) / (hi - lo))
            res["%s%d" % (m.upper(), k + 1)] = row
    return res


def fig_exp3(pos, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    mk = {"assistant": ("D", INK), "default": ("^", INK2), "no system prompt": ("*", "#1baf7a")}
    for i, a in enumerate(AXES):
        y = len(AXES) - 1 - i
        ax.plot([0, 1], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=0)
        for r, v in pos[a].items():
            ax.plot(v, y, mk[r][0], color=mk[r][1], ms=9 if r != "no system prompt" else 13, ls="none",
                    label=r if i == 0 else None, zorder=3)
    ax.set_yticks(range(len(AXES)), [a[:3] + " " + a[3:] for a in AXES][::-1])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("relative position on the direction: 0 = lowest role, 1 = highest role", color=INK2, fontsize=9)
    ax.set_title("Exp 3: where the Assistant sits on each direction (Assistant Axis §2.3.1)\n"
                 "paper, Qwen 3 32B, no system prompt: 0.03 from the edge on PC1; 0.27–0.50 on other PCs",
                 fontsize=9, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9, left=False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig3_exp3_assistant_position.png", dpi=200)
    plt.close(fig)


def ends_table(S, roles, out):
    """top 5 roles per end by cosine similarity, as in Assistant Axis Table 1; ** = also in that PC's Qwen list"""
    rows, data = [], {}
    for m in ("pca", "pga"):
        for k in range(K):
            c = S["%s_cos" % m][:, k]
            minus = [roles[i] for i in np.argsort(c)[:5]]
            plus = [roles[i] for i in np.argsort(c)[::-1][:5]]
            data["%s%d" % (m.upper(), k + 1)] = (minus, plus)
    aa_all = {r for v in AA_QWEN.values() for r in v["minus"] + v["plus"]}
    lines = ["| direction | − end (5 lowest cosine) | + end (5 highest cosine) |", "|---|---|---|"]
    for a, (mi, pl) in data.items():
        f = lambda rs: ", ".join("**%s**" % r if r in aa_all else r for r in rs)
        lines.append("| %s | %s | %s |" % (a[:3] + " " + a[3:], f(mi), f(pl)))
    lines += ["", "Assistant Axis Table 1, Qwen 3 32B (their hand labels):", "",
              "| PC | label | − end | + end |", "|---|---|---|---|"]
    for k, v in AA_QWEN.items():
        lines.append("| PC%d | %s | %s | %s |" % (k, v["label"], ", ".join(v["minus"]), ", ".join(v["plus"])))
    (out / "table_exp3_direction_ends.md").write_text("\n".join(lines) + "\n")
    # png version
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.axis("off")
    cell = [[a[:3] + " " + a[3:], ", ".join(mi), ", ".join(pl)] for a, (mi, pl) in data.items()]
    t = ax.table(cellText=cell, colLabels=["direction", "− end (5 lowest cosine)", "+ end (5 highest cosine)"],
                 loc="center", cellLoc="left", colWidths=[0.1, 0.45, 0.45])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(color=INK, weight="bold")
    ax.set_title("Exp 3: top 5 roles per end by cosine similarity (Qwen3-8B, layer 28, last token)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "table_exp3_direction_ends.png", dpi=200)
    plt.close(fig)
    return data, aa_all


def labels_table(r1, r2, out):
    lines = ["# Labels per direction", "",
             "Exp 1 themes (5 repeats, each on the same 10 + 10 roles) and Exp 2 labels (5 splits, each on a different 10 + 10). "
             "\"−\" and \"+\" are the ends by score sign.", ""]
    for a in AXES:
        lines += ["## %s" % (a[:3] + " " + a[3:]), "", "| | − end | + end |", "|---|---|---|"]
        for i, t in enumerate(r1[a]["themes"]):
            lines.append("| Exp 1 repeat %d | %s | %s |" % (i, t["minus"], t["plus"]))
        for i, t in enumerate(r2[a]["labels"]):
            lines.append("| Exp 2 split %d | %s | %s |" % (i, t["minus"], t["plus"]))
        lines += ["", "Exp 2 axis sentences:", ""] + ["- split %d: %s" % (i, t["axis"]) for i, t in enumerate(r2[a]["labels"])] + [""]
    (out / "table_labels.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="output/20-oct")
    a = ap.parse_args(argv)
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = dict(np.load(D / "scores.npz"))
    roles = [str(r) for r in S["roles"]]
    items1 = json.loads((D / "items" / "exp1.json").read_text())
    items2 = json.loads((D / "items" / "exp2.json").read_text())
    rho = fig_spearman(S, out)
    r1, r2 = exp1(D, items1), exp2(D, items2)
    fig_exp1(r1, out)
    fig_exp2(r2, out)
    pos = positions(S, roles)
    fig_exp3(pos, out)
    ends_table(S, roles, out)
    labels_table(r1, r2, out)
    summ = dict(spearman_pca_i_pga_j=rho.round(3).tolist(),
                var_share=dict(pca=S["pca_var_share"].round(4).tolist(), pga=S["pga_var_share"].round(4).tolist()),
                exp1={k: dict(theme_mean=float(np.mean(v["theme"])), theme_sd=float(np.std(v["theme"])),
                              contrast_mean=float(np.mean(v["contrast"])), contrast_sd=float(np.std(v["contrast"])))
                      for k, v in r1.items()},
                exp2={k: dict(held_acc=float(np.mean(v["held"])), held_ci=ci(round(np.mean(v["held"]) * 20 * len(v["held"])), 20 * len(v["held"])),
                              held_per_split=v["held"], seen_acc=float(np.mean(v["seen"])))
                      for k, v in r2.items()},
                exp3_position=pos)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
