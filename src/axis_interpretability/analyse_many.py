"""Exp 1, 2 and 4 over many directions (PCA 1-K and PGA 1-K, K = 50), summarised over 1-20 and 1-50.

Exp 4 score: Spearman rho between the judge's order (pole A -> pole B) and the true score of the 20 roles,
signed so that +1 means the judge's order is the true order. Pairs PCA k / PGA k are compared with a
Wilcoxon signed-rank test over k.

    python -m axis_interpretability.analyse_many --dir $PM/AXI_L28_K50 --out output/20-oct/k50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from axis_interpretability.analyse import COL, GRID, INK, INK2, MARK, MUTED, exp1, exp2, load, style

RANGES = (20, 50)


def exp4(D, items, sub="exp4_rank"):
    """Exp 4 (sub exp4_rank) and Exp 5 (sub exp5_anchor): Spearman rho of the judge's order per split"""
    rep = load(D, sub)
    out = {}
    for it in items:
        if it["id"] not in rep:
            continue
        o = rep[it["id"]]["reply"]["order"]
        sgn = 1 if it["A_is"] == "minus" else -1          # position 0 = closest to A
        rho = spearmanr([o.index(r) for r in it["spread"]], [sgn * it["spread_score"][r] for r in it["spread"]])[0]
        out.setdefault(it["axis"], []).append(float(rho))
    return out


PARTS = (("groups", "3 groups placed right: ρ of judge position\nvs group (5 near A, 10 middle, 5 near B)"),
         ("within_ends", "order within each end: ρ inside\nthe 5 roles nearest each pole (mean of 2)"),
         ("within_middle", "order within the middle: ρ inside\nthe 10 middle roles"))


def parts(D, items, sub):
    """split each ordering's 20 roles by true score into 5 near A / 10 middle / 5 near B and score three things:
    whether the groups are placed in the right order, and the order inside the ends and inside the middle"""
    rep = load(D, sub)
    out = {}
    for it in items:
        if it["id"] not in rep:
            continue
        o = rep[it["id"]]["reply"]["order"]
        sgn = 1 if it["A_is"] == "minus" else -1
        true = sorted(it["spread"], key=lambda r: sgn * it["spread_score"][r])       # pole A -> pole B
        pos = {r: o.index(r) for r in true}
        rho = lambda rs: spearmanr([pos[r] for r in rs], range(len(rs)))[0]
        grp = [0] * 5 + [1] * 10 + [2] * 5
        d = out.setdefault(it["axis"], dict(groups=[], within_ends=[], within_middle=[]))
        d["groups"].append(float(spearmanr([pos[r] for r in true], grp)[0]))
        d["within_ends"].append(float((rho(true[:5]) + rho(true[-5:])) / 2))
        d["within_middle"].append(float(rho(true[5:15])))
    return out


def fig_parts_one(P, K, title, name, out):
    """one row of figK3 as its own figure, no 'Exp n' anywhere"""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    x = np.arange(1, K + 1)
    for ax, (key, sub) in zip(axs, PARTS):
        for m in ("PCA", "PGA"):
            y = np.array([np.mean(P["%s%d" % (m, k)][key]) for k in x])
            e = np.array([np.std(P["%s%d" % (m, k)][key]) for k in x])
            ax.plot(x, y, "-", color=COL[m], lw=1.3)
            ax.plot(x, y, MARK[m], color=COL[m], ms=3.5, ls="none", label=m)
            ax.fill_between(x, y - e, y + e, color=COL[m], alpha=0.12, lw=0)
        ax.axhline(0, color=MUTED, ls="--", lw=1)
        ax.axvline(20.5, color=MUTED, ls=":", lw=1)
        ax.set_ylim(-0.8, 1.05)
        ax.set_title(sub, fontsize=9, color=INK)
        ax.set_xlabel("direction (PCA / PGA component number)", color=INK2, fontsize=9)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        style(ax)
    axs[0].set_ylabel("Spearman ρ\nmean ± SD over 5 splits", color=INK2, fontsize=9)
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def fig_parts(P4, P5, K, out):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey=True)
    x = np.arange(1, K + 1)
    for i, (P, nm) in enumerate(((P4, "Exp 4 (label)"), (P5, "Exp 5 (example roles)"))):
        for j, (key, title) in enumerate(PARTS):
            ax = axs[i, j]
            for m in ("PCA", "PGA"):
                y = np.array([np.mean(P["%s%d" % (m, k)][key]) for k in x])
                e = np.array([np.std(P["%s%d" % (m, k)][key]) for k in x])
                ax.plot(x, y, "-", color=COL[m], lw=1.3)
                ax.plot(x, y, MARK[m], color=COL[m], ms=3.5, ls="none", label=m)
                ax.fill_between(x, y - e, y + e, color=COL[m], alpha=0.12, lw=0)
            ax.axhline(0, color=MUTED, ls="--", lw=1)
            ax.axvline(20.5, color=MUTED, ls=":", lw=1)
            ax.set_ylim(-0.8, 1.05)
            if i == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("%s\nmean ± SD over 5 splits" % nm, color=INK2, fontsize=9)
            if i == 1:
                ax.set_xlabel("direction (PCA / PGA component number)", color=INK2, fontsize=9)
                ax.set_xticks([1, 10, 20, 30, 40, 50])
            style(ax)
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("What the judge's order of 20 roles gets right: the three groups, the order inside the ends, the order inside the middle",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figK3_exp4_exp5_order_parts.png", dpi=200)
    plt.close(fig)
    fig_parts_one(P4, K, "Judge orders 20 roles from along the axis, given the label written from the extreme roles:\n"
                  "the three groups, the order inside the ends, the order inside the middle",
                  "figK3a_order_parts_given_label.png", out)
    fig_parts_one(P5, K, "Judge orders 20 roles from along the axis, given 5 example roles at each pole (no label):\n"
                  "the three groups, the order inside the ends, the order inside the middle",
                  "figK3b_order_parts_given_example_roles.png", out)


def parts_table(P4, P5, out):
    L = ["| measure | directions | Exp 4 PCA | Exp 4 PGA | Exp 5 PCA | Exp 5 PGA |", "|---|---|---|---|---|---|"]
    res = {}
    for key, title in PARTS:
        for lo, hi in ((1, 3), (1, 10), (11, 20), (21, 50)):
            v = [np.mean([np.mean(P["%s%d" % (m, k)][key]) for k in range(lo, hi + 1)]) for P in (P4, P5) for m in ("PCA", "PGA")]
            res["%s_%d-%d" % (key, lo, hi)] = v
            L.append("| %s | %d–%d | %.2f | %.2f | %.2f | %.2f |" % (title.split(":")[0], lo, hi, *v))
    (out / "table_exp4_exp5_order_parts.md").write_text("\n".join(L) + "\n")
    return res


def table(S, r1, r2, r4, K, r5=None):
    rows = []
    for m in ("PCA", "PGA"):
        for k in range(1, K + 1):
            a = "%s%d" % (m, k)
            rows.append(dict(method=m, k=k, var_share=float(S["%s_var_share" % m.lower()][k - 1]),
                             theme=float(np.mean(r1[a]["theme"])), contrast=float(np.mean(r1[a]["contrast"])),
                             theme_sd=float(np.std(r1[a]["theme"])), contrast_sd=float(np.std(r1[a]["contrast"])),
                             held=float(np.mean(r2[a]["held"])), seen=float(np.mean(r2[a]["seen"])),
                             held_sd=float(np.std(r2[a]["held"])),
                             rho=float(np.mean(r4[a])), rho_sd=float(np.std(r4[a])), n_rank=len(r4[a]),
                             rho5=float(np.mean(r5[a])) if r5 else None, rho5_sd=float(np.std(r5[a])) if r5 else None,
                             n_anchor=len(r5[a]) if r5 else 0))
    return rows


def per(rows, m, key, n):
    return np.array([r[key] for r in rows if r["method"] == m and r["k"] <= n])


def line_panel(ax, rows, key, K, sd=None, chance=None, ylabel=""):
    x = np.arange(1, K + 1)
    for m in ("PCA", "PGA"):
        y = per(rows, m, key, K)
        ax.plot(x, y, "-", color=COL[m], lw=1.5, alpha=0.9)
        ax.plot(x, y, MARK[m], color=COL[m], ms=4.5, ls="none", label=m)
        if sd:
            e = per(rows, m, sd, K)
            ax.fill_between(x, y - e, y + e, color=COL[m], alpha=0.12, lw=0)
    if chance is not None:
        ax.axhline(chance, color=MUTED, ls="--", lw=1)
    ax.axvline(20.5, color=MUTED, ls=":", lw=1)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    style(ax)


def fig_lines(rows, K, out):
    n5 = rows[0]["rho5"] is not None
    fig, axs = plt.subplots(4 + n5, 1, figsize=(9, 10.5 + 2.5 * n5), sharex=True)
    line_panel(axs[0], rows, "theme", K, sd="theme_sd", ylabel="Exp 1 theme score (1–5)\nmean ± SD over\n2 ends × 5 repeats")
    line_panel(axs[1], rows, "contrast", K, sd="contrast_sd", ylabel="Exp 1 contrast score (1–5)\nmean ± SD over 5 repeats")
    line_panel(axs[2], rows, "held", K, sd="held_sd", chance=0.5,
               ylabel="Exp 2 share of held-out\nextreme roles at true end\n(mean ± SD over 5 splits\nof 20 roles)")
    line_panel(axs[3], rows, "rho", K, sd="rho_sd", chance=0.0,
               ylabel="Exp 4 Spearman ρ, judge order\nvs true score, 20 roles\nfrom the whole axis (mean ± SD, 5 splits)")
    if n5:
        line_panel(axs[4], rows, "rho5", K, sd="rho5_sd", chance=0.0,
                   ylabel="Exp 5 Spearman ρ, order of\n20 roles from the whole axis,\ngiven 5 + 5 example pole roles\n(mean ± SD, 5 splits)")
        axs[4].set_ylim(-0.6, 1.0)
    axs[0].set_ylim(0.8, 5.2)
    axs[1].set_ylim(0.8, 5.2)
    axs[2].set_ylim(0.3, 1.03)
    axs[3].set_ylim(-0.6, 1.0)
    axs[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    axs[-1].set_xlabel("direction (dotted line: end of directions 1–20)", color=INK2, fontsize=9)
    axs[-1].set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    fig.suptitle("PCA vs PGA directions 1–%d (Qwen3-8B, layer 28, last token; judge Claude Sonnet 5.5, blind to method)" % K,
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figK1_all_scores_by_direction.png", dpi=200)
    plt.close(fig)


def fig_separate(rows, K, out):
    """figK1 split into one figure per test (Exp 1, Exp 2, Exp 5), no 'Exp n' on the axes."""
    sub = "PCA vs PGA directions 1–%d (Qwen3-8B, layer 28, last token; judge Claude Sonnet 5.5, blind to method)" % K
    ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    xlab = "direction (PCA / PGA component number; dotted line: end of directions 1–20)"

    def save(fig, axs, title, name):
        axs[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
        axs[-1].set_xlabel(xlab, color=INK2, fontsize=9)
        axs[-1].set_xticks(ticks)
        fig.suptitle(title + "\n" + sub, fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(out / name, dpi=200)
        plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    line_panel(axs[0], rows, "theme", K, sd="theme_sd",
               ylabel="theme score (1–5):\ndo the 10 roles at one end\nshare one theme?\nmean ± SD, 2 ends × 5 repeats")
    line_panel(axs[1], rows, "contrast", K, sd="contrast_sd",
               ylabel="contrast score (1–5):\nare the two ends opposites\non one dimension?\nmean ± SD over 5 repeats")
    for ax in axs:
        ax.set_ylim(0.8, 5.2)
    save(fig, axs, "Judge rates the 10 roles at each end", "figK1a_rating_of_the_ends.png")

    if rows[0]["rho5"] is not None:
        fig, axs = plt.subplots(1, 1, figsize=(9, 3.6))
        line_panel(axs, rows, "rho5", K, sd="rho5_sd", chance=0.0,
                   ylabel="Spearman ρ, judge's order of\n20 roles from the whole axis\nvs their true score\nmean ± SD over 5 splits")
        axs.set_ylim(-0.6, 1.0)
        save(fig, [axs], "Judge orders 20 roles from along the axis, given 5 example roles at each pole (no label)",
             "figK1c_order_from_example_roles.png")

    fig, axs = plt.subplots(1, 1, figsize=(9, 3.6))
    line_panel(axs, rows, "held", K, sd="held_sd", chance=0.5,
               ylabel="share of 20 held-out extreme\nroles put at their true end\nmean ± SD over 5 splits")
    axs.set_ylim(0.3, 1.03)
    save(fig, [axs], "One judge labels the axis; a second sorts held-out roles into its two poles",
         "figK1b_label_then_sort_held_out.png")


def fig_vs_var(rows, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, yl in ((axs[0], "rho", "Exp 4 Spearman ρ (mean over 5 splits)"), (axs[1], "theme", "Exp 1 theme score (1–5)")):
        for m in ("PCA", "PGA"):
            ax.plot(per(rows, m, "var_share", 99) * 100, per(rows, m, key, 99), MARK[m], color=COL[m], ms=5, ls="none",
                    alpha=0.85, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("share of variance of the direction (%, log scale)", color=INK2, fontsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=9)
        style(ax)
    axs[0].axhline(0, color=MUTED, ls="--", lw=1)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Score against the direction's variance share (directions 1–50)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "figK2_scores_vs_variance_share.png", dpi=200)
    plt.close(fig)


def fig_match(rho, out):
    R = np.abs(np.array(rho))
    K = len(R)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = axs[0].imshow(R, cmap="Blues", vmin=0, vmax=1, origin="upper")
    axs[0].set_xlabel("PGA direction j", color=INK2, fontsize=9)
    axs[0].set_ylabel("PCA direction i", color=INK2, fontsize=9)
    axs[0].set_title("|Spearman ρ| of role scores, PCA i vs PGA j", fontsize=10, color=INK)
    t = [0, 9, 19, 29, 39, 49]
    axs[0].set_xticks(t, [i + 1 for i in t])
    axs[0].set_yticks(t, [i + 1 for i in t])
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    best = R.max(0)
    axs[1].plot(np.arange(1, K + 1), best, MARK["PGA"], color=COL["PGA"], ms=5, ls="none")
    axs[1].axvline(20.5, color=MUTED, ls=":", lw=1)
    axs[1].set_ylim(0, 1.02)
    axs[1].set_xlabel("PGA direction j", color=INK2, fontsize=9)
    axs[1].set_ylabel("max over PCA i of |ρ|", color=INK2, fontsize=9)
    axs[1].set_title("How closely each PGA direction repeats one PCA direction", fontsize=10, color=INK)
    style(axs[1])
    fig.tight_layout()
    fig.savefig(out / "figK0_pca_pga_match.png", dpi=200)
    plt.close(fig)
    return best


def summary(rows, best):
    lines = ["| measure | directions | PCA mean | PGA mean | PCA − PGA, median over k | Wilcoxon p (paired by k) |",
             "|---|---|---|---|---|---|"]
    names = dict(theme="Exp 1 theme score (1–5)", contrast="Exp 1 contrast score (1–5)",
                 held="Exp 2 held-out accuracy", seen="Exp 2 accuracy on roles the labeller saw", rho="Exp 4 Spearman ρ")
    if rows[0]["rho5"] is not None:
        names["rho5"] = "Exp 5 Spearman ρ (example roles, no label)"
    res = {}
    for key, nm in names.items():
        for n in RANGES:
            a, b = per(rows, "PCA", key, n), per(rows, "PGA", key, n)
            d = a - b
            p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
            res["%s_1-%d" % (key, n)] = dict(pca=float(a.mean()), pga=float(b.mean()), median_diff=float(np.median(d)), p=p)
            lines.append("| %s | 1–%d | %.2f | %.2f | %+.2f | %.2g |" % (nm, n, a.mean(), b.mean(), np.median(d), p))
    for n in RANGES:
        lines.append("| PGA directions repeating a PCA one (max \\|ρ\\| ≥ 0.8) | 1–%d | | %d of %d | | |" % (n, int((best[:n] >= 0.8).sum()), n))
    return lines, res


def themes_md(r1, rows, K, out):
    """Exp 1 themes only, one line per direction: the judge's theme for each end (repeats 0 and 1) and the scores"""
    sc = {(r["method"], r["k"]): r for r in rows}
    L = ["# Exp 1 themes per direction", "",
         "The judge rates the 10 most extreme roles at each end and names each end's theme in 2–5 words. "
         "Shown: repeats 0 and 1 of 5 (same roles, new shuffle and A/B flip); all 5 are in `labels_all_directions.md`. "
         "Theme score: mean over 2 ends × 5 repeats; contrast: mean over 5 repeats; both 1–5. "
         "The sign of a direction is arbitrary, so PCA k and PGA k ends need not correspond.", ""]
    for m in ("PCA", "PGA"):
        L += ["## %s" % m, "", "| dir. | var % | theme | contrast | − end | + end |", "|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            r, th = sc[(m, k)], r1["%s%d" % (m, k)]["themes"]
            L.append("| %d | %.1f | %.1f | %.1f | %s | %s |" % (k, 100 * r["var_share"], r["theme"], r["contrast"],
                     " / ".join(t["minus"] for t in th[:2]), " / ".join(t["plus"] for t in th[:2])))
        L.append("")
    (out / "themes_exp1.md").write_text("\n".join(L))


def labels_md(r1, r2, rows, K, out):
    """every label per direction: Exp 1 themes (5 repeats), Exp 2 labeller (5 splits; Exp 4 reuses these),
    and for directions 1-3 the Assistant Axis hand labels (Exp 3)"""
    from axis_interpretability.analyse import AA_QWEN
    sc = {(r["method"], r["k"]): r for r in rows}
    L = ["# Labels per direction (PCA and PGA 1–%d)" % K, "",
         "- **Exp 1:** the judge's theme for each end, 5 repeats on the same 10 + 10 most extreme roles.",
         "- **Exp 2 / Exp 4:** the labeller's poles and axis sentence, 5 splits on different 10 + 10 roles. "
         "Exp 2 sorts held-out extreme roles with this label; Exp 4 orders roles from the whole axis with it.",
         "- **Exp 3:** the Assistant Axis paper's hand labels (Qwen 3 32B, Table 1), directions 1–3 only.",
         "- \"−\" / \"+\" are the ends by score sign. The sign of a direction is arbitrary, so PCA k and PGA k "
         "ends need not correspond.",
         "- Each axis sentence runs from the labeller's pole A to pole B; the arrow says which end that is.",
         "- Each heading gives the direction's variance share, Exp 2 held-out accuracy and Exp 4 Spearman ρ.", ""]
    for k in range(1, K + 1):
        L.append("## Direction %d" % k)
        L.append("")
        if k in AA_QWEN:
            L += ["Exp 3, paper hand label for PC%d: **%s**" % (k, AA_QWEN[k]["label"]), ""]
        for m in ("PCA", "PGA"):
            a, r = "%s%d" % (m, k), sc[(m, k)]
            L += ["### %s %d — variance %.1f%%, Exp 2 %.0f%%, Exp 4 ρ = %.2f" % (m, k, 100 * r["var_share"], 100 * r["held"], r["rho"]),
                  "", "| source | − end | + end |", "|---|---|---|"]
            for i, t in enumerate(r1[a]["themes"]):
                L.append("| Exp 1 repeat %d | %s | %s |" % (i, t["minus"], t["plus"]))
            for i, t in enumerate(r2[a]["labels"]):
                L.append("| Exp 2/4 split %d | %s | %s |" % (i, t["minus"], t["plus"]))
            L += ["", "Axis sentences (Exp 2/4):", ""]
            L += ["%d. (%s → %s) %s" % (i, "−" if t["A_is"] == "minus" else "+", "+" if t["A_is"] == "minus" else "−", t["axis"])
                  for i, t in enumerate(r2[a]["labels"])] + [""]
    (out / "labels_all_directions.md").write_text("\n".join(L))


def ends_md(S, r2, rows, K, out):
    """the 20 most extreme roles at each end of every direction, in rank order, with their score"""
    from axis_interpretability.prompts import describe
    roles = [str(r) for r in S["roles"]]
    sc = {(r["method"], r["k"]): r for r in rows}
    L = ["# The roles at each end of every direction (PCA and PGA 1–%d)" % K, "",
         "- **Data:** Qwen3-8B, layer 28, last token, 276 roles. The score is the role's coordinate on the direction "
         "(PCA: projection; PGA: tangent coordinate at the Fréchet mean).",
         "- **Order:** rank 1 is the most extreme role at that end. The sign of a direction is arbitrary.",
         "- **Exp 1** used ranks 1–10 at each end (marked •). **Exp 2** used ranks 1–20: in each split the labeller saw "
         "a random 10 and the sorter got the other 10.",
         "- **Labels:** the line under each heading is the split-0 label from Exp 2. Every label is in "
         "`labels_all_directions.md`.", ""]
    for k in range(1, K + 1):
        L += ["## Direction %d" % k, ""]
        for m in ("PCA", "PGA"):
            a, r = "%s%d" % (m, k), sc[(m, k)]
            x = S["%s_score" % m.lower()][:, k - 1]
            o = np.argsort(x)
            lo, hi = o[:20], o[::-1][:20]
            lab = r2[a]["labels"][0]
            L += ["### %s %d — variance %.1f%%, Exp 2 %.0f%%, Exp 4 ρ = %.2f" % (m, k, 100 * r["var_share"], 100 * r["held"], r["rho"]),
                  "", "Label (Exp 2, split 0): − end *%s* ↔ + end *%s*" % (lab["minus"], lab["plus"]), "",
                  "| rank | − end | score | + end | score |", "|---|---|---|---|---|"]
            for i in range(20):
                b = "•" if i < 10 else ""
                L.append("| %d%s | %s | %.1f | %s | %.1f |" % (i + 1, b, roles[lo[i]], x[lo[i]], roles[hi[i]], x[hi[i]]))
            L.append("")
    L += ["## Role descriptions", "", "The first system prompt of each role, as the judges saw it.", ""]
    L += ["- **%s**: %s" % (r, describe(r)) for r in sorted(roles)]
    (out / "ends_all_directions.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="output/20-oct/k50")
    a = ap.parse_args(argv)
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = dict(np.load(D / "scores.npz"))
    K = S["pca_score"].shape[1]
    items1 = json.loads((D / "items" / "exp1.json").read_text())
    items2 = json.loads((D / "items" / "exp2.json").read_text())
    r1, r2, r4 = exp1(D, items1), exp2(D, items2), exp4(D, items2)
    f5 = D / "items" / "exp5.json"
    r5 = exp4(D, json.loads(f5.read_text()), "exp5_anchor") if (D / "exp5_anchor").exists() else None
    missing = [x for x in ["%s%d" % (m, k) for m in ("PCA", "PGA") for k in range(1, K + 1)]
               if x not in r1 or x not in r2 or len(r2[x]["held"]) < 5 or len(r4.get(x, [])) < 5]
    if missing:
        print("incomplete axes:", missing)
    rows = table(S, r1, r2, r4, K, r5)
    if r5:
        short = {x: len(v) for x, v in r5.items() if len(v) < 5}
        print("Exp 5 directions with fewer than 5 splits:", short)
    rho = json.loads((D / "scores_check.json").read_text())["spearman_pca_i_pga_j"]
    best = fig_match(rho, out)
    fig_lines(rows, K, out)
    fig_separate(rows, K, out)
    fig_vs_var(rows, out)
    lines, res = summary(rows, best)
    if r5:
        P4, P5 = parts(D, items2, "exp4_rank"), parts(D, json.loads(f5.read_text()), "exp5_anchor")
        fig_parts(P4, P5, K, out)
        print("\n".join((out / "table_exp4_exp5_order_parts.md").read_text().splitlines()) if parts_table(P4, P5, out) else "")
    labels_md(r1, r2, rows, K, out)
    themes_md(r1, rows, K, out)
    ends_md(S, r2, rows, K, out)
    (out / "table_summary.md").write_text("\n".join(lines) + "\n")
    hdr = "| direction | var % PCA | var % PGA | theme PCA | theme PGA | contrast PCA | contrast PGA | Exp 2 PCA | Exp 2 PGA | Exp 4 ρ PCA | Exp 4 ρ PGA | PGA max \\|ρ\\| with a PCA dir. |"
    tl = [hdr, "|" + "---|" * 12]
    for k in range(1, K + 1):
        p, g = rows[k - 1], rows[K + k - 1]
        tl.append("| %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.0f%% | %.0f%% | %.2f | %.2f | %.2f |" % (
            k, 100 * p["var_share"], 100 * g["var_share"], p["theme"], g["theme"], p["contrast"], g["contrast"],
            100 * p["held"], 100 * g["held"], p["rho"], g["rho"], best[k - 1]))
    (out / "table_per_direction.md").write_text("\n".join(tl) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(rows=rows, tests=res, pga_best_match=best.tolist(), missing=missing), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
