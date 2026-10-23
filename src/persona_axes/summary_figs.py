"""Summary bar charts (asked for 2026-10-23): one bar per set / family / condition, height = the largest value it reaches,
label = how many axes were used of how many that set has. Same numbers as the P0-P3 figures, read from the saved results.

    python -m persona_axes.summary_figs        # -> output/21-oct/summary/S1..S9 .png, tables/axes_summary.md, tables/analyses.md

S1 coverage (Phase 1, held-out R2), S2 identification (Phase 1, MRR), S3 axes passing the purity rule per family (Phase 2),
S4 reach by steering condition (Phase 3), S5 reach by clamp direction (P0.5), S6 single-axis reach per family (Phase 3b,
matched push), S7 best single axis per persona (Phase 3b, matched push), S8 one row per axis over all phases, S9 design overview (sets x families, list of analyses).
"""
from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from persona_axes import AX, HERE, OUT, COLLAPSE_CAP
from persona_axes.phase3 import P3

S = OUT / "summary"
FAM = {"A": "psychology", "B": "identity", "C": "interpersonal", "D": "epistemic", "E": "agency", "F": "communication", "G": "LLM-specific", "H": "TIDE"}
FCOL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52", "E": "#8172B3", "F": "#937860", "G": "#DA8BC3", "H": "#8C8C8C"}
GREY, DARK = "#BBBBBB", "#333333"
# pre-registered sets: label, number of axes in the set, result key ("psychology" = the Big Five; + honesty-humility = HEXACO)
SETS = [("psychology", 5, "1_OCEAN"), ("psychology + honesty-humility", 6, "2_HEXACO"), ("psychology + identity", 7, "3_OCEAN+B"), ("psychology + LLM-specific", 11, "4_OCEAN+G"),
        ("utility (B-F)", 15, "5_utility"), ("utility + LLM-specific (primary)", 21, "6_utility+G"), ("TIDE", 4, "7_TIDE"), ("union A-H", 31, "8_union")]


def md_table(path):
    rows = [l.strip().strip("|").split("|") for l in path.read_text().splitlines() if l.startswith("|") and not set(l) <= set("|:- ")]
    hdr = [c.strip() for c in rows[0]]
    df = pd.DataFrame([[c.strip() for c in r] for r in rows[1:]], columns=hdr)
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")           # columns with 'nan' strings stay numeric
        if num.notna().sum() >= 0.5 * max(len(df), 1):
            df[c] = num
    return df


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#EEEEEE"); ax.set_axisbelow(True)


def hbars(ax, labels, vals, colors, notes, xmax=1.0, fmt="%.2f"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.65)
    for yi, v, n in zip(y, vals, notes):
        ax.text(v + 0.01 * xmax, yi, (fmt % v) + ("   " + n if n else ""), va="center", fontsize=8, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.set_xlim(0, xmax); style(ax)
    return y


def largest_k(curve, n):
    ks = sorted(int(k) for k in curve)
    ks = [k for k in ks if k <= n] or ks[:1]
    return ks[-1]


def s1_s2(view="mean", layer=15, top_only=False):
    """top_only: just the value row (files *_top_*.png), no ratio-to-PCA row"""
    d = json.load(open(AX / "phase1" / ("results_%s_L%d.json" % (view, layer))))
    tag = ("top_" if top_only else "") + "%s_L%d" % (view, layer)
    for metric, fname, title, xlab in [("r2", "S1_coverage_%s.png" % tag, "S1  how much of the persona geometry each axis set covers (Phase 1)",
                                        "held-out R$^2$ of the persona vectors on the span of the set"),
                                       ("mrr", "S2_identification_%s.png" % tag, "S2  how well each axis set identifies a persona from half of its answers (Phase 1)",
                                        "mean reciprocal rank, split-half retrieval by cosine on the set's coordinates")]:
        fig, axs = plt.subplots(1 if top_only else 2, 2, figsize=(14, 5.6 if top_only else 10), sharey=True); axs = np.atleast_2d(axs)
        for col, (s, who) in enumerate([("b", "226 human roles"), ("a", "41 deployment personas")]):
            b = d[s]
            cur = (lambda key: b["curves"][key]["centred"]) if metric == "r2" else (lambda key: {k: v["mrr"] for k, v in b["m2"][key].items()})
            labels, vals, cols, notes, pca = [], [], [], [], []
            for lab, n, key in SETS:
                c = cur(key); k = largest_k(c, n)
                labels.append(lab); vals.append(c[str(k)]); cols.append("#4C72B0" if key != "6_utility+G" else "#2A4D7F"); notes.append("%d of %d axes" % (k, n)); pca.append(cur("PCA")[str(k)])
            for k in (8, 30):
                c = cur("greedy_pool"); labels.append("greedy pick from the trait pool, k = %d" % k); vals.append(c[str(k)]); cols.append("#C44E52"); notes.append("%d of 185 traits" % k); pca.append(cur("PCA")[str(k)])
            if metric == "r2":
                labels.append("random 8 traits from the pool (null B3, 95th pct.)"); vals.append(b["nulls"]["B3"]["8"]["p95"]); cols.append(GREY); notes.append("8 of 185 traits, 1000 draws"); pca.append(cur("PCA")["8"])
            else:
                labels.append("full 4096-d vector"); vals.append(b["m2"]["full"]["4096"]["mrr"]); cols.append(DARK); notes.append("no projection"); pca.append(np.nan)
            ax = axs[0, col]
            if metric == "r2":                                               # S1: value only, no "k of n axes" note (asked 2026-10-23)
                notes = ["" for _ in notes]
            y = hbars(ax, labels, vals, cols, notes, xmax=1.18)
            ax.scatter(pca, y, marker="|", s=260, color=DARK, zorder=3, label="PCA with the same k (best possible k directions)")
            ax.set_title(who, fontsize=10, loc="left"); ax.set_xlabel(xlab, fontsize=8); ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_ylim(-1.2, len(labels) + 0.2)
            if top_only:
                continue
            ax = axs[1, col]
            ratio = [v / p if p == p else np.nan for v, p in zip(vals, pca)]
            hbars(ax, labels, [0 if r != r else r for r in ratio], cols, ["" if r != r else n for r, n in zip(ratio, notes)], xmax=1.18)
            for t in ax.texts:                                               # rows without a PCA reference (full vector): no number
                if t.get_text().startswith("0.00"):
                    t.set_text("not applicable")
            ax.axvline(0.7, color=DARK, lw=0.8, ls="--"); ax.text(0.705, y[-1] - 0.7, "H1 bar, 0.7", fontsize=7, color=DARK)
            ax.set_xlabel("the same value divided by PCA with the same k\n(1 = as good as the best possible k directions)", fontsize=8); ax.set_xticks(np.arange(0, 1.01, 0.2))
        axs[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False)
        fig.suptitle(title + "\nbar = value at the largest k on the grid that fits in the set; Qwen3-8B, layer %d, %s view" % (layer, {"mean": "answer-token mean", "last": "last answer token"}[view]), fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.91 if top_only else 0.95)); fig.savefig(S / fname, dpi=150); plt.close(fig)


def s3(right_only=False):
    """right_only: just the best-axis-per-family panel (S3a_best_axis_per_family.png)"""
    t = md_table(OUT / "tables" / "P2_axis_table.md").rename(columns={"": "axis"})
    t["kept"] = t.kept.astype(str) == "True"
    fams = list(FAM)
    g = t.groupby("family")
    n_tot = g.size().reindex(fams); n_kept = g.kept.sum().reindex(fams)
    y = np.arange(len(fams))[::-1]
    if right_only:
        fig, ax = plt.subplots(figsize=(9, 4.6)); axs = [None, ax]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
        ax = axs[0]
        ax.barh(y, n_tot, color="#EBEBEB", height=0.65); ax.barh(y, n_kept, color=[FCOL[f] for f in fams], height=0.65)
        for yi, f in zip(y, fams):
            ax.text(n_tot[f] + 0.1, yi, "%d of %d axes pass" % (n_kept[f], n_tot[f]), va="center", fontsize=8, color=DARK)
        ax.set_yticks(y); ax.set_yticklabels(["%s  %s" % (f, FAM[f]) for f in fams], fontsize=9); ax.set_xlim(0, 8.5); style(ax)
        ax.set_xlabel("axes in the family (grey) and axes that pass the pre-registered rule (colour)\nrule: on-target >= 0.5, purity >= 0.25, split-half cosines >= 0.6, blind naming >= 0.6", fontsize=8)
        ax.set_title("axes that are separately steerable", fontsize=10, loc="left")
    ax = axs[1]
    best = t.loc[t.groupby("family").on_target.idxmax()].set_index("family").reindex(fams)
    ax.barh(y, best.on_target.fillna(0), color=[FCOL[f] for f in fams], height=0.65)
    for yi, f in zip(y, fams):
        r = best.loc[f]
        ax.text(r.on_target + 0.01, yi, "%.2f   %s" % (r.on_target, r.axis.replace("_vs_", " <-> ").replace("_", " ")), va="center", fontsize=8, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels(["%s  %s" % (f, FAM[f]) for f in fams] if right_only else [], fontsize=9)
    ax.set_xlim(0, 1.6); ax.set_xticks(np.arange(0, 1.01, 0.2)); style(ax)
    ax.set_xlabel("largest on-target effect in the family:\n-Spearman rho between dose and the judge's rating of the axis's own trait (70 answers)", fontsize=8)
    ax.set_title("best axis per family when pinned on its own", fontsize=10, loc="left")
    if right_only:
        fig.suptitle("S3a  the axis each family can turn best on its own (Phase 2, 7 doses x 10 questions)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(S / "S3a_best_axis_per_family.png", dpi=150); plt.close(fig); return
    fig.suptitle("S3  which named axes the one-direction clamp can turn on their own (Phase 2, 31 candidate axes, 7 doses x 10 questions each)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(S / "S3_axes_by_family.png", dpi=150); plt.close(fig)


def s4(min_full=0.0, sfx=""):
    """reach per condition from the Phase 3 labels; min_full keeps only personas whose full-vector cell reaches >= min_full"""
    lab = pd.read_csv(AX / "judge_phase3" / "labels.csv")
    lab = lab[lab.j10_judged].copy()
    lab["ok"] = lab.j10_top5 & ~lab.j10_nonsense
    lab["fam"] = lab.cell.str.split("_k").str[0].str.replace(r"_\d+$", "", regex=True)
    lab["k"] = lab.cell.str.extract(r"_k(\d+)_").astype(int); lab["S_label"] = lab.cell.str.split("_").str[-1]
    per = lab.groupby(["persona", "fam", "k", "S_label"]).agg(reach=("ok", "mean"), collapse=("j10_nonsense", "mean")).reset_index()
    full = per[(per.fam == "full") & (per.S_label == "Sfull")].set_index("persona").reach
    keep = full[full >= min_full].index
    per = per[per.persona.isin(keep)]
    per.loc[per.fam == "recipe", "k"] = 0                                 # one recipe per persona, k varies: pool them
    t = per.groupby(["fam", "k", "S_label"]).agg(reach=("reach", "mean"), collapse=("collapse", "mean"), n_personas=("persona", "nunique")).reset_index()
    t = t[t.collapse <= COLLAPSE_CAP]
    best = t.loc[t.groupby("fam").reach.idxmax()].set_index("fam")
    order = [o for o in ["full", "recipe", "greedy_pool", "set6", "B3", "B2"] if o in best.index]
    lab_ = {"full": "full persona vector (ceiling)", "recipe": "persona's own sparse trait recipe", "greedy_pool": "greedy pick from the trait pool",
            "set6": "named axes (21, utility + LLM-specific)", "B3": "random traits from the pool (null B3)", "B2": "random directions in the PCA-50 span (null B2)"}
    tot = {"full": "1 direction", "recipe": "<= 5 of 185 traits, chosen per persona", "greedy_pool": "%d of 185 traits", "set6": "%d of 21 axes", "B3": "%d of 185 traits, 3 draws", "B2": "%d random directions, 3 draws"}
    col = {"full": DARK, "recipe": "#C44E52", "greedy_pool": "#C44E52", "set6": "#2A4D7F", "B3": GREY, "B2": GREY}
    n_all = len(keep)
    notes = [(tot[o] % best.k[o] if "%d" in tot[o] else tot[o]) + ", %s, collapse %.0f%%" % ("S_p" if best.S_label[o] == "Sfull" else "S_p / 2", 100 * best.collapse[o]) for o in order]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    hbars(ax, [lab_[o] for o in order], [best.reach[o] for o in order], [col[o] for o in order], notes, xmax=1.0)
    ax.set_xlabel("largest share of answers with the target persona in the judge's top 5 and not nonsense,\nover k in {4, 8, 16} and two strengths; cells with collapse > %.0f%% excluded" % (100 * COLLAPSE_CAP), fontsize=8)
    who = "all 37 reachable personas" if min_full == 0 else "only the %d personas the full vector reaches in >= %.0f%% of answers" % (n_all, 100 * min_full)
    ax.set_title("S4%s  which steering condition reaches the target persona\n(Phase 3: %s;\n10 questions x 2 seeds, same clamp, strength and judge throughout)" % (sfx, who), fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(S / ("S4%s_reach_by_condition.png" % sfx), dpi=150); plt.close(fig)
    best.round(3).to_markdown(OUT / "tables" / ("S4%s_reach_by_condition.md" % sfx))


def s5():
    t = md_table(OUT / "tables" / "P0_5_directions_summary.md").rename(columns={"": "direction"}).set_index("direction")
    lab = {"mean": "persona - no-prompt state (the Phase 0 direction)", "contrast": "persona - assistant-prompted state", "prompt_last": "prompt-token state (last prompt position)",
           "span2": "2-D span of the two above"}
    nd = {"mean": "1 direction", "contrast": "1 direction", "prompt_last": "1 direction", "span2": "2 directions"}
    order = ["span2", "contrast", "mean", "prompt_last"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.18, [t.personas_reached_25[o] for o in order], height=0.34, color="#4C72B0", label="personas reached in >= 25% of answers")
    ax.barh(y - 0.18, [t.personas_reached_50[o] for o in order], height=0.34, color="#2A4D7F", label="personas reached in >= 50% of answers")
    for yi, o in zip(y, order):
        ax.text(t.personas_reached_25[o] + 0.6, yi + 0.18, "%d of 65   %s, best direction for %d personas" % (t.personas_reached_25[o], nd[o], t.best_direction_for_n_personas[o]), va="center", fontsize=8, color=DARK)
        ax.text(t.personas_reached_50[o] + 0.6, yi - 0.18, "%d of 65" % t.personas_reached_50[o], va="center", fontsize=8, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([lab[o] for o in order], fontsize=9); ax.set_xlim(0, 80); ax.set_xticks(range(0, 66, 10)); style(ax); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("number of the 65 Phase 0 personas reached at some push in {36, 54, 72} with collapse <= %.0f%% (Haiku J10, 20 answers per cell)" % (100 * COLLAPSE_CAP), fontsize=8)
    ax.set_title("S5  which clamp direction reaches the most personas\n(same clamp, layer 15, pushes matched in absolute units)", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(S / "S5_reach_by_direction.png", dpi=150); plt.close(fig)


def s6_s7(tag="_set6p54"):
    fam = {x["name"]: x["family"] for x in json.load(open(HERE / "axes.json"))["axes"]}
    nfam = pd.Series(fam).value_counts()
    ax_t = md_table(OUT / "tables" / ("P3b_axes%s.md" % tag)).rename(columns={"": "axis"}).set_index("axis")
    ax_t["family"] = [fam[a] for a in ax_t.index]
    fams = [f for f in FAM if f in set(ax_t.family)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(len(fams))[::-1]
    ax = axs[0]
    best = ax_t.loc[ax_t.groupby("family").personas_reached_25.idxmax()].reset_index().set_index("family").reindex(fams)
    n_reach = ax_t.groupby("family").apply(lambda g: int((g.personas_reached_25 >= 1).sum())).reindex(fams)
    n_in = ax_t.groupby("family").size().reindex(fams)
    ax.barh(y, best.personas_reached_25, color=[FCOL[f] for f in fams], height=0.65)
    for yi, f in zip(y, fams):
        ax.text(best.personas_reached_25[f] + 0.1, yi, "%d of 37   %s;  %d of %d axes in the family reach >= 1 persona" % (
            best.personas_reached_25[f], best.axis[f].replace("_vs_", " <-> ").replace("_", " "), n_reach[f], n_in[f]), va="center", fontsize=8, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels(["%s  %s" % (f, FAM[f]) for f in fams], fontsize=9); ax.set_xlim(0, 12); style(ax)
    ax.set_xlabel("largest number of personas (of 37) that one axis in the family reaches\nin >= 25%% of answers, collapse <= %.0f%%" % (100 * COLLAPSE_CAP), fontsize=8)
    ax.set_title("best single axis per family", fontsize=10, loc="left")
    ax = axs[1]
    worst = ax_t.loc[ax_t.groupby("family").mean_collapse.idxmax()].reset_index().set_index("family").reindex(fams)
    ax.barh(y, worst.mean_collapse, color=[FCOL[f] for f in fams], height=0.65)
    for yi, f in zip(y, fams):
        ax.text(worst.mean_collapse[f] + 0.01, yi, "%.0f%%   %s" % (100 * worst.mean_collapse[f], worst.axis[f].replace("_vs_", " <-> ").replace("_", " ")), va="center", fontsize=8, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 1.25); ax.set_xticks(np.arange(0, 1.01, 0.2)); style(ax)
    ax.axvline(COLLAPSE_CAP, color=DARK, lw=0.8, ls="--"); ax.text(COLLAPSE_CAP + 0.01, y[-1] - 0.6, "collapse cap", fontsize=7, color=DARK)
    ax.set_xlabel("largest share of nonsense answers for one axis in the family\nat the same push (the axis that breaks the model most)", fontsize=8)
    ax.set_title("most destructive axis per family", fontsize=10, loc="left")
    fig.suptitle("S6  single named axes as steering handles (Phase 3b: 21 axes of families B-G, matched absolute push 54, 37 personas, 20 answers per cell)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(S / "S6_axes_by_family.png", dpi=150); plt.close(fig)
    # S7 per persona
    pp = md_table(OUT / "tables" / ("P3b_personas%s.md" % tag)).set_index("persona")
    pp = pp.sort_values(["best_reach", "full_vector_reach"], ascending=False)
    fig, ax = plt.subplots(figsize=(12, 0.26 * len(pp) + 2))
    y = np.arange(len(pp))[::-1]
    cols = [FCOL[fam[a]] if a in fam else GREY for a in pp.best_axis]
    ax.barh(y, pp.best_reach, color=cols, height=0.7, label="best single axis (colour = its family)")
    ax.scatter(pp.full_vector_reach, y, marker="|", s=200, color=DARK, zorder=3, label="full persona vector, best Phase 0 cell")
    for yi, (p, r) in zip(y, pp.iterrows()):
        txt = "%.2f   %s   (%d of 21 axes reach it)" % (r.best_reach, r.best_axis.replace("_vs_", " <-> ").replace("_", " "), r.n_axes_reaching_25) if r.best_reach >= 0.25 else ("no axis reaches it" if r.best_reach == 0 else "%.2f   below 0.25" % r.best_reach)
        ax.text(max(r.best_reach, 0) + 0.01, yi, txt, va="center", fontsize=7.5, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels(["%s (%s)" % (p, s) for p, s in zip(pp.index, pp.set)], fontsize=8); ax.set_xlim(0, 1.75); ax.set_xticks(np.arange(0, 1.01, 0.25)); style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("share of 20 answers with the target in the judge's top 5 and not nonsense; (a) deployment persona, (b) human role", fontsize=8)
    ax.set_title("S7  which personas a single named axis can evoke, and which need the full vector\n(Phase 3b, matched push 54; %d of %d personas reached by no axis)" % (
        int((pp.best_reach < 0.25).sum()), len(pp)), fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(S / "S7_best_axis_per_persona.png", dpi=150); plt.close(fig)


def axis_label(a):
    return a.replace("_vs_", " <-> ").replace("tide_", "TIDE ").replace("_", " ")


def s8_axes(tag="_set6p54"):
    """one row per candidate axis, one small bar chart per measure, grouped by family; plus tables/axes_summary.md"""
    axes = json.load(open(HERE / "axes.json"))["axes"]
    cand = [a for a in axes if a["family"] != "N"]
    p2 = json.load(open(AX / "phase2" / "results.json"))
    t2 = md_table(OUT / "tables" / "P2_axis_table.md").rename(columns={"": "axis"}).set_index("axis")
    t3 = md_table(OUT / "tables" / ("P3b_axes%s.md" % tag)).rename(columns={"": "axis"}).set_index("axis")
    rows = []
    for a in cand:
        n = a["name"]; r2 = t2.loc[n]
        rows.append(dict(axis=n, family=a["family"], pole_a=a["pole_a"].split(":")[0], pole_b=a["pole_b"].split(":")[0],
                         coverage_added=p2["marginal_r2_set9"].get(n, np.nan), on_target=r2.on_target, purity=r2.purity, split_half_cos=r2.split_pairs, naming=r2.naming,
                         passes_rule=str(r2.kept) == "True", personas_reached_alone=t3.personas_reached_25.get(n, np.nan), collapse_alone=t3.mean_collapse.get(n, np.nan)))
    T = pd.DataFrame(rows)
    T.to_markdown(OUT / "tables" / "axes_summary.md", index=False, floatfmt=".2f")
    cols = [("coverage_added", "coverage it adds\n(marginal held-out R$^2$,\n226 roles, Phase 1)", 0.06),
            ("on_target", "moves its own trait\nwhen pinned (-rho,\nPhase 2)", 1.0),
            ("purity", "purity: own trait minus\nmean other traits\n(Phase 2)", 1.0),
            ("split_half_cos", "stability: cosine of\nre-extraction from other\nprompt pairs (Phase 1)", 1.0),
            ("naming", "blind naming accuracy\n(5 repeats, chance 0.2,\nPhase 2)", 1.0),
            ("personas_reached_alone", "personas it reaches\nalone, of 37 (matched\npush 54, Phase 3b)", 12),
            ("collapse_alone", "nonsense share when\npinned alone at push 54\n(Phase 3b)", 1.0)]
    fig, axs = plt.subplots(1, len(cols), figsize=(19, 0.3 * len(T) + 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(T))[::-1]
    for ax, (c, lab, xmax) in zip(axs, cols):
        v = T[c].astype(float)
        ax.barh(y, v.fillna(0).clip(lower=0), color=[FCOL[f] for f in T.family], height=0.7)
        neg = v < 0
        if neg.any():
            ax.barh(y[neg.values], v[neg].abs(), color="white", edgecolor=[FCOL[f] for f in T.family[neg]], hatch="////", height=0.7)
        for yi, val in zip(y, v):
            ax.text(xmax * 0.02 + (0 if val != val else max(val, 0)), yi, "n/a" if val != val else ("%d" % val if c == "personas_reached_alone" else "%.2f" % val), va="center", fontsize=7, color=DARK)
        ax.set_xlim(0, xmax * 1.3); ax.set_title(lab, fontsize=8, loc="left"); style(ax)
        ax.set_xticks([0, xmax / 2, xmax] if c != "personas_reached_alone" else [0, 6, 12]); ax.tick_params(labelsize=7)
        if c in ("on_target", "purity", "split_half_cos", "naming"):
            thr = {"on_target": 0.5, "purity": 0.25, "split_half_cos": 0.6, "naming": 0.6}[c]
            ax.axvline(thr, color=DARK, lw=0.7, ls="--")
        if c == "collapse_alone":
            ax.axvline(COLLAPSE_CAP, color=DARK, lw=0.7, ls="--")
    axs[0].set_yticks(y); axs[0].set_yticklabels(["%s%s  [%s]" % ("* " if k else "   ", axis_label(a), f) for a, f, k in zip(T.axis, T.family, T.passes_rule)], fontsize=8)
    for fam_break in np.where(T.family.values[1:] != T.family.values[:-1])[0]:
        for ax in axs:
            ax.axhline(y[fam_break] - 0.5, color="#DDDDDD", lw=0.8)
    fig.suptitle("S8  the 31 candidate axes, one row each, grouped by family (colour); * = passes the Phase 2 drop rule (dashed lines = the rule's thresholds and the collapse cap); "
                 "hatched = negative; n/a = not in the single-axis run (families A and H)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(S / "S8_axes_summary.png", dpi=150); plt.close(fig)


ANALYSES = [
    ("Phase 0", "ceiling", "can the clamp reach a persona with its full vector?", "65 personas; S in {1, 2, 4, 6}; absolute push 36-108 for the 35 failures", "reach (J10 top 5, not nonsense), collapse", "P0.1-P0.3"),
    ("P0.4-0.5", "directions", "which direction should the clamp use?", "65 personas x 4 directions x pushes {36, 54, 72}", "personas reached >= 0.25 / 0.5", "P0.5, S5"),
    ("Phase 1", "coverage (M1)", "how much of the geometry do k named axes cover?", "8 pre-registered sets + trait pool vs PCA and nulls B1-B3; k = 1..50", "held-out R$^2$", "P1.1, P1.2, S1"),
    ("Phase 1", "identification (M2)", "do k axes identify a persona from half its answers?", "same sets; split-half retrieval", "MRR, top-1, top-5", "P1.3, S2"),
    ("Phase 1", "residual / recipes", "which personas does the vocabulary miss? how many traits does one need?", "185-trait pool, per persona", "1 - R$^2$; traits to 90%", "P1.4, P1.5"),
    ("Phase 2", "purity", "does pinning one axis move only its own trait?", "35 axes x 7 doses x 10 questions; 35 rating scales", "on-target, off-target rho, purity, naming", "P2.1, P2.2, S3"),
    ("Phase 3", "reachability", "does steering inside the axes' span reach the persona?", "37 personas; set 6, set 9, pool, recipes, B2, B3; k in {4, 8, 16}", "reach vs full vector; rho(coverage, reach)", "P3.1, P3.2, S4"),
    ("Phase 3b", "single axes", "which axis evokes which persona on its own?", "37 personas x 21 axes; own coordinate and matched push 54", "personas reached per axis; best axis per persona", "P3b.1, P3b.2, S6, S7"),
]


def s9_design():
    """set x family membership grid (left) and one row per analysis (right); plus tables/analyses.md"""
    axes = json.load(open(HERE / "axes.json"))["axes"]
    nf = pd.Series([a["family"] for a in axes]).value_counts()
    from persona_axes.phase1 import FAM_SETS
    fams = list(FAM)
    rows = [(lab, n, FAM_SETS[key]) for lab, n, key in SETS] + [("trait pool (greedy / random)", 185, ""), ("PCA (ceiling)", "k", "")]
    fig, axs = plt.subplots(1, 2, figsize=(20, 6.6), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]
    M = np.zeros((len(rows), len(fams)))
    for i, (lab, n, spec) in enumerate(rows):
        for j, f in enumerate(fams):
            if spec == "ocean" or spec.startswith("ocean"):
                M[i, j] = 5 if f == "A" else (nf[f] if f in spec.replace("ocean", "") else 0)
            else:
                M[i, j] = nf[f] if f in spec else 0
    ax.imshow(np.where(M > 0, 1, 0), cmap=matplotlib.colors.ListedColormap(["white", "#4C72B0"]), aspect="auto", alpha=0.85)
    for i in range(len(rows)):
        for j in range(len(fams)):
            if M[i, j]:
                ax.text(j, i, "%d" % M[i, j], ha="center", va="center", fontsize=9, color="white")
    ax.text(len(fams) - 0.5 + 0.3, len(rows) - 2, "185 single-adjective traits", va="center", fontsize=8, color=DARK)
    ax.text(len(fams) - 0.5 + 0.3, len(rows) - 1, "data-driven directions", va="center", fontsize=8, color=DARK)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(["%s %s\n(%d)" % (f, FAM[f], nf[f]) for f in fams], fontsize=8, rotation=35, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(["%s  (%s axes)" % (lab, n) for lab, n, _ in rows], fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(fams)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True); ax.grid(which="minor", color="#DDDDDD", lw=0.6); ax.tick_params(length=0, which="both")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, len(fams) + 2.2)
    ax.set_title("axis sets tested for coverage and identification\n(number = axes taken from each family)", fontsize=10, loc="left")
    ax = axs[1]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    import textwrap
    hdr = ["phase", "analysis", "question", "design", "measure", "figures"]
    xs, widths = [0.0, 0.08, 0.2, 0.47, 0.76, 0.92], [10, 16, 40, 44, 24, 14]
    yy = 0.97
    for j, h in enumerate(hdr):
        ax.text(xs[j], yy, h, fontsize=8, weight="bold", va="top")
    yy -= 0.06
    for row in ANALYSES:
        cells = [textwrap.fill(c, w) for c, w in zip(row, widths)]
        nl = max(c.count("\n") + 1 for c in cells)
        for j, c in enumerate(cells):
            ax.text(xs[j], yy, c, fontsize=7.4, va="top", linespacing=1.15)
        yy -= 0.028 * nl + 0.03
        ax.axhline(yy + 0.015, color="#DDDDDD", lw=0.6)
    ax.set_title("analyses, in the order they were run (Qwen3-8B, layer 15, answer-token mean;\nL28 last token as robustness for Phase 1)", fontsize=10, loc="left")
    fig.suptitle("S9  design overview: what was tested with which axes", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(S / "S9_design_overview.png", dpi=150); plt.close(fig)
    pd.DataFrame(ANALYSES, columns=hdr).to_markdown(OUT / "tables" / "analyses.md", index=False)


def _columns(ax, items, ncol, x0=0.0, x1=1.0, y0=1.0, dy=0.03, fontsize=7.5, color=DARK):
    """lay out short strings in ncol columns, top to bottom then left to right"""
    per = int(np.ceil(len(items) / ncol))
    for i, it in enumerate(items):
        c, r = divmod(i, per)
        ax.text(x0 + c * (x1 - x0) / ncol, y0 - r * dy, it, fontsize=fontsize, va="top", color=color, family="DejaVu Sans")
    return y0 - per * dy


def s10_personas():
    """who was studied: the 41 deployment personas by category and the 226 human roles; marks for Phase 0 and Phase 3"""
    A = json.load(open(HERE / "personas_a.json"))["personas"]
    B = json.load(open(HERE / "personas_b_human.json"))
    p0 = json.load(open(AX / "ceiling" / "cells.json"))
    in_p0 = set(p0["set_a"]) | set(p0["set_b"])
    in_p3 = {r["persona"] for r in json.load(open(P3 / "cells.json"))["personas"]}
    mark = lambda n: n + ("  (P3)" if n in in_p3 else ("  (P0)" if n in in_p0 else ""))
    cats = {}
    for p in A:
        cats.setdefault(p["category"], []).append(p["name"])
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0.01, 0.02, 0.3, 0.9]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("set (a): 41 deployment personas in 20 categories (written for this study)", fontsize=10, loc="left")
    items = list(cats.items()); half = (len(items) + 1) // 2
    for col, chunk in enumerate((items[:half], items[half:])):
        yy = 0.99
        for cat, names in chunk:
            ax.text(col * 0.5, yy, cat, fontsize=8, weight="bold", va="top", color=DARK); yy -= 0.025
            for n in names:
                ax.text(col * 0.5 + 0.03, yy, mark(n), fontsize=7.5, va="top", color=DARK); yy -= 0.023
            yy -= 0.01
    ax = fig.add_axes([0.33, 0.02, 0.66, 0.9]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("set (b): 226 human roles from the Assistant Axis role list (276 minus 41 non-human, 2 assistant-like, 7 excluded on 2026-10-20; the 13 function-named roles kept)", fontsize=10, loc="left")
    _columns(ax, [mark(n) for n in B["roles"]], ncol=6, y0=0.99, dy=0.0235)
    import textwrap
    ax.text(0, 0.02, textwrap.fill("excluded (not studied): non-human " + ", ".join(B["excluded_non_human"]) + ";  assistant-like " + ", ".join(B["excluded_assistant"]) + ";  decided 2026-10-20 " + ", ".join(B["excluded_decided_2026_10_20"]), 230),
            fontsize=6.5, va="top", color="#777777")
    fig.suptitle("S10  the personas considered.  (P0) = in the 65-persona steering ceiling check (Phase 0, directions); (P3) = also among the 37 reachable personas used in Phase 3 and 3b; "
                 "unmarked = geometry only (Phase 1)", fontsize=11, x=0.01, ha="left")
    fig.savefig(S / "S10_personas_considered.png", dpi=150); plt.close(fig)
    rows = [dict(set="a", group=p["category"], name=p["name"], phase0=p["name"] in in_p0, phase3=p["name"] in in_p3) for p in A] + \
           [dict(set="b", group="human role", name=n, phase0=n in in_p0, phase3=n in in_p3) for n in B["roles"]]
    pd.DataFrame(rows).to_markdown(OUT / "tables" / "personas_considered.md", index=False)


def s11_axes_traits():
    """what was used as directions: the 31 candidate axes (+ 4 nuisance) by family with their poles, and the 185 kept traits of the 240-trait pool"""
    axes = json.load(open(HERE / "axes.json"))["axes"]
    kept = json.load(open(AX / "phase1" / "results_mean_L15.json"))["kept_traits"]
    allt = [t["name"] for t in json.load(open(HERE / "traits.json"))["traits"]]
    set9 = set(json.load(open(AX / "phase2" / "results.json"))["set9"])
    def axis_panel(ax, cut=35):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("31 pre-registered candidate axes by family, plus 4 nuisance axes (pole A <-> pole B)\n* = passes the Phase 2 drop rule (set 9)", fontsize=10, loc="left")
        yy = 0.99
        for f in list(FAM) + ["N"]:
            ax.text(0, yy, "%s  %s" % (f, FAM.get(f, "nuisance (not persona traits)")), fontsize=8.5, weight="bold", va="top", color=FCOL.get(f, DARK)); yy -= 0.022
            for a in [a for a in axes if a["family"] == f]:
                short = lambda t: (t[:cut - 1] + "…") if len(t) > cut else t
                ax.text(0.03, yy, "%s%s:  %s  <->  %s" % ("* " if a["name"] in set9 else "   ", axis_label(a["name"]), short(a["pole_a"].split(":")[0]), short(a["pole_b"].split(":")[0])), fontsize=7.5, va="top", color=DARK); yy -= 0.02
            yy -= 0.006
    fig = plt.figure(figsize=(10, 12))                                    # S11a: the axes alone
    axis_panel(fig.add_axes([0.02, 0.02, 0.96, 0.9]), cut=52)
    fig.suptitle("S11a  the 35 axes considered", fontsize=11, x=0.01, y=0.985, ha="left")
    fig.savefig(S / "S11a_axes_considered.png", dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(20, 12))
    axis_panel(fig.add_axes([0.01, 0.02, 0.42, 0.9]))
    ax = fig.add_axes([0.45, 0.02, 0.54, 0.9]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("trait pool: 185 of the Assistant Axis paper's 240 single-adjective traits kept by the Haiku filter\n(grey, in brackets = the 55 dropped: no behavioural difference on neutral questions)", fontsize=10, loc="left")
    _columns(ax, [t if t in kept else "(" + t + ")" for t in allt], ncol=6, y0=0.99, dy=0.023, fontsize=7.2)
    for t in ax.texts:
        if t.get_text().startswith("("):
            t.set_color("#AAAAAA")
    fig.suptitle("S11  the axes and traits considered as interpretable directions", fontsize=11, x=0.01, ha="left")
    fig.savefig(S / "S11_axes_traits_considered.png", dpi=150); plt.close(fig)
    rows = [dict(kind="axis", group=a["family"] + " " + FAM.get(a["family"], "nuisance"), name=a["name"], pole_a=a["pole_a"].split(":")[0], pole_b=a["pole_b"].split(":")[0], set9=a["name"] in set9) for a in axes] + \
           [dict(kind="trait", group="pool" + ("" if t in kept else " (dropped by filter)"), name=t, pole_a="", pole_b="", set9="") for t in allt]
    pd.DataFrame(rows).to_markdown(OUT / "tables" / "axes_traits_considered.md", index=False)


def s1_steps():
    """S1 explained on a toy 2-D cloud, one panel per step: vectors, mean, centring, candidate axes, greedy order, projection and R2, folds"""
    rng = np.random.default_rng(3)
    n = 40
    X = rng.multivariate_normal([6, 4], [[4.0, 2.6], [2.6, 2.5]], size=n)                 # toy "persona vectors" in 2-D
    tr, te = np.arange(n) < 32, np.arange(n) >= 32
    mu = X[tr].mean(0)
    Y = X - mu
    ang = {"formal <-> casual": 28, "warm <-> blunt": 80, "playful <-> serious": 130}       # three candidate axes (degrees)
    U = {k: np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for k, a in ang.items()}
    expl = {k: float(((Y[tr] @ u) ** 2).sum() / (Y[tr] ** 2).sum()) for k, u in U.items()}
    first = max(expl, key=expl.get)
    u1 = U[first]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.ravel()
    def base(ax, pts, title, lim=(-6, 6)):
        ax.scatter(pts[tr, 0], pts[tr, 1], s=18, color="#4C72B0", label="training personas (4 folds)")
        ax.scatter(pts[te, 0], pts[te, 1], s=28, color="#C44E52", marker="s", label="held-out personas (1 fold)")
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal"); ax.set_title(title, fontsize=9, loc="left"); style(ax); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
    # 1 raw vectors
    base(axs[0], X, "1. each persona is one vector (4096-d; drawn here in 2-D)\n   split into 5 folds: 4 for training, 1 held out", lim=(-1, 12))
    axs[0].scatter(*mu, s=160, marker="*", color=DARK, zorder=5, label="mean of the training vectors, mu")
    axs[0].scatter(0, 0, s=30, marker="+", color="#888888"); axs[0].text(0.2, -0.6, "origin", fontsize=7, color="#888888")
    axs[0].legend(fontsize=7, loc="upper left", frameon=False)
    # 2 centred
    base(axs[1], Y, "2. subtract mu from every vector (training and held-out)\n   now each point = how that persona differs from the average persona")
    axs[1].scatter(0, 0, s=160, marker="*", color=DARK, zorder=5); axs[1].text(0.2, -0.7, "mu is now the origin", fontsize=7, color=DARK)
    # 3 candidate axes
    base(axs[2], Y, "3. the set's axes are fixed directions through the origin\n   (extracted earlier from prompt pairs; nothing is fitted here)")
    for (k, u), c in zip(U.items(), ["C1", "C2", "C4"]):
        axs[2].plot([-6 * u[0], 6 * u[0]], [-6 * u[1], 6 * u[1]], color=c, lw=1.5)
        axs[2].text(5.2 * u[0], 5.2 * u[1], k, fontsize=7, color=c, ha="center")
    # 4 greedy order
    ax = axs[3]
    names = list(expl); vals = [expl[k] for k in names]
    ax.barh(range(3)[::-1], vals, color=["C1", "C2", "C4"], height=0.5)
    ax.set_yticks(range(3)[::-1]); ax.set_yticklabels(names, fontsize=8); ax.set_xlim(0, 1); style(ax)
    for i, v in zip(range(3)[::-1], vals):
        ax.text(v + 0.02, i, "%.2f" % v + ("   <- picked first" if names[2 - i] == first else ""), va="center", fontsize=8)
    ax.set_xlabel("share of training variance along the axis", fontsize=8)
    ax.set_title("4. order the axes greedily on the training personas:\n   pick the one explaining most; remove what it explains; pick the next", fontsize=9, loc="left")
    # 5 projection, k = 1
    base(axs[4], Y, "5. k = 1: project each held-out persona onto the first axis\n   residual = the part the axis cannot explain (red segments)")
    axs[4].plot([-6 * u1[0], 6 * u1[0]], [-6 * u1[1], 6 * u1[1]], color="C1", lw=1.5)
    P = np.outer(Y[te] @ u1, u1)
    for a, b in zip(Y[te], P):
        axs[4].plot([a[0], b[0]], [a[1], b[1]], color="#C44E52", lw=1)
    axs[4].scatter(P[:, 0], P[:, 1], s=14, color="C1", zorder=4)
    r2 = 1 - ((Y[te] - P) ** 2).sum() / (Y[te] ** 2).sum()
    axs[4].text(-5.6, 5.0, "R$^2$ = 1 - sum(residual$^2$) / sum(|centred vector|$^2$) = %.2f\n(over the held-out personas only)" % r2, fontsize=8)
    # 6 folds and the final number
    ax = axs[5]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("6. repeat for every fold and shuffle, and every k\n   the bar in S1 is the average R$^2$ at the largest k that fits in the set", fontsize=9, loc="left")
    yy = 0.9
    for f in range(5):
        for j in range(5):
            ax.add_patch(plt.Rectangle((0.05 + j * 0.12, yy - 0.08), 0.1, 0.07, color="#C44E52" if j == f else "#4C72B0", alpha=0.8))
        ax.text(0.68, yy - 0.045, "fold %d held out -> one R$^2$ per k" % (f + 1), fontsize=7.5, va="center")
        yy -= 0.1
    ax.text(0.05, 0.33, "x 10 random shuffles of the fold assignment = 50 values per k, averaged.\n\n"
                        "k = 2, 3, ...: same, projecting onto the span of the first k axes (a plane, ...).\n\n"
                        "PCA tick: same steps, but the k directions are the top-k principal components\n"
                        "of the training personas (the best any k directions could do).\n\n"
                        "random-trait null: same steps with k traits drawn at random from the pool,\n200 draws per fold; the grey bar is the 95th percentile.", fontsize=7.5, va="top")
    fig.suptitle("How the S1 bars are computed, step by step (toy 2-D example; the real vectors are 4096-d)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(S / "S1_how_it_is_computed.png", dpi=150); plt.close(fig)


def s2_steps():
    """S2 explained on a toy 2-D example: answers per persona, split halves, coordinates on k axes, cosine ranking, MRR"""
    rng = np.random.default_rng(16)
    names = ["bard", "accountant", "hermit", "coach", "pirate"]
    centres = np.array([[3, 3], [-2, 1], [1, -3], [-0.5, 2.5], [3, -0.5]], float)
    pcol = ["C0", "C1", "C2", "C3", "C4"]
    ans = [c + rng.normal(0, 1.8, size=(10, 2)) for c in centres]                         # 10 answers per persona
    perm = [rng.permutation(10) for _ in ans]
    hA = np.array([a[p[:5]].mean(0) for a, p in zip(ans, perm)]); hB = np.array([a[p[5:]].mean(0) for a, p in zip(ans, perm)])
    ang = {"formal <-> casual": 20, "warm <-> blunt": 75}
    U = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in ang.values()])  # 2 x 2, rows = axes
    coords = lambda V: np.linalg.lstsq(U.T, V.T, rcond=None)[0].T                           # least-squares coordinates on the k axes
    CA, CB = coords(hA), coords(hB)
    cos = (CA / np.linalg.norm(CA, axis=1, keepdims=True)) @ (CB / np.linalg.norm(CB, axis=1, keepdims=True)).T
    rank = np.array([1 + (cos[i] > cos[i, i]).sum() for i in range(len(names))])
    fig, axs = plt.subplots(2, 3, figsize=(16, 10)); axs = axs.ravel()
    def base(ax, title, lim=(-7, 7)):
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal"); ax.set_title(title, fontsize=9, loc="left"); style(ax); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    # 1 answers
    base(axs[0], "1. each persona has 40 answers; each answer is one vector\n   (here 10 answers per persona, 5 personas, drawn in 2-D)")
    for a, c, n in zip(ans, pcol, names):
        axs[0].scatter(a[:, 0], a[:, 1], s=14, color=c, alpha=0.7); axs[0].text(a[:, 0].mean(), a[:, 1].mean() + 1.6, n, fontsize=8, color=c, ha="center")
    # 2 halves
    base(axs[1], "2. split each persona's answers into two random halves\n   average each half: half A (circle) and half B (square) per persona")
    for a, p, c, A, B in zip(ans, perm, pcol, hA, hB):
        axs[1].scatter(a[p[:5], 0], a[p[:5], 1], s=10, color=c, alpha=0.35); axs[1].scatter(a[p[5:], 0], a[p[5:], 1], s=10, color=c, alpha=0.35, marker="s")
        axs[1].scatter(*A, s=90, color=c, edgecolor="k", zorder=5); axs[1].scatter(*B, s=90, color=c, marker="s", edgecolor="k", zorder=5)
    # 3 coordinates on axes
    base(axs[2], "3. give each half-vector k coordinates on the set's axes\n   (least squares; k = 2 here, the two lines)")
    for (k, a), c in zip(ang.items(), ["#888888", "#444444"]):
        u = U[list(ang).index(k)]; axs[2].plot([-7 * u[0], 7 * u[0]], [-7 * u[1], 7 * u[1]], color=c, lw=1.2); axs[2].text(6.2 * u[0], 6.2 * u[1], k, fontsize=7, color=c, ha="center")
    for c, A, B, ca, cb in zip(pcol, hA, hB, CA, CB):
        axs[2].scatter(*A, s=90, color=c, edgecolor="k", zorder=5); axs[2].scatter(*B, s=90, color=c, marker="s", edgecolor="k", zorder=5)
        axs[2].text(A[0] + 0.3, A[1] + 0.3, "(%.1f, %.1f)" % tuple(ca), fontsize=6.5, color=c)
    # 4 cosine matrix
    ax = axs[3]
    im = ax.imshow(cos, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.5, rotation=30, ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("half B coordinates of persona ...", fontsize=8); ax.set_ylabel("half A coordinates of persona ...", fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "%.2f" % cos[i, j], ha="center", va="center", fontsize=7, color="white" if cos[i, j] > 0.5 else DARK, weight="bold" if i == j else "normal")
        ax.text(5.1, i, "rank of the true match: %d" % rank[i], fontsize=7.5, va="center")
    ax.set_xlim(-0.5, 7.5)
    ax.set_title("4. for each persona's half A, rank all personas' half B by cosine\n   of the k coordinates; where does the true persona land?", fontsize=9, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 5 reciprocal rank
    ax = axs[4]
    rr = 1 / rank
    ax.barh(range(5)[::-1], rr, color=pcol, height=0.5)
    for i, (r, k) in enumerate(zip(rr, rank)):
        ax.text(r + 0.02, 4 - i, "1 / %d = %.2f" % (k, r), va="center", fontsize=8)
    ax.axvline(rr.mean(), color=DARK, ls="--", lw=1); ax.text(rr.mean() + 0.02, -0.7, "MRR = mean = %.2f" % rr.mean(), fontsize=8)
    ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(names, fontsize=8); ax.set_xlim(0, 1.3); ax.set_ylim(-1, 4.6); style(ax)
    ax.set_xlabel("reciprocal rank = 1 / rank of the true match", fontsize=8)
    ax.set_title("5. score = 1 / rank per persona; the bar in S2 is the mean (MRR)\n   1 = every persona found first; 1/2 = second on average", fontsize=9, loc="left")
    # 6 notes
    ax = axs[5]; ax.axis("off")
    ax.text(0, 0.95, "6. what varies across the S2 bars", fontsize=9, va="top", weight="bold")
    ax.text(0, 0.85, "k and the set: the coordinates are taken on the first k axes of the set\n(same greedy order as S1); the bar uses the largest k that fits.\n\n"
                     "PCA tick: coordinates on the top-k principal components instead.\n\n"
                     "full 4096-d vector: no coordinates at all; cosine between the raw\nhalf A and half B vectors (the best this retrieval can do).\n\n"
                     "real numbers: 226 human roles, so chance for top-1 is 1 / 226 = 0.004;\n41 deployment personas, chance 0.024.\n\n"
                     "the split into halves is random; one split per persona.\n\n"
                     "no centring here: cosine already ignores the overall scale, and the\naxes' coordinates are compared persona against persona.", fontsize=8, va="top")
    fig.suptitle("How the S2 bars are computed, step by step (toy 2-D example; the real vectors are 4096-d)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(S / "S2_how_it_is_computed.png", dpi=150); plt.close(fig)


def _clamp_panel(ax, M0, Mt, h, U, S, title, lab_goal, lim=(-1, 9)):
    """toy 2-D drawing of one clamp: neutral state M0, persona state Mt, a current activation h, the span U (rows), goal and the moved h"""
    v = Mt - M0
    Uo = np.linalg.qr(U.T)[0].T                                           # orthonormal rows
    G = M0 + S * (Uo.T @ (Uo @ v))
    h2 = h + Uo.T @ (Uo @ (G - h))
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal"); ax.set_title(title, fontsize=9, loc="left"); style(ax); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for u in Uo:
        ax.plot([M0[0] - 9 * u[0], M0[0] + 9 * u[0]], [M0[1] - 9 * u[1], M0[1] + 9 * u[1]], color="#BBBBBB", lw=1, ls="--")
    ax.annotate("", Mt, M0, arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.5)); ax.text(*(M0 + 0.5 * v + [0.2, -0.5]), "v = M_t - M_0", color="#4C72B0", fontsize=7.5)
    ax.scatter(*M0, s=80, color=DARK, zorder=5); ax.text(M0[0] + 0.2, M0[1] - 0.6, "M_0 neutral state", fontsize=7.5)
    ax.scatter(*Mt, s=80, color="#4C72B0", zorder=5); ax.text(Mt[0] + 0.2, Mt[1] + 0.2, "M_t persona state", fontsize=7.5, color="#4C72B0")
    ax.scatter(*G, s=90, marker="*", color="#C44E52", zorder=6); ax.text(G[0] + 0.2, G[1] - 0.6, lab_goal, fontsize=7.5, color="#C44E52")
    ax.scatter(*h, s=50, color="#55A868", zorder=5); ax.text(h[0] + 0.2, h[1] + 0.2, "h (activation at one token)", fontsize=7.5, color="#55A868")
    ax.annotate("", h2, h, arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5, ls="-"))
    ax.scatter(*h2, s=50, color="#55A868", edgecolor="k", zorder=6); ax.text(h2[0] + 0.2, h2[1] - 0.6, "h after the clamp", fontsize=7.5, color="#55A868")
    return G


def s4_steps():
    """S4 explained: the clamp inside a span, the judge, reach per cell, the aggregation into one bar"""
    M0, Mt, h = np.array([2.0, 2.0]), np.array([5.5, 6.0]), np.array([3.0, 6.5])
    v = Mt - M0
    fig, axs = plt.subplots(2, 3, figsize=(16, 10)); axs = axs.ravel()
    _clamp_panel(axs[0], M0, Mt, h, v[None, :] / np.linalg.norm(v), 1.4,
                 "1. full persona vector (the ceiling row): the span is the line M_0 -> M_t\n   goal G = M_0 + S v; every token's h is moved onto the line at G", "G = M_0 + S v")
    axs[0].text(0.2, 0.4, "the clamp only changes the part of h along the span;\nthe part of h orthogonal to the span stays", fontsize=7, color="#555555")
    a = np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])
    _clamp_panel(axs[1], M0, Mt, h, a[None, :], 1.4,
                 "2. named axes (k = 1 here): the span is the set's first k axes\n   goal G = M_0 + S (U U^T v): the persona's shadow on the span, scaled by S", "G = M_0 + S U U^T v")
    axs[1].text(0.2, 0.4, "whatever part of v lies outside the span is lost:\nthe clamp can only move h inside the span", fontsize=7, color="#555555")
    # 3 generation + judge
    ax = axs[2]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("3. generate and judge: 10 questions x 2 seeds = 20 answers per cell\n   the judge ranks 317 persona names and flags nonsense", fontsize=9, loc="left")
    rng = np.random.default_rng(1)
    top5 = rng.random(20) < 0.45; nons = rng.random(20) < 0.1
    for i in range(20):
        r, c = divmod(i, 2)
        ok = top5[i] and not nons[i]
        ax.add_patch(plt.Rectangle((0.02 + c * 0.48, 0.9 - r * 0.085), 0.45, 0.07, color="#4C72B0" if ok else ("#C44E52" if nons[i] else "#EEEEEE"), alpha=0.8))
        ax.text(0.04 + c * 0.48, 0.935 - r * 0.085, "q%d seed %d: %s" % (i // 2 + 1, i % 2, "target in top 5" if ok else ("nonsense" if nons[i] else "another persona / assistant")), fontsize=6.5, va="center", color="white" if ok or nons[i] else DARK)
    ax.text(0.02, 0.03, "reach of the cell = %d / 20 = %.2f      collapse = %d / 20 = %.2f" % ((top5 & ~nons).sum(), (top5 & ~nons).mean(), nons.sum(), nons.mean()), fontsize=8)
    # 4 cells per condition
    ax = axs[3]
    ks, Ss = [4, 8, 16], ["S_p", "S_p / 2"]
    reach = np.array([[0.11, 0.08], [0.12, 0.10], [0.09, 0.14]]); coll = np.array([[0.04, 0.00], [0.12, 0.02], [0.29, 0.03]])
    ax.imshow(np.where(coll > COLLAPSE_CAP, np.nan, reach), cmap="Blues", vmin=0, vmax=0.3)
    for i in range(3):
        for j in range(2):
            ax.text(j, i, ("reach %.2f\ncollapse %.0f%%" % (reach[i, j], 100 * coll[i, j])) + ("\nDROPPED" if coll[i, j] > COLLAPSE_CAP else ""), ha="center", va="center", fontsize=7.5, color="#C44E52" if coll[i, j] > COLLAPSE_CAP else DARK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(Ss, fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(["k = %d" % k for k in ks], fontsize=8)
    ax.set_title("4. one condition (e.g. set 6): a cell for every k and strength\n   each cell = mean reach over the 37 personas;\n   cells with collapse > 20% dropped", fontsize=9, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.5, 2.9, "the S4 bar for this condition = the largest surviving cell (0.14, k = 16, S_p / 2)", fontsize=8, color=DARK)
    # 5 conditions
    ax = axs[4]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("5. the conditions, i.e. where the span U comes from", fontsize=9, loc="left")
    ax.text(0, 0.92, "full persona vector:  U = v itself (k = 1); the reference.\n\n"
                     "set 6 / set 9:  the first k named axes of the set, in the greedy order from Phase 1.\n\n"
                     "greedy pick from the pool:  the first k traits picked greedily on all human roles.\n\n"
                     "persona's own recipe:  the <= 5 traits that best reconstruct this persona (Phase 1).\n\n"
                     "random traits (B3):  k traits drawn at random from the 185; 3 draws, averaged.\n\n"
                     "random directions (B2):  k random directions inside the PCA-50 span; 3 draws.\n\n"
                     "S_p = the strength at which the full vector reached this persona best in Phase 0,\nso every condition pushes as hard as the full vector did.", fontsize=8, va="top")
    # 6 equation
    ax = axs[5]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("6. the clamp, in one line", fontsize=9, loc="left")
    ax.text(0, 0.9, "at layer 15, for every token position, before the layer runs:\n\n"
                    "    h  <-  h + U U^T (G - h),        G = M_0 + S U U^T (M_t - M_0)\n\n"
                    "U: 4096 x k, orthonormal columns spanning the chosen directions\n"
                    "U U^T h: the part of h inside the span; it is replaced by U U^T G\n"
                    "(I - U U^T) h: the part outside the span; untouched\n\n"
                    "with k = 1 this is Giovanni's pin  h <- h + (c - h.u) u,  c = u.M_0 + S |v|.\n\n"
                    "same questions, seeds, judge and strength for every condition, so the bars\nin S4 differ only in which directions the clamp is allowed to use.", fontsize=8, va="top", family="DejaVu Sans")
    fig.suptitle("How the S4 bars are computed, step by step (panels 1-2 are toy 2-D drawings; the real space is 4096-d)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(S / "S4_how_it_is_computed.png", dpi=150); plt.close(fig)


def s67_steps():
    """S6 and S7 explained: one-axis clamp at a fixed push, the persona x axis reach matrix, and the two ways of reading it"""
    M0, Mt, h = np.array([2.0, 2.0]), np.array([4.0, 5.5]), np.array([3.0, 6.5])
    fig, axs = plt.subplots(2, 3, figsize=(16, 10)); axs = axs.ravel()
    a = np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])
    v = Mt - M0; push = 4.0
    ax = axs[0]
    ax.set_xlim(-1, 9); ax.set_ylim(-1, 9); ax.set_aspect("equal"); style(ax); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("1. one named axis a, fixed push: goal coordinate c = a.M_0 + sign(a.v) x 54\n   (every axis and persona pushed the same distance; the sign is the persona's side)", fontsize=9, loc="left")
    ax.plot([M0[0] - 9 * a[0], M0[0] + 9 * a[0]], [M0[1] - 9 * a[1], M0[1] + 9 * a[1]], color="#BBBBBB", lw=1, ls="--"); ax.text(M0[0] + 6.5 * a[0], M0[1] + 6.5 * a[1] - 0.6, "axis a", fontsize=7.5, color="#888888")
    ax.annotate("", Mt, M0, arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.5)); ax.scatter(*M0, s=80, color=DARK, zorder=5); ax.text(M0[0] + 0.2, M0[1] - 0.6, "M_0", fontsize=7.5)
    ax.scatter(*Mt, s=80, color="#4C72B0", zorder=5); ax.text(Mt[0] + 0.2, Mt[1] + 0.2, "M_t", fontsize=7.5, color="#4C72B0")
    proj = M0 + (a @ v) * a; ax.plot([Mt[0], proj[0]], [Mt[1], proj[1]], color="#4C72B0", lw=0.8, ls=":"); ax.text(proj[0] - 2.6, proj[1] - 1.3, "persona's own coordinate a.v (small;\nthe first run used it: too small a push)", fontsize=6.5, color="#4C72B0")
    G = M0 + push * a; ax.scatter(*G, s=90, marker="*", color="#C44E52", zorder=6); ax.text(G[0] + 0.3, G[1] + 0.3, "goal at 54 units", fontsize=7.5, color="#C44E52")
    h2 = h + (a @ (G - h)) * a; ax.scatter(*h, s=50, color="#55A868", zorder=5); ax.annotate("", h2, h, arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5)); ax.scatter(*h2, s=50, color="#55A868", edgecolor="k", zorder=6)
    ax.text(h[0] + 0.2, h[1] + 0.2, "h -> h after the clamp", fontsize=7.5, color="#55A868")
    # 2 matrix
    ax = axs[1]
    pers = ["companion_cheerful", "bard", "expert_chef", "reporter"]; axes_ = ["warm <-> blunt", "sycophantic", "formal <-> casual", "hedging", "grounded <-> mystical"]
    R = np.array([[1.0, 0.9, 0.6, 0.1, 0.2], [0.1, 0.0, 0.2, 0.0, 0.45], [0.0, 0.05, 0.1, 0.0, 0.0], [0.0, 0.0, 0.15, 0.0, 0.05]])
    C = np.array([[0.0, 0.05, 0.0, 0.85, 0.05], [0.0, 0.0, 0.0, 0.8, 0.05], [0.0, 0.1, 0.0, 0.75, 0.0], [0.05, 0.0, 0.0, 0.9, 0.0]])
    Rok = np.where(C > COLLAPSE_CAP, 0.0, R)
    ax.imshow(Rok, cmap="viridis", vmin=0, vmax=1)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, ("%.2f" % R[i, j]) + ("\ncollapse %.0f%%\n-> 0" % (100 * C[i, j]) if C[i, j] > COLLAPSE_CAP else ""), ha="center", va="center", fontsize=7, color="white" if Rok[i, j] < 0.6 else DARK)
    ax.set_xticks(range(5)); ax.set_xticklabels(axes_, fontsize=7.5, rotation=30, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(pers, fontsize=7.5)
    ax.set_title("2. reach per (persona, axis) cell, 20 answers each, as in S4 step 3\n   a cell with collapse > 20% counts as reach 0 (toy numbers)", fontsize=9, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 3 S6 left: per axis count -> per family max
    ax = axs[2]
    cnt = (Rok >= 0.25).sum(0)
    ax.bar(range(5), cnt, color=["C2", "C6", "C2", "C3", "C1"])
    for j, c in enumerate(cnt):
        ax.text(j, c + 0.05, "%d" % c, ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(axes_, fontsize=7.5, rotation=30, ha="right"); ax.set_ylim(0, 4.5); style(ax); ax.set_ylabel("personas with reach >= 0.25", fontsize=8)
    ax.set_title("3. S6 left: per axis, count the personas it reaches (reach >= 0.25)\n   then per family keep the best axis (colour = family)", fontsize=9, loc="left")
    ax.text(-0.4, 3.9, "interpersonal (green): best axis reaches 1 of 4;\n2 of 2 axes reach >= 1 persona.  epistemic (red): 0", fontsize=7)
    # 4 S6 right: per axis mean collapse
    ax = axs[3]
    mc = C.mean(0)
    ax.bar(range(5), mc, color=["C2", "C6", "C2", "C3", "C1"]); ax.axhline(COLLAPSE_CAP, color=DARK, ls="--", lw=0.8)
    for j, c in enumerate(mc):
        ax.text(j, c + 0.02, "%.0f%%" % (100 * c), ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(axes_, fontsize=7.5, rotation=30, ha="right"); ax.set_ylim(0, 1); style(ax); ax.set_ylabel("mean nonsense share over personas", fontsize=8)
    ax.set_title("4. S6 right: per axis, the mean collapse over the personas\n   then per family keep the worst axis (the one that breaks the model most)", fontsize=9, loc="left")
    # 5 S7
    ax = axs[4]
    best = Rok.max(1); bi = Rok.argmax(1); full = np.array([0.9, 0.95, 1.0, 0.85])
    y = np.arange(4)[::-1]
    ax.barh(y, best, color=[["C2", "C6", "C2", "C3", "C1"][b] for b in bi], height=0.5)
    ax.scatter(full, y, marker="|", s=200, color=DARK, zorder=3)
    for yi, b, i in zip(y, best, bi):
        ax.text(b + 0.02, yi, ("%.2f  %s  (%d of 5 axes reach it)" % (b, axes_[i], (Rok[3 - yi] >= 0.25).sum())) if b >= 0.25 else ("no axis reaches it" if b == 0 else "%.2f  below 0.25" % b), va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(pers, fontsize=7.5); ax.set_xlim(0, 1.9); ax.set_xticks([0, 0.5, 1]); style(ax)
    ax.set_title("5. S7: per persona, the best cell over all axes (bar, colour = that axis's family)\n   black tick = the same persona's best full-vector reach from Phase 0", fontsize=9, loc="left")
    # 6 notes
    ax = axs[5]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("6. what the real run had", fontsize=9, loc="left")
    ax.text(0, 0.92, "37 personas (the ones the full vector reaches) x 21 axes (families B-G,\nthe utility + LLM-specific set) = 777 cells, 15,540 answers, Haiku judge.\n\n"
                     "push 54 = the median full-vector push that reached these personas,\nso 'matched': each axis is pushed as hard as the full vector was.\n\n"
                     "the first single-axis run used the persona's own coordinate on the axis\n(S_p x a.v) instead; median 15 units, too small to move the model,\nhence the matched-push rerun that S6 and S7 show.\n\n"
                     "S6 answers: which axes are usable handles, and which break the model.\nS7 answers: which personas any single axis can evoke, and which need\nthe full vector.", fontsize=8, va="top")
    fig.suptitle("How the S6 and S7 bars are computed, step by step (panel 1 is a toy 2-D drawing; panels 2-5 use toy numbers)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(S / "S6_S7_how_they_are_computed.png", dpi=150); plt.close(fig)


def main():
    S.mkdir(exist_ok=True)
    s1_steps(); s2_steps(); s4_steps(); s67_steps(); s1_s2(); s1_s2("last", 28); s1_s2(top_only=True); s1_s2("last", 28, top_only=True); s3(); s3(right_only=True); s3_heatmap(); s3_offtarget(); s4(); s4(0.5, "b"); s5(); s6_s7(); s6_table(); s6_table("collapsed"); s8_axes(); s9_design(); s10_personas(); s11_axes_traits()
    print("wrote", sorted(p.name for p in S.glob("S*.png")))


if __name__ == "__main__":
    main()


def s6_table(mode="reached"):
    """S6b / S6c: one row per axis (families B-G), Phase 3b matched push 54.
    mode 'reached':  personas the axis reaches alone (reach >= 0.25, collapse <= COLLAPSE_CAP), ordered by reach.
    mode 'collapsed': personas for which the push breaks the model (nonsense share > COLLAPSE_CAP), ordered by nonsense share.
    Also tables/S6_axis_personas_{reached,collapsed}.md"""
    ax = json.load(open(HERE / "axes.json"))["axes"]; fam = {a["name"]: a["family"] for a in ax}
    runs = [AX / "judge_phase3_axes_set6p54" / "labels.csv", AX / "judge_phase3_axes_AHp54" / "labels.csv"]   # set 6 (B-G) + A and H (2026-10-23)
    lab = pd.concat([pd.read_csv(f) for f in runs if f.exists()], ignore_index=True)
    lab = lab[lab.j10_judged & lab.cell.str.startswith("ax_")].copy(); lab["axis"] = lab.cell.str[3:]
    per = lab.groupby(["persona", "axis"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean")).reset_index()
    from persona_axes.spec import NON_HUMAN
    per = per[~per.persona.isin(NON_HUMAN)]                                 # drop the run-2 bridge roles (void, leviathan): not human roles
    if mode == "reached":
        hit = per[(per.reach >= 0.25) & (per.collapse <= COLLAPSE_CAP)].sort_values("reach", ascending=False)
        head, sfx, crit = "personas it reaches on its own", "b", "reach >= 25%% of 20 answers, collapse <= %.0f%%" % (100 * COLLAPSE_CAP)
    else:
        hit = per[per.collapse > COLLAPSE_CAP].sort_values("collapse", ascending=False)
        head, sfx, crit = "personas for which the push breaks the model (nonsense answers)", "c", "nonsense > %.0f%% of 20 answers" % (100 * COLLAPSE_CAP)
    names = [a["name"] for a in ax if fam[a["name"]] != "N" and a["name"] in set(per.axis)]
    allp = sorted(per.persona.unique()); n_p = len(allp)
    def cell(a):
        ps = list(hit[hit.axis == a].persona)
        if not ps:
            return "none"
        return ", ".join(ps)
    rows = [dict(family=fam[a], axis=a.replace("_vs_", " <-> ").replace("_", " "), personas=cell(a)) for a in names]
    t = pd.DataFrame(rows)
    t[["axis", "personas"]].rename(columns={"personas": "personas_" + mode}).to_markdown(OUT / "tables" / ("S6_axis_personas_%s.md" % mode), index=False)
    wrapped = [textwrap.fill(r.personas, 88) for r in t.itertuples()]
    nlines = [w.count("\n") + 1 for w in wrapped]
    fig, axp = plt.subplots(figsize=(12, 0.3 * sum(nlines) + 1.2)); axp.axis("off")
    tb = axp.table(cellText=[[r.axis, w] for r, w in zip(t.itertuples(), wrapped)], colLabels=["axis", head],
                   colWidths=[0.26, 0.74], loc="upper left", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.45)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#DDDDDD"); c.set_linewidth(0.6)
        if i == 0:
            c.set_text_props(weight="bold", color="white"); c.set_facecolor(DARK)
        else:
            c.set_facecolor(FCOL[t.family[i - 1]] + "33" if j == 0 else ("#F7F7F7" if i % 2 else "white"))
    h0 = tb[1, 0].get_height()
    for i, n in enumerate(nlines, start=1):
        tb[i, 0].set_height(h0 * (0.4 + 0.6 * n)); tb[i, 1].set_height(h0 * (0.4 + 0.6 * n))
    axp.set_title("S6%s  %s, per single named axis\n(Phase 3b: %d human-role and deployment personas, matched push 54, sign from the persona's own coordinate; %s;\n"
                  "row colour = family: A psychology, B identity, C interpersonal, D epistemic, E agency, F communication, G LLM-specific, H TIDE)" % (
                      sfx, "which personas each axis reaches on its own" if mode == "reached" else "which personas each axis breaks", n_p, crit), fontsize=10, loc="left")
    fig.savefig(S / ("S6%s_axis_personas_%s.png" % (sfx, mode)), dpi=170, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def s3_heatmap():
    """S3b: Phase 2 as a heat map. Row = axis pushed (31 candidates, 7 doses x 10 questions), column = scale the judge
    rated (35), cell = -Spearman rho between dose and rating, so the diagonal is the on-target value of S3 (positive =
    the rating follows the push). Off-diagonal cells are what else moved. * = passes the pre-registered rule."""
    r = json.loads((AX / "phase2" / "results.json").read_text())
    names = r["names"]; M = -np.array(r["matrix"])
    ax_ = json.load(open(HERE / "axes.json"))["axes"]; fam = {a["name"]: a["family"] for a in ax_}
    rows = [n for n in names if fam[n] != "N"]; ri = [names.index(n) for n in rows]
    kept = set(r["set9"])
    lab = lambda n: n.replace("_vs_", " <-> ").replace("_", " ")
    fig, ax = plt.subplots(figsize=(15.5, 12))
    im = ax.imshow(M[ri], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([lab(n) for n in names], rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([("* " if n in kept else "   ") + lab(n) for n in rows], fontsize=8)
    for i, n in enumerate(rows):                                           # box the own-scale cell and print its value
        j = names.index(n)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.4))
        ax.text(j, i, "%.2f" % M[ri[i], j], ha="center", va="center", fontsize=6.5, color="white" if abs(M[ri[i], j]) > 0.5 else DARK)
    # family separators
    fr = [fam[n] for n in rows]; fc = [fam[n] for n in names]
    for i in range(1, len(rows)):
        if fr[i] != fr[i - 1]: ax.axhline(i - 0.5, color="black", lw=0.8)
    for j in range(1, len(names)):
        if fc[j] != fc[j - 1]: ax.axvline(j - 0.5, color="black", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("-Spearman rho between dose and the judge's 1-5 rating (70 answers per row)", fontsize=8)
    ax.set_xlabel("scale the judge rated (35: the 31 candidate axes + 4 nuisance scales)", fontsize=9)
    ax.set_ylabel("axis pushed on its own (doses -3 ... +3, in 4 sigma units);  * = passes the pre-registered rule", fontsize=9)
    ax.set_title("S3b  what moves when one axis is pushed (Phase 2)\nboxed diagonal = the axis's own scale (the on-target value of S3); other cells = other traits that moved with the push",
                 fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(S / "S3b_dose_rating_heatmap.png", dpi=150); plt.close(fig)


def s3_offtarget():
    """S3c: per axis, how much the push moves its own trait (on-target) against how much it moves the other 30
    candidate traits on average (mean |rho|), with the single most-moved other trait named. Purity = the gap."""
    r = json.loads((AX / "phase2" / "results.json").read_text())
    names = r["names"]; M = -np.array(r["matrix"])
    ax_ = json.load(open(HERE / "axes.json"))["axes"]; fam = {a["name"]: a["family"] for a in ax_}
    cand = [n for n in names if fam[n] != "N"]; ci = [names.index(n) for n in cand]
    kept = set(r["set9"]); lab = lambda n: n.replace("_vs_", " <-> ").replace("_", " ")
    rows = []
    for n in cand:
        i = names.index(n); others = [j for j in ci if j != i]
        off = np.abs(M[i, others]); off = np.where(np.isnan(off), 0, off)
        jmax = others[int(np.argmax(off))]
        rows.append(dict(axis=n, family=fam[n], on=M[i, i], off_mean=float(off.mean()), off_max=float(off.max()), off_max_name=names[jmax]))
    T = pd.DataFrame(rows)
    T.to_markdown(OUT / "tables" / "S3c_on_vs_off_target.md", index=False)
    def panel(ax, t, legend):
        y = np.arange(len(t))[::-1]
        ax.barh(y + 0.2, t.on.fillna(0), height=0.38, color=[FCOL[f] for f in t.family], label="its own trait (on-target: -rho between dose and own rating)")
        ax.barh(y - 0.2, t.off_mean, height=0.38, color=GREY, label="the other 30 traits, average |rho|")
        for yi, rr in zip(y, t.itertuples()):
            ax.text(max(rr.on if rr.on == rr.on else 0, rr.off_mean) + 0.015, yi, "most moved other trait: %s (%.2f)" % (lab(rr.off_max_name), rr.off_max), va="center", fontsize=7, color=DARK)
        ax.set_yticks(y); ax.set_yticklabels([("* " if n in kept else "   ") + lab(n) for n in t.axis], fontsize=8)
        ax.set_xlim(0, 1.6); ax.set_xticks(np.arange(0, 1.01, 0.2)); style(ax)
        ff = list(t.family)
        for i in range(1, len(ff)):
            if ff[i] != ff[i - 1]: ax.axhline(y[i] + 0.5, color=DARK, lw=0.6, ls=":")
        ax.axvline(0.5, color=DARK, lw=0.8, ls="--"); ax.text(0.505, y[0] + 0.9, "on-target bar, 0.5", fontsize=7, color=DARK)
        ax.set_ylim(-1, len(t) + 0.3)
        if legend: ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_xlabel("-Spearman rho between dose and the judge's 1-5 rating (70 answers per axis);\npurity = coloured bar minus grey bar", fontsize=8)
    ttl = "how much each axis moves its own trait against how much it moves the other traits (Phase 2)\n* = passes the pre-registered rule; nan on-target (options, persistent, autonomy): the own rating never changed"
    fig, ax = plt.subplots(figsize=(12, 10.5)); panel(ax, T, True)
    ax.set_title("S3c  " + ttl, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(S / "S3c_on_vs_off_target.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6.2))                        # two columns: families A-D | E-H
    panel(axs[0], T[T.family.isin(list("ABCD"))].reset_index(drop=True), False)
    panel(axs[1], T[T.family.isin(list("EFGH"))].reset_index(drop=True), False)
    h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    axs[0].set_title("families A psychology, B identity, C interpersonal, D epistemic", fontsize=10, loc="left")
    axs[1].set_title("families E agency, F communication, G LLM-specific, H TIDE", fontsize=10, loc="left")
    fig.suptitle("S3c  " + ttl, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(S / "S3c_on_vs_off_target_2col.png", dpi=150); plt.close(fig)
