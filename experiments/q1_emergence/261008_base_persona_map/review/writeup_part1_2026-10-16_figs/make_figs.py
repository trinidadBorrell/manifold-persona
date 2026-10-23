"""Figures for the 16/10 write-up and slides. One question per figure, the finding as the title,
the number on every bar, 95% intervals where the result file has them. Reads saved result files only.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/review/writeup_part1_2026-10-16_figs/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parent
EXP = OUT.parent.parent
E0 = EXP.parent / "261005_base_incontext_elicitation"
for f in ("/System/Library/Fonts/Supplemental/Charter.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)

TEAL, PURPLE, ORANGE, GREY = "#2F7F7A", "#6E5BA8", "#D08A3C", "#A8A8A8"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Charter", "Georgia", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.titlepad": 14, "axes.unicode_minus": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "grid.color": "#E2E2E2"})
EB = dict(ecolor="black", capsize=4, elinewidth=1.1, capthick=1.1)
BAR = dict(edgecolor="black", linewidth=0.8)

N = json.load(open(EXP / "results/naming_v2.json"))
B = json.load(open(EXP / "results/broken/finish_261016/report.json"))
E = json.load(open(E0 / "results.json"))
JUDGES = [("gpt", "GPT-5.6-Sol judge", TEAL), ("claude", "Claude Sonnet 5.5 judge", PURPLE)]
SHIFTS = ["shift6", "shift10", "shift14"]


def hit5(cell, judge, run="run_main"):
    return N[f"{run}/{cell}"]["results"]["none"][judge]["all_replies"]


def bars(ax, x, vals, lo, hi, color, label=None, width=0.36, pad=0.8):
    vals = np.asarray(vals)
    err = None
    if lo is not None:
        lo, hi = np.asarray(lo), np.asarray(hi)
        err = [np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)]
    ax.bar(x, vals, width=width, color=color, label=label, yerr=err, error_kw=EB, **BAR)
    tops = hi if lo is not None else vals
    for xi, v, t in zip(x, vals, tops):
        ax.text(xi, t + pad, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5)


def finish(fig, ax, name, caption, legend_cols=0, leg_y=-0.14):
    if legend_cols:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, leg_y), ncol=legend_cols, frameon=True,
                  edgecolor="#999999", fancybox=False, fontsize=11)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [a.get_tightbbox(r) for a in fig.axes] + [a.get_legend().get_window_extent(r) for a in fig.axes if a.get_legend()]
    y0 = min(b.transformed(inv).y0 for b in boxes)
    fig.text(0.5, y0 - 0.03, caption, ha="center", va="top", fontsize=10, color="#555555")
    fig.savefig(OUT / name)
    # slide copy: the slide title carries the finding
    ax.set_title("")
    (OUT / "slide").mkdir(exist_ok=True)
    fig.savefig(OUT / "slide" / name)
    plt.close(fig)
    print("wrote", name)


# Figure 1: which prompt lets the judges name the role?
conds = [("desc", "Description only\nfails the rule"), ("ex2", "Description +\n2 same-role examples\npasses on the first sample"),
         ("ex2_filler", "Description +\nneutral text, same length\nfails the rule")]
x = np.arange(len(conds))
fig, ax = plt.subplots(figsize=(8.5, 5))
nulls = []
for k, (j, lab, col) in enumerate(JUDGES):
    d = [hit5(c, j) for c, _ in conds]
    nulls += [v["null_mean"] * 100 for v in d]
    bars(ax, x + (k - .5) * .38, [v["macro_hit5"] * 100 for v in d], [v["ci95"][0] * 100 for v in d],
         [v["ci95"][1] * 100 for v in d], col, lab, width=.38)
ax.axhline(20, ls="--", color="black", lw=1.1, label="My pass bar: lower end of the interval at 20%")
ax.axhspan(min(nulls), max(nulls), color=GREY, alpha=.35, lw=0,
           label=f"Level from shuffled role labels ({min(nulls):.1f}-{max(nulls):.1f}%)")
ax.set_xticks(x, [l for _, l in conds])
ax.set_ylim(0, 65)
ax.grid(axis="y")
ax.set_ylabel("Replies where the judge's top 5 guesses\ninclude the right role (%)")
ax.set_title("Observed naming rises from 15-17% to 38-42% with same-role examples")
finish(fig, ax, "fig1_naming_by_prompt.png",
       "OLMo-2 7B base model, first sample of replies. 1,800 replies per prompt (20 roles x 5 wordings x 18 questions).\n"
       "Average over the 20 roles. Error bars: 95% intervals over roles and questions.", 2,
       leg_y=-0.30)

# Figure 2: whose role do the judges name when the examples come from another role?
fig, ax = plt.subplots(figsize=(8.5, 5))
xj = np.arange(2)
series = [("Description only: described role", GREY), ("Other role's examples: described role", TEAL),
          ("Other role's examples: example role", ORANGE)]
for s, (lab, col) in enumerate(series):
    vals, lo, hi, dots = [], [], [], []
    for j, _, _ in JUDGES:
        if s == 0:
            v = hit5("desc", j)
            vals.append(v["macro_hit5"] * 100); lo.append(v["ci95"][0] * 100); hi.append(v["ci95"][1] * 100); dots.append([])
        elif s == 1:
            per = [hit5(f"ex2_other/{sh}", j)["macro_hit5"] * 100 for sh in SHIFTS]
            vals.append(np.mean(per)); dots.append(per)
        else:
            per = [N[f"run_main/ex2_other/{sh}"]["donor_hit5"][j] * 100 for sh in SHIFTS]
            vals.append(np.mean(per)); dots.append(per)
    xs = xj + (s - 1) * .27
    if s == 0:
        bars(ax, xs, vals, lo, hi, col, lab, width=.27)
    else:
        bars(ax, xs, vals, None, None, col, lab, width=.27, pad=2.2)
        for xi, ds in zip(xs, dots):
            ax.scatter([xi] * len(ds), ds, s=14, color="black", zorder=3)
ax.set_xticks(xj, [l for _, l, _ in JUDGES])
ax.set_ylim(0, 40)
ax.grid(axis="y")
ax.set_ylabel("Replies where the judge's top 5\nguesses include that role (%)")
ax.set_title("The judges name the example role more often than the described role")
finish(fig, ax, "fig2_example_role_wins.png",
       "Other-role prompt: the role's description + 2 examples of another role (3 different example roles per role, 1,800 replies each).\n"
       "Other-role bars: mean of the 3 sets; dots: each set. The example-role rate has no interval yet;\n"
       "the gap between example role and described role is 20-26 points in every set, every 95% interval above 0.",
       3, leg_y=-0.12)

# Figure 3: which roles carry the ex2 result?
roles = list(hit5("ex2", "gpt")["role_hit5"])
g = hit5("ex2", "gpt")["role_hit5"]
c = hit5("ex2", "claude")["role_hit5"]
order = sorted(roles, key=lambda r: (g[r] + c[r]) / 2)
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(8.5, 9))
for k, (j, lab, col) in enumerate(JUDGES):
    d = g if j == "gpt" else c
    v = [d[r] * 100 for r in order]
    yy = y + (k - .5) * .4
    ax.barh(yy, v, height=.4, color=col, label=lab, **BAR)
    for yi, vi in zip(yy, v):
        ax.text(vi + .8, yi, f"{vi:.0f}%", va="center", fontsize=9)
ax.set_yticks(y, order)
ax.set_xlim(0, 110)
ax.grid(axis="x")
ax.set_xlabel("Replies where the judge's top 5 guesses include the right role (%)")
ax.set_title("Some roles are named almost every time, others almost never")
finish(fig, ax, "fig3_roles_ex2.png",
       "Description + 2 same-role examples, first sample, 90 replies per role. Sorted by the mean of the two judges.\n"
       "No per-role interval: the 90 replies share wordings and questions.",
       2, leg_y=-0.07)

# Figure 4: how many replies are broken?
cells = [("run_main/desc", "Description\nonly"), ("run_main/ex2", "Same-role\nexamples"), ("run_main/ex2_filler", "Neutral\ntext"),
         ("run_main/ex2_other", "Other-role\nexamples\n(1 example role)"), ("run_rep/desc", "Description\nonly\nsecond sample"), ("run_rep/ex2", "Same-role\nexamples\nsecond sample")]
xb = np.arange(len(cells))
fig, ax = plt.subplots(figsize=(10.5, 4.8))
R = B["rates"]
bars(ax, xb, [R[k]["final_rate"] * 100 for k, _ in cells], [R[k]["final_rate_crossed_ci95"][0] * 100 for k, _ in cells],
     [R[k]["final_rate_crossed_ci95"][1] * 100 for k, _ in cells], TEAL, "Broken, 16/10 definition (provisional)", width=.55)
ax.set_xticks(xb, [l for _, l in cells], fontsize=10.5)
ax.set_ylim(0, 22)
ax.grid(axis="y")
ax.set_ylabel("Broken replies (%), lower = better")
ax.set_title("A provisional judge marks 5-11% of replies as broken")
finish(fig, ax, "fig4_broken_replies.png",
       "1,800 replies per bar. My own label on 80 replies, a GPT-6.1-Sol judge on the rest. No judge reached my agreement bar\n"
       "(kappa 0.6) on both of my label sets, so these rates are provisional. Error bars: 95% intervals, assuming the labels are right.", 2, leg_y=-0.2)

# Figure 5 (context): E0, OLMo-2-0425-1B base vs final on the same plain-text prompt
sets = [("set1", "Example set 1"), ("set2", "Example set 2"), ("set3", "Example set 3"), ("zero", "No examples")]
xe = np.arange(len(sets) + 1)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for k, (m, lab, col) in enumerate([("base", "1B base", TEAL), ("final", "1B final (Instruct), same prompt", PURPLE)]):
    d = [E[f"{m}/{s}_neutral"] for s, _ in sets]
    bars(ax, xe[:-1] + (k - .5) * .38, [v["top5"] * 100 for v in d], [v["lo"] * 100 for v in d],
         [v["hi"] * 100 for v in d], col, lab, width=.38)
nat = E["final/native"]
bars(ax, [xe[-1] + .19], [nat["top5"] * 100], [nat["lo"] * 100], [nat["hi"] * 100], ORANGE,
     "1B final, own chat template", width=.38)
ax.axhline(2, ls=":", color="black", lw=1.1, label="Chance (about 2%)")
ax.set_xticks(xe, [l for _, l in sets] + ["Chat template"])
ax.set_ylim(0, 70)
ax.grid(axis="y")
ax.set_ylabel("Replies where the judge's top 5\nguesses include the right role (%)")
ax.set_title("The 1B model's naming rate depends on the example set")
finish(fig, ax, "fig5_e0_1b.png",
       "OLMo-2 1B, 60 roles x 2 questions = 120 replies per bar, GPT-5.6-Sol judge picking from 276 roles.\n"
       "Each example set shows three made-up guests answering in character. Error bars: 95% intervals over roles.", 2,
       leg_y=-0.12)

# Figure 6: how much of the internal-state decoding survives when the prompt is removed?
C = json.load(open(EXP / "results/part2/part2_context_run_main.json"))
groups = [("desc", "Description only"), ("ex2", "Description +\n2 same-role examples"), ("ex2_filler", "Description +\nneutral text")]
series = [("R", "Reply, read with its full prompt", TEAL), ("c2", "Same reply, description and examples removed", ORANGE),
          ("c3", "Prompt alone, then one fixed neutral sentence", PURPLE), ("text_joint_tfidf_length_stop", "Reply words, length and stop reason only", GREY)]
xg = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
w = .2
for k, (key, lab, col) in enumerate(series):
    d = [C[g][key] for g, _ in groups]
    bars(ax, xg + (k - 1.5) * w, [v["acc_mean_50"] * 100 for v in d], [v["ci95"][0] * 100 for v in d],
         [v["ci95"][1] * 100 for v in d], col, lab, width=w, pad=1.0)
ax.axhline(10, ls="--", color="black", lw=1.1, label="My floor for the lower end (10%)")
ax.axhline(5, ls=":", color="black", lw=1.1, label="Chance (5%)")
ax.set_xticks(xg, [l for _, l in groups])
ax.set_ylim(0, 112)
ax.grid(axis="y")
ax.set_ylabel("Held-out replies where the classifier\npicks the right role of 20 (%)")
ax.set_title("With the prompt removed, the same replies give much lower role decoding")
for t in ax.texts:
    t.set_fontsize(8.5)
finish(fig, ax, "fig6_internal_states_prompt_removed.png",
       "OLMo-2 7B base. Coloured bars: layer 17 activations averaged over the reply (or over the fixed sentence).\n"
       "Grey bars: reply words, length and stop reason. A simple classifier is trained on 3 wordings x 9 questions and\n"
       "tested on the other 2 wordings x 9 questions, 50 splits. Error bars: 95% intervals. First sample of replies.", 2,
       leg_y=-0.16)
