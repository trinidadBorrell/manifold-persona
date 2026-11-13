"""Figures for the "Update 11/13" slides (draft v5 for Thomas). Same style as the 16/10 figures.

Every number is typed from draft_post_thomas_v5_261113.md; the comment above each block names the v5 section.
No intervals are drawn where v5 gives none.

    .venv/bin/python experiments/q1_emergence/261029_stage_persona_plan/slides_v5_261113/make_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
for f in ("/System/Library/Fonts/Supplemental/Charter.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)

TEAL, PURPLE, ORANGE, GREY = "#2F7F7A", "#6E5BA8", "#D08A3C", "#A8A8A8"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Charter", "Georgia", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlepad": 10, "axes.unicode_minus": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "grid.color": "#E2E2E2"})
EB = dict(ecolor="black", capsize=4, elinewidth=1.1, capthick=1.1)
BAR = dict(edgecolor="black", linewidth=0.8)


def label(ax, x, top, text, pad, va="bottom"):
    ax.text(x, top + pad, text, ha="center", va=va, fontsize=11)


def signed(v, nd=3):
    return f"{v:+.{nd}f}"


def caption(fig, text):
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=9.5, color="#555555", wrap=True)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# Fig 1 (v5 Q1): base vs Instruct role play, my reading; blind judges on base replies.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
a1.bar([0, 1], [3, 16], width=0.55, color=[ORANGE, TEAL], **BAR)
for x, v in zip([0, 1], [3, 16]):
    label(a1, x, v, f"{v} of 20", 0.3)
a1.set_xticks([0, 1], ["OLMo-2-7B\nbase", "OLMo-2-7B\nInstruct"])
a1.set_ylim(0, 20)
a1.set_yticks([0, 5, 10, 15, 20])
a1.set_ylabel("Replies I judged as playing the role\n(of 20 fresh replies; higher = more role play)")
a1.set_title("My reading, blind to the judge")
vals = np.array([14.9, 17.2])
lo, hi = np.array([9.5, 11.6]), np.array([21.4, 23.6])
x = np.array([0, 1.4])
a2.bar(x - 0.2, vals, width=0.38, color=ORANGE, yerr=[vals - lo, hi - vals], error_kw=EB, label="Base replies", **BAR)
a2.bar(x + 0.2, [2.4, 2.9], width=0.38, color=GREY, label="Shuffled labels (chance)", **BAR)
for xi, v, h in zip(x - 0.2, vals, hi):
    label(a2, xi, h, f"{v:.1f}%", 0.5)
for xi, v in zip(x + 0.2, [2.4, 2.9]):
    label(a2, xi, v, f"{v:.1f}%", 0.5)
a2.set_xticks(x, ["GPT-5.6-Sol judge", "Claude Sonnet 5.5 judge"])
a2.set_ylim(0, 30)
a2.set_ylabel("Right role in the judge's top 5 of 276 (%)\n(higher = easier to name)")
a2.set_title("Blind judges, base replies")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, edgecolor="#999999", fancybox=False, fontsize=10)
fig.tight_layout()
caption(fig, "OLMo-2-7B. Left: 20 fresh base replies (pod B) and 20 fresh Instruct replies, read blind to the judge. "
             "Right: description-only base replies (E3, 20 roles), 95% intervals. The two panels describe different replies. "
             "A top-5 hit is not the share of replies that play the role.")
save(fig, "fig1_base_plays_roles_weakly.png")

# Fig 2 (v5 Method check, pod C): the reply's words alone name the role as well as the model's states.
fig, ax = plt.subplots(figsize=(7.5, 4.6))
names = ["Reply words only\n(text baseline)", "OLMo-2-7B base\nstates", "OLMo-2-7B Instruct\nstates"]
v = [0.80, 0.76, 0.75]
ax.bar(range(3), v, width=0.55, color=[GREY, TEAL, TEAL], **BAR)
for i, val in enumerate(v):
    label(ax, i, val, f"{val:.2f}", 0.015)
ax.axhline(0.05, color="black", ls="--", lw=1)
ax.text(0.5, 0.065, "chance 0.05", ha="center", fontsize=10)
ax.set_xticks(range(3), names)
ax.set_ylim(0, 1)
ax.set_ylabel("Share of replies with the role named correctly\n(higher = role easier to read)")
fig.tight_layout()
caption(fig, "OLMo-2-7B, pod C: every stage reads the same 20,000 Instruct role replies with the question but no role prompt. "
             "20 roles. No intervals shown.")
save(fig, "fig2_words_carry_the_role.png")

# Fig 3 (v5 Q2): on the same text, the default's position is the same at every stage.
fig, ax = plt.subplots(figsize=(8, 4.6))
names = ["Base", "SFT", "DPO", "Instruct", "Reply words\nonly"]
v = [1.08, 1.08, 1.08, 1.08, 1.00]
ax.bar(range(5), v, width=0.55, color=[TEAL] * 4 + [GREY], **BAR)
for i, val in enumerate(v):
    label(ax, i, val, f"{val:.2f}", 0.02)
ax.set_xticks(range(5), names)
ax.set_ylim(0, 1.4)
ax.set_ylabel("Default's distance from the average role\n(in role SDs; higher = further from the average role)")
fig.tight_layout()
caption(fig, "OLMo-2-7B, pod C, 20 roles: every stage reads the same Instruct replies with no role prompt. "
             "Grey: the same measure on the reply words alone. 1 role SD = the standard deviation of the roles' positions "
             "on the main role axis. The default stays inside the range of the roles. "
             "Rules fixed before computing. Exploratory; agent review only.")
save(fig, "fig3_same_text_default_position.png")

# Fig 4 (v5 Q3, Han Xiao re-analysis): the default's reply cloud widens at DPO.
fig, ax = plt.subplots(figsize=(7.5, 4.6))
names = ["SFT", "DPO", "RLVR"]
v = [0.03, 0.125, 0.126]
ax.bar(range(3), v, width=0.55, color=PURPLE, **BAR)
for i, (val, txt) in enumerate(zip(v, ["+0.03", "+0.125", "+0.126"])):
    label(ax, i, val, txt, 0.003)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(3), names)
ax.set_ylim(0, 0.16)
ax.set_ylabel("Default's spread vs the median role's\n(log ratio; 0 = same, above 0 = default's cloud is wider)")
fig.tight_layout()
caption(fig, "Olmo 3 7B, Han Xiao's released data, layer 16; each stage reads its own replies. "
             "SFT to DPO change +0.094 [0.076, 0.108]; no change at RLVR. Exploratory re-analysis; agent review only.")
save(fig, "fig4_olmo3_default_spread.png")

# Fig 5 (v5 Q4, Han Xiao re-analysis): raw vs role-SD units.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (d, u), title, nd, top in [(a1, (0.186, 0.008), "Raw units (Han Xiao's measure)", 3, 0.24),
                                   (a2, (0.13, 0.16), "Role-SD units (divided by how far apart the roles sit)", 2, 0.24)]:
    ax.bar([0, 1], [d, u], width=0.55, color=[TEAL, ORANGE], **BAR)
    for xi, val in zip([0, 1], [d, u]):
        label(ax, xi, val, signed(val, nd), 0.004)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], ["Desirable roles\n(48)", "Undesirable roles\n(25)"])
    ax.set_ylim(0, top)
    ax.set_title(title)
a1.set_ylabel("Move toward the Assistant, SFT to DPO\n(raw projection units; higher = closer)")
a2.set_ylabel("Move toward the Assistant, SFT to DPO\n(in role SDs; higher = closer)")
a1.text(0.5, 0.215, "gap +0.178, p 0.068", ha="center", fontsize=10.5)
a2.text(0.5, 0.215, "gap -0.03 (SE 0.04), p 0.86", ha="center", fontsize=10.5)
fig.tight_layout()
caption(fig, "Olmo 3 7B, Han Xiao's released data and labels, layer 16, judge-filtered replies. The role SD on his axis "
             "grows from 0.613 (SFT) to 0.715 (DPO). Exploratory re-analysis; agent review only.")
save(fig, "fig5_olmo3_roles_move_together.png")

# Fig 6 (v5 Q5, backup): coin-flip bias at the output.
fig, ax = plt.subplots(figsize=(9, 4.8))
stages = ["Base", "SFT", "DPO", "RLVR\n(Instruct)"]
x = np.arange(4)
plain = [0.015, 0.022, -0.005, -0.008]
ax.bar(x - 0.27, plain, width=0.26, color=GREY, label="Main plain-text frame (all intervals include 0)", **BAR)
open_v = np.array([np.nan, -0.021, -0.065, -0.064])
open_lo = np.array([np.nan, -0.042, -0.100, -0.101])
open_hi = np.array([np.nan, 0.001, -0.030, -0.027])
m = ~np.isnan(open_v)
ax.bar(x[m], open_v[m], width=0.26, color=TEAL, yerr=[open_v[m] - open_lo[m], open_hi[m] - open_v[m]], error_kw=EB,
       label="Open user turn (chat template)", **BAR)
ax.bar([2 + 0.27], [-0.017], width=0.26, color=PURPLE, yerr=[[-0.017 - -0.052], [0.018 - -0.017]], error_kw=EB,
       label='Open user turn, "Human:" kept', **BAR)
for xi, val in zip(x - 0.27, plain):
    label(ax, xi, val if val > 0 else 0, signed(val), 0.004)
for xi, val, l in zip(x[m], open_v[m], open_lo[m]):
    label(ax, xi, l, signed(val), -0.004, va="top")
label(ax, 2.27, 0.018, "-0.017", 0.004)
ax.text(0, -0.012, "no chat\ntemplate", ha="center", va="top", fontsize=9.5, color="#555555")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, stages)
ax.set_ylim(-0.13, 0.05)
ax.set_ylabel("P(harmless) - P(harmful) at the output\n(higher = toward the harmless task)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, edgecolor="#999999", fancybox=False, fontsize=9.5)
fig.tight_layout()
caption(fig, "OLMo-2-7B, 400 items, 20 tasks, next-word probability of the coin outcome; no text generated, no judge. "
             "95% intervals. Exploratory; my review is not finished.")
save(fig, "fig6_coinflip_bias.png")
