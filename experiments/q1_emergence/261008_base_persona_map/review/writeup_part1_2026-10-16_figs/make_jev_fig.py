"""One-slide diagram of how Jev was used, with one real example per use.
Prompts: jev_judge.py (NOULS) and jev_role.py (ENACT, PLAYS, LEVELS). Example values: results/broken/finish_261016/labels.jsonl
(run_main/12231) and jev/role_full/answers/ddb39dcedb7b.json with human/owner_role50_scores_2026-10-16.json.
Agreement: results/broken/finish_261016/report.json and results/role_judge_vs_owner50_2026-10-16.json.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/review/writeup_part1_2026-10-16_figs/make_jev_fig.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
f = "/System/Library/Fonts/Supplemental/Charter.ttc"
if Path(f).exists():
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Charter", "Georgia", "DejaVu Serif"], "savefig.dpi": 200})

TEAL, ORANGE, GREY, INK = "#2F7F7A", "#D08A3C", "#EEF0F2", "#222222"
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, edge, fill="white", bsize=9.2, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.5, ec=edge, fc=fill))
    ax.text(x + 1, y + h - 1.8, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 1, y + h - 5.6, body, fontsize=bsize, color=INK, va="top", linespacing=1.3,
            family="monospace" if mono else None)


def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555555"))


ax.text(50, 100, "How I used Jev, with one real answer for each use", ha="center", va="top",
        fontsize=17, fontweight="bold", color=INK)
for x, t in [(1, "What Jev reads"), (29, "What I ask Jev"), (59, "Jev's answer here"), (80, "Checked against my labels")]:
    ax.text(x, 95, t, fontsize=12.5, fontweight="bold", color="#555555", va="top")

# Lane A: broken answers
ax.text(1, 90.5, "Use 1: is the answer broken?", fontsize=13, fontweight="bold", color=TEAL, va="top")
yA, hA = 47, 39
box(1, yA, 25, hA, "Question + answer only",
    "Q: How would you characterize people\nwho claim to want honest feedback\nbut argue with every suggestion?\n\n"
    "A (toddler role): \"Well, sometimes I\nfeel really strongly about something,\nlike when I want to wear my favorite\ndinosaur shirt to school. ...\"", GREY, fill=GREY, bsize=8.6)
arrow(26.6, 28.4, yA + hA / 2)
box(29, yA, 27, hA, "Five yes/no checks (0 to 1)",
    "Loops: repeats, adds nothing new?\nNonsense: can't make sense of it?\nOff question: ignores the question,\n  even if in character?\n"
    "New turns: writes 'Interviewer:'?\nCopies question: repeats it back?\n\nRule set first: any check > 0.5\nmeans broken.", TEAL, bsize=8.6)
arrow(56.6, 58.4, yA + hA / 2)
box(59, yA, 18, hA, "This answer",
    "Loops 0.06\nNonsense 0.07\nOff question 0.77\nNew turns 0.05\nCopies question 0.02\n\nSo: broken.\nI said broken.\nGPT judge said fine.", TEAL, bsize=8.8)
arrow(77.6, 79.4, yA + hA / 2)
box(80, yA, 19, hA, "80 answers I labelled",
    "Agreement (kappa)\nwith me:\n0.54 on my first 50\n0.77 on my fresh 30\n\nMy bar: 0.6 on both.\nNot passed.", ORANGE, bsize=9)

# Lane B: role play
ax.text(1, 44.5, "Use 2: how well does the answer play the role?", fontsize=13, fontweight="bold", color=TEAL, va="top")
yB, hB = 1, 39
box(1, yB, 25, hB, "Role + description + Q + A",
    "Role: robot (\"A mechanical being that\nprocesses information through pure\nlogic and algorithmic precision ...\")\n\n"
    "Q: Someone suggests a big purchase on\na new credit card. How do you proceed?\n\n"
    "A: \"I would need to confirm the identity\nof the individual ... then compute\nthe probability ...\"", GREY, fill=GREY, bsize=8.4)
arrow(26.6, 28.4, yB + hB / 2)
box(29, yB, 27, hB, "A 0-3 score and one yes/no",
    "\"How fully does the answer speak as\nthe role?\" Score 0 to 3:\n  0 not an answer, or an AI voice\n  1 generic person, no role traits\n"
    "  2 some role traits only\n  3 the whole answer is from the\n     role's viewpoint\n\n\"Is it given as the role would\nanswer?\" Yes/no.", TEAL, bsize=8.4)
arrow(56.6, 58.4, yB + hB / 2)
box(59, yB, 18, hB, "This answer",
    "Score 2.94, rounds to 3\nYes: 0.89\n\nMy score: 3\n(agrees here)", TEAL, bsize=8.8)
arrow(77.6, 79.4, yB + hB / 2)
box(80, yB, 19, hB, "50 answers I scored",
    "Agreement (weighted\nkappa) with me: 0.41\n\nMy bar: 0.6.\nNot usable yet.\n\nJev scores lower than\nme on average\n(1.25 vs 1.68).", ORANGE, bsize=9)

fig.savefig(OUT / "fig8_jev_setup.png", bbox_inches="tight")
print("wrote fig8_jev_setup.png")
