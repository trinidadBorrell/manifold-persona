"""One-slide diagram of the E3 work: prompts, model, answers, and the four checks with their results.
Numbers are copied from the write-up (writeup_part1_2026-10-16.md), which links each to its result file.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/review/writeup_part1_2026-10-16_figs/make_workflow_fig.py
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


def box(x, y, w, h, title, body, edge, fill="white", tsize=12.5, bsize=10.5, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.6, ec=edge, fc=fill))
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=tsize, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.2, y + h - (6.0 if title else 2.4), body, fontsize=bsize, color=INK, va="top", linespacing=1.35,
            family="monospace" if mono else None)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555555"))


ax.text(50, 99, "How the answers were made and checked", ha="center", va="top", fontsize=18, fontweight="bold", color=INK)

# 1. prompts
box(1, 43, 22, 48, "1. Four kinds of prompt",
    "Plain interview text,\nno chat format.\n\n"
    "a) role description only\n"
    "b) + 2 example answers\n    from the same role\n"
    "c) + 2 example answers\n    from another role\n"
    "d) + neutral notes of\n    the same length", GREY, fill=GREY, bsize=10)
box(1, 1, 22, 38, "Example prompt (b)",
    "About the guest: a pirate\ncaptain who has sailed\nthe seven seas ...\n\n"
    "Interviewer: <question>\nGuest: Arrr, matey! ...\n\n"
    "Interviewer: <question>\nGuest: When inspiration\nstrikes, I ...\n\n"
    "Interviewer: <test question>\nGuest:", GREY, bsize=7.6, mono=True)

# 2. model and answers
arrow(23.8, 67, 26.5, 67)
box(27, 46, 16, 42, "2. OLMo-2 7B base",
    "Writes the guest's\nanswer.\n\n20 roles x 5 wordings\nx 18 questions =\n1,800 answers per\nprompt, plus a second\nsample for (a) and (b).", TEAL, bsize=9.8)

# 3. the four checks
ax.text(48, 93, "3. Checks", fontsize=13.5, fontweight="bold", color=INK, va="top")
ax.text(76, 93, "Results", fontsize=13.5, fontweight="bold", color=INK, va="top")
checks = [
    ("Answer test (Ivan's test)",
     "GPT-5.6-Sol and Claude Sonnet 5.5\nread only the answer and guess\nthe role from all 276.",
     "Description only: 15-17%\nSame-role examples: 38-42%\nOther role's examples: judges\nname that other role (25-30%)", TEAL),
    ("Inside the model",
     "Layer 17 signals while it answers.\nA simple classifier guesses which\nof the 20 roles it was given.",
     "With the prompt: 52%\nPrompt hidden: 15%\nPrompt alone: 92%\n(chance 5%)", TEAL),
    ("Broken answers",
     "My own labels on 80 answers vs\nthree judges: GPT-6.1-Sol,\nOpus 5.5 and Jev.",
     "Judges say 5-11% are broken.\nNo judge agreed with me\nwell enough (bar: kappa 0.6).", ORANGE),
    ("Role-play score",
     "My own 0-3 scores on 50 answers\nvs Jev.",
     "Jev agreement: 0.41\n(bar 0.6). Not usable yet.", ORANGE),
]
ys = [68, 46, 24, 2]
for (title, how, res, col), y in zip(checks, ys):
    arrow(43.8, 67, 47, y + 9.5)
    box(48, y, 24, 19, title, how, col, tsize=11.5, bsize=9.3)
    arrow(72.8, y + 9.5, 75.3, y + 9.5)
    box(76, y, 23, 19, "", res, col, bsize=9.6)
    
fig.savefig(OUT / "fig7_workflow.png", bbox_inches="tight")
print("wrote fig7_workflow.png")
