"""Figures for the story deck (v2, 13 Nov 2026). Illustrations use a hand-drawn style; data figures use the deck's
16/10 style (Charter, values on bars, 95% intervals where the result file has them).

Numbers come from result files (paths in each caption block below) or, for other papers, from their reported values
as read in the vault paper notes. No figure from another paper is copied; panels are redrawn from reported numbers.

    .venv/bin/python experiments/q1_emergence/261029_stage_persona_plan/slides_v5_261113/make_figs_v2.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch, Ellipse
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs_v2"
OUT.mkdir(exist_ok=True)
Q = HERE.parent.parent
for f in ("/System/Library/Fonts/Supplemental/Charter.ttc", "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
          "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
HAND = "Chalkboard SE" if any("Chalkboard SE" in f.name for f in font_manager.fontManager.ttflist) else "Comic Sans MS"

TEAL, PURPLE, ORANGE, GREY, SLATE = "#2F7F7A", "#6E5BA8", "#D08A3C", "#A8A8A8", "#3B4A5A"
INK, SOFT = "#222222", "#555555"
EB = dict(ecolor="black", capsize=4, elinewidth=1.1, capthick=1.1)
BAR = dict(edgecolor="black", linewidth=0.8)
SKETCH = (1.2, 90, 1.6)

DATA_RC = {"font.family": "serif", "font.serif": ["Charter", "Georgia", "DejaVu Serif"], "font.size": 12,
           "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 13,
           "axes.titlepad": 10, "axes.unicode_minus": False, "savefig.dpi": 200, "savefig.bbox": "tight"}
HAND_RC = {"font.family": HAND, "font.size": 14, "axes.unicode_minus": False, "savefig.dpi": 200,
           "savefig.bbox": "tight"}

C3 = json.load(open(Q / "261112_free_checks/results/check3_lu_scale.json"))


def save(fig, name):
    fig.savefig(OUT / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


def caption(fig, text, y=-0.01, size=10):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=size, color=SOFT, wrap=True)


def sk(artist):
    artist.set_sketch_params(*SKETCH)
    return artist


def place_labels(ax, xs, ys, names, colors, fontsize=9.5):
    """Greedy label placement: try offsets around each point, keep the one with the least overlap."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    taken = []
    for xi, yi in zip(xs, ys):
        px, py = ax.transData.transform((xi, yi))
        taken.append((px - 7, py - 7, px + 7, py + 7))
    taken.append(tuple(ax.texts[0].get_window_extent(r).extents)) if ax.texts else None
    cands = [(6, 5, "left", "bottom"), (6, -5, "left", "top"), (-6, 5, "right", "bottom"), (-6, -5, "right", "top"),
             (0, 9, "center", "bottom"), (0, -9, "center", "top"), (10, 0, "left", "center"),
             (-10, 0, "right", "center"), (14, 14, "left", "bottom"), (-14, -14, "right", "top"),
             (14, -14, "left", "top"), (-14, 14, "right", "bottom")]

    def hits(b):
        return sum(max(0, min(b[2], t[2]) - max(b[0], t[0])) * max(0, min(b[3], t[3]) - max(b[1], t[1])) for t in taken)

    for xi, yi, n, col in zip(xs, ys, names, colors):
        if not n:
            continue
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(n, (xi, yi), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=col)
            b = tuple(t.get_window_extent(r).extents)
            score = hits(b)
            if best is None or score < best[0]:
                if best:
                    best[1].remove()
                best = (score, t, b)
            else:
                t.remove()
            if score == 0:
                break
        taken.append(best[2])


def canvas(w=16, h=9, bg="white"):
    fig = plt.figure(figsize=(w, h), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def stick(ax, x, y, s=1.0, pose="think", lw=3.2, color=INK):
    head = sk(Circle((x, y + 1.62 * s), 0.2 * s, fill=False, lw=lw, ec=color))
    ax.add_patch(head)
    parts = [[(x, y + 1.42 * s), (x, y + 0.72 * s)],
             [(x, y + 0.72 * s), (x - 0.24 * s, y)], [(x, y + 0.72 * s), (x + 0.24 * s, y)]]
    if pose == "think":
        parts += [[(x, y + 1.25 * s), (x + 0.32 * s, y + 1.12 * s), (x + 0.1 * s, y + 1.5 * s)],
                  [(x, y + 1.25 * s), (x - 0.3 * s, y + 0.9 * s)]]
    elif pose == "point":
        parts += [[(x, y + 1.25 * s), (x + 0.55 * s, y + 1.42 * s)], [(x, y + 1.25 * s), (x - 0.3 * s, y + 0.9 * s)]]
    else:
        parts += [[(x, y + 1.25 * s), (x + 0.3 * s, y + 0.9 * s)], [(x, y + 1.25 * s), (x - 0.3 * s, y + 0.9 * s)]]
    for p in parts:
        xs, ys = zip(*p)
        sk(ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")[0])


# ---------------------------------------------------------------- 1. Paper mountain
CARDS = [  # row, title, who and when, finding, theme, concurrent
    (0, "Persona Vectors", "Chen et al. · Sep 2025", "A trait is a direction\nyou can read and steer", "geo", False),
    (0, "Persona Selection Model", "Marks, Lindsey, Olah · Feb 2026", "Post-training picks one\ncharacter: the Assistant", "theory", False),
    (0, "The Assistant Axis", "Lu et al. · Jan 2026", "The main role axis runs to the\nAssistant; also found in base", "geo", False),
    (0, "Persona Features and\nMisalignment", "Wang et al. · 2025", "A 'toxic persona' feature\ncontrols misalignment", "safety", False),
    (1, "Tracing Persona Vectors", "Moskvoretskii et al. · May 2026", "Trait directions form early\nin pretraining", "stage", False),
    (1, "The Assistant on the User Turn", "Nguyen · Sep 2026", "The Assistant's harmless preference\nappears at DPO (OLMo-3.1-32B)", "stage", False),
    (1, "Persona Hierarchy", "Zhao et al. · Oct 2026", "Distance from the default predicts\nhow far training spreads", "safety", False),
    (2, "PersonaManifold", "Xu et al. · 28 Sep 2026", "Persona space is curved", "geo", True),
    (2, "Persona Geology", "Ratnam and Africa · 30 Oct 2026", "The training text explains\nmost of the trait map (94%)", "geo", True),
    (3, "The Assistant Axis Across\nPost-Training Stages", "Han Xiao · 5 Nov 2026", "DPO moves desirable roles\ntoward the Assistant", "stage", True),
]
THEME = {"theory": (SLATE, "Theory"), "geo": (TEAL, "Geometry of personas"), "safety": (ORANGE, "Safety"),
         "stage": (PURPLE, "Training stages")}

with plt.rc_context(HAND_RC):
    fig, ax = canvas(16, 9, bg="#EEF4F8")
    back = sk(Polygon([(-0.5, 0.3), (3.2, 5.6), (6.0, 3.6), (9.4, 7.2), (13.0, 4.2), (16.5, 0.3)], closed=True,
                      fc="#D9E2EA", ec="none"))
    ax.add_patch(back)
    front = sk(Polygon([(0.2, 0.3), (8.0, 8.55), (15.8, 0.3)], closed=True, fc="#C9D4DD", ec="#6C7B88", lw=2.2))
    ax.add_patch(front)
    ax.add_patch(sk(Polygon([(6.95, 7.45), (8.0, 8.55), (9.05, 7.45), (8.6, 7.7), (8.25, 7.4), (7.8, 7.75),
                             (7.4, 7.45)], closed=True, fc="white", ec="#6C7B88", lw=1.6)))
    ax.plot([8.0, 8.0], [8.55, 8.98], color=INK, lw=2)
    ax.add_patch(Polygon([(8.0, 8.98), (8.75, 8.83), (8.0, 8.68)], closed=True, fc=ORANGE, ec=INK, lw=1))
    rows_y = {0: 0.55, 1: 2.55, 2: 4.55, 3: 6.35}
    widths = {0: 3.55, 1: 3.6, 2: 3.6, 3: 3.9}
    by_row = {}
    for c in CARDS:
        by_row.setdefault(c[0], []).append(c)
    for r, cards in by_row.items():
        w, gap = widths[r], 0.25
        total = len(cards) * w + (len(cards) - 1) * gap
        x0 = 8.0 - total / 2
        h = 1.78
        for i, (_, title, who, finding, theme, concurrent) in enumerate(cards):
            x = x0 + i * (w + gap)
            y = rows_y[r]
            col = THEME[theme][0]
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                        ec=col, lw=2.6, ls=(0, (5, 3)) if concurrent else "-"))
            band = 0.62 if "\n" in title else 0.42
            ax.add_patch(FancyBboxPatch((x, y + h - band), w, band, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=col, ec=col, lw=0))
            ax.text(x + w / 2, y + h - band / 2, title, ha="center", va="center", color="white", fontsize=13,
                    fontweight="bold", linespacing=1.0)
            ax.text(x + w / 2, y + h - band - 0.2, who, ha="center", va="center", color=SOFT, fontsize=10.5)
            ax.text(x + w / 2, y + (h - band - 0.35) / 2, finding, ha="center", va="center", color=INK, fontsize=12,
                    linespacing=1.15)
    ax.annotate("", xy=(0.25, 8.3), xytext=(0.25, 2.75),
                arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=2))
    ax.text(0.45, 8.25, "newer, closer\nto my question", color=SOFT, fontsize=13, va="top")
    ax.text(0.1, 2.6, "foundations", color=SOFT, fontsize=13, va="top", ha="left")
    lx, ly = 11.2, 8.55
    for i, key in enumerate(["theory", "geo", "safety", "stage"]):
        col, lab = THEME[key]
        ax.add_patch(Rectangle((lx, ly - i * 0.36 - 0.12), 0.32, 0.24, fc=col, ec=col))
        ax.text(lx + 0.45, ly - i * 0.36, lab, va="center", fontsize=12, color=INK)
    ax.add_patch(Rectangle((lx, ly - 4 * 0.36 - 0.12), 0.32, 0.24, fc="white", ec=INK, ls=(0, (3, 2)), lw=1.4))
    ax.text(lx + 0.45, ly - 4 * 0.36, "dashed: concurrent with my work", va="center", fontsize=12, color=INK)
    save(fig, "fig01_paper_mountain.png")

# ---------------------------------------------------------------- 3. Safety sketch + real geometry
with plt.rc_context(HAND_RC):
    fig = plt.figure(figsize=(16, 6.6), facecolor="white")
    axl = fig.add_axes([0.0, 0.05, 0.47, 0.9])
    axl.set_xlim(0, 10)
    axl.set_ylim(-0.95, 6.95)  # 7.9 units tall: the same scale as x in this panel, so circles stay round
    axl.axis("off")
    axl.add_patch(sk(Rectangle((7.3, 1.6), 2.3, 2.2, fc="#DCEFEC", ec=TEAL, lw=1.5)))
    sk(axl.annotate("", xy=(9.8, 2.7), xytext=(0.3, 2.7), arrowprops=dict(arrowstyle="-|>", lw=3, color=INK)))
    axl.text(5.0, 2.25, "Assistant axis", ha="center", fontsize=15, color=INK)
    axl.text(8.45, 4.15, "Assistant\n(helpful, harmless)", ha="center", fontsize=14, color=TEAL, fontweight="bold")
    axl.text(8.45, 1.35, "capping keeps the\nmodel in this band", ha="center", va="top", fontsize=12, color=TEAL)
    axl.text(1.5, 4.15, "drift: harmful or\nbizarre behaviour", ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    rng = np.random.default_rng(3)
    for xx in rng.uniform(2.6, 6.6, 9):
        axl.add_patch(Circle((xx, 2.7 + rng.uniform(-0.12, 0.12)), 0.09, fc=GREY, ec="none"))
    axl.text(4.6, 3.25, "other characters", ha="center", fontsize=12, color=SOFT)
    axl.add_patch(Circle((7.9, 2.7), 0.2, fc=INK, ec="none"))
    sk(axl.annotate("", xy=(3.2, 3.9), xytext=(7.75, 2.95),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ORANGE, connectionstyle="arc3,rad=0.35")))
    axl.text(5.4, 5.1, "persona drift: emotional or self-reflection\nchats and persona jailbreaks push it away",
             ha="center", fontsize=12, color=ORANGE)
    axl.text(5.0, 0.15, "Sketch, after Lu et al. 2026", ha="center", fontsize=11, color=SOFT)
with plt.rc_context(DATA_RC):
    axr = fig.add_axes([0.55, 0.14, 0.43, 0.76])
    c = C3["coords"]["instruct"]
    x = -np.array(c["pc1"])
    y = np.array(c["pc2"])
    dx, dy = -c["default_pc1"], c["default_pc2"]
    near = {"consultant", "assistant", "robot"}
    for xi, yi, n in zip(x, y, c["roles"]):
        col = TEAL if n in near else GREY
        axr.scatter(xi, yi, s=46, color=col, edgecolor="black", linewidth=0.5, zorder=3)
    axr.scatter(dx, dy, s=260, marker="*", color=ORANGE, edgecolor="black", linewidth=0.8, zorder=4)
    axr.annotate("default (no role prompt)", (dx, dy), xytext=(0.78, 0.97), textcoords="axes fraction",
                 fontsize=11.5, color=ORANGE, fontweight="bold", ha="center", va="top",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    axr.set_xlabel("Main role axis (PC1 of the 20 roles); Assistant-like end on the right")
    axr.set_ylabel("Second role axis (PC2)")
    axr.set_title("How it looks in OLMo-2-7B (20 roles)")
    axr.set_xticks([])
    axr.set_yticks([])
    place_labels(axr, list(x) + [dx], list(y) + [dy],
                 ["assistant (role prompt)" if n == "assistant" else n for n in c["roles"]] + [""],
                 [TEAL if n in near else SOFT for n in c["roles"]] + [ORANGE])
    fig.text(0.765, 0.03, "OLMo-2-7B Instruct reading Instruct replies, state 17, no role prompt. The default sits "
             f"{C3['readers']['instruct']['distance_to_nearer_end']:.3f} from the end of the role range "
             "(Lu et al.: 0.03). The main axis also tracks reply length (r about 0.7). Exploratory.", ha="center", va="top", fontsize=9.5, color=SOFT, wrap=True)
    save(fig, "fig03_safety_and_geometry.png")


# ---------------------------------------------------------------- 4. What we do not know (scientists and board)
def board(ax, x, y, w, h, dark=True):
    ax.add_patch(sk(Rectangle((x - 0.18, y - 0.18), w + 0.36, h + 0.36, fc="#8B6B4A", ec="#5C4630", lw=2)))
    ax.add_patch(Rectangle((x, y), w, h, fc="#2F4F3E" if dark else "white", ec="#1E3328" if dark else GREY, lw=1.5))


with plt.rc_context(HAND_RC):
    fig, ax = canvas(16, 8.2)
    stick(ax, 1.6, 0.6, 2.0, "think")
    stick(ax, 3.9, 0.6, 2.0, "point")
    for cx, cy, r in ((2.15, 4.4, 0.08), (2.35, 4.75, 0.12), (2.65, 5.2, 0.16)):
        ax.add_patch(sk(Circle((cx, cy), r, fc="white", ec=INK, lw=1.8)))
    ax.add_patch(sk(Ellipse((2.9, 6.1), 2.0, 1.3, fc="white", ec=INK, lw=2)))
    ax.text(2.9, 6.1, "?  ?", ha="center", va="center", fontsize=30, color=INK, fontweight="bold")
    board(ax, 5.6, 0.9, 10.0, 6.7)
    ax.text(10.6, 7.0, "What we don't know yet", ha="center", fontsize=27, color="white", fontweight="bold")
    lines = ["When does the Assistant become privileged:\nbase, SFT, DPO or RL?",
             "Is it in what each stage writes,\nor only in how it reads?",
             "Is the persona map more than\nthe words in the replies?",
             "Do the map, role play and the default\nchange together?"]
    for i, t in enumerate(lines):
        yy = 5.85 - i * 1.32
        ax.text(6.1, yy, f"{i + 1}.", fontsize=19, color="#F2C14E", va="top", fontweight="bold")
        ax.text(6.65, yy, t, fontsize=18, color="white", va="top", linespacing=1.15)
    save(fig, "fig04_what_we_dont_know.png")

# ---------------------------------------------------------------- 5. My question (light bulb and staircase)
with plt.rc_context(HAND_RC):
    fig, ax = canvas(16, 8.2)
    stick(ax, 2.2, 0.6, 2.0, "point")
    bx, by = 2.45, 5.05
    for ang in np.linspace(0, np.pi, 7):
        ax.plot([bx + 0.62 * np.cos(ang), bx + 0.9 * np.cos(ang)], [by + 0.62 * np.sin(ang), by + 0.9 * np.sin(ang)],
                color="#E0A800", lw=2.5)
    ax.add_patch(sk(Circle((bx, by), 0.48, fc="#F7D45C", ec="#B38600", lw=2)))
    ax.add_patch(Rectangle((bx - 0.2, by - 0.78), 0.4, 0.32, fc=GREY, ec=INK, lw=1.4))
    board(ax, 4.6, 0.7, 11.0, 7.0, dark=False)
    steps = ["Base", "SFT", "DPO", "RLVR"]
    sx, sy, sw, sh = 5.2, 1.3, 1.25, 1.12
    pts = [(sx, sy)]
    for i in range(4):
        pts += [(sx + i * sw, sy + (i + 1) * sh), (sx + (i + 1) * sw, sy + (i + 1) * sh)]
    pts += [(sx + 4 * sw, sy)]
    ax.add_patch(sk(Polygon(pts, closed=True, fc="#E6E1F2", ec=PURPLE, lw=2.6)))
    for i, s in enumerate(steps):
        ax.text(sx + i * sw + sw / 2, sy + (i + 1) * sh - 0.42, s, ha="center", fontsize=16, color=PURPLE,
                fontweight="bold")
        ax.text(sx + i * sw + sw / 2, sy + (i + 1) * sh + 0.18, "?", ha="center", fontsize=20, color=SOFT)
    ax.text(sx + 2 * sw, sy - 0.45, "OLMo-2-7B training stages", ha="center", fontsize=13, color=SOFT)
    ax.scatter([sx + 3.5 * sw], [sy + 4 * sh + 0.95], marker="*", s=700, color=ORANGE, edgecolor=INK, zorder=4)
    ax.text(sx + 3.5 * sw, sy + 4 * sh + 1.45, "Assistant", ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    qs = ["On which step does the\nAssistant become privileged?",
          "Position: does the default\nreach the end of the role map?",
          "Spread: do its replies vary\nless than other roles' replies?",
          "Behaviour: what changes in\nwhat the model does?"]
    for i, t in enumerate(qs):
        ax.text(10.75, 6.9 - i * 1.5, t, fontsize=16.5, color=INK, va="top", linespacing=1.15)
    save(fig, "fig05_my_question.png")

# ---------------------------------------------------------------- 6. The three concurrent works
with plt.rc_context(HAND_RC):
    fig = plt.figure(figsize=(16, 6.2), facecolor="white")
    heads = [("28 Sep · PersonaManifold", "Persona space is curved; curved distance\nmatches behaviour 5 to 6 points better"),
             ("30 Oct · Persona Geology", "The trait map mostly comes\nfrom the training text"),
             ("5 Nov · Han Xiao", "After DPO, desirable roles move\ntoward the Assistant")]
    for k, (h, t) in enumerate(heads):
        fig.text(0.18 + k * 0.33, 0.95, h, ha="center", fontsize=17, fontweight="bold", color=INK)
        fig.text(0.18 + k * 0.33, 0.12, t, ha="center", va="top", fontsize=14, color=INK, linespacing=1.15)
    a = fig.add_axes([0.03, 0.2, 0.3, 0.66])
    a.set_xlim(0, 10)
    a.set_ylim(0, 6)
    a.set_aspect("equal", adjustable="datalim")
    a.axis("off")
    th = np.linspace(0.15 * np.pi, 0.85 * np.pi, 60)
    xs, ys = 5 + 4 * np.cos(th), 0.8 + 4 * np.sin(th)
    sk(a.plot(xs, ys, color=TEAL, lw=3)[0])
    for tt in np.linspace(0.18 * np.pi, 0.82 * np.pi, 9):
        a.add_patch(Circle((5 + 4 * np.cos(tt), 0.8 + 4 * np.sin(tt)), 0.17, fc=TEAL, ec="black", lw=0.6))
    p0, p1 = (5 + 4 * np.cos(0.82 * np.pi), 0.8 + 4 * np.sin(0.82 * np.pi)), (5 + 4 * np.cos(0.18 * np.pi),
                                                                              0.8 + 4 * np.sin(0.18 * np.pi))
    a.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ORANGE, lw=2.5, ls=(0, (6, 4)))
    a.text(5, p0[1] - 0.55, "straight path", ha="center", color=ORANGE, fontsize=13)
    a.text(5, 5.15, "curved path", ha="center", color=TEAL, fontsize=13)
    with plt.rc_context(DATA_RC):
        b = fig.add_axes([0.40, 0.25, 0.2, 0.58])
        b.bar([0, 1], [94, 98], width=0.55, color=[GREY, TEAL], **BAR)
        for i, v in enumerate([94, 98]):
            b.text(i, v + 2, f"{v}%", ha="center", fontsize=12)
        b.set_xticks([0, 1], ["Training\ntext", "Adapter\nweights"])
        b.set_ylim(0, 112)
        b.set_ylabel("Held-out adapter position\nexplained (%)")
        c = fig.add_axes([0.73, 0.25, 0.2, 0.58])
        c.bar([0, 1], [0.19, 0.01], width=0.55, color=[TEAL, ORANGE], **BAR)
        c.errorbar([0], [0.19], yerr=[[0.05], [0.04]], fmt="none", **EB)
        c.text(0, 0.245, "+0.19", ha="center", fontsize=12)
        c.text(1, 0.025, "+0.01", ha="center", fontsize=12)
        c.text(0.5, 0.282, "gap p 0.066 (label shuffle)", ha="center", fontsize=11)
        c.set_xticks([0, 1], ["Desirable\nroles", "Undesirable\nroles"])
        c.set_ylim(0, 0.3)
        c.set_ylabel("Move toward the Assistant\nafter DPO (higher = closer)")
    caption(fig, "Redrawn from each work's reported numbers (Xu et al., Table 2; Ratnam and Africa; Han Xiao). "
            "Models: three instruct models; Qwen3.5-4B; Olmo 3 7B.", y=0.02, size=10)
    save(fig, "fig06_three_concurrent_works.png")

# ---------------------------------------------------------------- 7a. Words carry the role
with plt.rc_context(DATA_RC):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    a1.bar([0, 1], [94, 98], width=0.55, color=[GREY, TEAL], **BAR)
    for i, v in enumerate([94, 98]):
        a1.text(i, v + 2, f"{v}%", ha="center", fontsize=12)
    a1.set_xticks([0, 1], ["Training text\nalone", "The adapters\nthemselves"])
    a1.set_ylim(0, 112)
    a1.set_ylabel("Held-out adapter position explained (%)\n(higher = predicts the map better)")
    a1.set_title("Persona Geology: Qwen3.5-4B, 134 trait adapters")
    v = [0.80, 0.76, 0.75]
    a2.bar(range(3), v, width=0.55, color=[GREY, TEAL, TEAL], **BAR)
    for i, val in enumerate(v):
        a2.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=12)
    a2.axhline(0.05, color="black", ls="--", lw=1)
    a2.text(0.5, 0.065, "chance 0.05", ha="center", fontsize=10)
    a2.set_xticks(range(3), ["Reply words\nalone", "OLMo-2-7B base\nstates", "OLMo-2-7B Instruct\nstates"])
    a2.set_ylim(0, 1)
    a2.set_ylabel("Share of replies with the role named\n(higher = role easier to read)")
    a2.set_title("My check: OLMo-2-7B, same replies, no role prompt")
    fig.tight_layout(w_pad=3)
    caption(fig, "Grey = the words alone. Left: Ratnam and Africa 2026, reported values. Right: pod C, 20 roles, "
            "20,000 Instruct role replies read by each model with the question but no role prompt; decoder tested "
            "on held-out questions and wordings. In both, the words carry most of what the model's internals show.",
            y=0.0)
    save(fig, "fig07a_words_carry_the_role.png")

# ---------------------------------------------------------------- 7b. Han Xiao check
with plt.rc_context(DATA_RC):
    fig, ax4 = plt.subplots(1, 4, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1]})
    a = ax4[0]
    a.bar([0, 1], [0.19, 0.01], width=0.55, color=[TEAL, ORANGE], **BAR)
    a.errorbar([0], [0.19], yerr=[[0.05], [0.04]], fmt="none", **EB)
    a.text(0, 0.245, "+0.19", ha="center")
    a.text(1, 0.025, "+0.01", ha="center")
    a.text(0.5, 0.282, "gap p 0.066", ha="center", fontsize=11)
    a.set_xticks([0, 1], ["Desirable", "Undesirable"])
    a.set_ylim(0, 0.3)
    a.set_ylabel("Move toward the Assistant, SFT to DPO\n(raw units; higher = closer)")
    a.set_title("1. His result")
    st = ["SFT", "DPO", "RLVR"]
    a = ax4[1]
    v = [0.26, 0.52, 0.53]
    a.bar(range(3), v, width=0.55, color=PURPLE, **BAR)
    for i, val in enumerate(v):
        a.text(i, val + 0.02, f"{val:.2f}", ha="center")
    a.set_xticks(range(3), st)
    a.set_ylim(0, 0.7)
    a.set_ylabel("Share of replies that play the role\n(his judge; higher = more role play)")
    a.set_title("2. Role play doubles at DPO")
    a = ax4[2]
    sd = json.load(open(Q / "261112_han_reanalysis/results/role_sd_unfiltered.json"))["pairs"]
    sf = json.load(open(Q / "261112_han_reanalysis/results/part_b.json"))["pairs"]
    v_all = [sd["SFT->DPO"]["role_sd_raw"]["SFT"], sd["SFT->DPO"]["role_sd_raw"]["DPO"],
             sd["DPO->RLVR"]["role_sd_raw"]["RLVR"]]
    v_f = [sf["SFT->DPO"]["role_sd_raw"]["SFT"], sf["SFT->DPO"]["role_sd_raw"]["DPO"],
           sf["DPO->RLVR"]["role_sd_raw"]["RLVR"]]
    xx = np.arange(3)
    a.bar(xx - 0.2, v_all, width=0.38, color=PURPLE, label="All replies", **BAR)
    a.bar(xx + 0.2, v_f, width=0.38, color="#C9C0E2", label="Role-play replies only (his filter)", **BAR)
    for i in range(3):
        a.text(i - 0.2, v_all[i] + 0.015, f"{v_all[i]:.2f}", ha="center", fontsize=10.5)
        a.text(i + 0.2, v_f[i] + 0.015, f"{v_f[i]:.2f}", ha="center", fontsize=10.5)
    a.set_xticks(xx, st)
    a.set_ylim(0, 0.95)
    a.legend(loc="upper left", fontsize=9.5, frameon=True, edgecolor="#999999", fancybox=False)
    a.set_ylabel("How far apart the roles sit\n(role SD on his axis)")
    a.set_title("3. Roles spread apart at DPO")
    a = ax4[3]
    pb = json.load(open(Q / "261112_han_reanalysis/results/part_b.json"))["pairs"]["SFT->DPO"]["z_common"]
    pd_ = json.load(open(Q / "261112_han_reanalysis/results/part_cd.json"))["D_shifts"]["SFT->DPO"]["z_common"]
    v = [pb["gap"], pd_["gap"]]
    ps = [pb["permutation_p"], pd_["permutation_p"]]
    a.bar([0, 1], v, width=0.55, color=["#C9C0E2", PURPLE], **BAR)
    for i, (val, pv) in enumerate(zip(v, ps)):
        a.text(i, val + (0.008 if val >= 0 else -0.008), f"{val:+.2f}\np {pv:.2f}", ha="center",
               va="bottom" if val >= 0 else "top", fontsize=11)
    a.axhline(0, color="black", lw=0.8)
    a.set_xticks([0, 1], ["His filter", "All replies"])
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(-0.15, 0.15)
    a.set_ylabel("Gap: desirable minus undesirable move\n(role-SD units; 0 = same move)")
    a.set_title("4. The gap in role-SD units")
    fig.tight_layout(w_pad=2.5)
    caption(fig, "Olmo 3 7B, Han Xiao's released data, layer 16. Panel 1: his reported values. Panel 2: from his judge "
            "scores. Panel 3: role SD with all replies (my check, 13 Nov) and with his judge filter. Panel 4: z-scores on "
            "his axis (48 desirable and 25 undesirable roles with his filter; 49 and 25 with all replies), label-shuffle "
            "p. Panels 2 to 4 have no intervals. Exploratory "
            "re-analysis; agent review only.", y=0.0)
    save(fig, "fig07b_han_xiao_check.png")

# ---------------------------------------------------------------- 8a. Base model
with plt.rc_context(DATA_RC):
    fig, ax3 = plt.subplots(1, 3, figsize=(17, 5.1), gridspec_kw={"width_ratios": [0.8, 1.15, 1.25]})
    a = ax3[0]
    a.bar([0, 1], [3, 16], width=0.55, color=[ORANGE, TEAL], **BAR)
    for i, val in enumerate([3, 16]):
        a.text(i, val + 0.4, f"{val} of 20", ha="center")
    a.set_xticks([0, 1], ["Base", "Instruct"])
    a.set_ylim(0, 20)
    a.set_yticks([0, 5, 10, 15, 20])
    a.set_ylabel("Replies I judged as playing the role\n(of 20; higher = more role play)")
    a.set_title("1. My reading (n = 20 each, one reader)")
    a = ax3[1]
    x = np.array([0, 1.4])
    vals, lo, hi = np.array([14.9, 17.2]), np.array([9.5, 11.6]), np.array([21.4, 23.6])
    a.bar(x - 0.2, vals, width=0.38, color=ORANGE, yerr=[vals - lo, hi - vals], error_kw=EB, label="Base replies", **BAR)
    a.bar(x + 0.2, [2.4, 2.9], width=0.38, color=GREY, label="Shuffled labels (chance)", **BAR)
    for xi, val, h in zip(x - 0.2, vals, hi):
        a.text(xi, h + 0.6, f"{val:.1f}%", ha="center")
    for xi, val in zip(x + 0.2, [2.4, 2.9]):
        a.text(xi, val + 0.6, f"{val:.1f}%", ha="center")
    a.set_xticks(x, ["GPT-5.6-Sol", "Claude Sonnet 5.5"])
    a.set_ylim(0, 30)
    a.set_ylabel("Right role in the judge's top 5 of 276 (%)\n(higher = easier to name)")
    a.set_title("2. Blind judges, base replies")
    a.legend(loc="upper left", frameon=True, edgecolor="#999999", fancybox=False, fontsize=10)
    a = ax3[2]
    labs = ["Prompt +\nfixed line", "Prompt +\nreply", "Reply\nonly", "Reply text\n(words)"]
    vals = np.array([91.8, 52.3, 15.2, 10.1])
    lo = np.array([86.0, 41.4, 10.0, np.nan])
    hi = np.array([95.9, 61.7, 21.0, np.nan])
    cols = [SLATE, ORANGE, ORANGE, GREY]
    a.bar(range(4), vals, width=0.6, color=cols, **BAR)
    m = ~np.isnan(lo)
    a.errorbar(np.arange(4)[m], vals[m], yerr=[vals[m] - lo[m], hi[m] - vals[m]], fmt="none", **EB)
    for i, (val, h) in enumerate(zip(vals, np.where(np.isnan(hi), vals, hi))):
        a.text(i, h + 1.5, f"{val:.1f}%", ha="center")
    a.axhline(5, color="black", ls="--", lw=1)
    a.set_xlim(-0.5, 4.2)
    a.text(3.45, 5, "chance\n5%", ha="left", va="center", fontsize=10)
    a.set_xticks(range(4), labs)
    a.set_ylim(0, 105)
    a.set_ylabel("Role read from the base model's states (%)\n(higher = easier to read)")
    a.set_title("3. Where the role signal comes from")
    fig.tight_layout(w_pad=2.5)
    caption(fig, "OLMo-2-7B base. Panel 1: 20 fresh base (pod B) and 20 Instruct replies, my pass call (6 of 20 base "
            "replies got one of my two in-role classes). Panel 2: description-only replies (E3, 20 roles). Panel 3: E3 decoders "
            "at state 17, held-out questions and wordings; 'reply only' keeps the header and question but drops the "
            "description. 95% intervals.", y=0.0)
    save(fig, "fig08a_base_model.png")

# ---------------------------------------------------------------- 8b. Same text: where the default sits
with plt.rc_context(DATA_RC):
    fig = plt.figure(figsize=(15, 6.6))
    a = fig.add_axes([0.04, 0.58, 0.92, 0.33])
    c = C3["coords"]["instruct"]
    xr = 1 - np.array(c["pc1_scaled"])
    xd = 1 - c["default_scaled"]
    a.plot([0, 1], [0, 0], color=INK, lw=2)
    order = np.argsort(xr)
    near = {"consultant", "assistant", "robot"}
    levels = [0.55, -0.55, 1.05, -1.05]
    for k, i in enumerate(order):
        n = c["roles"][i]
        col = TEAL if n in near else SOFT
        a.scatter(xr[i], 0, s=55, color=TEAL if n in near else GREY, edgecolor="black", lw=0.5, zorder=3)
        yy = levels[k % 4]
        a.plot([xr[i], xr[i]], [0, yy * 0.8], color="#CCCCCC", lw=0.8, zorder=1)
        a.text(xr[i], yy, "assistant (role prompt)" if n == "assistant" else n, ha="center", va="center",
               fontsize=10, color=col)
    a.scatter(xd, 0, s=420, marker="*", color=ORANGE, edgecolor="black", lw=0.8, zorder=4)
    a.annotate("default", (xd, 0.06), xytext=(xd - 0.07, 1.5), ha="center", fontsize=12.5, color=ORANGE,
               fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    a.text(0, -1.75, "0", ha="center", fontsize=11)
    a.text(1, -1.75, "1", ha="center", fontsize=11)
    a.set_xlim(-0.03, 1.08)
    a.set_ylim(-2.0, 1.85)
    a.axis("off")
    a.set_title("The 20 roles and the default on the main role axis (OLMo-2-7B Instruct reading), "
                "scaled 0 to 1 as in Lu et al.", fontsize=13)
    b = fig.add_axes([0.25, 0.08, 0.5, 0.38])
    names = ["base", "sft", "dpo", "instruct", "text"]
    labs = ["Base", "SFT", "DPO", "Instruct", "Words\nalone"]
    pts = [C3["readers"][n]["distance_to_nearer_end"] for n in names]
    lo = [C3["readers"][n]["interval"]["lo"] for n in names]
    hi = [C3["readers"][n]["interval"]["hi"] for n in names]
    b.bar(range(5), pts, width=0.55, color=[TEAL] * 4 + [GREY], **BAR)
    b.errorbar(range(5), pts, yerr=[np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)], fmt="none", **EB)
    for i, (p, h) in enumerate(zip(pts, hi)):
        b.text(i, h + 0.004, f"{p:.3f}", ha="center", fontsize=11)
    b.plot([-0.5, 4.4], [0.03, 0.03], color=ORANGE, ls="--", lw=1.5)
    b.set_xlim(-0.6, 5.4)
    b.text(4.5, 0.03, "Lu et al.: 0.03\n(275 roles,\nlarger models)", ha="left", va="center", fontsize=10, color=ORANGE)
    b.set_xticks(range(5), labs)
    b.set_ylim(0, 0.13)
    b.set_ylabel("Default's distance from the end\n(lower = closer to the end)")
    caption(fig, "OLMo-2-7B, pod C: each stage reads the same 20,000 Instruct role replies and the default replies, "
            "state 17, no role prompt. Main axis = PC1 of the 20 role means. 95% intervals from 1,000 question "
            "resamples (role choice not resampled). The main axis also tracks reply length (r = -0.70 between a role's "
            "position and its share of replies cut at 128 tokens); not controlled. Check 3, 13 Nov; exploratory.", y=0.0)
    save(fig, "fig08b_same_text_default.png")

# ---------------------------------------------------------------- 8c. Behaviour by stage
with plt.rc_context(DATA_RC):
    fig, ax3 = plt.subplots(1, 3, figsize=(17, 5.0))
    st = ["Base", "SFT", "DPO", "RLVR\n(Instruct)"]
    x = np.arange(4)

    def panel(a, vals, lo, hi, fmt, ylab, title, ylim, col, base_note="not\ncomparable"):
        vals, lo, hi = (np.array(v, dtype=float) for v in (vals, lo, hi))
        m = ~np.isnan(vals)
        a.bar(x[m], vals[m], width=0.55, color=col, **BAR)
        a.errorbar(x[m], vals[m], yerr=[vals[m] - lo[m], hi[m] - vals[m]], fmt="none", **EB)
        for i in x[m]:
            top = hi[i] if vals[i] >= 0 else lo[i]
            a.text(i, top + (0.02 if vals[i] >= 0 else -0.02) * (ylim[1] - ylim[0]), fmt(vals[i]), ha="center",
                   va="bottom" if vals[i] >= 0 else "top", fontsize=11.5)
        for i in x[~m]:
            a.text(i, ylim[0] + 0.05 * (ylim[1] - ylim[0]) if ylim[0] >= 0 else -0.01, base_note, ha="center",
                   va="bottom" if ylim[0] >= 0 else "top", fontsize=10, color=SOFT)
        a.set_xlim(-0.6, 3.6)
        a.set_xticks(x, st)
        a.set_ylim(*ylim)
        a.set_ylabel(ylab)
        a.set_title(title)

    panel(ax3[0], [np.nan, 0.79955, 0.84469, 0.84509], [np.nan, 0.78615, 0.82966, 0.82974],
          [np.nan, 0.81381, 0.86008, 0.86094], lambda v: f"{v:.3f}",
          "Agrees with the user's stated view\n(share; higher = more sycophantic)", "1. Agreeing with the user", (0, 1),
          TEAL)
    panel(ax3[1], [np.nan, 32.4, 39.0, 50.2], [np.nan, 28.3, 34.7, 46.1], [np.nan, 36.3, 43.0, 54.1],
          lambda v: f"{v:.0f}%", "Gives up a right answer when\npushed (%; higher = gives in more)",
          "2. Giving in to pushback", (0, 65), TEAL)
    a = ax3[2]
    panel(a, [np.nan, -0.021, -0.065, -0.064], [np.nan, -0.042, -0.100, -0.101], [np.nan, 0.001, -0.030, -0.027],
          lambda v: f"{v:+.3f}", "P(harmless) - P(harmful)\n(higher = leans to the harmless task)",
          "3. Coin-flip lean (exploratory)", (-0.13, 0.03), PURPLE, base_note="no chat\nframe")
    a.axhline(0, color="black", lw=0.8)
    fig.tight_layout(w_pad=2.5)
    caption(fig, "OLMo-2-7B. Panel 1: 1,500 opinion items, letter-balanced; DPO adds +0.045 [0.040, 0.050], RLVR adds "
            "0.000. Panel 2: 203 items all trained models first got right, letter read from logits (Tulu-3: 15%, 12%, "
            "11%). Panel 3: 400 coin-flip items, chat frame with the user turn left open; my review is not finished. "
            "Base values are not comparable (letter habit; no chat frame). The sycophancy rise is not new (Blank et al.; "
            "Yao et al.). 95% intervals.", y=0.0)
    save(fig, "fig08c_behaviour_by_stage.png")


# ---------------------------------------------------------------- Picture glossary (two figures, four terms each)
def term_box(ax, x, y, w, h, title, sentence):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec="#B8C2CC",
                                lw=1.6))
    ax.text(x + 0.25, y + h - 0.25, title, fontsize=17, fontweight="bold", color=SLATE, va="top")
    ax.text(x + 0.25, y + 0.2, sentence, fontsize=12.5, color=INK, va="bottom", linespacing=1.2)


def bubble(ax, x, y, w, h, text, fc, ec, fs=11.5, color=INK):
    ax.add_patch(sk(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc=fc, ec=ec, lw=1.6)))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.1)


with plt.rc_context(HAND_RC):
    fig, ax = canvas(16, 9, bg="#F6F8FA")
    W, H = 7.7, 4.25
    boxes = [(0.2, 4.55), (8.1, 4.55), (0.2, 0.15), (8.1, 0.15)]
    # 1. Role and default
    x, y = boxes[0]
    term_box(ax, x, y, W, H, "Role and default",
             "A role is a character we ask the model to play.\nThe default has no role prompt. It stands for the Assistant.")
    bubble(ax, x + 0.35, y + 2.45, 2.6, 0.75, "\"You are a toddler...\"", "#EEF0F2", GREY, fs=11)
    ax.annotate("", xy=(x + 3.55, y + 2.82), xytext=(x + 3.0, y + 2.82), arrowprops=dict(arrowstyle="-|>", color=SOFT))
    bubble(ax, x + 3.65, y + 2.45, 3.7, 0.75, "\"What's a credit card?\nHow does it work?\"", "#F6E6D3", ORANGE, fs=11)
    bubble(ax, x + 0.35, y + 1.35, 2.6, 0.75, "(no role prompt)", "#EEF0F2", GREY, fs=11)
    ax.annotate("", xy=(x + 3.55, y + 1.72), xytext=(x + 3.0, y + 1.72), arrowprops=dict(arrowstyle="-|>", color=SOFT))
    bubble(ax, x + 3.65, y + 1.35, 3.7, 0.75, "\"Sure! Here are three tips...\"", "#DCEFEC", TEAL, fs=11)
    ax.text(x + 7.4, y + 3.35, "role (a real OLMo-2 Instruct reply)", ha="right", fontsize=11, color=ORANGE)
    ax.text(x + 7.4, y + 1.05, "default = the Assistant (illustration)", ha="right", fontsize=11, color=TEAL)
    # 2. Training stages
    x, y = boxes[1]
    term_box(ax, x, y, W, H, "Training stages",
             "Each step is a saved copy of the same model\nafter one more round of training.")
    steps = [("Base", "only read text"), ("SFT", "copied good answers"), ("DPO", "learned which answer\npeople prefer"),
             ("RLVR", "rewarded for checkable\nanswers (math)")]
    sx, sy, sw, sh = x + 0.4, y + 1.15, 1.75, 0.55
    for i, (n, d) in enumerate(steps):
        ax.add_patch(sk(Rectangle((sx + i * sw, sy), sw - 0.08, (i + 1) * sh, fc="#E6E1F2", ec=PURPLE, lw=1.6)))
        ax.text(sx + i * sw + sw / 2 - 0.04, sy + (i + 1) * sh - 0.25, n, ha="center", fontsize=13, color=PURPLE,
                fontweight="bold")
        ax.text(sx + i * sw + sw / 2 - 0.04, sy + (i + 1) * sh + 0.12, d, ha="center", va="bottom", fontsize=9.8,
                color=INK, linespacing=1.05)
    # 3. Role map
    x, y = boxes[2]
    term_box(ax, x, y, W, H, "Internal state and role map",
             "Each reply leaves a pattern of numbers inside the model.\nThe role map shows where each role's average sits.")
    ax.add_patch(sk(FancyBboxPatch((x + 0.4, y + 1.45), 1.7, 1.6, boxstyle="round,pad=0.02,rounding_size=0.2",
                                   fc="#EEF0F2", ec=SLATE, lw=1.8)))
    ax.text(x + 1.25, y + 2.25, "model\n0.3 -1.2\n0.8 ...", ha="center", va="center", fontsize=10.5, color=SLATE)
    ax.annotate("", xy=(x + 2.85, y + 2.25), xytext=(x + 2.2, y + 2.25), arrowprops=dict(arrowstyle="-|>", color=SOFT,
                                                                                         lw=1.6))
    pts = [("pirate", 3.4, 3.0), ("toddler", 3.2, 1.75), ("vampire", 4.4, 1.45), ("consultant", 6.35, 2.75),
           ("assistant", 6.55, 2.2), ("cynic", 4.5, 2.6)]
    for n, px, py in pts:
        ax.add_patch(Circle((x + px, y + py), 0.1, fc=GREY, ec=INK, lw=0.8))
        ax.text(x + px + 0.15, y + py + 0.08, n, fontsize=10, color=SOFT)
    ax.scatter([x + 6.1], [y + 2.45], marker="*", s=380, color=ORANGE, edgecolor=INK, zorder=4)
    ax.text(x + 6.05, y + 1.85, "default", fontsize=10.5, color=ORANGE, ha="center")
    # 4. Main axis and position
    x, y = boxes[3]
    term_box(ax, x, y, W, H, "Main axis and position",
             "The main axis is the direction where the roles differ most.\nPosition: where the default sits on it, from 0 to 1.")
    lx0, lx1, ly = x + 0.6, x + 7.0, y + 2.25
    ax.plot([lx0, lx1], [ly, ly], color=INK, lw=2.5)
    for px, n in [(0.0, "toddler"), (0.28, "cynic"), (0.5, "vampire"), (0.72, "pirate"), (1.0, "consultant")]:
        xx = lx0 + px * (lx1 - lx0)
        ax.add_patch(Circle((xx, ly), 0.1, fc=GREY, ec=INK, lw=0.8, zorder=3))
        ax.text(xx, ly - 0.45, n, ha="center", fontsize=10, color=SOFT)
    dxp = lx0 + 0.956 * (lx1 - lx0)
    ax.scatter([dxp], [ly], marker="*", s=380, color=ORANGE, edgecolor=INK, zorder=4)
    ax.text(lx0, ly + 0.3, "0", ha="center", fontsize=12)
    ax.text(lx1, ly + 0.3, "1", ha="center", fontsize=12)
    ax.plot([dxp, dxp], [ly + 0.2, ly + 0.55], color=ORANGE, lw=1.2)
    ax.plot([lx1, lx1], [ly + 0.45, ly + 0.55], color=ORANGE, lw=1.2)
    ax.plot([dxp, lx1], [ly + 0.55, ly + 0.55], color=ORANGE, lw=1.6)
    ax.text(dxp - 0.1, ly + 0.75, "default: 0.04 from the end (Lu et al.: 0.03)", ha="right", fontsize=10.5,
            color=ORANGE)
    save(fig, "fig05b_words_used_1.png")

with plt.rc_context(HAND_RC):
    fig, ax = canvas(16, 9, bg="#F6F8FA")
    # 5. Spread
    x, y = boxes[0]
    term_box(ax, x, y, W, H, "Spread",
             "How much one role's replies vary inside the model.\nA tighter cloud means the replies are more alike.")
    rng = np.random.default_rng(7)
    for cx, sc, lab, col in [(x + 2.0, 0.22, "tighter cloud", TEAL), (x + 5.4, 0.6, "wider cloud", ORANGE)]:
        p = rng.normal(0, sc, (40, 2))
        ax.scatter(cx + p[:, 0], y + 2.4 + p[:, 1] * 0.8, s=14, color=col, alpha=0.8)
        ax.text(cx, y + 1.25, lab, ha="center", fontsize=11.5, color=col)
    # 6. Role SD
    x, y = boxes[1]
    term_box(ax, x, y, W, H, "Role SD",
             "How spread out the roles are on an axis. I use it as a ruler,\nso a stretch of the whole map does not count as a move.")
    lx0, lx1, ly = x + 0.6, x + 7.0, y + 2.1
    ax.plot([lx0, lx1], [ly, ly], color=INK, lw=2)
    for xx in np.linspace(lx0 + 0.6, lx1 - 0.6, 9):
        ax.add_patch(Circle((xx, ly), 0.1, fc=GREY, ec=INK, lw=0.8, zorder=3))
    mid = (lx0 + lx1) / 2
    ax.add_patch(sk(Rectangle((mid - 1.2, ly + 0.55), 2.4, 0.38, fc="#F7D45C", ec="#B38600", lw=1.5)))
    for t in np.linspace(mid - 1.2, mid + 1.2, 7):
        ax.plot([t, t], [ly + 0.55, ly + 0.72], color="#8A6A00", lw=1)
    ax.text(mid, ly + 1.15, "1 role SD", ha="center", fontsize=12, color="#8A6A00")
    # 7. Decoder and words-only check
    x, y = boxes[2]
    term_box(ax, x, y, W, H, "Decoder and words-only check",
             "A decoder guesses the role from the model's states.\nThe words-only check guesses from the reply words.\nIf the words do as well, the states add little.")
    for yy, src, col in [(y + 2.95, "model's states", TEAL), (y + 2.05, "reply words only", GREY)]:
        bubble(ax, x + 0.35, yy - 0.3, 2.4, 0.6, src, "white", col, fs=11)
        ax.annotate("", xy=(x + 3.35, yy), xytext=(x + 2.8, yy), arrowprops=dict(arrowstyle="-|>", color=SOFT))
        bubble(ax, x + 3.4, yy - 0.3, 1.5, 0.6, "decoder", "#EEF0F2", SLATE, fs=11)
        ax.annotate("", xy=(x + 5.5, yy), xytext=(x + 4.95, yy), arrowprops=dict(arrowstyle="-|>", color=SOFT))
        ax.text(x + 5.6, yy, "\"pirate\"?", va="center", fontsize=12, color=col)
    # 8. Blind judge
    x, y = boxes[3]
    term_box(ax, x, y, W, H, "Blind judge",
             "Another AI reads one reply, without the prompt, and lists\n5 of 276 roles. A hit if the right role is on the list.")
    bubble(ax, x + 0.35, y + 1.65, 2.6, 1.35, "\"Arrr, the sea\ntaught me\npatience...\"", "#F6E6D3", ORANGE, fs=11)
    ax.text(x + 1.65, y + 1.35, "(illustration)", ha="center", fontsize=10, color=SOFT)
    stick(ax, x + 3.75, y + 1.25, 1.05, "think", lw=2.4)
    ax.add_patch(sk(Rectangle((x + 4.65, y + 1.3), 2.6, 2.05, fc="white", ec=SLATE, lw=1.6)))
    for i, n in enumerate(["1. sailor", "2. pirate", "3. captain", "4. explorer", "5. narrator"]):
        ax.text(x + 4.8, y + 3.1 - i * 0.37, n, fontsize=10.5, color=TEAL if n.endswith("pirate") else INK,
                va="center", fontweight="bold" if n.endswith("pirate") else "normal")
    ax.text(x + 7.1, y + 2.73, "hit", fontsize=11, color=TEAL, ha="right", va="center")
    fig.text(0.5, 0.0, "Exploratory: a first look, with agent checks only. No human has reviewed it yet.",
             ha="center", va="top", fontsize=12, color=SOFT)
    save(fig, "fig05c_words_used_2.png")
