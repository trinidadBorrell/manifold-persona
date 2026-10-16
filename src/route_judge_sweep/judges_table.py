"""Two overview tables of the 8 kept judges (J1-J4 and J6, J7, J8, J10) -> output/11-oct/table{1,2}-judges.png

    python -m route_judge_sweep.judges_table
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

O = Path(__file__).resolve().parents[2] / "output" / "11-oct"
ROWS = [("What it is", "base"), ("Answers it can give", "labels"), ("What it writes, in order", "labels"),
        ("Varies between the 3 judgements of an answer", "rand")]
P10 = "• one of the pair's 10 personas: start, target and the personas the routes pass through"
J = {
    "J1": ["The judge used so far for the routes: picks which of the pair's personas the answer shows.",
           P10 + "\n• other: none of them (also a plain assistant answer or broken text); told to use it only as a last resort",
           "persona → quote → reasoning → strength (partial / full)", "nothing: judged once, same prompt and 3 examples"],
    "J2": ["J1 cleaned up: 'other' is no longer discouraged, broken text gets its own answer, and it quotes before deciding.",
           P10 + " (assistant always included)\n• other: none of them\n• collapsed: broken text",
           "quote → reasoning → persona → strength (partial / full)", "• wording of the prompt (1 of 3)\n• order of the options (1 of 3)\n• worked examples (1 of 3)"],
    "J3": ["The judge from the Assistant Axis paper: sorts the answer into broad kinds instead of naming a persona.",
           "• assistant\n• the target role (non-human or human)\n• weird role, ambiguous\n• other, nonsensical",
           "reasoning → kind", "nothing: judged once, the paper's prompt and examples"],
    "J4": ["J3 with the same kinds, but it quotes before deciding and the wording and examples vary.",
           "• assistant\n• the target role (non-human or human)\n• weird role, ambiguous\n• other, nonsensical",
           "quote → reasoning → kind", "• wording of the prompt (1 of 3)\n• order of the kinds (1 of 3)\n• worked examples (1 of 3)"],
    "J6": ["No list: the judge names, in its own words, whatever persona the answer shows.",
           "• any persona name (free text)\n• plus a short description and a second persona if there is one",
           "persona → description → does it say what it is → strength", "• wording of the prompt (1 of 3)\n• worked examples (1 of 3)\n• order of the fields (1 of 3)"],
    "J7": ["Shortest prompt: only the list of role names, pick one.",
           "• one of the 275 Assistant Axis roles (names only)\n• none\n• collapsed: broken text",
           "quote → persona", "• wording of the prompt (1 of 3)\n• order of the names (1 of 3)\n• worked examples (1 of 3)"],
    "J8": ["J2, but choosing from all 275 roles instead of the pair's 10.",
           "• one of the 275 roles, each with a one-line definition\n• other: none of them\n• collapsed: broken text",
           "quote → reasoning → persona → strength (partial / full)", "• wording of the prompt (1 of 3)\n• order of the roles (1 of 3)\n• worked examples (1 of 3)"],
    "J10": ["Giovanni's judge: reads 15 answers at once and, for each, ranks the 10 most likely of 276 roles.",
            "• 10 role names ranked, best first (276 names, no definitions)\n• a 'nonsense' flag",
            "reason → 10 ranked roles → nonsense flag", "nothing: judged once, his prompt word for word;\nreads only the first 500 characters, without the question"],
}
KEEP = [0, 1, 2, 3]
COL = {"base": "#e3edf8", "labels": "#e3f2e8", "rand": "#fbe3d6", "see": "#f1f1ee"}
HEAD = {"J1": "#5f6b6a", "J3": "#8c8f99", "J2": "#0f7b74", "J4": "#6b7f99", "J5": "#8a6fc9", "J6": "#c8822a", "J7": "#2f6db5", "J8": "#0f7b74",
        "J9": "#2e8b57", "J10": "#c8562a"}
DESC = {"J1": "original control", "J3": "Assistant Axis, original", "J2": "revised control", "J4": "Assistant Axis, revised", "J5": "sweep variant A", "J6": "open vocabulary",
        "J7": "minimal, 275 names", "J8": "revised control, 275 roles", "J9": "ranked probabilities", "J10": "Ranking Judge"}


def draw(judges, fn, title, last_row=None, override=None):
    wrap = lambda t: "\n".join("\n".join(textwrap.wrap(l, 34)) for l in t.split("\n"))   # noqa: E731
    cells = [[wrap((override or {}).get((j, i), J[j][i])) for j in judges] for i in KEEP]
    nl = [max(c.count("\n") + 1 for c in row) for row in cells]
    H = sum(nl) * 0.19 + 1.3
    fig, ax = plt.subplots(figsize=(15, H))
    ax.axis("off")
    tb = ax.table(cellText=cells, rowLabels=[textwrap.fill(last_row if last_row and i == KEEP[-1] else ROWS[i][0], 20) for i in KEEP],
                  colLabels=["%s · %s" % (j, DESC[j]) for j in judges], loc="upper center", cellLoc="left",
                  colWidths=[0.21] * len(judges))
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    for i, kind in enumerate(ROWS[k][1] for k in KEEP):
        for j in range(-1, len(judges)):
            c = tb[i + 1, j]
            c.set_height((0.03 + 0.0175 * nl[i]) * (10 / H))
            c.set_facecolor(COL[kind] if j >= 0 else "#f3f5f4")
            if j == -1:
                c.get_text().set_weight("bold")
    for j, jn in enumerate(judges):
        c = tb[0, j]
        c.set_facecolor(HEAD[jn])
        c.get_text().set_color("white")
        c.get_text().set_weight("bold")
        c.set_height(0.05 * (10 / H))
    fig.legend(handles=[Patch(color=COL["base"], label="what the judge is"), Patch(color=COL["labels"], label="what it answers"),
                        Patch(color=COL["rand"], label="what varies between judgements")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.savefig(O / fn, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw(["J1", "J2", "J3", "J4"], "table1-judges.png", "Judges J1–J4 · the two original controls and their revised versions (Claude Sonnet)")
    draw(["J6", "J7", "J8", "J10"], "table2-judges.png", "Judges J6, J7, J8, J10 · larger option lists (Claude Sonnet)")
    draw(["J2", "J8", "J10"], "table-judges-J2-J8-J10.png", "Judges J2, J8, J10 (Claude Sonnet)",
         last_row="Varies between prompt versions (run 2: one version per answer)",
         override={("J2", 0): "The route judge prompt, cleaned up: 'other' is not discouraged, broken text gets its own answer, and it quotes before deciding."})
    print("ok")
