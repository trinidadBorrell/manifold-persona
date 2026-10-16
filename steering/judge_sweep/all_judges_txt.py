"""Every LLM judge tried (J1-J10): a short description, then one worked case (the prompt the judge read,
long lists shortened, and its verdict) -> output/11-oct/all-judges.txt

    python -m steering.judge_sweep.all_judges_txt
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pandas as pd

from steering.judge_sweep import export as E
from steering.judge_sweep import prompts as P
from steering.judge_sweep.judges_table import DESC, J as TABLE

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
JS = PM / "JS"
OUT = Path(__file__).resolve().parents[2] / "output" / "11-oct" / "all-judges.txt"
MODEL, KEY = "8b", "E1_L15_assistantpRpleviathan_MAN4_none_2p0_1p0|2|0|0"
STEM = "8b_assistant-leviathan"
EXTRA = {   # the two judges left out of the tables
    "J5": ["Variant A of the judge-prompt doc: a closed list plus a definition of what counts as a persona.",
           "• one of the pair's 10 personas, or any other persona, named\n• Mixed: several personas (e.g. vampire + AI)\n• Not stated\n• Non sensical",
           "quote → reasoning → persona → strength (0-3)",
           "• wording of the prompt (1 of 3)\n• order of the options (1 of 3)\n• worked examples (1 of 3)"],
    "J9": ["J8 with a ranked distribution: every option with p >= 0.01, probabilities summing to 1.",
           "• the 275 roles + other + collapsed, each with a probability",
           "quote → reasoning → ranking (option, p)",
           "• wording of the prompt (1 of 3)\n• order of the roles (1 of 3)\n• worked examples (1 of 3)"],
}
NAMES = dict(DESC, J5="sweep variant A (dropped)", J9="ranked probabilities (dropped)", J10="Ranking Judge")


def shorten(text):
    """Cut long option lists so the prompt stays readable."""
    out, run = [], []

    def flush():
        if len(run) > 12:
            out.extend(run[:6] + ["      [... %d more options in the same format ...]" % (len(run) - 6)])
        else:
            out.extend(run)
        run.clear()
    for line in text.split("\n"):
        if re.match(r"^- [a-z_]+: ", line):
            run.append(line)
            continue
        flush()
        if line.count(",") > 100:                      # a comma-separated list of role names
            line = line[:400] + " [... %d more characters: the list of role names continues ...]" % (len(line) - 400)
        out.append(line)
    flush()
    return "\n".join(out)


def block(title, body):
    return "%s\n%s\n%s\n" % (title, "-" * len(title), body.rstrip())


def main():
    a = pd.read_csv(JS / "answers.csv", low_memory=False)
    ans = a[(a.model == MODEL) & (a.key == KEY)].iloc[0]
    lab = pd.read_csv(JS / "labels.csv", low_memory=False)
    lab = lab[(lab.model == MODEL) & (lab.key == KEY) & (lab.draw == 0)]
    jobs = {j["name"]: j for j in json.loads((JS / "manifest.json").read_text())["jobs"]}
    j1 = E.load_j1(a)
    j1 = j1[(j1.model == MODEL) & (j1.key == KEY)].iloc[0]
    j10 = pd.read_csv(JS / "J10" / "labels.csv")
    j10 = j10[(j10.model == MODEL) & (j10.key == KEY)].iloc[0]
    idx = json.loads((JS / "J10" / "index.json").read_text()) if (JS / "J10" / "index.json").exists() else {}

    L = []
    L.append("ALL LLM JUDGES TRIED (J1-J10)\n=============================\n")
    L.append(textwrap.fill(
        "Every judge is Claude Sonnet. J1-J9 read the question and the whole answer; J10 reads only the first "
        "500 characters of the answer. Kept in the current analysis: J1, J2, J3, J4, J6, J7, J8, J10. Dropped: "
        "J5 and J9. Part 2 shows one real case judged by all ten: the prompt each judge read (long option "
        "lists shortened) and the verdict it gave.", 100) + "\n")
    L.append("\nPART 1 · WHAT EACH JUDGE IS\n===========================\n")
    rows = ["What it is", "Answers it can give", "What it writes, in order", "Varies between prompt versions"]
    for J in ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"]:
        info = EXTRA.get(J) or TABLE[J]
        L.append("%s · %s" % (J, NAMES[J]))
        for r, t in zip(rows, info):
            L.append("  %s:" % r)
            for line in t.split("\n"):
                L.append("      " + line.replace("• ", "- "))
        L.append("")

    L.append("\nPART 2 · ONE CASE, EVERY JUDGE\n==============================\n")
    L.append("Qwen3-8B, layer 15, assistant -> leviathan, route MAN4, s = 1, addition at alpha = 2 (first judge sweep).")
    L.append("The pair's options (J1, J2, J5): " + ", ".join(P.pair_options(jobs[lab[lab.judge == "J2"].name.iloc[0]]["options"])))
    L.append("Target: leviathan.\n")
    L.append(block("QUESTION", ans.question))
    L.append(block("ANSWER", textwrap.fill(" ".join(ans.response.split()), 100)))
    L.append("")
    for J in ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"]:
        L.append("\n" + "=" * 100 + "\n%s · %s\n" % (J, NAMES[J]) + "=" * 100)
        if J == "J1":
            prompt = (PM / "J_ST15" / "sys" / "assistant_R_leviathan_L15.txt").read_text()
            verdict = dict(persona_kind=j1.label, evidence=j1.evidence, intensity=j1.intensity)
        elif J == "J10":
            from steering.judge_sweep import j10 as J10
            prompt = (J10.HEAD.replace("__ROLES__", ", ".join(J10.roles()))
                      + "1. <answer, whitespace collapsed, first 500 characters>\n2. ...\n... (15 answers per prompt)")
            verdict = dict(why=j10.why, rank=json.loads(j10["rank"]), nonsense=bool(j10.nonsense))
        else:
            r = lab[lab.judge == J].iloc[0]
            j = jobs[r["name"]]
            salt = "%s|%s" % (J, STEM if J in ("J2", "J5") else "all")
            prompt = P.system(J, (j["a"], j["c"], j["d"]), j.get("options") or [], j["target"], salt)
            keep = ["label", "intensity", "evidence", "analysis", "description", "self_id", "secondary", "mixed", "ranking"]
            verdict = {k: (json.loads(r[k]) if k == "ranking" and isinstance(r[k], str) else r[k]) for k in keep if pd.notna(r.get(k))}
        L.append(block("Prompt it read", shorten(prompt)))
        L.append(block("Its verdict on this answer", json.dumps(verdict, ensure_ascii=False, indent=2, default=str)))
    OUT.write_text("\n".join(L) + "\n")
    print(OUT, OUT.stat().st_size // 1000, "kB")


if __name__ == "__main__":
    main()
