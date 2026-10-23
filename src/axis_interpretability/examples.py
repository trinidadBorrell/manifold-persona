"""Real judge prompts and replies for Exp 1, 2, 4 and 5 on a few directions, into output/20-oct/examples/.

    python -m axis_interpretability.examples --dir $PM/AXI_L28_K50 --out output/20-oct/examples

Needs analyse_many's summary.json for the scores in each heading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PICK = ["PCA1", "PGA1", "PGA2", "PCA10", "PGA10", "PCA40", "PGA40"]
END = {"minus": "−", "plus": "+"}
OTHER = {"minus": "plus", "plus": "minus"}


def name(a):
    return a[:3] + " " + a[3:]


def head(rows, a):
    r = rows[(a[:3], int(a[3:]))]
    return "variance %.1f%%; Exp 1 theme %.1f, contrast %.1f; Exp 2 held-out %.0f%%; Exp 4 ρ %.2f" % (
        100 * r["var_share"], r["theme"], r["contrast"], 100 * r["held"], r["rho"])


def block(title, text, lang=""):
    return ["**%s**" % title, "", "```" + lang, text, "```", ""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="output/20-oct/examples")
    ap.add_argument("--summary", default="output/20-oct/k50/summary.json", help="analyse_many output")
    a = ap.parse_args(argv)
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {(r["method"], r["k"]): r for r in json.loads(Path(a.summary).read_text())["rows"]}
    it1 = {x["id"]: x for x in json.loads((D / "items" / "exp1.json").read_text())}
    it2 = {x["id"]: x for x in json.loads((D / "items" / "exp2.json").read_text())}
    intro = ["Directions shown: %s. They are chosen to span good to bad: direction 1, the one-sided PGA 2, "
             "direction 10 and direction 40." % ", ".join(name(x) for x in PICK),
             "Every prompt and reply below is the real one, read from `$PM/AXI_L28_K50`. The full prompt is shown each time, "
             "so each example can be read on its own.", ""]

    # Exp 1
    L = ["# Exp 1: real prompts and replies (repeat 0)", ""] + intro
    for ax in PICK:
        c = json.loads((D / "exp1" / ("exp1_%s_r0.json" % ax)).read_text())
        it = it1["exp1_%s_r0" % ax]
        L += ["## %s" % name(ax), "", head(rows, ax), "", "Group A is the %s end." % END[it["A_is"]], ""]
        L += block("Prompt", c["prompt"]) + block("Reply", json.dumps(c["reply"], indent=1, ensure_ascii=False), "json")
    (out / "exp1_examples.md").write_text("\n".join(L))

    # Exp 2
    L = ["# Exp 2: real prompts and replies (split 0)", ""] + intro
    for ax in PICK:
        it = it2["exp2_%s_s0" % ax]
        lab = json.loads((D / "exp2_label" / ("exp2_%s_s0.json" % ax)).read_text())
        srt = json.loads((D / "exp2_sort" / ("exp2_%s_s0_held.json" % ax)).read_text())
        to_end = {"A": it["A_is"], "B": OTHER[it["A_is"]]}
        res = [(r, END[it["truth"][r]], srt["reply"][r]["pole"], to_end[srt["reply"][r]["pole"]] == it["truth"][r]) for r in it["held"]]
        L += ["## %s" % name(ax), "", head(rows, ax), "", "Pole A is the %s end." % END[it["A_is"]], ""]
        L += block("Labeller prompt", lab["prompt"]) + block("Labeller reply", json.dumps(lab["reply"], indent=1, ensure_ascii=False), "json")
        L += block("Sorter prompt (20 held-out roles)", srt["prompt"]) + block("Sorter reply", json.dumps(srt["reply"], indent=1, ensure_ascii=False), "json")
        L += ["**Result: %d of 20 correct.**" % sum(x[3] for x in res), "", "| role | true end | judge said | correct |", "|---|---|---|---|"]
        L += ["| %s | %s | %s | %s |" % (r, t, p, "yes" if ok else "**no**") for r, t, p, ok in res] + [""]
    (out / "exp2_examples.md").write_text("\n".join(L))

    # Exp 4
    L = ["# Exp 4: real prompts and replies (split 0)", ""] + intro + [
        "The label is the Exp 2 labeller's label for the same split, shown in each prompt.", ""]
    for ax in PICK:
        it = it2["exp2_%s_s0" % ax]
        c = json.loads((D / "exp4_rank" / ("exp2_%s_s0.json" % ax)).read_text())
        o = c["reply"]["order"]
        sgn = 1 if it["A_is"] == "minus" else -1
        true = sorted(it["spread"], key=lambda r: sgn * it["spread_score"][r])
        rho = spearmanr([o.index(r) for r in it["spread"]], [sgn * it["spread_score"][r] for r in it["spread"]])[0]
        L += ["## %s" % name(ax), "", head(rows, ax), "",
              "Pole A is the %s end, so the true order runs from the %s end to the %s end." % (END[it["A_is"]], END[it["A_is"]], END[OTHER[it["A_is"]]]), ""]
        L += block("Prompt", c["prompt"]) + block("Reply", json.dumps(c["reply"], indent=1, ensure_ascii=False), "json")
        L += ["**Result: Spearman ρ = %.2f.**" % rho, "", "| position | judge's order | its true score | true order | its true score |",
              "|---|---|---|---|---|"]
        L += ["| %d | %s | %.1f | %s | %.1f |" % (i + 1, o[i], it["spread_score"][o[i]], true[i], it["spread_score"][true[i]]) for i in range(len(o))] + [""]
    (out / "exp4_examples.md").write_text("\n".join(L))

    # Exp 5
    it5 = {x["id"]: x for x in json.loads((D / "items" / "exp5.json").read_text())}
    L = ["# Exp 5: real prompts and replies (split 0)", ""] + intro + [
        "No label: the judge gets 5 example roles at each pole (drawn from the 20 most extreme) and orders 20 other roles "
        "from along the whole axis.", ""]
    for ax in PICK:
        it = it5["exp5_%s_s0" % ax]
        f = D / "exp5_anchor" / ("exp5_%s_s0.json" % ax)
        if not f.exists():
            L += ["## %s" % name(ax), "", "(no valid reply after 10 tries)", ""]
            continue
        c = json.loads(f.read_text())
        o = c["reply"]["order"]
        sgn = 1 if it["A_is"] == "minus" else -1
        true = sorted(it["spread"], key=lambda r: sgn * it["spread_score"][r])
        rho = spearmanr([o.index(r) for r in it["spread"]], [sgn * it["spread_score"][r] for r in it["spread"]])[0]
        r = rows[(ax[:3], int(ax[3:]))]
        L += ["## %s" % name(ax), "", head(rows, ax) + "; Exp 5 ρ %.2f" % r["rho5"], "",
              "Pole A is the %s end." % END[it["A_is"]], ""]
        L += block("Prompt", c["prompt"]) + block("Reply", json.dumps(c["reply"], indent=1, ensure_ascii=False), "json")
        L += ["**Result: Spearman ρ = %.2f.**" % rho, "", "| position | judge's order | its true score | true order | its true score |",
              "|---|---|---|---|---|"]
        L += ["| %d | %s | %.1f | %s | %.1f |" % (i + 1, o[i], it["spread_score"][o[i]], true[i], it["spread_score"][true[i]]) for i in range(len(o))] + [""]
    (out / "exp5_examples.md").write_text("\n".join(L))
    print("wrote", sorted(p.name for p in out.glob("*.md")))


if __name__ == "__main__":
    main()
