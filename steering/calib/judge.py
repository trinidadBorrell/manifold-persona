"""Judge batches for C / E1 / E2 (plan 2026-10-09) and the E2 trait judge.

persona judge: same prompt as every steering arm (judge_prompt.system_for_route: route_judge_v3 + the
assistant fix); options = the pair's options from cells.json (start, the routes' personas, target), so
every route of a pair is judged against the same list. Reference answers (no prompt / A / B prompted)
are judged too, as a check that the prompted personas are recognised.
trait judge (E2): reads the time-stamped hypothesis of the direction and says whether the answer leans
to the - end description, the + end description, or neither.

    python -m steering.calib.judge persona --dir CAL --out J_CAL
    python -m steering.calib.judge trait   --dir DIR --out J_DIRT --hyp hypotheses_E2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from steering.ace_study.judge_prompt import system_for_route
import steering.route_judge_v2 as _v2
import steering.route_judge_v3 as _v3

_orig_def = _v2.definition


def _definition(role):
    """`default` (the no-persona Assistant Axis role) has no eval_prompt; it is the model as itself."""
    if role == "default":
        return ("The model's own default identity with no persona: an AI assistant answering as itself, "
                "without taking on any character.")
    return _orig_def(role)


_v2.definition = _v3.definition = _definition

BATCH = 100

TRAIT_SYS = """You are evaluating which of two descriptions an instruct-tuned LLM's [response] leans towards. You will see one
[question] and one [response]. Evaluate ONLY the [response].

THE TWO ENDS OF THIS DIRECTION
- minus: {minus}
- plus: {plus}

Decide whether the [response] leans to the minus description, the plus description, or neither (no clear lean,
both equally, or incoherent text). Judge what the response says and how it says it, not whether it mentions
these words.

FIELDS
- persona_kind: one of minus, plus, neither.
- evidence: a VERBATIM quote of at most 25 words from the [response] that decides your answer.
- intensity: 1 (slight lean) or 2 (clear lean) for minus/plus; null for neither.

Answer with a single JSON object and nothing else, with keys: persona_kind, evidence, intensity."""


def key(r):
    return "%s|%d|%d|%d" % (r["cell_id"], int(r["q_idx"]), int(r["para"]), int(r["sample"]))


def write(D, stem, sys_text, options, rows, groups):
    (D / "sys" / (stem + ".txt")).write_text(sys_text)
    n = 0
    for i in range(0, len(rows), BATCH):
        (D / "batches" / ("%s_%03d.json" % (stem, i // BATCH))).write_text(json.dumps(rows[i:i + BATCH], ensure_ascii=False))
        n += 1
    groups.append(dict(stem=stem, n=n, options=options))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["persona", "trait"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hyp", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    spec = json.loads((d / "cells.json").read_text())
    cell = {c["cell_id"]: c for c in spec["cells"]}
    g = pd.read_csv(d / "generations.csv")
    g = g[~g.collapsed].copy()
    g["key"] = [key(r) for _, r in g.iterrows()]
    g["pair"] = [cell[c]["pair"] for c in g.cell_id]
    D = Path(a.out)
    for s in ("batches", "sys", "labels_strict"):
        (D / s).mkdir(parents=True, exist_ok=True)
    groups = []
    for (pair, L), dr in g.groupby(["pair", "layer"]):
        pk = "%s|L%d" % (pair, L)
        stem = ("%s_L%d" % (pair, L)).replace(">", "-").replace(".", "_")
        rows = dr[["key", "question", "response"]].to_dict("records")
        if a.kind == "persona":
            P = spec["pairs"][pk]["options"]
            write(D, stem, system_for_route(P), P + ["other"], rows, groups)
        else:
            if spec["exp"] != "E2":
                continue
            k = spec["pairs"][pk]["direction"]
            h = json.loads(Path(a.hyp).read_text())["directions"]["L%d" % L][str(k)]
            write(D, stem, TRAIT_SYS.format(minus=h["minus"], plus=h["plus"]), ["minus", "plus", "neither"], rows, groups)
    (D / "manifest.json").write_text(json.dumps(dict(kind=a.kind, groups=groups)))
    print("%s: %d answers (collapsed not sent), %d batches over %d groups"
          % (a.kind, len(g), sum(x["n"] for x in groups), len(groups)))


if __name__ == "__main__":
    main()
