"""Phase 2 judges (plan v2 s1 M4), Claude Sonnet 5.5 through the interactive API.

    python -m persona_axes.rate --stage rate   [--dry-run]   # every dose answer rated 1-5 on all 35 scales, 12 per call
    python -m persona_axes.rate --stage name   [--dry-run]   # blind naming: label the poles, then match the label (5 repeats)

rate: one JSON per (answer, group) in $PM/AX21/rate/calls/; resumable. The scale block is the cached system prompt.
name: $PM/AX21/rate/naming/<axis>_r<rep>_{label,match}.json. The labeller sees 10 roles per end (by coordinate)
and 3 steered answers at s = -2 and s = +2 to the same questions, with the axis name hidden; the matcher sees
the written labels and 5 candidate axis descriptions (the true one + 4 distractors) and picks one. Chance 20%.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd

from persona_axes import AX, HERE, LAYER, VIEW
from persona_axes.api import SONNET, Runner

GROUPS = [range(0, 12), range(12, 24), range(24, 35)]

RATE_SYS = """You rate one answer written by a language model on several behavioural scales. For each scale, 1 means the answer clearly shows pole A, 5 means it clearly shows pole B, 3 means neither or balanced. Judge only what the text shows; if a scale does not apply to this answer, give 3. Read the whole answer.

THE SCALES
%s

Reply with one JSON object and nothing else, one entry per scale asked for, in this form:
{"<scale>": {"note": "<at most 12 words of evidence>", "r": <1-5>}, ...}"""

LABEL_PROMPT = """A language model was pushed along one hidden direction in its activation space. Below are (1) the 10 roles that sit at each end of this direction, and (2) three answers to the same three questions when the model is pushed toward end A and toward end B.

ROLES AT END A: %s
ROLES AT END B: %s

ANSWERS PUSHED TOWARD END A:
%s

ANSWERS PUSHED TOWARD END B:
%s

Name the two poles of this direction as behavioural dispositions (not topics). Reply with one JSON object: {"pole_A": "<at most 8 words>", "pole_B": "<at most 8 words>", "why": "<one sentence>"}"""

MATCH_PROMPT = """Someone described the two poles of a behavioural direction as:
  pole A: %s
  pole B: %s

Which ONE of the following candidate directions is it? Each candidate is written as "pole A <-> pole B". Pick the single best match.

%s

Reply with one JSON object: {"choice": <number 1-5>, "why": "<one sentence>"}"""


def scales():
    return json.loads((HERE / "axes.json").read_text())["axes"]


def scale_block(idx, ax):
    return "\n".join("- %s: A = %s | B = %s" % (ax[i]["name"], ax[i]["pole_a"], ax[i]["pole_b"]) for i in idx)


DOSE_DIR = "dose"           # set by --dose-dir; "dose" = pre-registered sigma units, "dose_x4" = 4 sigma units (addendum 2026-10-21)


def answers():
    rows = []
    for f in sorted((AX / DOSE_DIR).glob("gen_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    return pd.DataFrame(rows)


def stage_rate(a):
    ax = scales()
    df = answers()
    if a.axes:
        df = df[df.axis.isin(a.axes.split(","))]
    D = AX / ("rate" if DOSE_DIR == "dose" else "rate_" + DOSE_DIR) / "calls"
    D.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in df.itertuples():
        for g, idx in enumerate(GROUPS):
            f = D / ("%s_s%+d_q%d_g%d.json" % (r.axis, r.s, r.qi, g))
            if not f.exists():
                todo.append((f, r, g))
    sys_cache = {g: RATE_SYS % scale_block(idx, ax) for g, idx in enumerate(GROUPS)}
    R = Runner(SONNET, workers=12)
    prompts = ["QUESTION: %s\n\nANSWER:\n%s\n\nRate this answer on every scale listed in your instructions." % (r.question, r.response) for _, r, _ in todo]
    print("rate: %d calls (%d answers), estimate $%.2f + cached system prompts + thinking" % (len(todo), len(df), R.estimate(prompts, 350)))
    if a.dry_run:
        print(sys_cache[0][:800]); print(prompts[0][:500] if prompts else ""); return

    def ok_for(g):
        names = [ax[i]["name"] for i in GROUPS[g]]
        return lambda v: all(isinstance(v.get(n), dict) and v[n].get("r") in (1, 2, 3, 4, 5) for n in names)

    def one(t):
        f, r, g = t
        user = "QUESTION: %s\n\nANSWER:\n%s\n\nRate this answer on every scale listed in your instructions." % (r.question, r.response)
        v, raw, stop = R.call(user, system=sys_cache[g], max_tokens=1500, effort="medium", ok=ok_for(g))
        if v is None:
            print("FAILED %s (stop=%s)" % (f.name, stop), flush=True); return False
        f.write_text(json.dumps(dict(axis=r.axis, s=int(r.s), qi=int(r.qi), group=g, reply=v), ensure_ascii=False))
        return True
    res = R.map(one, todo)
    print("done: %d ok, %d failed, $%.2f, refusals %d" % (sum(res), len(res) - sum(res), R.spent, R.refusals))


def stage_name(a):
    ax = scales()
    names = [x["name"] for x in ax]
    V = AX / "vectors"
    A = np.load(V / ("axes_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    rn = json.loads((V / "roles_names.json").read_text())
    Rv = np.load(V / ("roles_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    coord = Rv @ A.T                                                    # 276 x 35
    df = answers()
    D = AX / ("rate" if DOSE_DIR == "dose" else "rate_" + DOSE_DIR) / "naming"
    D.mkdir(parents=True, exist_ok=True)
    R = Runner(SONNET, workers=8)
    tasks = [(i, rep) for i in range(len(names)) for rep in range(5) if not (D / ("%s_r%d_match.json" % (names[i], rep))).exists()]
    print("name: %d axes x reps = %d label + match pairs" % (len(names), len(tasks)))
    if a.dry_run:
        return

    def one(t):
        i, rep = t
        rng = random.Random(1000 * i + rep)
        lo = [rn[j] for j in np.argsort(coord[:, i])[:10]]              # low coordinate = pole A (pos pole)
        hi = [rn[j] for j in np.argsort(coord[:, i])[-10:]]
        # the axis vector is pole_a - pole_b, so HIGH coordinate = pole A. End A := high end.
        endA, endB = hi, lo
        qs = rng.sample(range(10), 3)
        d = df[df.axis == names[i]]
        ansA = "\n".join("Q: %s\nA: %s" % (d[(d.s == 2) & (d.qi == q)].question.iloc[0], d[(d.s == 2) & (d.qi == q)].response.iloc[0][:600]) for q in qs)
        ansB = "\n".join("Q: %s\nA: %s" % (d[(d.s == -2) & (d.qi == q)].question.iloc[0], d[(d.s == -2) & (d.qi == q)].response.iloc[0][:600]) for q in qs)
        rng.shuffle(endA); rng.shuffle(endB)
        v, raw, stop = R.call(LABEL_PROMPT % (", ".join(endA), ", ".join(endB), ansA, ansB), max_tokens=600, effort="medium",
                              ok=lambda v: isinstance(v.get("pole_A"), str) and isinstance(v.get("pole_B"), str))
        if v is None:
            print("FAILED label %s r%d" % (names[i], rep), flush=True); return False
        (D / ("%s_r%d_label.json" % (names[i], rep))).write_text(json.dumps(dict(axis=names[i], rep=rep, endA=endA, endB=endB, qs=qs, reply=v), ensure_ascii=False))
        others = rng.sample([j for j in range(len(names)) if j != i], 4)
        cands = others + [i]; rng.shuffle(cands)
        block = "\n".join("%d. %s <-> %s" % (k + 1, ax[j]["pole_a"], ax[j]["pole_b"]) for k, j in enumerate(cands))
        m, raw, stop = R.call(MATCH_PROMPT % (v["pole_A"], v["pole_B"], block), max_tokens=900, effort="medium",   # prose before the JSON needs room
                              ok=lambda v: v.get("choice") in (1, 2, 3, 4, 5))
        if m is None:
            print("FAILED match %s r%d" % (names[i], rep), flush=True); return False
        (D / ("%s_r%d_match.json" % (names[i], rep))).write_text(json.dumps(dict(axis=names[i], rep=rep, candidates=[names[j] for j in cands],
                                                                               correct=cands.index(i) + 1, reply=m), ensure_ascii=False))
        return True
    res = R.map(one, tasks)
    print("done: %d ok, %d failed, $%.2f" % (sum(res), len(res) - sum(res), R.spent))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rate", "name"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--axes", default=None, help="comma-separated axis names (pilot); default all")
    ap.add_argument("--dose-dir", default="dose")
    a = ap.parse_args(argv)
    global DOSE_DIR
    DOSE_DIR = a.dose_dir
    (stage_rate if a.stage == "rate" else stage_name)(a)


if __name__ == "__main__":
    main()
