"""Answer browser data for the C and E1 pages: every answer with its setting and judge label, plus the
judge prompts.

    python -m calibration.answers --exp CAL --judge J_CAL --out <dir>
    python -m calibration.answers --exp ST15,ST19 --judge J_ST15,J_ST19 --out <dir>
    python -m calibration.answers --root <ROOT>/qwen3_14b --exp CAL --judge J_CAL --out <dir> --name cal14b

Writes <dir>/answers/index.json (questions, system prompts, pairs with routes and judge options),
<dir>/answers/judge.json (subagent instructions, the judge system prompt of every pair, one sample
judge input) and one <dir>/answers/<pair>_L<L>.json per pair and layer. Collapsed answers (not judged)
are cut to their first COLLAPSED_CHARS characters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
COLLAPSED_CHARS = 600
FIELDS = ["route", "cond", "alpha", "s", "q", "sample", "sys_role", "label", "cls", "intensity", "evidence",
          "collapsed", "n_new", "response"]


def stem(pair, L):
    return ("%s_L%d" % (pair, L)).replace(">", "-").replace(".", "_")


def num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT), help="run folder (Qwen3-14B: ROOT/qwen3_14b)")
    ap.add_argument("--name", default="answers", help="output subfolder")
    a = ap.parse_args(argv)
    run = Path(a.root)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    index = dict(questions=None, system_prompts={}, pairs={}, files={})
    judge = dict(instructions=(ROOT / "JUDGE_AGENT_STRICT.md").read_text(), prompts={}, sample_input=None)
    for exp, jd in zip(a.exp.split(","), a.judge.split(",")):
        spec = json.loads((run / exp / "cells.json").read_text())
        index["questions"] = [q[0] for q in spec["questions"]]
        index["system_prompts"].update({k: v for k, v in spec["system_prompts"].items() if v})
        gen = pd.read_csv(run / exp / "generations.csv", low_memory=False)
        gen["key"] = ["%s|%d|%d|%d" % k for k in zip(gen.cell_id, gen.q_idx, gen.para, gen["sample"])]
        rows = pd.read_csv(run / exp / "rows_measured.csv", low_memory=False,
                           usecols=["key", "pair", "start", "target", "route", "cond", "s", "label", "cls"])
        g = gen.merge(rows, on="key", how="left", validate="one_to_one")
        lab = {}
        for line in (run / jd / "judgements.jsonl").read_text().splitlines():
            r = json.loads(line)
            lab[r["key"]] = (r.get("intensity"), r.get("evidence") or "")
        for f in sorted((run / jd / "sys").glob("*.txt")):
            judge["prompts"][f.stem] = f.read_text()
        if judge["sample_input"] is None:
            b = sorted((run / jd / "batches").glob("*.json"))[0]
            judge["sample_input"] = dict(batch=b.name, items=json.loads(b.read_text())[:2])
        for (pair, L), d in g.groupby(["pair", "layer"]):
            pk = "%s|L%d" % (pair, L)
            p = spec["pairs"][pk]
            st = stem(pair, L)
            index["pairs"][st] = dict(layer=int(L), set=p["set"], start=p["start"], target=p["target"],
                                      routes=p["routes"], options=p["options"], n=len(d))
            recs = []
            for r in d.itertuples(index=False):
                it, ev = lab.get(r.key, (None, ""))
                txt = r.response if isinstance(r.response, str) else ""
                if r.collapsed and len(txt) > COLLAPSED_CHARS:
                    txt = txt[:COLLAPSED_CHARS] + " […cut: collapsed into repeated text]"
                recs.append([r.route, r.cond, num(r.alpha), num(r.s), int(r.q_idx), int(r.sample),
                             r.sys_role if isinstance(r.sys_role, str) else None,
                             r.label if isinstance(r.label, str) else None, r.cls, it, ev,
                             bool(r.collapsed), int(r.n_new), txt])
            # U+FFFD (broken tokens in the model's own output) kept as a JSON escape so the file stays clean UTF-8 text
            (out / (st + ".json")).write_text(json.dumps(dict(fields=FIELDS, rows=recs), ensure_ascii=False)
                                              .replace("�", "\\ufffd"))
            index["files"][st] = st + ".json"
        print(exp, len(g), "answers,", g.groupby(["pair", "layer"]).ngroups, "files", flush=True)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    (out / "judge.json").write_text(json.dumps(judge, ensure_ascii=False))


if __name__ == "__main__":
    main()
