"""Batches and judge instructions for the judge prompt sweep (plan 2026-10-14-judge-prompt-sweep.md).

Answers: 4 far pairs, routes LIN/PMG/MAN4/FL, alpha = 2 (+ the start at s = 0 and the prompted A/B
references), 8B layer 15 and 14B layer 21; collapsed answers included. Each (model, pair) is shuffled
and cut into 3 batches b = 0, 1, 2. A job = (judge, model, pair, draw, b); its version (a, c, d) comes
from the Latin square in prompts.version.

    python -m route_judge_sweep.build --out $PM/JS
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from route_judge_sweep import prompts as P

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
MODELS = {"8b": (PM / "ST15", 15), "14b": (PM / "qwen3_14b" / "ST21", 21)}
PAIRS = ["assistant.R.vampire", "assistant.R.leviathan", "grader.R.tree", "scheduler.R.void"]
ROUTES = ["LIN", "PMG", "MAN4", "FL"]
ALPHA = 2.0
N_BATCH = 3


def key(r):
    return "%s|%d|%d|%d" % (r["cell_id"], int(r["q_idx"]), int(r["para"]), int(r["sample"]))


def answers(d):
    spec = json.loads((d / "cells.json").read_text())
    cells = pd.DataFrame(spec["cells"])
    g = pd.read_csv(d / "generations.csv", low_memory=False)
    g = g.merge(cells[["cell_id", "pair", "route", "s", "target", "start"]], on="cell_id")
    g = g[g.pair.isin(PAIRS)]
    keep = ((g.route.isin(ROUTES + ["start"]) & (g.alpha == ALPHA)) | (g.route == "ref"))
    g = g[keep].copy()
    g["key"] = [key(r) for _, r in g.iterrows()]
    return spec, g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PM / "JS"))
    a = ap.parse_args(argv)
    D = Path(a.out)
    for s in ("batches", "sys", "labels"):
        (D / s).mkdir(parents=True, exist_ok=True)
    jobs, meta = [], []
    for m, (d, L) in MODELS.items():
        spec, g = answers(d)
        meta.append(g.assign(model=m)[["key", "model", "layer", "pair", "start", "target", "route", "s", "alpha",
                                      "q_idx", "para", "sample", "sys_role", "question", "response", "collapsed"]])
        for pair, dp in g.groupby("pair"):
            opts = P.pair_options(spec["pairs"]["%s|L%d" % (pair, L)]["options"])
            target = dp.target.iloc[0]
            stem = "%s_%s" % (m, pair.replace(".R.", "-"))
            rows = dp[["key", "question", "response"]].to_dict("records")
            random.Random(stem).shuffle(rows)
            for b in range(N_BATCH):
                bname = "%s_b%d" % (stem, b)
                (D / "batches" / (bname + ".json")).write_text(json.dumps(rows[b::N_BATCH], ensure_ascii=False))
                for J, spec_j in P.JUDGES.items():
                    for draw in range(spec_j["draws"]):
                        v = P.version(draw, b)
                        # salt: the option order and example draw depend on the pair options, not the batch
                        salt = "%s|%s" % (J, stem if J in ("J2", "J5") else "all")
                        name = "%s_%s_d%d" % (J, bname, draw)
                        (D / "sys" / (name + ".txt")).write_text(P.system(J, v, opts, target, salt))
                        jobs.append(dict(name=name, judge=J, model=m, layer=L, pair=pair, draw=draw, b=b,
                                         a=v[0], c=v[1], d=v[2], batch=bname, n=len(rows[b::N_BATCH]),
                                         options=opts if J in ("J2", "J5") else None, target=target))
    pd.concat(meta).to_csv(D / "answers.csv", index=False)
    (D / "manifest.json").write_text(json.dumps(dict(jobs=jobs), indent=0))
    n = sum(j["n"] for j in jobs)
    print("%d answers, %d jobs, %d judgements (+ J6 stage 2)" % (len(pd.concat(meta)), len(jobs), n))


if __name__ == "__main__":
    main()
