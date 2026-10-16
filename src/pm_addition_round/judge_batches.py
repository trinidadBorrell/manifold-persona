"""Subagent-judge batches (plan 2026-10-08, 1.5 and 2.5).

Same judge as every ACE-study arm (judge_prompt.system_for_route: route_judge_v3 + the assistant
fix) and the same 100-answer batches. Only the option list differs:

  --kind pm  PersonaManifold arms. Options per (route, layer) = source, the personas the geodesic
             shortest path passes through, target (G1/summary.json). pm_linear uses the same list
             (same endpoints; a straight line passes through no persona) so the arms compare.
  --kind pc  Component steering. Options per (start, method, component) = start, then the personas
             on the two routes (towards the minus end, then the plus end), ending at each target
             (PC/cells.json). The start-only cells (s = 0) are judged in every group of their start,
             under that group's options; their keys are prefixed with the group.

Writes JUDGE/{sys,batches}/, manifest.json; subagents write JUDGE/labels/; collect_labels.py reads.

    python -m pm_addition_round.judge_batches --kind pm --gen S1 --g1 G1 --out JUDGE_PM
    python -m pm_addition_round.judge_batches --kind pc --gen PC --out JUDGE_PC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ace_steering.common import REP4_COLLAPSED, route_slug
from ace_steering.judge_prompt import system_for_route

BATCH = 100


def pm_key(r):
    return "%s|%d|%d|%d" % (r["cell_id"], int(r["prompt_idx"]), int(r["q_idx"]), int(r["sample"]))


def write_group(D, stem, P, rows, groups):
    (D / "sys" / (stem + ".txt")).write_text(system_for_route(P))
    n = 0
    for i in range(0, len(rows), BATCH):
        (D / "batches" / ("%s_%03d.json" % (stem, i // BATCH))).write_text(
            json.dumps(rows[i:i + BATCH], ensure_ascii=False))
        n += 1
    groups.append(dict(stem=stem, n=n, options=P + ["other"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pm", "pc"], required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--g1", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default=None, help="only these strengths (comma list), e.g. the follow-up cells")
    a = ap.parse_args(argv)
    D = Path(a.out)
    for s in ("batches", "sys", "labels"):
        (D / s).mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.gen) / "generations.csv")
    if a.alphas:
        d = d[d.alpha.isin([float(x) for x in a.alphas.split(",")])]
    if "aliased_from" in d:
        d = d[d.aliased_from.isna()]
    n_all, n_coll = len(d), int((d.deg_rep4 > REP4_COLLAPSED).sum())
    d = d[d.deg_rep4 <= REP4_COLLAPSED].copy()
    d["key"] = [pm_key(r) for _, r in d.iterrows()]
    groups = []
    if a.kind == "pm":
        opts = {}
        for rec in json.loads((Path(a.g1) / "summary.json").read_text()):
            for r, v in rec["routes"].items():
                opts["%s|L%d" % (r, rec["layer"])] = list(v["path_personas"])
        for (route, L), dr in d.groupby(["route", "layer"]):
            write_group(D, "%s_L%d" % (route_slug(route), L), opts["%s|L%d" % (route, L)],
                        dr[["key", "question", "response"]].to_dict("records"), groups)
    else:
        spec = json.loads((Path(a.gen) / "cells.json").read_text())
        for gk, P in spec["options"].items():
            start, method, comp = gk.split("|")
            m = (d.start == start) & (((d.method == method) & (d.component == int(comp))) | (d.method == "none"))
            dr = d[m].copy()
            stem = "%s_%s_c%s" % (start, method, comp)
            dr["key"] = [stem + "#" + k for k in dr.key]
            write_group(D, stem, list(P), dr[["key", "question", "response"]].to_dict("records"), groups)
    (D / "manifest.json").write_text(json.dumps(dict(kind=a.kind, groups=groups)))
    print("%d answers, %d collapsed (not sent), %d batches over %d groups"
          % (n_all, n_coll, sum(g["n"] for g in groups), len(groups)))


if __name__ == "__main__":
    main()
