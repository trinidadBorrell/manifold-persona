"""Write subagent-judge batches for any set of routes and layers.

One batch = up to 100 non-collapsed answers of ONE (route, layer), because the judge prompt is
per route and layer. Writes, under output/ace-steering-29_09/judge_subagent/<tag>/:
  batches/<slug>_L<L>_<i>.json   [{key, question, response}]
  sys/<slug>_L<L>.txt            the judge prompt (judge_prompt.system_for_route: assistant fix where it applies)
  manifest.json, args_groups.json  the groups the workflow script iterates over

    python -m steering.ace_study.subagent_batches --tag B --gen step2 \\
        --routes "bard>assistant,vampire>assistant" --layers 19,25
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from steering.ace_study.common import OUT_ROOT, REP4_COLLAPSED, route_slug
from steering.ace_study.grid import route_personas
from steering.ace_study.judge import key_of
from steering.ace_study.judge_prompt import system_for_route

BATCH = 100


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="judge_subagent/<tag>/ and judgements_<tag>_subagent.jsonl")
    ap.add_argument("--gen", required=True, help="generation folder, e.g. step2, step3_A, step5")
    ap.add_argument("--routes", default=None, help="comma list; default: every route in the generations")
    ap.add_argument("--layers", default=None, help="comma list; default: every layer")
    ap.add_argument("--exclude-tags", default=None,
                    help="comma list of earlier tags whose batched keys are skipped (judge a step in pieces)")
    a = ap.parse_args(argv)
    D = OUT_ROOT / "judge_subagent" / a.tag
    (D / "batches").mkdir(parents=True, exist_ok=True)
    (D / "sys").mkdir(exist_ok=True)
    d = pd.read_csv(OUT_ROOT / a.gen / "generations.csv")
    if a.routes:
        d = d[d.route.isin(a.routes.split(","))]
    if a.layers:
        d = d[d.layer.isin([int(x) for x in a.layers.split(",")])]
    n_all = len(d)
    n_coll = int((d.deg_rep4 > REP4_COLLAPSED).sum())
    d = d[d.deg_rep4 <= REP4_COLLAPSED].copy()
    d["key"] = [key_of(r) for _, r in d.iterrows()]
    if a.exclude_tags:
        seen = set()
        for t in a.exclude_tags.split(","):
            for f in (OUT_ROOT / "judge_subagent" / t / "batches").glob("*.json"):
                seen |= {it["key"] for it in json.loads(f.read_text())}
        d = d[~d.key.isin(seen)]
    groups = []
    for (route, L), dr in d.groupby(["route", "layer"]):
        slug, P = route_slug(route), route_personas(route, int(L))
        (D / "sys" / ("%s_L%d.txt" % (slug, L))).write_text(system_for_route(P))
        rows = dr[["key", "question", "response"]].to_dict("records")
        n = 0
        for i in range(0, len(rows), BATCH):
            (D / "batches" / ("%s_L%d_%03d.json" % (slug, L, i // BATCH))).write_text(
                json.dumps(rows[i:i + BATCH], ensure_ascii=False))
            n += 1
        groups.append(dict(route=route, slug=slug, L=int(L), n=n, options=P + ["other"]))
    (D / "manifest.json").write_text(json.dumps(dict(gen=a.gen, groups=groups)))
    (D / "args_groups.json").write_text(json.dumps(groups, separators=(",", ":")))
    print("%s: %d answers in the selection, %d collapsed (not sent), %d new to judge in %d batches over %d route-layers"
          % (a.tag, n_all, n_coll, len(d), sum(g["n"] for g in groups), len(groups)))


if __name__ == "__main__":
    main()
