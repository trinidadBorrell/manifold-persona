"""Turn a subagent-judge workflow journal into judgements_<step>_subagent.jsonl.

The workflow (Claude Code subagents on Sonnet, one per 100-answer batch, the route's judge
prompt read verbatim as their instructions) records each agent's structured result in its
journal. This keeps the last result per batch, checks it against the batch files (every key
labelled once, options_seen == the route's option list), writes the same record shape the
OpenRouter judge writes, adds the mechanically `collapsed` rows, and lists what is missing.

    python -m ace_steering.subagent_labels --step 1 --journal <run>/journal.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ace_steering.common import OUT_ROOT, REP4_COLLAPSED
from ace_steering.judge import key_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None, help="step 1/2 layout (judge_subagent/step<N>)")
    ap.add_argument("--tag", default=None, help="subagent_batches.py layout (judge_subagent/<tag>)")
    ap.add_argument("--journal", required=True, nargs="+")
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    sub = a.tag or ("step%d" % a.step)
    bdir = root / "judge_subagent" / sub
    man = json.loads((bdir / "manifest.json").read_text())
    gen_sub = man.get("gen", sub)
    if "batches" not in man:                  # step 2: {groups: [{route, slug, L, n, options}]}
        man = dict(options={}, batches=[])
        for g in json.loads((bdir / "manifest.json").read_text())["groups"]:
            gk = "%s|L%d" % (g["route"], g["L"])
            man["options"][gk] = g["options"]
            for i in range(g["n"]):
                name = "%s_L%d_%03d" % (g["slug"], g["L"], i)
                man["batches"].append(dict(name=name, route=gk, items=str(bdir / "batches" / (name + ".json"))))
    want = {}
    for b in man["batches"]:
        for it in json.loads(Path(b["items"]).read_text()):
            want[it["key"]] = b
    labels, seen_bad, n_results = {}, set(), 0
    for jp in a.journal:
        for line in Path(jp).read_text().splitlines():
            d = json.loads(line)
            if d.get("type") != "result" or not isinstance(d.get("result"), dict):
                continue
            res = d["result"]
            if "labels" not in res:
                continue
            n_results += 1
            for l in res["labels"]:
                k = l.get("key")
                if k not in want:
                    continue
                opts = man["options"][want[k]["route"]]
                if [x.strip().lower() for x in res.get("options_seen", [])] != opts:
                    seen_bad.add(want[k]["name"])
                if l.get("persona_kind") in opts:
                    labels[k] = l                     # last result wins
    df = (pd.read_csv(root / gen_sub / "generations.csv") if gen_sub
          else pd.DataFrame(columns=["route", "layer", "deg_rep4", "cell_id", "prompt_idx", "q_idx", "sample"]))
    df = df.assign(key=[key_of(r) for _, r in df.iterrows()]) if len(df) else df.assign(key=[])
    out = root / "judge" / ("judgements_%s_subagent.jsonl" % sub)
    missing = [k for k in want if k not in labels]
    with out.open("w") as fh:
        covered = set(man["options"])          # "route" (step 1) or "route|L<layer>" (later)
        coll = df[df.deg_rep4 > REP4_COLLAPSED]
        coll = coll[[(r in covered) or ("%s|L%d" % (r, L) in covered) for r, L in zip(coll.route, coll.layer)]]
        for _, r in coll.iterrows():
            fh.write(json.dumps(dict(key=r.key, persona_kind="collapsed", intensity=None, evidence="",
                                     analysis="rep4>%d" % REP4_COLLAPSED, error=None, model=None)) + "\n")
        for k, l in labels.items():
            fh.write(json.dumps(dict(key=k, persona_kind=l["persona_kind"], intensity=l.get("intensity"),
                                     evidence=l.get("evidence", ""), analysis="", error=None,
                                     model="claude-code-subagent:sonnet")) + "\n")
    (bdir / "missing_keys.json").write_text(json.dumps(missing))
    print("results read %d | labelled %d of %d | missing %d | batches whose options_seen did not match: %s"
          % (n_results, len(labels), len(want), len(missing), sorted(seen_bad) or "none"))
    print("->", out)


if __name__ == "__main__":
    main()
