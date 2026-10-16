"""Collect subagent-judge outputs (labels/<batch>.json) into judgements.jsonl.

Each judge subagent reads sys/<group>.txt (the judge prompt, verbatim) and batches/<batch>.json
([{key, question, response}]) and writes labels/<batch>.json =
  {"options_seen": [...], "labels": [{"key", "persona_kind", "intensity", "evidence"}]}.
A label counts only if its persona_kind is one of the group's options. Prints what is missing, so
the missing batches can be re-dispatched.

    python -m steering.personamanifold.collect_labels --judge JUDGE_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--labels", default="labels_strict,labels",
                    help="label dirs in priority order: a batch comes from the first dir that has it")
    a = ap.parse_args(argv)
    J = Path(a.judge)
    man = json.loads((J / "manifest.json").read_text())
    want, opts_of = {}, {}
    for g in man["groups"]:
        for i in range(g["n"]):
            name = "%s_%03d" % (g["stem"], i)
            opts_of[name] = [o.lower() for o in g["options"]]
            for it in json.loads((J / "batches" / (name + ".json")).read_text()):
                want[it["key"]] = name
    got, bad_batches = {}, []
    for name in sorted(set(want.values())):
        f = next((J / d / (name + ".json") for d in a.labels.split(",") if (J / d / (name + ".json")).exists()), None)
        if f is None:
            continue
        try:
            res = json.loads(f.read_text())
        except json.JSONDecodeError:
            bad_batches.append(name)
            continue
        for l in res.get("labels", []):
            k, pk = l.get("key"), str(l.get("persona_kind", "")).strip().lower()
            if want.get(k) == name and pk in opts_of[name]:
                got[k] = dict(key=k, persona_kind=pk, intensity=l.get("intensity"),
                              evidence=l.get("evidence", ""), model="claude-code-subagent:sonnet",
                              pass_=f.parent.name)
    with (J / "judgements.jsonl").open("w") as fh:
        for r in got.values():
            fh.write(json.dumps(r) + "\n")
    missing = sorted({want[k] for k in want if k not in got})
    (J / "missing_batches.json").write_text(json.dumps(missing))
    print("labelled %d of %d | batches incomplete: %d %s | unreadable: %s"
          % (len(got), len(want), len(missing), missing[:10], bad_batches))


if __name__ == "__main__":
    main()
