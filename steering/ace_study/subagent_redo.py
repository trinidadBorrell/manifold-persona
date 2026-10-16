"""Re-batch the answers a subagent pass missed, 10 per batch (long answers make agents stop early).

Reads judge_subagent/<tag>/missing_keys.json (written by subagent_labels.py), writes
batches/<slug>_L<L>_redo<NNN>.json and prints the workflow args (suffix "redo").

    python -m steering.ace_study.subagent_redo --tag D1
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict

from steering.ace_study.common import OUT_ROOT

SIZE = 10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    D = OUT_ROOT / "judge_subagent" / a.tag
    missing = set(json.loads((D / "missing_keys.json").read_text()))
    groups = OrderedDict()
    for g in json.loads((D / "manifest.json").read_text())["groups"]:
        items = []
        for i in range(g["n"]):
            items += [it for it in json.loads((D / "batches" / ("%s_L%d_%03d.json" % (g["slug"], g["L"], i))).read_text())
                      if it["key"] in missing]
        for j in range(0, len(items), SIZE):
            (D / "batches" / ("%s_L%d_redo%03d.json" % (g["slug"], g["L"], j // SIZE))).write_text(
                json.dumps(items[j:j + SIZE], ensure_ascii=False))
            groups.setdefault((g["slug"], g["L"]), dict(slug=g["slug"], L=g["L"], n=0, options=g["options"]))["n"] += 1
    print("%d missing -> %d redo batches" % (len(missing), sum(x["n"] for x in groups.values())))
    print(json.dumps(dict(dir=str(D), suffix="redo", groups=list(groups.values())), separators=(",", ":")))


if __name__ == "__main__":
    main()
