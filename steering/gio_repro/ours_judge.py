"""J10 (Giovanni's ranking judge, exact) on our-method answers: build chunks, then score recall@k.

    python -m steering.gio_repro.ours_judge build --dir $PM/GIO/ours/spy
    python -m steering.gio_repro.ours_judge score --dir $PM/GIO/ours/spy
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from steering.judge_sweep import j10


def rows(d):
    r = []
    for f in sorted(d.glob("gen_*.jsonl")):
        for line in open(f):
            x = json.loads(line)
            x["key"] = "%s|%s|%d" % (x["alpha"], x["s"], x["qi"])
            r.append(x)
    return r


def build(d):
    R = rows(d)
    random.Random(0).shuffle(R)
    (d / "rank").mkdir(exist_ok=True)
    head = j10.HEAD.replace("__ROLES__", ", ".join(j10.roles()))
    index = {}
    for n in range((len(R) + 14) // 15):
        text = head
        for i, x in enumerate(R[n * 15:(n + 1) * 15], 1):
            index["%d:%d" % (n, i)] = x["key"]
            text += "%d. %s\n" % (i, " ".join(x["response"].split())[:500])
        (d / "rank" / ("chunk_%03d.txt" % n)).write_text(text)
    (d / "rank" / "index.json").write_text(json.dumps(index))
    print(len(R), "answers,", (len(R) + 14) // 15, "chunks")


def score(d):
    persona = d.name
    R = {x["key"]: x for x in rows(d)}
    idx = json.loads((d / "rank" / "index.json").read_text())
    rank = {}
    for f in sorted((d / "rank").glob("verdict_*.json")):
        n = int(f.stem.split("_")[1])
        for i, v in json.loads(f.read_text()).items():
            rank[idx["%d:%s" % (n, i)]] = ([str(r).strip().lower().replace(" ", "_") for r in v["rank"]], bool(v.get("nonsense")))
    res = []
    for a, s in sorted({(x["alpha"], x["s"]) for x in R.values()}):
        rk = [rank[k] for k, x in R.items() if (x["alpha"], x["s"]) == (a, s) and k in rank]
        res.append(dict(alpha=a, s=s, n=len(rk), **{"r@%d" % k: float(np.mean([persona in r[:k] for r, _ in rk])) for k in (1, 5, 10)},
                        asst1=float(np.mean([r[:1] == ["assistant"] for r, _ in rk])),
                        nonsense=float(np.mean([ns for _, ns in rk])), top1={}))
        for r, _ in rk:
            res[-1]["top1"][r[0]] = res[-1]["top1"].get(r[0], 0) + 1
    (d / "results.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]](Path(sys.argv[3]))
