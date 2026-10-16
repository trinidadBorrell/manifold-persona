"""J10 for the Qwen3-8B clamp test (qwen3_pin.py): chunks exactly as make_rank_chunks.py, then recall@k.

    python -m giovanni_repro.q3_judge build --dir $PM/GIO8B
    python -m giovanni_repro.q3_judge score --dir $PM/GIO8B
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from route_judge_sweep import j10


def rows(d):
    return [json.loads(l) for f in sorted(d.glob("gen_*.jsonl")) for l in open(f)]


def build(d):
    R = rows(d)
    for r in R:
        r["key"] = "L%d|%s|%g|%d" % (r["layer"], r["target"], r["s"], r["qi"])
    random.Random(0).shuffle(R)
    (d / "rank").mkdir(exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    head = j10.HEAD.replace("__ROLES__", ", ".join(j10.roles()))
    index, jobs = {}, []
    for n in range((len(R) + 14) // 15):
        text = head
        for i, r in enumerate(R[n * 15:(n + 1) * 15], 1):
            index["%d:%d" % (n, i)] = r["key"]
            text += "%d. %s\n" % (i, " ".join(r["response"].split())[:500])
        (d / "rank" / ("chunk_%03d.txt" % n)).write_text(text)
        jobs.append(dict(name="J10_chunk_%03d" % n, judge="J10", kind="rank", batch="chunk_%03d" % n))
    (d / "rank" / "index.json").write_text(json.dumps(index))
    (d / "manifest.json").write_text(json.dumps(dict(jobs=jobs)))
    print(len(R), "answers,", len(jobs), "chunks")


def score(d):
    idx = json.loads((d / "rank" / "index.json").read_text())
    out = []
    for f in sorted((d / "rank").glob("verdict_*.json")):
        n = int(f.stem.split("_")[1])
        for i, v in json.loads(f.read_text()).items():
            k = idx.get("%d:%s" % (n, i))
            if not k or not isinstance(v, dict) or not v.get("rank"):
                continue
            L, t, s, q = k.split("|")
            rk = [str(x).strip().lower().replace(" ", "_") for x in v["rank"]]
            out.append(dict(layer=int(L.lstrip("L")), target=t, s=float(s), r1=t in rk[:1], r5=t in rk[:5], r10=t in rk[:10],
                            nonsense=bool(v.get("nonsense")), clean5=(t in rk[:5]) and not v.get("nonsense"),
                            top1=rk[0]))
    df = pd.DataFrame(out)
    df.to_csv(d / "scores.csv", index=False)
    g = df.groupby(["layer", "target", "s"])[["r1", "r5", "nonsense", "clean5"]].mean().round(2)
    print(g.to_string())
    print(df.groupby(["layer", "s"])[["r1", "r5", "nonsense", "clean5"]].mean().round(2).to_string())


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]](Path(sys.argv[3]))
