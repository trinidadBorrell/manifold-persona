"""Judge the reproduction with Giovanni's ranking judge (exact: make_rank_chunks.py) and our J8.

    python -m steering.gio_repro.judge build --out $PM/GIO/spy
    python -m steering.gio_repro.judge score --out $PM/GIO/spy

J10: his header and the 276 role names, whitespace collapsed, clipped to 500 characters, all answers
shuffled with seed 0, 15 per chunk -> rank/chunk_NN.txt; verdicts rank/verdict_NN.json.
J8: our revised control with the 275 roles (judge sweep J8, draw 0), one batch -> j8/.
Scored like his score_rank.py: recall@k = share of answers whose persona is in the first k names.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from steering.judge_sweep import j10
from steering.judge_sweep import prompts as P


def rows(out):
    r = []
    for f in sorted(out.glob("gen_steer_*.jsonl")) + [out / "gen_ceiling.jsonl"]:
        for line in open(f):
            x = json.loads(line)
            x["key"] = "%s|%s|%d" % (x["cell"], x["alpha"], x["qi"])
            r.append(x)
    return r


def build(out, persona):
    R = rows(out)
    # J10, exactly as make_rank_chunks.py
    d = out / "rank"
    d.mkdir(exist_ok=True)
    rr = list(R)
    random.Random(0).shuffle(rr)
    head = j10.HEAD.replace("__ROLES__", ", ".join(j10.roles()))
    index = {}
    for n in range((len(rr) + 14) // 15):
        block = rr[n * 15:(n + 1) * 15]
        text = head
        for i, x in enumerate(block, 1):
            index["%d:%d" % (n, i)] = x["key"]
            text += "%d. %s\n" % (i, " ".join(x["response"].split())[:500])
        (d / ("chunk_%03d.txt" % n)).write_text(text)
    (d / "index.json").write_text(json.dumps(index))
    # J8, our revised control with the 275 roles
    j = out / "j8"
    for s in ("sys", "batches", "labels"):
        (j / s).mkdir(parents=True, exist_ok=True)
    (j / "sys" / "J8_gio.txt").write_text(P.system("J8", P.version(0, 0), None, persona, "J8|all"))
    items = [dict(key=x["key"], question=x["question"], response=x["response"]) for x in R]
    random.Random(1).shuffle(items)
    (j / "batches" / "gio.json").write_text(json.dumps(items, ensure_ascii=False))
    print("%d answers; J10 %d chunks; J8 1 batch" % (len(R), len(index) // 15 + (len(index) % 15 > 0)))


def score(out, persona):
    R = {x["key"]: x for x in rows(out)}
    idx = json.loads((out / "rank" / "index.json").read_text())
    rank = {}
    for f in sorted((out / "rank").glob("verdict_*.json")):
        n = int(f.stem.split("_")[1])
        for i, v in json.loads(f.read_text()).items():
            rank[idx["%d:%s" % (n, i)]] = ([str(r).strip().lower().replace(" ", "_") for r in v["rank"]], bool(v.get("nonsense")))
    lab = {l["key"]: l for l in json.loads((out / "j8" / "labels" / "J8_gio.json").read_text())["labels"]}
    res = []
    for cell in sorted({(x["cell"], x["alpha"]) for x in R.values()}):
        ks = [k for k, x in R.items() if (x["cell"], x["alpha"]) == cell]
        rk = [rank[k] for k in ks if k in rank]
        lb = [str(lab[k].get("persona_kind", "")).lower() for k in ks if k in lab]
        res.append(dict(cell=cell[0], s=cell[1], n=len(ks),
                        **{"r@%d" % k: float(np.mean([persona in r[:k] for r, _ in rk])) for k in (1, 5, 10)},
                        asst1=float(np.mean([r[:1] == ["assistant"] for r, _ in rk])),
                        nonsense=float(np.mean([ns for _, ns in rk])),
                        j8_persona=float(np.mean([l == persona for l in lb])),
                        j8_labels={l: lb.count(l) for l in sorted(set(lb))}))
    (out / "results.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    o = Path(sys.argv[3]) if sys.argv[2] == "--out" else None
    persona = o.name
    {"build": build, "score": score}[sys.argv[1]](o, persona)
