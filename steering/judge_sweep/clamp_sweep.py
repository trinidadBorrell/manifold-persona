"""Judge-sweep inputs for the clamp answers (option A, STC15 / qwen3_14b/STC21) -> $PM/JSC.

    python -m steering.judge_sweep.clamp_sweep build     # answers.csv, batches, sys, rank chunks, manifest
    python -m steering.judge_sweep.clamp_sweep labels    # labels.csv in the JS format (+ J1, J10 files)

Only the new answers (steered and s = 0) are judged; the prompted A / B and no-prompt references are the
same answers as in JS, so their labels are copied from JS (draw 0), from the control judge (J1) and J10.
Draw 0 of J2-J9 with the same versions as JS (prompts.version(0, b)), J1 = the control prompt of the pair
(J_ST15 / J_ST21 sys), collapsed answers (4-gram rule) not sent to J1 and counted collapsed, as in JS.
J10 = Giovanni's ranking judge, same build as j10.py (15 per chunk, 500 characters, seed 0).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from steering.judge_sweep import j10 as J10
from steering.judge_sweep import prompts as P
from steering.judge_sweep.build import ALPHA, PAIRS, PM, ROUTES, key

import os
V2 = os.environ.get("CLAMP_V") == "2"     # version 2 (clamp_build2): routes in mean answer-token units, S = 1, 4
SV = os.environ.get("CLAMP_S", "1")       # version 2: which strength this judging set holds
JS, OUT = PM / "JS", PM / (("JSD_S" + SV) if V2 else "JSC")
if os.environ.get("GIO_L"):
    OUT = PM / ("JSG_L%s_S%s" % (os.environ["GIO_L"], SV))
_S = "STD" if V2 else "STC"
DIRS = {"8b": (PM / (_S + "15"), 15, PM / "J_ST15"), "14b": (PM / "qwen3_14b" / (_S + "21"), 21, PM / "qwen3_14b" / "J_ST21")}
N_BATCH = 3


def gio_answers():
    """run 2 (steering.gio_repro.qwen3_route): our routes with his clamp in his setup, one layer and strength
    per judging set (GIO_L, CLAMP_S); model label '8b'."""
    L = int(os.environ["GIO_L"])
    rows = [json.loads(l) for f in sorted((PM / "GIO8B_R").glob("gen_L%d_*_S%s.jsonl" % (L, SV))) for l in open(f)]
    g = pd.DataFrame(rows)
    g["start"] = "default"
    g["cell_id"] = ["G8_L%d_%s_%s_S%s_%s" % (L, r.pair.replace(".", "p"), r.route, SV, str(r.s).replace(".", "p")) for r in g.itertuples()]
    g["q_idx"], g["para"], g["sample"] = g.qi, 0, 0
    g["key"] = [key(r) for _, r in g.iterrows()]
    g["model"], g["layer"], g["sys_role"] = "8b", L, None
    from steering.followups.common import degeneracy
    g["collapsed"] = [degeneracy(t)["rep4"] > 10 for t in g.response]     # 140-token answers: 4-gram rule scaled from 40 at 1024
    return g


def new_answers():
    if os.environ.get("GIO_L"):
        return gio_answers()
    out = []
    for m, (d, L, _) in DIRS.items():
        spec = json.loads((d / "cells.json").read_text())
        cells = pd.DataFrame(spec["cells"])
        if V2:     # read the finished chunks directly, so one strength can be judged before the other is done
            g = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted((d / "rows").glob("c*.csv")) if not f.name.endswith(".tmp.csv")],
                          ignore_index=True)
        else:
            g = pd.read_csv(d / "generations.csv", low_memory=False)
        g = g.merge(cells[["cell_id", "pair", "route", "s", "target", "start"] + (["strength"] if V2 else [])], on="cell_id")
        if V2:
            g = g[g.strength == float(SV)]
        g["key"] = [key(r) for _, r in g.iterrows()]
        g["model"], g["layer"], g["sys_role"] = m, L, None
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build():
    for s in ("batches", "sys", "labels", "rank"):
        (OUT / s).mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(JS / "answers.csv", low_memory=False)
    refs = old[old.route == "ref"] if not os.environ.get("GIO_L") else old.iloc[0:0]   # run 2 has no prompted references
    new = new_answers()
    cols = list(old.columns)
    allans = pd.concat([new.reindex(columns=cols), refs], ignore_index=True)
    allans.to_csv(OUT / "answers.csv", index=False)
    jobs = []
    opts_of = {(j["model"], j["pair"]): j["options"] for j in json.loads((JS / "manifest.json").read_text())["jobs"] if j["judge"] == "J2"}
    for m, (d, L, jdir) in DIRS.items():
        if os.environ.get("GIO_L") and m != "8b":
            continue
        for pair in PAIRS:
            dp = new[(new.model == m) & (new.pair == pair)]
            stem = "%s_%s" % (m, pair.replace(".R.", "-"))
            rows = dp[["key", "question", "response"]].to_dict("records")
            random.Random(stem).shuffle(rows)
            target = dp.target.iloc[0]
            opts = opts_of[(m, pair)]
            for b in range(N_BATCH):
                bname = "%s_b%d" % (stem, b)
                part = rows[b::N_BATCH]
                (OUT / "batches" / (bname + ".json")).write_text(json.dumps(part, ensure_ascii=False))
                for J in P.JUDGES:
                    v = P.version(0, b)
                    salt = "%s|%s" % (J, stem if J in ("J2", "J5") else "all")
                    name = "%s_%s_d0" % (J, bname)
                    (OUT / "sys" / (name + ".txt")).write_text(P.system(J, v, opts, target, salt))
                    jobs.append(dict(name=name, judge=J, model=m, layer=L, pair=pair, draw=0, b=b, a=v[0], c=v[1], d=v[2],
                                     batch=bname, n=len(part), options=opts if J in ("J2", "J5") else None, target=target))
                # J1: the control prompt of the pair, collapsed answers not sent (as in JS)
                keep = set(dp.key[~dp.collapsed.astype(str).eq("True")])
                p1 = [x for x in part if x["key"] in keep]
                n1 = "J1_%s_d0" % bname
                (OUT / "batches" / (n1 + ".json")).write_text(json.dumps(p1, ensure_ascii=False))
                (OUT / "sys" / (n1 + ".txt")).write_text((jdir / "sys" / ("%s_L%d.txt" % (pair.replace(".", "_"), L))).read_text())
                jobs.append(dict(name=n1, judge="J1", model=m, layer=L, pair=pair, draw=0, b=b, batch=n1, n=len(p1), target=target))
    # J10 chunks over every new answer, as j10.build
    rows = new[["model", "key", "response"]].to_dict("records")
    random.Random(J10.SEED).shuffle(rows)
    head = J10.HEAD.replace("__ROLES__", ", ".join(J10.roles()))
    index = {}
    for n in range((len(rows) + J10.PER_CHUNK - 1) // J10.PER_CHUNK):
        text = head
        for i, r in enumerate(rows[n * J10.PER_CHUNK:(n + 1) * J10.PER_CHUNK], 1):
            index["%d:%d" % (n, i)] = dict(model=r["model"], key=r["key"])
            text += "%d. %s\n" % (i, " ".join(str(r["response"]).split())[:J10.CLIP])
        (OUT / "rank" / ("chunk_%03d.txt" % n)).write_text(text)
        jobs.append(dict(name="J10_chunk_%03d" % n, judge="J10", kind="rank", batch="chunk_%03d" % n))
    (OUT / "rank" / "index.json").write_text(json.dumps(index))
    (OUT / "manifest.json").write_text(json.dumps(dict(jobs=jobs), indent=0))
    print("%d new answers (+%d references reused), %d jobs" % (len(new), len(refs), len(jobs)))


def labels():
    """labels.csv (JS format, judges J1-J9) and J10/labels.csv for JSC: new answers from the API labels,
    references copied from JS (their answers are the same)."""
    from steering.judge_sweep import collect as CO
    from steering.judge_sweep.export import load_answers, load_j1
    jobs = json.loads((OUT / "manifest.json").read_text())["jobs"]
    ans = pd.read_csv(OUT / "answers.csv", low_memory=False)
    rows, bad = [], 0
    for j in jobs:
        f = OUT / "labels" / (j["name"] + ".json")
        if j["judge"] == "J10" or not f.exists():
            continue
        labs = json.loads(f.read_text()).get("labels", [])
        for lab in labs:
            if j["judge"] == "J1":
                r = dict(key=lab.get("key"), evidence=lab.get("evidence"), label=CO.norm(lab.get("persona_kind")),
                         intensity=lab.get("intensity"))
                e = None
            else:
                r, e = CO.check(dict(j, options=j.get("options") or []), lab)
            bad += e is not None
            rows.append({**{c: j.get(c) for c in ("name", "judge", "model", "pair", "draw", "b", "a", "c", "d")}, **r})
    new = pd.DataFrame(rows)
    # J1: collapsed answers were not sent; they count as collapsed, as in JS
    nn = ans[ans.route != "ref"]
    j1keys = set(zip(new[new.judge == "J1"].model, new[new.judge == "J1"].key))
    extra = [dict(name="J1", judge="J1", model=r.model, pair=r.pair, draw=0, key=r.key, label="collapsed",
                  analysis="Not judged: the 4-gram rule flagged it as collapsed.")
             for r in nn.itertuples() if (r.model, r.key) not in j1keys and str(r.collapsed) == "True"]
    # references: the same answers as in JS -> copy JS labels (draw 0) and the J1 control labels
    old = pd.read_csv(JS / "labels.csv", low_memory=False)
    refkeys = set(zip(ans[ans.route == "ref"].model, ans[ans.route == "ref"].key))
    isref = lambda df: pd.Series([(m, k) in refkeys for m, k in zip(df.model, df.key)], index=df.index)   # noqa: E731
    oldref = old[(old.draw == 0) & isref(old)]
    j1 = load_j1(load_answers())
    j1ref = j1[isref(j1)]
    out = pd.concat([new, pd.DataFrame(extra), oldref, j1ref], ignore_index=True)
    out.to_csv(OUT / "labels.csv", index=False)
    # J10: verdicts of the new chunks + JS J10 labels of the references
    (OUT / "J10").mkdir(exist_ok=True)
    idx = json.loads((OUT / "rank" / "index.json").read_text())
    r10 = []
    for f in sorted((OUT / "rank").glob("verdict_*.json")):
        n = int(f.stem.split("_")[1])
        for i, v in json.loads(f.read_text()).items():
            m = idx.get("%d:%s" % (n, i))
            if m and isinstance(v, dict) and len(v.get("rank", [])) >= 5:
                rk = [str(x).strip().lower().replace(" ", "_") for x in v["rank"]]
                r10.append(dict(**m, rank=json.dumps(rk), why=v.get("why"), nonsense=bool(v.get("nonsense"))))
    o10 = pd.read_csv(JS / "J10" / "labels.csv")
    o10 = o10[isref(o10)]
    pd.concat([pd.DataFrame(r10), o10], ignore_index=True).to_csv(OUT / "J10" / "labels.csv", index=False)
    print("labels: %d new (%d invalid), %d J1-collapsed, %d reference rows; J10 %d new" % (len(new), bad, len(extra),
          len(oldref) + len(j1ref), len(r10)))


if __name__ == "__main__":
    {"build": build, "labels": labels}[sys.argv[1]]()
