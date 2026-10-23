"""Judge inputs for a set of steered answers (ceiling, later Phase 3), in the route_judge_sweep.api_judge layout:

    python -m persona_axes.judge_build --stage ceiling        # -> $PM/AX21/judge_ceiling/{answers.csv, rank/, batches/, sys/, manifest.json}
    API_BUDGET=25 python -m route_judge_sweep.api_judge --dir $PM/AX21/judge_ceiling --jobs manifest.json --workers 16
    python -m persona_axes.judge_build --stage ceiling --collect   # -> labels.csv (one row per answer: J10 rank, nonsense, J8 label)

J10 = Giovanni's ranking judge (route_judge_sweep.j10: 15 answers per chunk, first 500 characters, ranks 10
names, nonsense flag), option list = 276 roles + the 41 set (a) personas. J8 = J2 over the same 317 names
with one-line definitions (route_judge_sweep.prompts.j2_system), on a 25% random subsample of the answers
(seed 0), version (0, 0, 0).
"""
from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from route_judge_sweep import j10 as J10
from route_judge_sweep import prompts as P

from persona_axes import AX, HERE

J8_SHARE, PER_BATCH = 0.25, 60


def option_lines():
    lines = dict(P.role_lines())
    for p in json.loads((HERE / "personas_a.json").read_text())["personas"]:
        lines[p["name"]] = p["line"]
    return lines


def build(stage, no_j8=False, pattern="gen_*.jsonl", out=None):
    """pattern/out: judge a subset of a stage's files into its own judge dir (e.g. the matched-push single-axis run)"""
    src, D = AX / stage, AX / ("judge_" + (out or stage))
    for s in ("batches", "sys", "labels", "rank"):
        (D / s).mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(src.glob(pattern)):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            r["key"] = "%s|%s|%d|%d" % (r["persona"], r["cell"], r["seed"], r["qi"])
            rows.append(r)
    ans = pd.DataFrame(rows)
    ans.to_csv(D / "answers.csv", index=False)
    lines = option_lines()
    names = sorted(lines)
    jobs = []
    # J10 chunks
    items = ans[["key", "response"]].to_dict("records")
    random.Random(J10.SEED).shuffle(items)
    head = J10.HEAD.replace("__ROLES__", ", ".join(names))
    index = {}
    for n in range((len(items) + J10.PER_CHUNK - 1) // J10.PER_CHUNK):
        text = head
        for i, r in enumerate(items[n * J10.PER_CHUNK:(n + 1) * J10.PER_CHUNK], 1):
            index["%d:%d" % (n, i)] = dict(key=r["key"])
            text += "%d. %s\n" % (i, " ".join(str(r["response"]).split())[:J10.CLIP])
        (D / "rank" / ("chunk_%03d.txt" % n)).write_text(text)
        jobs.append(dict(name="J10_chunk_%03d" % n, judge="J10", kind="rank", batch="chunk_%03d" % n))
    (D / "rank" / "index.json").write_text(json.dumps(index))
    # J8 on a 25% subsample (dropped with --no-j8: budget decision 2026-10-21, J8/J10 agreement 90-97%)
    sub = [] if no_j8 else ans.sample(frac=J8_SHARE, random_state=0)[["key", "question", "response"]].to_dict("records")
    P.role_lines = lambda: lines                      # J8 definitions must cover the set (a) names
    P.usable = lambda r: r in P.curated() and P.curated()[r]["intensity"] > 0   # worked examples only from curated roles
    for b in range(0, len(sub), PER_BATCH):
        bname = "J8_b%03d" % (b // PER_BATCH)
        (D / "batches" / (bname + ".json")).write_text(json.dumps(sub[b:b + PER_BATCH], ensure_ascii=False))
        (D / "sys" / (bname + ".txt")).write_text(P.j2_system(names, (0, 0, 0), "J8|ax21", defs="lines"))
        jobs.append(dict(name=bname, judge="J8", batch=bname, n=len(sub[b:b + PER_BATCH])))
    (D / "manifest.json").write_text(json.dumps(dict(jobs=jobs), indent=0))
    (D / "options.json").write_text(json.dumps(names))
    print("%d answers, %d J10 chunks, %d J8 batches (%d answers) -> %s" % (
        len(ans), sum(j["judge"] == "J10" for j in jobs), sum(j["judge"] == "J8" for j in jobs), len(sub), D))


def collect(stage):
    D = AX / ("judge_" + stage)
    ans = pd.read_csv(D / "answers.csv")
    index = json.loads((D / "rank" / "index.json").read_text())
    j10 = {}
    for f in sorted((D / "rank").glob("verdict_*.json")):
        n = int(f.stem.split("_")[1])
        v = json.loads(f.read_text())
        for i, item in v.items():
            k = index.get("%d:%s" % (n, i))
            if k and isinstance(item, dict):
                j10[k["key"]] = dict(rank=item.get("rank") or [], nonsense=bool(item.get("nonsense")))
    j8 = {}
    for f in sorted((D / "labels").glob("J8_b*.json")):
        for x in json.loads(f.read_text()).get("labels", []):
            j8[x.get("key")] = x.get("persona_kind")
    out = []
    for r in ans.itertuples():
        t = j10.get(r.key)
        pos = (t["rank"].index(r.persona) + 1) if t and r.persona in t["rank"] else None
        out.append(dict(key=r.key, persona=r.persona, set=r.set, cell=r.cell, S=getattr(r, "S", None), seed=r.seed, qi=r.qi,
                        j10_judged=t is not None, j10_rank=pos, j10_nonsense=None if t is None else t["nonsense"],
                        j10_top5=None if t is None else (pos is not None and pos <= 5 and not t["nonsense"]),
                        j10_mrr=None if t is None else (0.0 if (pos is None or t["nonsense"]) else 1.0 / pos),
                        j8_label=j8.get(r.key), j8_reached=None if r.key not in j8 else (j8[r.key] == r.persona),
                        j8_collapsed=None if r.key not in j8 else (j8[r.key] == "collapsed")))
    df = pd.DataFrame(out)
    df.to_csv(D / "labels.csv", index=False)
    print("labels: %d answers, J10 on %d, J8 on %d" % (len(df), df.j10_judged.sum(), df.j8_label.notna().sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--no-j8", action="store_true")
    ap.add_argument("--glob", default="gen_*.jsonl", help="which gen files of the stage to judge")
    ap.add_argument("--out", default=None, help="judge dir suffix (default: the stage name)")
    a = ap.parse_args(argv)
    collect(a.stage) if a.collect else build(a.stage, a.no_j8, a.glob, a.out)


if __name__ == "__main__":
    main()
