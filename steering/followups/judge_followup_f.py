"""Judge the exploratory follow-up F (replace arm: t > 1 and subspace rank 16 / 256).

Same judge, prompt, options and validation as judge_followups.py; results go to
judge/judgements_F.jsonl. The spend cap is SHARED with the main judge: money already
recorded in judge/judgements.jsonl counts against --max-usd.

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.judge_followup_f --yes
"""
import argparse
import glob
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from steering.followups import judge_followups as J
from steering.followups.common import OUT_ROOT, ROUTES, Geom


def spent(path):
    s = 0.0
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                s += float(json.loads(line).get("cost") or 0.0)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    files = sorted(glob.glob(str(root / "followup_F" / "cells" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(root / "followup_F" / "generations.parquet", index=False)
    df = df[df.kind == "identity"].copy()
    df["key"] = [J.key_of(r) for _, r in df.iterrows()]
    out = root / "judge" / "judgements_F.jsonl"
    done = J.done_keys(out)
    coll = df[df.deg_rep4 > J.REP4_COLLAPSED]
    todo = df[(df.deg_rep4 <= J.REP4_COLLAPSED) & ~df.key.isin(done)]
    G = Geom()
    personas = {r: J.route_personas(G, r) for r in ROUTES}
    systems = {r: J.system_for(personas[r]) for r in ROUTES}
    prior = spent(root / "judge" / "judgements.jsonl") + spent(out)
    print("F: %d identity rows, %d collapsed, %d to judge; spent so far $%.2f of $%.2f"
          % (len(df), len(coll), len(todo), prior, a.max_usd))
    if not a.yes:
        return
    client = J.OpenRouterJudge((J._REPO / "token" / "open-router.txt").read_text().strip(),
                               J.MODELS["openrouter"])
    budget = J.Budget(a.max_usd, prior)
    lock = threading.Lock()
    with out.open("a") as fh:
        for _, r in coll[~coll.key.isin(done)].iterrows():
            fh.write(json.dumps(dict(key=r.key, route=r.route, persona_kind="collapsed",
                                     error=None, cost=0.0)) + "\n")

        def work(r):
            res = J.judge_one(client, systems[r.route], r.question, r.response,
                              set(personas[r.route]) | {"other"}, budget)
            res.update(key=r.key, route=r.route)
            with lock:
                fh.write(json.dumps(res, default=str) + "\n")
                fh.flush()
            return res

        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for f in as_completed([ex.submit(work, r) for _, r in todo.iterrows()]):
                f.result()
    print("F judged; total spent $%.2f" % budget.spent)


if __name__ == "__main__":
    main()
