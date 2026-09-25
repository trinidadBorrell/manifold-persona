"""Judge the per-answer-start run (output/steering-per-answer-25_09).

Same judge, prompt, route options and validation as judge_followups.py. The $30 spend cap
is shared with every earlier judging run (their recorded cost counts against it).

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.judge_per_answer --yes
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from steering.followups import judge_followups as J
from steering.followups.common import Geom
from steering.followups.judge_followup_f import spent

ROUTE = "validator>vampire"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="output/steering-per-answer-25_09")
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    df = pd.read_csv(run / "generations.csv")
    df["key"] = ["q%d|s%d|%s|%g" % (r.q_idx, r["sample"], r.arm, r.alpha) for _, r in df.iterrows()]
    out = run / "judgements.jsonl"
    done = J.done_keys(out)
    coll = df[df.collapsed.astype(bool)]
    todo = df[~df.collapsed.astype(bool) & ~df.key.isin(done)]
    G = Geom()
    options = J.route_personas(G, ROUTE)
    system = J.system_for(options)
    (run / "judge_system_prompt.txt").write_text(system)
    fix = Path("output/steering-fix-24_09/judge")
    prior = spent(fix / "judgements.jsonl") + spent(fix / "judgements_F.jsonl") + spent(out)
    print("%d rows, %d collapsed, %d to judge; spent so far $%.2f of $%.2f"
          % (len(df), len(coll), len(todo), prior, a.max_usd))
    if not a.yes:
        return
    client = J.OpenRouterJudge((J._REPO / "token" / "open-router.txt").read_text().strip(),
                               J.MODELS["openrouter"])
    budget = J.Budget(a.max_usd, prior)
    lock = threading.Lock()
    with out.open("a") as fh:
        for _, r in coll[~coll.key.isin(done)].iterrows():
            fh.write(json.dumps(dict(key=r.key, persona_kind="collapsed", error=None, cost=0.0)) + "\n")

        def work(r):
            res = J.judge_one(client, system, r.question, r.response, set(options) | {"other"}, budget)
            res.update(key=r.key)
            with lock:
                fh.write(json.dumps(res, default=str) + "\n")
                fh.flush()

        with ThreadPoolExecutor(max_workers=12) as ex:
            for f in as_completed([ex.submit(work, r) for _, r in todo.iterrows()]):
                f.result()
    print("judged; total spent $%.2f" % budget.spent)


if __name__ == "__main__":
    main()
