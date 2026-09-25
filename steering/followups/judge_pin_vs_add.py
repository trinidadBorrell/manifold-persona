"""Judge the pin-vs-add answers (output/pin-vs-add-25_09) with the same judge as every other run.

Same prompt, route options (source + nearest-8 knots + target + other) and validation as
judge_followups.py. Results go to output/steering-story-25_09/judge_pin_vs_add.jsonl; the $30 cap
is shared with all earlier judging.

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.judge_pin_vs_add --yes
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
from steering.run_steering import INTROSPECTIVE_QUESTIONS

SRC = Path("output/pin-vs-add-25_09")
OUT = Path("output/steering-story-25_09")
ROUTES = ["assistant>vampire", "assistant>bard"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(SRC / ("responses_%s.csv" % r.replace(">", "-"))) for r in ROUTES],
                   ignore_index=True)
    df["question"] = [INTROSPECTIVE_QUESTIONS[int(q)] for q in df.q]
    df["key"] = ["%s|%s|%g|%d|%d" % (r.route, r.method, r.strength, r.q, r["sample"]) for _, r in df.iterrows()]
    out = OUT / "judge_pin_vs_add.jsonl"
    done = J.done_keys(out)
    todo = df[~df.key.isin(done)]
    G = Geom()
    opts = {r: J.route_personas(G, r) for r in ROUTES}
    systems = {r: J.system_for(opts[r]) for r in ROUTES}
    prior = sum(spent(p) for p in [Path("output/steering-fix-24_09/judge/judgements.jsonl"),
                                   Path("output/steering-fix-24_09/judge/judgements_F.jsonl"),
                                   Path("output/steering-per-answer-25_09/judgements.jsonl"), out])
    print("%d answers, %d to judge; spent so far $%.2f of $%.2f" % (len(df), len(todo), prior, a.max_usd))
    if not a.yes:
        return
    client = J.OpenRouterJudge((J._REPO / "token" / "open-router.txt").read_text().strip(),
                               J.MODELS["openrouter"])
    budget = J.Budget(a.max_usd, prior)
    lock = threading.Lock()
    with out.open("a") as fh:
        def work(r):
            res = J.judge_one(client, systems[r.route], r.question, r.response,
                              set(opts[r.route]) | {"other"}, budget)
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
