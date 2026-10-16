"""Judge the ACE study's answers: the follow-ups' judge, with this study's options.

The rubric, client, retry/validation and spend accounting are
`steering/followups/judge_followups.py`, unchanged: route_judge_v3's
examples_prioritize prompt, anthropic/claude-sonnet-5 via OpenRouter,
reasoning off, system prompt cached, one call per answer, interactive API.

What differs is the option list: A, the 8 manifold_v1 knots of THIS route at
THIS layer, B, and `other` (grid.route_personas) -- the personas the manifold
arm actually steers through. One list per (route, layer), shared by all three
arms, so "% of answers in category X" is the same measurement for each arm.

Collapsed answers (rep4 > REP4_COLLAPSED) are recorded as `collapsed` and not sent.

    python -m steering.ace_study.judge --step 1              # dry run: estimates + one prompt
    python -m steering.ace_study.judge --step 1 --yes        # spends
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from steering.ace_study.common import OUT_ROOT, REP4_COLLAPSED, route_slug
from steering.ace_study.grid import route_personas
from steering.followups.judge_followups import (CHARS_PER_TOKEN, MODELS, Budget, OpenRouterJudge,
                                                done_keys, judge_one, system_for)
from steering.route_judge_v2 import user_msg

_REPO = Path(__file__).resolve().parents[2]
CAP_USD = 180.0          # the whole ACE study, both steps (plan, approved 2026-09-29)
USD_PER_CALL = 0.0025    # plan's estimate for 512-token answers


def key_of(r):
    return "%s|%d|%d|%d" % (r["cell_id"], int(r["prompt_idx"]), int(r["q_idx"]), int(r["sample"]))


def spent_so_far(judge_dir):
    tot = 0.0
    for p in Path(judge_dir).glob("judgements_*.jsonl"):
        for line in p.read_text().splitlines():
            try:
                tot += float(json.loads(line).get("cost") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pending")
    ap.add_argument("--max-usd", type=float, default=CAP_USD)
    ap.add_argument("--yes", action="store_true", help="actually call the API (costs money)")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    sub = "smoke" if a.step == 0 else "step%d" % a.step
    jdir = root / "judge"
    jdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / sub / "generations.csv")
    df = df.assign(key=[key_of(r) for _, r in df.iterrows()])

    pairs = sorted({(r, int(L)) for r, L in zip(df.route, df.layer)})
    personas = {p: route_personas(*p) for p in pairs}
    systems = {p: system_for(personas[p]) for p in pairs}
    (jdir / "prompts").mkdir(exist_ok=True)
    for (r, L), s in systems.items():
        (jdir / "prompts" / ("system_%s_L%d.txt" % (route_slug(r), L))).write_text(s)
    opts_p = jdir / "route_options.json"
    old = json.loads(opts_p.read_text()) if opts_p.exists() else {}
    opts_p.write_text(json.dumps({**old, **{"%s|L%d" % p: v for p, v in personas.items()}}, indent=1))

    res_path = jdir / ("judgements_%s.jsonl" % sub)
    done = done_keys(res_path)
    coll = df[df.deg_rep4 > REP4_COLLAPSED]
    todo = df[(df.deg_rep4 <= REP4_COLLAPSED) & ~df.key.isin(done)]
    if a.limit is not None:
        todo = todo.head(a.limit)
    ins = np.array([(len(systems[(r.route, int(r.layer))]) + len(user_msg(r.question, r.response)))
                    / CHARS_PER_TOKEN for _, r in todo.iterrows()]) if len(todo) else np.zeros(1)
    prior = spent_so_far(jdir)
    print("answers %d | collapsed (not sent) %d | already judged %d | pending calls %d"
          % (len(df), len(coll), len(done), len(todo)))
    print("input tokens/call ~%.0f (chars/%.1f) | est. cost $%.2f at $%.4f/call | spent so far $%.2f of cap $%.0f"
          % (ins.mean(), CHARS_PER_TOKEN, len(todo) * USD_PER_CALL, USD_PER_CALL, prior, a.max_usd))
    if not a.yes:
        if len(todo):
            r = todo.iloc[0]
            print("\n----- SYSTEM (%s, L%d) -----\n%s" % (r.route, r.layer, systems[(r.route, int(r.layer))]))
            print("\n----- USER -----\n" + user_msg(r.question, r.response))
        print("\nDRY RUN -- no API calls. Pass --yes to spend.")
        return
    model = MODELS["openrouter"]
    client = OpenRouterJudge((_REPO / "token" / "open-router.txt").read_text().strip(), model)
    budget = Budget(a.max_usd, prior)
    lock = threading.Lock()
    with res_path.open("a") as fh:
        for _, r in coll[~coll.key.isin(done)].iterrows():
            fh.write(json.dumps(dict(key=r.key, persona_kind="collapsed", intensity=None,
                                     evidence="", analysis="rep4>%d" % REP4_COLLAPSED,
                                     error=None, raw="", model=None, cost=0.0)) + "\n")
        fh.flush()

        def work(r):
            p = (r.route, int(r.layer))
            allowed = set(personas[p]) | {x.replace("_", " ") for x in personas[p]} | {"other"}
            res = judge_one(client, systems[p], r.question, r.response, allowed, budget)
            if res.get("persona_kind"):
                res["persona_kind"] = res["persona_kind"].replace(" ", "_")   # coral reef -> coral_reef
            res.update(key=r.key, model=model)
            with lock:
                fh.write(json.dumps(res, default=str) + "\n")
                fh.flush()
            return res

        n = 0
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(work, r) for _, r in todo.iterrows()]
            for f in as_completed(futs):
                n += 1
                res = f.result()
                if n % 100 == 0 or res["error"]:
                    print("  %d/%d  $%.2f  %s -> %s %s" % (n, len(todo), budget.spent, res["key"],
                                                          res["persona_kind"], res["error"] or ""),
                          flush=True)
    print("wrote %s | spent $%.2f total (cap $%.0f)" % (res_path, budget.spent, a.max_usd))
    (jdir / "spend.json").write_text(json.dumps(dict(spent_usd=budget.spent, cap_usd=a.max_usd)))


if __name__ == "__main__":
    main()
