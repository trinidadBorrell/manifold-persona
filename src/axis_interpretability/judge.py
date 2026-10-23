"""Run the judges through the Claude API (claude-sonnet-5-5, interactive requests, not Batches).
One JSON per call: {"id", "model", "prompt", "reply", "usage"}. Calls already on disk are skipped.

    python -m axis_interpretability.judge --dir $PM/AXI_L28 --stage exp1|label|sort|rank|anchor [--only PCA1_s0] [--dry-run]

`sort` needs the labels: it sorts the held-out half (the score) and the seen half (upper bound only).
`rank` (Exp 4) needs the labels too: it orders 20 roles from along the whole axis, from pole A to pole B.
`anchor` (Exp 5) needs no label: 5 + 5 example roles at the poles, then the same kind of ordering.
Keys and client: route_judge_sweep.api_judge.clients (token/). Stops at API_BUDGET (default $10).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from axis_interpretability import prompts as P
from route_judge_sweep.api_judge import clients

MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.0, 10.0          # $ per M tokens
BUDGET = float(os.environ.get("API_BUDGET", 10))
OUT_EST = {"exp1": 700, "label": 500, "sort": 1600, "rank": 900, "anchor": 900}


def ok_rate(v, _):
    return all(isinstance(v.get(f), int) and 1 <= v[f] <= 5 for f in ("score_A", "score_B", "score_contrast"))


def ok_label(v, _):
    return all(isinstance(v.get(f), str) and v[f].strip() for f in ("pole_A", "pole_B", "axis"))


def ok_sort(v, roles):
    return set(v) >= set(roles) and all(isinstance(v[r], dict) and v[r].get("pole") in ("A", "B") for r in roles)


def ok_rank(v, roles):
    o = v.get("order")
    return isinstance(o, list) and len(o) == len(roles) and set(o) == set(roles)


def tasks(D, stage):
    """(out path, prompt, validator, extra) for every call of the stage"""
    if stage == "exp1":
        for it in json.loads((D / "items" / "exp1.json").read_text()):
            yield D / "exp1" / (it["id"] + ".json"), P.rate(it["A"], it["B"]), ok_rate, None
    elif stage == "anchor":
        for it in json.loads((D / "items" / "exp5.json").read_text()):
            yield D / "exp5_anchor" / (it["id"] + ".json"), P.anchor(it["A"], it["B"], it["spread"]), ok_rank, it["spread"]
    elif stage == "label":
        for it in json.loads((D / "items" / "exp2.json").read_text()):
            yield D / "exp2_label" / (it["id"] + ".json"), P.label(it["A"], it["B"]), ok_label, None
    else:
        for it in json.loads((D / "items" / "exp2.json").read_text()):
            f = D / "exp2_label" / (it["id"] + ".json")
            if not f.exists():
                continue
            lab = json.loads(f.read_text())["reply"]
            if stage == "rank":
                yield D / "exp4_rank" / (it["id"] + ".json"), P.rank(lab, it["spread"]), ok_rank, it["spread"]
                continue
            for half in ("held", "seen"):
                yield D / "exp2_sort" / ("%s_%s.json" % (it["id"], half)), P.sort(lab, it[half]), ok_sort, it[half]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stage", required=True, choices=("exp1", "label", "sort", "rank", "anchor"))
    ap.add_argument("--only", default=None, help="substring of the call id, e.g. PCA1_s0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tries", type=int, default=4, help="attempts per call until the reply passes the check")
    a = ap.parse_args(argv)
    D = Path(a.dir)
    todo = [t for t in tasks(D, a.stage) if not t[0].exists() and (a.only is None or a.only in t[0].name)]
    est_in = sum(len(P.SYSTEM + t[1]) / 3.5 for t in todo)
    est = est_in / 1e6 * PRICE_IN + len(todo) * OUT_EST[a.stage] / 1e6 * PRICE_OUT
    print("%s: %d calls, ~%d input tokens, estimate $%.2f (budget $%.0f)" % (a.stage, len(todo), est_in, est, BUDGET))
    if a.dry_run:
        if todo:
            print("---- first prompt ----\n" + todo[0][1])
        return
    anthropic, cl = clients()
    lock, spent = threading.Lock(), {"$": 0.0, "n": 0}

    def one(t):
        path, prompt, ok, extra = t
        for _ in range(a.tries):
            with lock:
                if spent["$"] > BUDGET:
                    raise SystemExit("budget reached")
            try:
                r = cl[0].messages.create(model=MODEL, max_tokens=4000, system=P.SYSTEM,
                                          messages=[{"role": "user", "content": prompt}])
            except anthropic.APIError as e:      # the client already retried 8 times; wait and try again
                print("API error on %s: %s" % (path.name, str(e)[:200]), flush=True)
                time.sleep(30)
                continue
            u = dict(input=r.usage.input_tokens, output=r.usage.output_tokens)
            with lock:
                spent["$"] += u["input"] / 1e6 * PRICE_IN + u["output"] / 1e6 * PRICE_OUT
            if r.stop_reason == "refusal":
                print("refusal on %s" % path.name, flush=True)
            text = "".join(b.text for b in r.content if b.type == "text")
            m = re.search(r"\{.*\}", text, re.S)
            try:
                v = json.loads(m.group(0))
            except Exception:
                continue
            if ok(v, extra):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(id=path.stem, model=MODEL, prompt=prompt, reply=v, raw=text, usage=u),
                                           ensure_ascii=False, indent=1))
                with lock:
                    spent["n"] += 1
                    if spent["n"] % 10 == 0:
                        print("%d/%d, $%.2f" % (spent["n"], len(todo), spent["$"]), flush=True)
                return True
        print("FAILED after %d tries:" % a.tries, path.name, flush=True)
        return False

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, todo))
    print("done: %d ok, %d failed, $%.2f" % (sum(res), len(res) - sum(res), spent["$"]))


if __name__ == "__main__":
    main()
