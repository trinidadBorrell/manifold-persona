"""Big Five judge through the Claude API (claude-sonnet-5-5, interactive calls, not Batches), one answer
per request, same rating instructions as the subagent judge (paper_measures.BF_SYS).

    python -m steering.personamanifold.bigfive_api --gens $PM/PCC/S1/generations.csv --out $PM/PCC/S1/bigfive.csv [--limit 200]

Resumable: answers already in --out are skipped. Prints token use and cost (Sonnet 5.5: $2 / $10 per M).
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from steering.personamanifold.paper_measures import BF_SYS, TRAITS

MODEL = "claude-sonnet-5-5"
TOK = Path(__file__).resolve().parents[2] / "token"
KEYS = [p for p in (TOK / "anthropic-workspace.txt", TOK / "anthropic-lastest.txt", TOK / "anthropic.txt", TOK / "anthropic-2.txt")
        if p.exists() and p.read_text().strip().startswith("sk-ant")]   # anthropic-workspace.txt: a workspace-scoped key
BUDGET = 60.0      # stop before spending more than this (USD); the account holds about $100
ASK = ('Question: {q}\n\nAnswer:\n[ANSWER START]\n{a}\n[ANSWER END]\n\nRate this answer. Reply with ONE JSON object and '
       'nothing else: {{"evidence": "<quote, at most 25 words>", "openness": 1-5, "conscientiousness": 1-5, '
       '"extraversion": 1-5, "agreeableness": 1-5, "neuroticism": 1-5, "degenerate": true|false}}')


def key_of(r, alpha):
    """the same key as paper_measures.gens(), so the page can join on it"""
    return "a%s|%s|%d|%d|%d" % (alpha, r["cell_id"], r["q_idx"], r["prompt_idx"], r["sample"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", required=True, help="strength label, as in PM_MEAS_RUNS")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    import anthropic
    clients = [anthropic.Anthropic(api_key=k.read_text().strip(), max_retries=8) for k in KEYS]
    cur = {"i": 0}     # which key is in use; switch to the next one if this one is refused (auth / credits)
    g = pd.read_csv(a.gens, low_memory=False)
    g["key"] = [key_of(r, a.alpha) for r in g.to_dict("records")]
    out = Path(a.out)
    done = set(pd.read_csv(out).key) if out.exists() else set()
    todo = [r for r in g.to_dict("records") if r["key"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    lock, use = threading.Lock(), {"in": 0, "out": 0, "fail": 0}
    rows = []

    def one(r):
        msg = ASK.format(q=r["question"], a=str(r["response"]))
        for attempt in range(3):
            if use["in"] / 1e6 * 2 + use["out"] / 1e6 * 10 > BUDGET:
                raise SystemExit("budget reached")
            try:
                resp = clients[cur["i"]].messages.create(model=MODEL, max_tokens=300, system=BF_SYS,
                                                         messages=[{"role": "user", "content": msg}])
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError) as e:
                with lock:
                    if cur["i"] + 1 < len(clients) and ("credit" in str(e).lower() or not isinstance(e, anthropic.BadRequestError)):
                        cur["i"] += 1
                        print("key %d refused (%s); switching to key %d" % (cur["i"], type(e).__name__, cur["i"] + 1), flush=True)
                        continue
                raise
            with lock:
                use["in"] += resp.usage.input_tokens
                use["out"] += resp.usage.output_tokens
            text = "".join(b.text for b in resp.content if b.type == "text")
            m = re.search(r"\{.*\}", text, re.S)
            try:
                j = json.loads(m.group(0))
                if all(isinstance(j.get(t), int) and 1 <= j[t] <= 5 for t in TRAITS):
                    return dict(key=r["key"], **{t: j[t] for t in TRAITS}, degenerate=bool(j.get("degenerate")),
                                evidence=j.get("evidence"))
            except Exception:
                pass
        with lock:
            use["fail"] += 1
        return None

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            x = f.result()
            if x:
                rows.append(x)
            if i % 200 == 0 or i == len(futs):
                df = pd.DataFrame(rows)
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                rows = []
                cost = use["in"] / 1e6 * 2 + use["out"] / 1e6 * 10
                print("%d/%d judged, %d failed, %d in / %d out tokens, $%.2f so far" % (i, len(futs), use["fail"], use["in"], use["out"], cost), flush=True)


if __name__ == "__main__":
    main()
