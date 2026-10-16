"""Run any sweep judge through the Claude API (claude-sonnet-5-5, interactive requests, not Batches):
the judge's own instructions as the system prompt, a slice of its batch as the user message, the
same labels file the subagent judge writes (labels/<job>.json, {"labels": [{key, ...fields}]}).

    python -m route_judge_sweep.api_judge --dir $PM/JSC --jobs manifest.json [--per 15] [--workers 24]

J10 (rank chunks) is handled the same way: its chunk text is the user message, its verdict JSON is kept.
Key: token/anthropic-lastest.txt (+ token/anthropic-workspace.txt as anthropic-workspace-id), fallbacks
token/anthropic.txt, anthropic-2.txt. Stops at the budget. Sonnet 5.5: $2 / $10 per M tokens.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "claude-sonnet-5-5"
TOK = Path(__file__).resolve().parents[2] / "token"
BUDGET = float(__import__("os").environ.get("API_BUDGET", 60))
ASK = ("Label every item below by following your instructions. Each item is "
       '{"key", "question", "response"}; read each response to its end.\n\nITEMS:\n%s\n\n'
       'Reply with ONE JSON object and nothing else: {"labels": [ {"key": "<copied verbatim>", ...the fields '
       "your instructions ask for, in their order} , ... ]}, exactly one entry per item.")


def clients():
    import anthropic
    # token/anthropic-workspace.txt holds a workspace-scoped key (works without a header); then the others
    ks = [p for p in (TOK / "anthropic-workspace.txt", TOK / "anthropic-lastest.txt", TOK / "anthropic.txt", TOK / "anthropic-2.txt")
          if p.exists() and p.read_text().strip().startswith("sk-ant")]
    return anthropic, [anthropic.Anthropic(api_key=k.read_text().strip(), max_retries=8) for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--jobs", required=True, help="manifest json: {jobs: [{name, batch, sys?, kind?}]}")
    ap.add_argument("--per", type=int, default=15)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    D = Path(a.dir)
    anthropic, cl = clients()
    cur, lock, use = {"i": 0}, threading.Lock(), {"in": 0, "out": 0}

    def call(system, user, max_tokens):
        for _ in range(4):
            if use["in"] / 1e6 * 2 + use["out"] / 1e6 * 10 > BUDGET:
                raise SystemExit("budget reached")
            try:
                # the judge's instructions are the same for every slice of its batch: cache them
                r = cl[cur["i"]].messages.create(model=MODEL, max_tokens=max_tokens,
                                                 system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                                 messages=[{"role": "user", "content": user}])
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                with lock:
                    if cur["i"] + 1 < len(cl):
                        cur["i"] += 1
                        continue
                raise
            with lock:   # cost-weighted input: cache writes 1.25x, cache reads 0.1x
                use["in"] += r.usage.input_tokens + 1.25 * (r.usage.cache_creation_input_tokens or 0) + 0.1 * (r.usage.cache_read_input_tokens or 0)
                use["out"] += r.usage.output_tokens
            text = "".join(b.text for b in r.content if b.type == "text")
            m = re.search(r"\{.*\}", text, re.S)
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
        return None

    def part(j, i):
        """one request: a J10 chunk, or items i:i+per of a judge batch -> their labels"""
        if j.get("kind") == "rank":
            return call("You follow the instructions in the user message exactly.", (D / "rank" / (j["batch"] + ".txt")).read_text(), 4000)
        system = (D / "sys" / ((j.get("sys") or j["name"]) + ".txt")).read_text()
        items = json.loads((D / "batches" / (j["batch"] + ".json")).read_text())[i:i + a.per]
        v = call(system, ASK % json.dumps(items, ensure_ascii=False), 400 * len(items) + 500)
        got = {x.get("key"): x for x in (v or {}).get("labels", [])}
        return [got[x["key"]] for x in items if x["key"] in got]

    jobs = [j for j in json.loads((D / a.jobs).read_text())["jobs"] if not (D / "labels" / (j["name"] + ".json")).exists()]
    (D / "labels").mkdir(exist_ok=True)
    tasks = []
    for j in jobs:
        if j.get("kind") == "rank":
            tasks.append((j, 0))
        else:
            n = len(json.loads((D / "batches" / (j["batch"] + ".json")).read_text()))
            tasks += [(j, i) for i in range(0, n, a.per)]
    res = {}
    # phase 1 writes each judge's instructions to the cache (one request per distinct prompt);
    # phase 2 sends the other slices, which then read the cache at a tenth of the price
    first, seen = [], set()
    for j, i in tasks:
        sysname = None if j.get("kind") == "rank" else (j.get("sys") or j["name"])
        txt = None if sysname is None else (D / "sys" / (sysname + ".txt")).read_text()
        if txt is None or txt not in seen:
            first.append((j, i))
            if txt is not None:
                seen.add(txt)
    rest = [t for t in tasks if t not in first]
    k = 0
    for phase in (first, rest):
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(part, j, i): (j["name"], i) for j, i in phase}
            for f in as_completed(futs):
                res[futs[f]] = f.result()
                k += 1
                if k % 25 == 0 or k == len(tasks):
                    print("%d/%d requests, %d in / %d out (cost-weighted) tokens, $%.2f" % (k, len(tasks), use["in"], use["out"],
                          use["in"] / 1e6 * 2 + use["out"] / 1e6 * 10), flush=True)
    for j in jobs:
        if j.get("kind") == "rank":
            v = res.get((j["name"], 0))
            if v:
                (D / "rank" / (j["batch"].replace("chunk", "verdict") + ".json")).write_text(json.dumps(v))
                (D / "labels" / (j["name"] + ".json")).write_text("{}")
            continue
        labs = [x for (n, i), v in sorted(res.items(), key=lambda t: (t[0][0], t[0][1])) if n == j["name"] for x in (v or [])]
        (D / "labels" / (j["name"] + ".json")).write_text(json.dumps({"labels": labs}, ensure_ascii=False))
    print("done; $%.2f" % (use["in"] / 1e6 * 2 + use["out"] / 1e6 * 10))


if __name__ == "__main__":
    main()
