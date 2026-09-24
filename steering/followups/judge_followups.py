"""Route judge for exps 4-5: per-route option lists, interactive API, resumable.

Plan: plans/2026-09-24-steering-followups.md (Execution; Metric).

WHAT IS JUDGED. Each identity response of Study A and Study B, against its
ROUTE's option list: the source persona, the route's knots, the target persona,
and `other`. The prompt is route_judge_v3's `build_system` in the
"examples_prioritize" configuration -- the one steering/dose_judge.py
validated and the last judged runs used -- so nothing about the rubric is new
(the plan puts a new rubric out of scope).

  knots       the nearest-rule k = 8 knots of the Study A route c_A -> c_B
              (grid.Interventions, the same object exp4 steered through). One
              fixed list per route: the manifold_v1/v2 arms and Study B's
              per-question paths thread other personas, but a per-arm list
              would make "target share" a different measurement per arm.
  collapsed   NOT a judge option. A response with rep4 > REP4_COLLAPSED is
              labelled `collapsed` mechanically and never sent (plan:
              "collapsed text is its own judge class, never dropped"); the
              judge would otherwise have to guess at "the the the".
  canary      not judged by default -- its metric is the string "Paris"
              (plan, Metric). --include-canary judges it too.

BACKENDS (no call is made without --yes):
  openrouter  (default) anthropic/claude-sonnet-5 via OpenRouter, key from
              token/open-router.txt -- the path every earlier judged run used.
              No temperature (Sonnet 5 rejects sampling params); reasoning off
              with {"reasoning": {"enabled": false}}; the system prompt is a
              content-block list with cache_control on it, because a
              top-level cache_control would place the breakpoint on the last
              block, i.e. the per-request user message, and cache nothing.
  anthropic   claude-sonnet-5 via the official SDK, interactive Messages API
              (never Batches -- user rule): thinking disabled, system prompt
              cached with cache_control, no temperature.

Usage is recorded per row (input, cached, cache-write, output tokens, and
OpenRouter's cost when it reports one) so the real bill can be computed.

RESUMABLE. Results append to <out>/judgements.jsonl, one line per response,
keyed by (study, cell_id, unit_index, sample); rows already present with no
error are skipped on the next run.

DRY RUN (--dry-run, the default without --yes) prints ONE full example prompt
and writes token estimates, calling nothing: input by the tokenizer-free
chars/3.5 heuristic; output from the rubric's max_tokens (ceiling) and from the
length of the raw judge replies already stored under prompts/ablation/*/
(typical). The SDK's count_tokens endpoint is deliberately NOT used, even
with a key present: it is an API call.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups.common import GEOM_DEFAULT, K_DEFAULT, OUT_ROOT, REP4_COLLAPSED, ROUTES, Geom
from steering.followups.grid import Interventions
from steering.route_judge_v2 import parse, user_msg
from steering.route_judge_v3 import build_system

MAX_TOKENS = 600                      # route_judge_v2/v3's own ceiling
CHARS_PER_TOKEN = 3.5
MODELS = {"openrouter": "anthropic/claude-sonnet-5", "anthropic": "claude-sonnet-5"}
_REPO = Path(__file__).resolve().parents[2]


def route_personas(G, route, k=K_DEFAULT):
    """source + nearest-k knots (chord order) + target."""
    A, B, cA, cB = G.route(route)
    p = Interventions(G, k=k).path(route, "nearest", cA, cB, "cA")
    return [A] + [G.names[int(i)] for i in p.centroid_idx] + [B]


def system_for(personas):
    return build_system(personas, with_examples=True, prioritize=True, persona_examples=False)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

class OpenRouterJudge:
    def __init__(self, key, model):
        import requests
        self.model = model
        self._s = requests.Session()
        self._s.headers.update({"Authorization": "Bearer " + key,
                                "Content-Type": "application/json"})

    def body(self, system, user):
        return {"model": self.model, "max_tokens": MAX_TOKENS,
                "reasoning": {"enabled": False},
                # no response_format: not guaranteed for Anthropic models on OpenRouter;
                # parse() is tolerant of prose around the JSON (review 2026-09-24 #11)
                "usage": {"include": True},
                "messages": [{"role": "system",
                              "content": [{"type": "text", "text": system,
                                           "cache_control": {"type": "ephemeral"}}]},
                             {"role": "user", "content": user}]}

    def __call__(self, system, user):
        r = self._s.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                         json=self.body(system, user))
        r.raise_for_status()
        j = r.json()
        if not j.get("choices"):
            raise RuntimeError("no choices: %s" % str(j)[:200])
        u = j.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        usage = dict(input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                     cache_read_tokens=det.get("cached_tokens"),
                     cache_write_tokens=det.get("cache_write_tokens"), cost=u.get("cost"))
        return j["choices"][0]["message"].get("content"), usage


class AnthropicJudge:
    def __init__(self, model):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def __call__(self, system, user):
        m = self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS, thinking={"type": "disabled"},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        u = m.usage
        usage = dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                     cache_read_tokens=getattr(u, "cache_read_input_tokens", None),
                     cache_write_tokens=getattr(u, "cache_creation_input_tokens", None),
                     cost=None)
        txt = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
        return txt, usage


# Sonnet 5 list price, used only when a backend returns no cost (anthropic direct)
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 2e-6, 10e-6, 0.2e-6


def usage_cost(u):
    if u.get("cost") is not None:
        return float(u["cost"])
    i, o, c = (u.get("input_tokens") or 0), (u.get("output_tokens") or 0), (u.get("cache_read_tokens") or 0)
    return i * PRICE_IN + o * PRICE_OUT + c * PRICE_CACHE_READ


class Budget:
    """Running spend against the user's cap, shared by all worker threads.

    Every billed attempt counts, including ones whose reply failed to parse
    (review 2026-09-24 #1). No new call starts once the cap is reached.
    """

    def __init__(self, cap_usd, spent=0.0):
        self.cap, self.spent, self.lock = float(cap_usd), float(spent), threading.Lock()

    def ok(self):
        with self.lock:
            return self.spent < self.cap

    def add(self, usd):
        with self.lock:
            self.spent += usd


def judge_one(client, system, q, a, allowed, budget, retries=4):
    last, spent = None, 0.0
    usage_all = dict(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    for k in range(retries):
        if not budget.ok():
            last = "spend cap reached"
            break
        try:
            txt, usage = client(system, user_msg(q, a))
            c = usage_cost(usage)
            spent += c
            budget.add(c)
            for kk in usage_all:
                usage_all[kk] += usage.get(kk) or 0
            d = parse(txt)
            kind = str(d.get("persona_kind") or "").strip().lower()
            if kind not in allowed:                   # review 2026-09-24 #3
                raise ValueError("persona_kind %r not in route options" % d.get("persona_kind"))
            return dict(persona_kind=kind, intensity=d.get("intensity"),
                        evidence=d.get("evidence", ""), analysis=d.get("analysis", ""),
                        error=None, raw=txt, cost=spent, attempts=k + 1, **usage_all)
        except Exception as e:                       # noqa: BLE001 - report, never crash
            last = repr(e)
            time.sleep(1.5 * (k + 1))
    return dict(persona_kind=None, intensity=None, evidence="", analysis="", error=last, raw="",
                cost=spent, attempts=retries, **usage_all)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_generations(root):
    frames = []
    for sub in ("studyA", "studyB"):
        for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            p = Path(root) / sub / ("generations" + ext)
            if p.exists():
                frames.append(fn(p))
                break
    if not frames:
        raise SystemExit("no generations under %s/studyA|studyB -- run stage collect" % root)
    return pd.concat(frames, ignore_index=True)


def key_of(r):
    return "%s|%s|%d|%d" % (r["study"], r["cell_id"], int(r["unit_index"]), int(r["sample"]))


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error") is None:
                keys.add(d["key"])
    return keys


def typical_output_tokens():
    """Char length of the raw judge replies already on disk -> tokens (chars/3.5)."""
    lens = []
    for p in (_REPO / "prompts" / "ablation").glob("*/*.md"):
        t = p.read_text(errors="ignore")
        m = re.search(r"## (?:6\. RAW JUDGE REPLY|raw)\s*\n(?:```\s*\n)?(.*?)(?:\n```|\Z)", t, re.S)
        if m and m.group(1).strip().startswith("{"):
            lens.append(len(m.group(1).strip()))
    if not lens:
        return None
    x = np.array(lens) / CHARS_PER_TOKEN
    return dict(n_replies=len(lens), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--out-dir", default=None, help="default <run-root>/judge")
    ap.add_argument("--geom", default=str(GEOM_DEFAULT))
    ap.add_argument("--backend", default="openrouter", choices=list(MODELS))
    ap.add_argument("--model", default=None, help="default per backend: %s" % MODELS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--include-canary", action="store_true")
    ap.add_argument("--max-calls", type=int, default=5000,
                    help="refuse to start if more calls than this are pending")
    ap.add_argument("--max-usd", type=float, default=30.0,
                    help="hard spend cap in USD across all runs of this judge (user: $30)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually call the API (costs money)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir or Path(a.run_root) / "judge")
    out.mkdir(parents=True, exist_ok=True)
    model = a.model or MODELS[a.backend]

    G = Geom(a.geom)
    personas = {r: route_personas(G, r) for r in ROUTES}
    systems = {r: system_for(personas[r]) for r in ROUTES}
    (out / "route_options.json").write_text(json.dumps(personas, indent=1))
    for r in ROUTES:
        (out / ("system_%s.txt" % r.replace(">", "-"))).write_text(systems[r])

    df = load_generations(a.run_root)
    if not a.include_canary:
        df = df[df.kind == "identity"]
    df = df.assign(key=[key_of(r) for _, r in df.iterrows()])
    res_path = out / "judgements.jsonl"
    done = done_keys(res_path)
    coll = df[df.deg_rep4 > REP4_COLLAPSED]
    todo = df[(df.deg_rep4 <= REP4_COLLAPSED) & ~df.key.isin(done)]
    print("responses: %d  collapsed (not sent): %d  already judged: %d  pending calls: %d"
          % (len(df), len(coll), len(done), len(todo)))

    # --- estimates (always written) ---
    typ = typical_output_tokens()
    est = []
    for _, r in todo.iterrows():
        n_in = (len(systems[r.route]) + len(user_msg(r.question, r.response))) / CHARS_PER_TOKEN
        est.append(dict(key=r.key, route=r.route, input_tokens_est=round(n_in),
                        system_tokens_est=round(len(systems[r.route]) / CHARS_PER_TOKEN),
                        output_tokens_max=MAX_TOKENS,
                        output_tokens_typical=round(typ["mean"]) if typ else None))
    ins = np.array([e["input_tokens_est"] for e in est]) if est else np.zeros(1)
    summary = dict(model=model, backend=a.backend, heuristic="chars/%.1f" % CHARS_PER_TOKEN,
                   n_requests=len(est), n_collapsed_not_sent=int(len(coll)),
                   input_tokens_per_request=dict(mean=float(ins.mean()), median=float(np.median(ins)),
                                                 max=float(ins.max())),
                   system_tokens_per_route={r: round(len(systems[r]) / CHARS_PER_TOKEN)
                                            for r in ROUTES},
                   output_tokens_per_request=dict(max=MAX_TOKENS, typical=typ),
                   total_input_tokens_est=float(ins.sum()) if est else 0.0,
                   total_output_tokens_typical=(typ["mean"] * len(est)) if typ else None,
                   total_output_tokens_max=MAX_TOKENS * len(est),
                   note="no API was called; count_tokens deliberately not used")
    (out / "token_estimates.json").write_text(json.dumps(dict(summary=summary, requests=est),
                                                          indent=1))
    print(json.dumps(summary, indent=1))

    if a.dry_run or not a.yes:
        if len(todo):
            r = todo.iloc[0]
            client = OpenRouterJudge("<redacted>", model) if a.backend == "openrouter" else None
            print("\n===== EXAMPLE REQUEST (%s, %s) =====" % (r.route, r.key))
            if client is not None:
                b = client.body(systems[r.route], user_msg(r.question, r.response))
                print(json.dumps({k: v for k, v in b.items() if k != "messages"}, indent=1))
            print("\n----- SYSTEM -----\n" + systems[r.route])
            print("\n----- USER -----\n" + user_msg(r.question, r.response))
        print("\nDRY RUN -- no API calls. Pass --yes to spend.")
        return summary

    if len(todo) > a.max_calls:
        raise SystemExit("%d pending calls exceed --max-calls %d" % (len(todo), a.max_calls))
    if a.backend == "openrouter":
        client = OpenRouterJudge((_REPO / "token" / "open-router.txt").read_text().strip(), model)
    else:
        client = AnthropicJudge(model)
    prior = 0.0
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            try:
                prior += float(json.loads(line).get("cost") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    budget = Budget(a.max_usd, prior)
    print("spend so far $%.2f of cap $%.2f" % (prior, a.max_usd))
    lock = threading.Lock()
    with res_path.open("a") as fh:
        # collapsed rows are recorded once, as their own class, never sent
        for _, r in coll[~coll.key.isin(done)].iterrows():
            fh.write(json.dumps(dict(key=r.key, route=r.route, persona_kind="collapsed",
                                     intensity=None, evidence="", analysis="rep4>%d" % REP4_COLLAPSED,
                                     error=None, raw="", model=None)) + "\n")
        fh.flush()

        def work(r):
            res = judge_one(client, systems[r.route], r.question, r.response,
                            set(personas[r.route]) | {"other"}, budget)
            res.update(key=r.key, route=r.route, model=model, backend=a.backend)
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
                if n % 25 == 0 or res["error"]:
                    print("  %d/%d  %s -> %s %s" % (n, len(todo), res["key"], res["persona_kind"],
                                                    res["error"] or ""), flush=True)
    print("wrote %s   spent $%.2f (cap $%.2f)" % (res_path, budget.spent, a.max_usd))
    (out / "spend.json").write_text(json.dumps(dict(spent_usd=budget.spent, cap_usd=a.max_usd)))


if __name__ == "__main__":
    main()
