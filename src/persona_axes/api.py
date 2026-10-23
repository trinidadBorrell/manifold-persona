"""Claude API helper shared by the persona_axes judges: interactive requests (never Batches), one JSON
reply per call, a dollar budget (API_BUDGET, default $10) and a running cost print.
Keys: route_judge_sweep.api_judge.clients (files in token/, never printed)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}      # $ per M tokens in / out
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"


class Runner:
    def __init__(self, model, budget=None, workers=12):
        from route_judge_sweep.api_judge import clients
        self.anthropic, self.cl = clients()
        self.model, self.workers = model, workers
        self.budget = float(os.environ.get("API_BUDGET", 10)) if budget is None else budget
        self.lock, self.spent, self.n, self.refusals = threading.Lock(), 0.0, 0, 0

    def estimate(self, prompts, out_tokens):
        pin, pout = PRICE[self.model]
        n_in = sum(len(p) for p in prompts) / 3.5
        return n_in / 1e6 * pin + len(prompts) * out_tokens / 1e6 * pout

    def _create(self, system, user, max_tokens, effort):
        kw = dict(model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": user}])
        if system:
            kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if effort:
            kw["output_config"] = {"effort": effort}
        try:
            return self.cl[0].messages.create(**kw)
        except TypeError:                                   # older SDK: output_config not a named parameter
            kw.pop("output_config", None)
            return self.cl[0].messages.create(extra_body={"output_config": {"effort": effort}} if effort else None, **kw)

    def call(self, user, system=None, max_tokens=1500, effort=None, ok=None, tries=4):
        """-> (parsed JSON or None, raw text, stop_reason). Retries until `ok(parsed)` passes."""
        pin, pout = PRICE[self.model]
        raw, stop = "", None
        for _ in range(tries):
            with self.lock:
                if self.spent > self.budget:
                    raise SystemExit("budget $%.0f reached" % self.budget)
            try:
                r = self._create(system, user, max_tokens, effort)
            except self.anthropic.APIError as e:
                print("API error: %s" % str(e)[:160], flush=True)
                time.sleep(20)
                continue
            u = r.usage
            with self.lock:
                self.spent += (u.input_tokens + 1.25 * (u.cache_creation_input_tokens or 0) + 0.1 * (u.cache_read_input_tokens or 0)) / 1e6 * pin \
                    + u.output_tokens / 1e6 * pout
                self.n += 1
                if r.stop_reason == "refusal":
                    self.refusals += 1
                if self.n % 50 == 0:
                    print("%d calls, $%.2f, %d refusals" % (self.n, self.spent, self.refusals), flush=True)
            stop = r.stop_reason
            raw = "".join(b.text for b in r.content if b.type == "text")
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                v = json.loads(m.group(0)) if m else json.loads(raw.strip())
            except Exception:
                v = None
            if v is not None and (ok is None or ok(v)):
                return v, raw, stop
        return None, raw, stop

    def map(self, fn, items):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(fn, items))
