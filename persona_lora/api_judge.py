"""Stateless API judge: one call per item, every response cached to disk.

Same protocol as local_judge.py -- fresh system+user per item, free-form JSON
with `analysis` before `score`. Uses the Batches API (half price) and caches the
shared system prompt.

Spending is bounded twice: --max-usd is checked against a pre-flight estimate
from count_tokens before anything is submitted, and against actual usage after.
Every raw response is written to the cache directory keyed by sha256 of
model+system+user, so a re-run costs nothing for items already judged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .judge_protocol import CATEGORIES

JSON_RE = re.compile(r"\{.*\}", re.S)

# $/token, batch rate = half of these
PRICES = {
    "claude-sonnet-5": (2.00e-6, 10.00e-6),   # intro rate through 2026-08-31
    "claude-haiku-4-5": (1.00e-6, 5.00e-6),
    "claude-opus-5": (5.00e-6, 25.00e-6),
}
MAX_TOKENS = 400


def key(model: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    for part in (model, system, user):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


def parse(text: str):
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("score") not in CATEGORIES:
        return None
    return {"analysis": str(obj.get("analysis", ""))[:1200], "score": obj["score"]}


def cost(usage, model, batch: bool) -> float:
    pin, pout = PRICES[model]
    d = 0.5 if batch else 1.0
    n_in = getattr(usage, "input_tokens", 0) or 0
    n_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    n_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    n_out = getattr(usage, "output_tokens", 0) or 0
    return d * (n_in * pin + n_read * pin * 0.1 + n_write * pin * 1.25 + n_out * pout)


def load_rows(paths, cache: Path, model: str):
    seen, rows = set(), []
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_key"] = key(model, r["system"], r["user"])
            r["_src"] = Path(p).stem
            if r["_key"] in seen:
                continue
            seen.add(r["_key"])
            rows.append(r)
    todo = [r for r in rows if not (cache / f"{r['_key']}.json").exists()]
    return rows, todo


def estimate(client, model, rows, batch: bool) -> tuple[float, int, int]:
    """Pre-flight cost using the real tokenizer. count_tokens is free."""
    sample = rows[: min(8, len(rows))]
    counts = [
        client.messages.count_tokens(
            model=model,
            system=[{"type": "text", "text": r["system"]}],
            messages=[{"role": "user", "content": r["user"]}],
        ).input_tokens
        for r in sample
    ]
    n_in = sum(counts) / len(counts)
    sys_tok = client.messages.count_tokens(
        model=model,
        system=[{"type": "text", "text": rows[0]["system"]}],
        messages=[{"role": "user", "content": "."}],
    ).input_tokens
    pin, pout = PRICES[model]
    d = 0.5 if batch else 1.0
    # assume no cache hit: the pessimistic bound
    per = d * (n_in * pin + MAX_TOKENS * pout)
    return per * len(rows), int(n_in), int(sys_tok)


def submit(client, model, rows, cache: Path, chunk: int):
    """Submit in chunks; write each raw response to the cache as it lands."""
    spent = 0.0
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        reqs = [
            Request(
                custom_id=r["_key"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    thinking={"type": "disabled"},
                    system=[
                        {
                            "type": "text",
                            "text": r["system"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": r["user"]}],
                ),
            )
            for r in part
        ]
        b = client.messages.batches.create(requests=reqs)
        print(f"batch {b.id}  {len(part)} items  submitted", flush=True)
        while True:
            b = client.messages.batches.retrieve(b.id)
            if b.processing_status == "ended":
                break
            print(f"  {b.processing_status}  "
                  f"done={b.request_counts.succeeded} "
                  f"proc={b.request_counts.processing}", flush=True)
            time.sleep(30)

        n_ok = 0
        for res in client.messages.batches.results(b.id):
            if res.result.type != "succeeded":
                (cache / f"{res.custom_id}.err.json").write_text(
                    json.dumps({"type": res.result.type}))
                continue
            msg = res.result.message
            spent += cost(msg.usage, model, batch=True)
            text = "".join(c.text for c in msg.content if c.type == "text")
            (cache / f"{res.custom_id}.json").write_text(json.dumps({
                "key": res.custom_id, "model": msg.model, "text": text,
                "usage": msg.usage.model_dump(),
            }))
            n_ok += 1
        print(f"  cached {n_ok}/{len(part)}   running spend ${spent:.3f}", flush=True)
    return spent


def collect(rows, cache: Path, out: Path):
    n_fail = 0
    with open(out, "w") as f:
        for r in rows:
            p = cache / f"{r['_key']}.json"
            if not p.exists():
                continue
            got = parse(json.loads(p.read_text())["text"])
            if got is None:
                n_fail += 1
                got = {"analysis": "", "score": None}
            f.write(json.dumps({"id": r["id"], "src": r["_src"], **got}) + "\n")
    print(f"wrote {out}  ({n_fail} unparseable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5", choices=list(PRICES))
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ANTHROPIC_BATCH_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    rows, todo = load_rows(args.batches, cache, args.model)
    print(f"{len(rows)} items, {len(rows)-len(todo)} already cached, {len(todo)} to judge")
    if not todo:
        collect(rows, cache, Path(args.out))
        return

    est, n_in, sys_tok = estimate(client, args.model, todo, batch=True)
    print(f"system prompt {sys_tok} tok  (needs >=1024 to cache on Sonnet)")
    print(f"mean input {n_in} tok, max output {MAX_TOKENS} tok")
    print(f"upper-bound cost for {len(todo)} items: ${est:.2f}  (cap ${args.max_usd:.2f})")
    if est > args.max_usd:
        n_fit = int(args.max_usd / (est / len(todo)))
        sys.exit(f"ABORT: estimate exceeds cap. ${args.max_usd:.2f} covers ~{n_fit} items.")
    if args.dry_run:
        print("dry run: nothing submitted")
        return

    spent = submit(client, args.model, todo, cache, args.chunk)
    print(f"actual spend ${spent:.3f} (estimate was ${est:.2f})")
    collect(rows, cache, Path(args.out))


if __name__ == "__main__":
    main()
