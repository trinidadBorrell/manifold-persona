"""Run the route judge over the A->B (pair) arms.

Plan: plans/2026-09-08-steering-rebuild.md.

The D.1.3 judge in `steering.judge` answers "is this the Assistant?" and returns
`assistant` for 85-95% of A->B rows at every alpha, which is true and tells us
nothing about how far along its route the identity has travelled. This driver
runs `steering.route_judge` instead: per route, the option list is that route's
own personas, so the verdict is a POSITION rather than a category.

Only the pair arms are judged. The axis arms have no route to be positioned on;
they keep the paper's D.1.3 scores.

RESUME. Verdicts are appended to a jsonl as they land, keyed by row. A rerun
reads that file and skips what it already has, so an interrupted run costs
nothing to finish. The checkpoint is keyed to the OPTION LIST as well as the
row (`--ckpt-tag`): changing the judge's options changes what a verdict means,
and silently resuming across that change would mix two rubrics in one column.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from steering import route_judge as RJ
from steering.judge import _client


def parse_route_verdict(content, options: list) -> dict:
    """One reply -> {score, analysis}, never raising.

    NOT `steering.judge.parse_verdict`: that one validates `score` against the
    D.1.3 category set, so every legitimate route verdict (`navigator`,
    `moderator`, ...) fails it as out-of-schema. The valid enum here is the
    route's own option list, which differs per route.
    """
    text = next((b.text for b in content if getattr(b, "type", None) == "text"), None)
    if text is None:
        return {"score": None, "analysis": "NO_TEXT_BLOCK"}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {"score": None, "analysis": "BAD_JSON: %s" % text[:160]}
    if not isinstance(parsed, dict):
        return {"score": None, "analysis": "NOT_AN_OBJECT: %s" % text[:120]}
    score = parsed.get("score")
    if score not in options:
        return {"score": None, "analysis": "OUT_OF_SCHEMA: %r" % (score,)}
    return {"score": score, "analysis": parsed.get("analysis", "")}


def waypoints_by_route(manifest: dict) -> dict:
    """(A, B) -> the manifold route's interior centroid names.

    The linear arm is judged against the SAME option list as the manifold arm
    for the same (A, B) -- that comparison is the point of the experiment, so
    the waypoints come from the manifold entry regardless of which arm the row
    belongs to.
    """
    out = {}
    for r in manifest.get("routes", []):
        if r.get("kind") != "pair":
            continue
        out[(r["A"], r["B"])] = list(r.get("manifold", {}).get("centroid_roles", []))
    return out


def row_key(row) -> str:
    return "%s|%s|%s|%.3f|%d|%d" % (row.arm, row.path_a, row.path_b,
                                    row.alpha, row.system_idx, row.question_idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="run's data/ directory")
    ap.add_argument("--ckpt-tag", default="v2",
                    help="checkpoint suffix; bump whenever the option list or "
                         "rubric changes, so a resume cannot mix rubrics")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--redo-empty-analysis", action="store_true",
                    help="also re-judge rows that have a verdict but no analysis "
                         "text -- the rows whose reasoning a parser bug dropped")
    ap.add_argument("--dry-run", action="store_true",
                    help="count rows and print one prompt; no network, no cost")
    args = ap.parse_args()

    data = Path(args.data)
    gens = pd.read_parquet(data / "judged_L19.parquet")
    pair = gens[gens["target_distance"] == "pair"].copy()
    manifest = json.loads((data / "path_manifest.json").read_text())
    wps = waypoints_by_route(manifest)

    missing = {(a, b) for a, b in zip(pair.path_a, pair.path_b)} - set(wps)
    if missing:
        raise SystemExit("no manifest waypoints for routes: %s" % sorted(missing))

    ckpt = data / ("route_judge_ckpt_%s.jsonl" % args.ckpt_tag)
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec

    # A record whose score is None is a FAILED call, not a finished one: retry
    # it. The last record for a key wins on reload, so a retry supersedes it.
    def pending(r):
        rec = done.get(row_key(r))
        if rec is None or rec.get("score") is None:
            return True
        return args.redo_empty_analysis and not (rec.get("analysis") or "").strip()

    todo = [r for r in pair.itertuples() if pending(r)]
    if args.limit:
        todo = todo[:args.limit]

    print("pair rows: %d   already judged: %d   to judge: %d"
          % (len(pair), len(pair) - len([r for r in pair.itertuples()
                                         if pending(r)]), len(todo)))
    print("checkpoint: %s" % ckpt)

    if args.dry_run:
        if todo:
            r = todo[0]
            print("\n--- SYSTEM (%s -> %s) ---\n%s"
                  % (r.path_a, r.path_b,
                     RJ.build_system(r.path_a, wps[(r.path_a, r.path_b)], r.path_b)))
            print("\n--- USER ---\n%s"
                  % RJ.USER_TEMPLATE.format(question=r.question, response=r.response))
        for (a, b), w in sorted(wps.items()):
            if (a, b) in {(x, y) for x, y in zip(pair.path_a, pair.path_b)}:
                print("OPTIONS %-12s -> %-10s : %s"
                      % (a, b, RJ.build_options(a, w, b)))
        return

    client = _client()
    lock = threading.Lock()
    fh = ckpt.open("a")
    cost = {"in": 0, "out": 0, "cr": 0, "cw": 0, "n": 0}

    def one(r):
        key = row_key(r)
        p = RJ.params(r.path_a, wps[(r.path_a, r.path_b)], r.path_b,
                      r.question, r.response)
        for attempt in range(4):
            try:
                msg = client.messages.create(**p)
                break
            except Exception as exc:                      # noqa: BLE001
                if attempt == 3:
                    return {"key": key, "score": None,
                            "analysis": "API_ERROR: %s" % str(exc)[:160]}
                import time
                time.sleep(2 ** attempt)
        v = parse_route_verdict(msg.content,
                                RJ.build_options(r.path_a,
                                                 wps[(r.path_a, r.path_b)], r.path_b))
        u = msg.usage
        rec = {"key": key, "score": v["score"], "analysis": v["analysis"],
               "in": u.input_tokens, "out": u.output_tokens,
               "cr": getattr(u, "cache_read_input_tokens", 0) or 0,
               "cw": getattr(u, "cache_creation_input_tokens", 0) or 0,
               # The raw reply, so a parser bug can be repaired from the
               # checkpoint instead of re-buying 2,400 API calls.
               "raw": next((b.text for b in msg.content
                            if getattr(b, "type", None) == "text"), None)}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            for k in ("in", "out", "cr", "cw"):
                cost[k] += rec[k]
            cost["n"] += 1
            if cost["n"] % 100 == 0:
                print("  %d/%d" % (cost["n"], len(todo)), flush=True)
        return rec

    # Rows are grouped by route so the cached system block is written once per
    # route and read by the hundreds of rows behind it, rather than expiring
    # between interleaved routes.
    todo.sort(key=lambda r: (r.path_a, r.path_b))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(one, todo):
            done[rec["key"]] = rec
    fh.close()

    # Sonnet pricing, USD per million tokens.
    usd = (cost["in"] * 3 + cost["out"] * 15 + cost["cr"] * 0.3
           + cost["cw"] * 3.75) / 1e6
    print("tokens in=%d out=%d cache_read=%d cache_write=%d  ~$%.2f"
          % (cost["in"], cost["out"], cost["cr"], cost["cw"], usd))

    keys = [row_key(r) for r in pair.itertuples()]
    pair["route_score"] = [done.get(k, {}).get("score") for k in keys]
    pair["route_analysis"] = [done.get(k, {}).get("analysis") for k in keys]
    pair["route_score"] = pair["route_score"].astype("object")
    out = data / ("route_judged_%s_L19.parquet" % args.ckpt_tag)
    pair.to_parquet(out, index=False)
    print("wrote %s  (%d rows, %d scored)"
          % (out, len(pair), pair["route_score"].notna().sum()))
    print(pair["route_score"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
