"""Judge generations with concurrent interactive calls instead of the Batches API.

The Batches API trades control for a 50% discount; when a batch stalls (ours sat
18.5 h at 0/13) that discount buys nothing. This path drives the same _params /
schema / parse code as judge.py, just with a thread pool, so throughput is ours
to set.

Resumable: every result is appended to a JSONL checkpoint keyed by the row's
IDENTITY (arm, alpha, role, system prompt, question), so a crash or Ctrl-C loses
only in-flight calls and re-running skips completed rows. Never by row index:
run_steering rebuilds the generations parquet by re-globbing and sorting the
shards, so adding an arm shifts every row and an index-keyed checkpoint would
attach stale verdicts to different responses.
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering.judge import _client, _params, parse_verdict, row_keys, CATEGORIES
import steering.judge as _judge   # module handle, so --model can rebind JUDGE_MODEL

_print_lock = threading.Lock()


def judge_one(client, key, role, question, response, max_attempts=4):
    """One judged row. Returns a dict; never raises -- failures come back tagged.

    `max_attempts` is 4 on top of the SDK's own `max_retries`, not 6: the two
    nest multiplicatively, so 6 x 4 was up to 24 attempts per row against a
    model that is already refusing.
    """
    for attempt in range(max_attempts):
        try:
            msg = client.messages.create(**_params(role, question, response))
            return {"key": key, **parse_verdict(msg.content),
                    "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens}
        except Exception as e:
            if attempt == max_attempts - 1:
                return {"key": key, "judge_score": None,
                        "judge_analysis": "ERROR_%s" % type(e).__name__,
                        "in_tok": 0, "out_tok": 0}
            time.sleep(min(2 ** attempt, 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True, help="run dir")
    # The judge MODEL is a recorded deviation whatever we pick (the paper used
    # deepseek-v3), so it is a flag rather than a constant. Measured 2026-09-02:
    # Sonnet answered 0/3 trivial calls in 80 s while Haiku answered 3/3 in 4.6 s,
    # and 19,000 rows on Sonnet projected to 45 h at 0.1 rows/s.
    ap.add_argument("--model", default=None,
                    help="override the judge model (default: judge.JUDGE_MODEL)")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N rows (test mode)")
    ap.add_argument("--sample", type=int, default=0, help="stratified sample of N rows (test mode)")
    ap.add_argument("--tag", default="", help="suffix for output files (keeps tests separate)")
    # Every other API module in this tree is inert without --yes. This one was
    # not, so a mistyped --generations could spend the full grid's budget with
    # no confirmation step.
    ap.add_argument("--yes", action="store_true",
                    help="required to spend money; without it this is a dry run")
    args = ap.parse_args()
    if args.model:
        # _params() reads judge.JUDGE_MODEL at call time, so setting it on the
        # module is enough and the rubric/schema stay untouched.
        _judge.JUDGE_MODEL = args.model
        print("judge model overridden -> %s" % args.model, flush=True)

    run_dir = Path(args.out)
    df = pd.read_parquet(args.generations)

    if args.sample:
        # stratified: spread across arm x alpha, plus the degenerate cases
        df["_empty"] = df.response.astype(str).str.strip() == ""
        parts = [g.head(1) for _, g in df[~df._empty].groupby(["arm", "alpha"])]
        parts.append(df[df._empty].head(2))
        sel = pd.concat(parts).head(args.sample)
        if len(sel) < args.sample:                      # top up randomly
            rest = df.drop(sel.index).sample(args.sample - len(sel), random_state=0)
            sel = pd.concat([sel, rest])
        df = sel.drop(columns=["_empty"])
    elif args.limit:
        df = df.head(args.limit)

    keys = row_keys(df)

    ckpt = run_dir / "data" / ("judge_ckpt%s.jsonl" % args.tag)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "key" not in r:
                raise SystemExit(
                    "%s is a POSITION-KEYED checkpoint from before the identity "
                    "fix. Its verdicts cannot be matched to rows now that the "
                    "frame may have been reordered. Delete it and re-judge, or "
                    "move it aside if you need it for forensics." % ckpt)
            done[r["key"]] = r
    todo = [(k, r) for k, r in zip(keys, df.itertuples()) if k not in done]
    print("rows=%d already_done=%d todo=%d workers=%d"
          % (len(df), len(done), len(todo), args.workers), flush=True)

    if not args.yes:
        raise SystemExit(
            "DRY RUN - nothing sent.\n  rows: %d\n  already judged: %d\n"
            "  to judge: %d\n  rough cost: $%.2f\nRe-run with --yes to actually "
            "spend this." % (len(df), len(done), len(todo),
                             len(todo) * (1100 * 2e-6 + 120 * 10e-6)))

    client = _client().with_options(max_retries=4, timeout=120.0)
    t0 = time.time()
    n_done = 0
    with open(ckpt, "a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, client, k, r.target_role, r.question, r.response): k
                for k, r in todo}
        for fut in as_completed(futs):
            res = fut.result()
            done[res["key"]] = res
            with _print_lock:
                ck.write(json.dumps(res) + "\n"); ck.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print("  %d/%d  %.1f rows/s  eta %.1f min"
                          % (n_done, len(todo), n_done / el,
                             (len(todo) - n_done) / max(n_done / el, 1e-9) / 60), flush=True)

    out = df.copy()
    # .get, not done[k]: a row whose in-flight call was lost to a Ctrl-C has no
    # record, and a bare lookup would raise here instead of reporting it as
    # unjudged alongside every other failure.
    out["judge_score"] = [done.get(k, {}).get("judge_score") for k in keys]
    out["judge_analysis"] = [done.get(k, {}).get("judge_analysis") for k in keys]
    p = run_dir / "data" / ("judged_L19%s.parquet" % args.tag)
    out.to_parquet(p, index=False)

    bad = int(out.judge_score.isna().sum())
    invalid = int((~out.judge_score.isin(CATEGORIES) & out.judge_score.notna()).sum())
    it = sum(done.get(k, {}).get("in_tok", 0) for k in keys)
    ot = sum(done.get(k, {}).get("out_tok", 0) for k in keys)
    print("wrote %s (%d rows)" % (p, len(out)))
    print("failed=%d out_of_schema=%d" % (bad, invalid))
    print("tokens: %d in / %d out | cost @intro $2/$10: $%.2f" % (it, ot, it*2e-6 + ot*10e-6))
    print(out.judge_score.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
