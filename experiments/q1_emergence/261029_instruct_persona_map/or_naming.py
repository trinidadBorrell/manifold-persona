"""Pod A D1 blind naming through OpenRouter (THRESHOLDS.md D1): one reply per request, the two E3 judges, a 5%
blind repeat audit at the end of each batch, and one sync retry per malformed answer. It writes the payload that
analysis_naming.load_payload reads. The key is read from ~/.config/openrouter/api_key (through or_judge.call)
and never written. The work folder holds the item keys and must be outside the repo; judges see only an opaque
id and the reply text.

    python or_naming.py prep    WORK --replicates 0,1,2     # requests per judge; no call
    python or_naming.py cost    WORK                        # dollar estimate; no call
    python or_naming.py pilot   WORK --n 20                 # a few sync calls per judge: real tokens and cost
    python or_naming.py submit  WORK                        # one batch per judge (parts of 5,000)
    python or_naming.py poll    WORK
    python or_naming.py retry   WORK                        # one sync retry per malformed answer
    python or_naming.py payload WORK --approval "REF"       # WORK/payload.json for analysis.py --naming
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3 = HERE.parent / "261008_base_persona_map"
CANDIDATES_PATH = E3 / "candidates.txt"
QUESTIONS_PATH = E3 / "questions_40.json"
TEMPLATE_PATH = HERE / "prompt_naming_openrouter.txt"
ROWS_PATH = HERE / "run_full" / "rows.jsonl"
# The D1 judge pair (THRESHOLDS v1.5): GPT-6.1-Terra at reasoning effort high, as E3 ran GPT-5.6-Sol, and Claude
# Sonnet 5.5 without extended thinking, as in E3.
# max_tokens: OpenRouter holds credit as if every answer used it all; the judge check's longest answers were 713
# tokens (Terra, reasoning included) and 42 (Sonnet), and a cut-off answer gets the one retry.
JUDGES = {"GPT-6.1-Terra": dict(model="openai/gpt-6.1-terra", settings=dict(reasoning={"effort": "high"}, max_tokens=1200)),
          "Claude Sonnet 5.5": dict(model="anthropic/claude-sonnet-5.5", settings=dict(max_tokens=100))}
# Names restricted to the list by the schema, as the prompt asks: in the pilot, Claude Sonnet 5.5 answered 1 of 20
# replies with a name outside the list ("entity") and repeated it on a second call.
SCHEMA = {"type": "json_schema", "json_schema": {"name": "naming", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["names"],
    "properties": {"names": {"type": "array", "items": {"type": "string",
                                                        "enum": (E3 / "candidates.txt").read_text().splitlines()}}}}}}
MARKERS = ("[REPLY START]", "[REPLY END]")
REPEAT_SHARE, REPEAT_SEED, ID_SEED = 0.05, 20261030, "poda-naming-261030"
MAX_RETRY_SHARE = 0.10  # more malformed answers than this needs the owner's go (sync calls cost twice the batch)
# OpenRouter :batch prices per 1M tokens (input, cached input, output), read from /api/v1/models on 30/10/2026
BATCH_PRICE = {"openai/gpt-6.1-terra": (0.5, 0.025, 2.5), "anthropic/claude-sonnet-5.5": (1.0, 0.05, 5.0)}

spec = importlib.util.spec_from_file_location("or_judge", HERE.parent / "261029_judge_validation/or_judge.py")
orj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orj)  # call, content_of, submit_parts, DONE, FAILED, CHUNK: shared with the rubric judges


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


@functools.cache
def catalog():
    names = CANDIDATES_PATH.read_text().splitlines()
    assert len(names) == len(set(names)) == 276
    return names


@functools.cache
def template():
    return TEMPLATE_PATH.read_text().replace("{candidates}", "\n".join(catalog()))


def prompt(reply):
    if any(m in reply for m in MARKERS):
        raise SystemExit("a reply contains a marker")
    return template().replace("{reply}", reply)


def body(judge, reply):
    j = JUDGES[judge]
    return dict(model=j["model"], messages=[{"role": "user", "content": prompt(reply)}], response_format=SCHEMA,
                usage={"include": True}, **j["settings"])


def key_of(r):
    return json.dumps([r["role"], r["format"], r["wording"], r["question"], r["replicate"]], separators=(",", ":"))


def opaque(key):
    return sha(f"{ID_SEED}:{key}")[:16]


def parse(text):
    """Five distinct catalog names, best first, or None. Case is ignored, as in the E3 check."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        names = json.loads(t).get("names")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        return None
    names = [x.strip().lower() for x in names]
    return names if len(names) == 5 and len(set(names)) == 5 and set(names) <= set(catalog()) else None


def fingerprint():
    """What must not change between prep, submit, retry and payload."""
    return dict(judges=JUDGES, schema_sha256=sha(json.dumps(SCHEMA, sort_keys=True)),
                template_sha256=sha(TEMPLATE_PATH.read_bytes()), candidates_sha256=sha(CANDIDATES_PATH.read_bytes()))


def work_dir(path):
    w = Path(path).resolve()
    if w.is_relative_to(REPO):
        raise SystemExit("keep the naming work folder (it holds the answer keys) outside the repo")
    return w


def load(w):
    m = json.loads((w / "manifest.json").read_text())
    if m["fingerprint"] != fingerprint() or m["items_sha256"] != sha((w / "items.jsonl").read_bytes()):
        raise SystemExit("prompt, judges, settings or items changed since prep; the answers would mix two setups")
    return m, [json.loads(l) for l in (w / "items.jsonl").read_text().splitlines()]


def slug(judge):
    return JUDGES[judge]["model"].replace("/", "__")


def cmd_prep(a):
    w = work_dir(a.work)
    if any((w / slug(j) / f).exists() for j in JUDGES for f in ("batch.json", "submitting.json")):
        raise SystemExit("a batch was already submitted from these requests; prep would overwrite what ran")
    reps = sorted(int(x) for x in a.replicates.split(","))
    if reps not in ([0, 1, 2], [0, 1, 2, 3, 4]):
        raise SystemExit("THRESHOLDS allows replicates 0-2 or 0-4 only")
    rows = [json.loads(l) for l in ROWS_PATH.read_text().splitlines()]
    items = [dict(id=opaque(key_of(r)), key=key_of(r), reply=r["reply"]) for r in rows if r["replicate"] in reps]
    if len({it["id"] for it in items}) != len(items):
        raise SystemExit("opaque ids collide")
    for it in items:
        prompt(it["reply"])  # marker check
    rep = sorted(random.Random(f"{REPEAT_SEED}:{len(items)}").sample([it["id"] for it in items],
                                                                       math.ceil(REPEAT_SHARE * len(items))))
    w.mkdir(parents=True, exist_ok=True)
    (w / "items.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    by_id = {it["id"]: it for it in items}
    for j in JUDGES:
        d = w / slug(j)
        d.mkdir(exist_ok=True)
        reqs = [dict(custom_id=it["id"], body=body(j, it["reply"])) for it in items]
        reqs += [dict(custom_id=f"rep_{i}", body=body(j, by_id[i]["reply"])) for i in rep]  # blind: ids look alike
        (d / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
    manifest = dict(replicates=reps, n_items=len(items), repeat_ids=rep, fingerprint=fingerprint(),
                    items_sha256=sha((w / "items.jsonl").read_bytes()), rows_sha256=sha(ROWS_PATH.read_bytes()),
                    code_sha256=sha(Path(__file__).read_bytes()), prompt_example_sha256=sha(prompt("{reply}")))
    (w / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(items)} items + {len(rep)} blind repeats x {len(JUDGES)} judges -> {w}")


def cmd_cost(a):
    """Dollar estimate at batch prices, ~4 characters per token. The shared prefix (instructions and list) may be
    cached; the output (reasoning) assumption is the main unknown."""
    w = work_dir(a.work)
    m, items = load(w)
    n = m["n_items"] + len(m["repeat_ids"])
    prefix = len(TEMPLATE_PATH.read_text().split("{reply}")[0].replace("{candidates}", "\n".join(catalog()))) / 4
    reply = sum(len(it["reply"]) for it in items) / len(items) / 4
    for j, out_tokens in (("GPT-6.1-Terra", (165, 713)), ("Claude Sonnet 5.5", (28, 42))):
        pin, pcache, pout = BATCH_PRICE[JUDGES[j]["model"]]
        inp = n * (prefix + reply + 20) / 1e6
        cached = n * (prefix * pcache + (reply + 20) * pin) / 1e6
        lo, hi = (n * t / 1e6 * pout for t in out_tokens)
        print(f"{j}: {n} requests, input {inp:.1f}M tokens (${inp * pin:.0f} uncached, ${cached:.0f} if the prefix "
              f"caches), output ${lo:.0f}-${hi:.0f} ({out_tokens[0]}-{out_tokens[1]} tokens each)")


def cmd_pilot(a):
    """A few sync calls per judge to measure tokens, cost and answer validity before the batch (kept apart from the
    run's answers). The full-run estimate halves the sync cost, the :batch price."""
    from concurrent.futures import ThreadPoolExecutor
    w = work_dir(a.work)
    m, items = load(w)
    pick = random.Random(f"pilot:{REPEAT_SEED}").sample(items, a.n)
    report, n_requests = {}, m["n_items"] + len(m["repeat_ids"])
    for j in JUDGES:
        with ThreadPoolExecutor(10) as ex:
            got = list(ex.map(lambda it: orj.call("POST", "/chat/completions", body(j, it["reply"])), pick))
        rows = []
        for it, (code, resp) in zip(pick, got):
            text, usage = orj.content_of(resp)
            rows.append(dict(id=it["id"], http=code, valid=parse(text) is not None, cost=usage.get("cost"),
                             prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                             reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             error=resp.get("error") if code != 200 else None))
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        mean = sum(costs) / len(costs) if costs else None
        report[j] = dict(rows=rows, valid=sum(r["valid"] for r in rows), n=len(rows), spent=sum(costs),
                         mean_sync_cost=mean, full_batch_estimate=mean * n_requests / 2 if mean is not None else None)
        print(f"{j}: {report[j]['valid']}/{len(rows)} valid, spent ${report[j]['spent']:.3f}, completion tokens "
              f"{[r['completion_tokens'] for r in rows]}, full batch about ${report[j]['full_batch_estimate'] or 0:.0f}")
    (w / "pilot.json").write_text(json.dumps(report, indent=1))


def cmd_submit(a):
    w = work_dir(a.work)
    load(w)
    for j in JUDGES:
        d = w / slug(j)
        b = json.loads((d / "batch.json").read_text()) if (d / "batch.json").exists() else None
        if b is not None and len(b["parts"]) >= len(b["planned"]):
            print(j, "already submitted")
            continue
        if (d / "submitting.json").exists():
            print(j, "an earlier submit got no answer; check the OpenRouter dashboard, then delete", d / "submitting.json")
            continue
        reqs = [json.loads(l) for l in (d / "requests.jsonl").read_text().splitlines()]
        orj.submit_parts(JUDGES[j]["model"], d, [(f"chunk{k // orj.CHUNK + 1:02d}", reqs[k:k + orj.CHUNK])
                                                 for k in range(0, len(reqs), orj.CHUNK)])


def cmd_poll(a):
    w = work_dir(a.work)
    for j in JUDGES:
        d = w / slug(j)
        if (d / "results.json").exists() or not (d / "batch.json").exists():
            continue
        b = json.loads((d / "batch.json").read_text())
        if len(b["parts"]) < len(b["planned"]):
            print(j, f"only {len(b['parts'])} of {len(b['planned'])} parts submitted; rerun submit first")
            continue
        polled = [orj.call("GET", f"/batches/{x['id']}") for x in b["parts"]]
        if any(code != 200 for code, _ in polled):
            print(j, "poll failed:", [code for code, _ in polled])
            continue
        sts = [str(r.get("status", "")).lower() for _, r in polled]
        print(j, sts, [r.get("request_counts") for _, r in polled])
        if any(s in orj.FAILED for s in sts):
            (d / "batch_failed.json").write_text(json.dumps([r for _, r in polled], indent=1))
            print(j, "a part FAILED; nothing saved; see batch_failed.json")
        elif all(s in orj.DONE for s in sts):
            results = [x for _, r in polled for x in (r.get("results") or [])]
            sent = sum(1 for _ in (d / "requests.jsonl").read_text().splitlines())
            if len(results) != sent:
                print(j, f"finished, but {len(results)} of {sent} results are inline; not saved")
                continue
            (d / "results.json").write_text(json.dumps(dict(parts=b["parts"], results=results), ensure_ascii=False))


def answers(d):
    """custom_id -> [batch text, retry text]; names are parsed again from the raw text each time."""
    got = {}
    for r in json.loads((d / "results.json").read_text())["results"]:
        got[r["custom_id"]] = [orj.content_of(r.get("response") or {})[0]]
    if (d / "retries.json").exists():
        for cid, v in json.loads((d / "retries.json").read_text()).items():
            got.setdefault(cid, [None]).append(v["text"])
    return got


def names_of(texts):
    return next((x for x in map(parse, texts) if x is not None), None)


def cmd_retry(a):
    w = work_dir(a.work)
    m, items = load(w)
    reply = {it["id"]: it["reply"] for it in items}
    for j in JUDGES:
        d = w / slug(j)
        if not (d / "results.json").exists():
            continue
        cids = [json.loads(l)["custom_id"] for l in (d / "requests.jsonl").read_text().splitlines()]
        got, rpath = answers(d), d / "retries.json"
        done = json.loads(rpath.read_text()) if rpath.exists() else {}
        todo = [c for c in cids if parse(got.get(c, [None])[0]) is None and c not in done]
        if len(todo) + len(done) > MAX_RETRY_SHARE * len(cids) and not a.allow_many:
            print(j, f"{len(todo) + len(done)} malformed of {len(cids)}: over {MAX_RETRY_SHARE:.0%}; not retried. "
                     "Re-run with --allow-many after the owner's go.")
            continue
        for c in todo:
            code, resp = orj.call("POST", "/chat/completions", body(j, reply[c.removeprefix("rep_")]))
            text, usage = orj.content_of(resp)
            done[c] = dict(http=code, text=text, usage=usage, valid=parse(text) is not None,
                           error=resp.get("error") if code != 200 else None)
            rpath.write_text(json.dumps(done, indent=1, ensure_ascii=False))
        print(j, "retried", len(done), "valid after retry", sum(v["valid"] for v in done.values()))


def cmd_payload(a):
    w = work_dir(a.work)
    m, items = load(w)
    keys = {it["id"]: it["key"] for it in items}
    judges, problems = [], []
    for j in JUDGES:
        got = answers(w / slug(j))
        pred = {keys[i]: names_of(got.get(i, [None])) for i in keys}
        records = [dict(item_key=keys[i], repeat_id=f"rep_{i}", predictions=names_of(got.get(f"rep_{i}", [None])))
                   for i in m["repeat_ids"]]
        bad = sum(v is None for v in pred.values()) + sum(r["predictions"] is None for r in records)
        if bad:
            problems.append(f"{j}: {bad} answers still malformed after one retry")
        judges.append(dict(id=j, predictions=pred, repeat_audit=dict(
            blind=True, frozen_before_judging=True, records=records,
            reference=f"or_naming manifest {sha((w / 'manifest.json').read_bytes())}")))
    if problems:
        raise SystemExit("; ".join(problems) + ". D1 needs five valid names for every panel item; nothing written.")
    roles = sorted({json.loads(k)[0] for k in keys.values()})
    payload = dict(approval=dict(owner_approved=True, reference=a.approval, approved_before_judging=True,
                                 frozen_before_labels=True),
                   panel=dict(roles=roles, format="chat", questions=json.loads(QUESTIONS_PATH.read_text()),
                              wordings=list(range(5)), replicates=m["replicates"]),
                   judges=judges)
    (w / "payload.json").write_text(json.dumps(payload, indent=1))
    print("wrote", w / "payload.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "cost", "pilot", "submit", "poll", "retry", "payload"):
        p = sub.add_parser(name)
        p.add_argument("work")
        if name == "prep":
            p.add_argument("--replicates", required=True, help="0,1,2 or 0,1,2,3,4 (the approved judge panel)")
        if name == "pilot":
            p.add_argument("--n", type=int, default=20)
        if name == "retry":
            p.add_argument("--allow-many", action="store_true")
        if name == "payload":
            p.add_argument("--approval", required=True, help="where the owner approved the panel, prompt and judges")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
