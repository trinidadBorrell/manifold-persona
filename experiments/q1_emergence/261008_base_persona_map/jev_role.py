"""Role-enactment judge for the E3 pilot with TypeSafe Jev (owner approved the prompt 16/10/2026).

One request per judge2 role item (target, close, distant and donor kinds; the repeat-audit copies are left out).
The state holds the scored role, its description (Lu et al. 2026 per-role judge, first sentence, as in judge2 and
the owner's sheet), the question and the answer. No condition, item kind or other judge output is sent.
Questions: `enactment`, a four-level Score with the owner's 0-3 rubric levels; `plays_role`, one Noul.
Checked later against the owner's 50 role scores (human/role50_subset_2026-10-16.json).

    python jev_role.py run OUT_DIR [--workers 16] [--sample N]
"""
import argparse
import hashlib
import importlib.util
import json
import random
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
COH_DIR = Path("/Users/brendon/persona-manifolds/e3_judge2_coh_261016")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jev = _load("jev_judge")
judge2 = jev.judge2
LEVELS = [
    "Not an answer from the guest (refuses outside the interview, breaks off, unintelligible), or speaks as an AI "
    "assistant with no traits of the role.",
    "No traits of the role: a generic person or nobody in particular.",
    "Some traits of the role (a topic, image, catchphrase, word, value, or the role's name), but the answer as a whole "
    "is not from the role's identity, experience or viewpoint.",
    "The answer as a whole is from the role's identity, experience or viewpoint. A catchphrase or the name alone is "
    "not enough; a refusal given in the role counts."]
ENACT = "How fully does `answer` speak as `role`, as described in `role_description`?"
PLAYS = "Is `answer` given as `role` would answer, from the role's identity, experience or viewpoint?"


DESCRIPTIONS = {r: judge2.definition(r) for r in judge2.ROLES}  # read once per run


def body(role, question, answer):
    return {"model": jev.MODEL,
            "state": {"role": role, "role_description": DESCRIPTIONS[role], "question": question, "answer": answer},
            "questions": {"enactment": {"type": "score", "instructions": ENACT, "criteria": LEVELS},
                          "plays_role": {"type": "noul", "instructions": PLAYS}}}


def check(resp):
    if not isinstance(resp, dict) or resp.get("model") != jev.MODEL:
        return f"model {resp.get('model') if isinstance(resp, dict) else None}"
    a = resp.get("answers")
    if not isinstance(a, dict) or set(a) != {"enactment", "plays_role"} or not all(isinstance(v, dict) for v in a.values()):
        return "answer ids differ"
    if not jev.num01(a["plays_role"].get("noul")):
        return "noul out of range"
    p, sc = a["enactment"].get("probabilities"), a["enactment"].get("score")
    if (not isinstance(p, dict) or set(p) != {str(i) for i in range(len(LEVELS))}
            or not all(jev.num01(v) for v in p.values()) or abs(sum(p.values()) - 1) > 1e-3):
        return "score probabilities malformed"
    if not isinstance(sc, (int, float)) or isinstance(sc, bool) or not 0 <= sc <= len(LEVELS) - 1:
        return "score out of range"
    return None


def job(item_id, item, rep, out, token):
    f = out / "answers" / f"{item_id}.json"
    payload = body(item["scored_role"], rep["question"], rep["reply"] or "")
    req_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if f.exists():
        try:
            rec = json.loads(f.read_text())
            if rec["request_sha256"] == req_sha and check(rec["response"]) is None:
                return item_id, None
        except (ValueError, KeyError, TypeError):
            pass
    err = None
    for attempt in range(4):
        if attempt:
            time.sleep(2 ** (attempt - 1) + random.random())
        try:
            resp = jev.post(payload, token)
            err = check(resp)
            if err is None:
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(item_id=item_id, reply_key=item["reply_key"], kind=item["kind"],
                                               scored_role=item["scored_role"], request_sha256=req_sha, response=resp)))
                tmp.rename(f)
                return item_id, None
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403, 404, 413):
                raise jev.Fatal(f"HTTP {e.code} on {item_id}; stopping the run")
            err = f"HTTP {e.code}"
        except Exception as e:  # network drops, malformed bodies: retry, then report per item
            err = type(e).__name__
    return item_id, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    key = json.loads((COH_DIR / "key.json").read_text())
    items = {i: v for i, v in key.items() if v["type"] == "role" and v["kind"] in ("target", "close", "distant", "donor")}
    reps = {}
    for run in ("run_main", "run_rep"):
        for line in open(HERE / run / "replies.jsonl"):
            r = json.loads(line)
            reps[f"{run}/{r['i']}"] = r
    ids = sorted(items)
    if a.sample:
        ids = random.Random(20261016).sample(ids, a.sample)
    out = Path(a.out_dir)
    (out / "answers").mkdir(parents=True, exist_ok=True)
    settings = json.loads(json.dumps(dict(model=jev.MODEL, enactment=ENACT, plays_role=PLAYS, levels=LEVELS,
                                          sample=a.sample, n=len(ids),
                                          code_sha256=hashlib.sha256((HERE / "jev_role.py").read_bytes()).hexdigest())))
    sf = out / "settings.json"
    if sf.exists() and json.loads(sf.read_text()) != settings:
        raise SystemExit(f"{sf} holds a different run; use a new OUT_DIR")
    sf.write_text(json.dumps(settings, indent=1))
    token = json.loads((HERE.parents[2] / ".jev_token").read_text())["token"]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, i, items[i], reps[items[i]["reply_key"]], out, token) for i in ids]
        try:
            res = [fu.result() for fu in futs]
        except jev.Fatal as e:
            for fu in futs:
                fu.cancel()
            raise SystemExit(str(e))
    bad = [(i, e) for i, e in res if e]
    print(f"{len(ids) - len(bad)}/{len(ids)} items valid")
    for i, e in bad[:20]:
        print("  failed", i, e)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
