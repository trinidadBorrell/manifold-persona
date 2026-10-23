"""Second broken-reply judge for the E3 pilot: TypeSafe Jev (owner decision 16/10/2026).

One request per reply. The state holds only the question and the answer (no role, no condition).
Four Noul questions, one condition each (Jev docs: one yes/no condition per Noul, edge cases in criteria),
plus one five-level coherence Score. Fixed before any output: broken = any Noul > 0.5.
Model pinned to jev-1.13.0, the version evaluated by Deusser et al. 2026 (arXiv 2609.37647).
Replies: the judge2 coherence set (10,800; key.json items of type "coh" in JUDGE_DIR).
The API key is read from REPO/.jev_token at run time and never printed or written.

    python jev_judge.py run JUDGE_DIR OUT_DIR [--workers 16] [--sample 20 --exclude LABELS.json]
"""
import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("judge2", HERE / "judge2.py")
judge2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge2)
REPO = HERE.parents[2]
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
NOULS = {
    "loops": ("Does `answer` keep repeating the same content without adding anything new?",
              "The same sentence or idea comes back again and again, and later sentences add nothing new.",
              "Each sentence adds something. Starting several sentences the same way for emphasis is not looping."),
    "nonsense": ("Is `answer` hard or impossible to make sense of?",
                 "Garbled words, broken grammar, or sentences that do not connect.",
                 "A reader can follow it, even if it is wrong, odd, or stops mid-sentence at the end."),
    "off_question": ("Does `answer` ignore `question` and talk about something else?",
                     "It talks about something other than the question, even if it stays in character.",
                     "It responds to the question, even if it is wrong, odd, rude, short, or in a strange voice."),
    "new_turns": ("Does `answer` write new interview turns or switch to another speaker?",
                  "It writes lines such as 'Interviewer:' or a new question, or starts another speaker.",
                  "It is one answer from one speaker."),
    "copies_question": ("Does `answer` copy `question` back?",
                        "It repeats the question, in full or almost in full.",
                        "It does not repeat the question."),
}  # owner definition of broken, 16/10 (THRESHOLDS.md); round 2 used "new_turns" and a looser off_question
INJECTION_HINT = judge2.INJECTION_HINT  # recorded per run, not blocked (same rule as judge2 prep)
LEVELS = ["Not readable language; makes no sense at all.",
          "Mostly does not make sense; only fragments connect.",
          "Partly makes sense, but is confused or rambling.",
          "Makes sense, with small lapses.",
          "A clear, connected answer to the question with no sign of confusion."]


def body(question, answer):
    qs = {k: {"type": "noul", "instructions": ins, "criteria": {"true": t, "false": f}}
          for k, (ins, t, f) in NOULS.items()}
    qs["coherence"] = {"type": "score", "instructions": "How coherent is `answer` as a reply to `question`?",
                       "criteria": LEVELS}
    return {"model": MODEL, "state": {"question": question, "answer": answer}, "questions": qs}


def num01(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1


def check(resp):
    if not isinstance(resp, dict) or resp.get("model") != MODEL:
        return f"model {resp.get('model') if isinstance(resp, dict) else None}"
    a = resp.get("answers")
    if not isinstance(a, dict) or set(a) != set(NOULS) | {"coherence"} or not all(isinstance(v, dict) for v in a.values()):
        return "answer ids differ"
    if not all(num01(a[k].get("noul")) for k in NOULS):
        return "noul out of range"
    p = a["coherence"].get("probabilities")
    if (not isinstance(p, dict) or set(p) != {str(i) for i in range(len(LEVELS))}
            or not all(num01(v) for v in p.values()) or abs(sum(p.values()) - 1) > 1e-3):
        return "score probabilities malformed"
    sc = a["coherence"].get("score")
    if not isinstance(sc, (int, float)) or isinstance(sc, bool) or not 0 <= sc <= len(LEVELS) - 1:
        return "score out of range"
    return None


class Fatal(Exception):
    pass


def post(payload, token):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def job(rk, rep, out, token):
    f = out / "answers" / (rk.replace("/", "_") + ".json")
    payload = body(rep["question"], rep["reply"] or "")
    req_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if f.exists():
        try:
            rec = json.loads(f.read_text())
            if rec["request_sha256"] == req_sha and check(rec["response"]) is None:
                return rk, None
        except (ValueError, KeyError, TypeError):
            pass
    err = None
    for attempt in range(4):
        if attempt:
            time.sleep(2 ** (attempt - 1) + random.random())
        try:
            resp = post(payload, token)
            err = check(resp)
            if err is None:
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(reply_key=rk, request_sha256=req_sha, response=resp)))
                tmp.rename(f)
                return rk, None
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403, 404, 413):
                raise Fatal(f"HTTP {e.code} on {rk}; stopping the run")
            err = f"HTTP {e.code}"
        except Exception as e:  # network drops, malformed bodies: retry, then report per reply
            err = type(e).__name__
    return rk, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("judge_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--exclude", default=None, help="owner label JSON; its replies are left out of --sample")
    a = ap.parse_args()
    key = json.loads((Path(a.judge_dir) / "key.json").read_text())
    keys = sorted({v["reply_key"] for v in key.values() if v["type"] == "coh"})
    reps = {}
    for run in ("run_main", "run_rep"):
        for line in open(HERE / run / "replies.jsonl"):
            r = json.loads(line)
            reps[f"{run}/{r['i']}"] = r
    if a.sample:
        skip = set()
        if a.exclude:
            skip = {f"run_main/{v['item']}" for v in json.loads(Path(a.exclude).read_text()).values()}
        keys = random.Random(20261016).sample([k for k in keys if k not in skip], a.sample)
    out = Path(a.out_dir)
    (out / "answers").mkdir(parents=True, exist_ok=True)
    settings = dict(model=MODEL, url=URL, nouls=NOULS, levels=LEVELS,
                    code_sha256=hashlib.sha256((HERE / "jev_judge.py").read_bytes()).hexdigest(),
                    sample=a.sample, exclude=a.exclude, keys=keys,
                    injection_hint_keys=[k for k in keys if INJECTION_HINT.search(reps[k]["reply"] or "")])
    sf = out / "settings.json"
    settings = json.loads(json.dumps(settings))  # tuples become lists, as in the saved file
    if sf.exists() and json.loads(sf.read_text()) != settings:
        raise SystemExit(f"{sf} holds a different run; use a new OUT_DIR")
    sf.write_text(json.dumps(settings, indent=1))
    token = json.loads((REPO / ".jev_token").read_text())["token"]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, rk, reps[rk], out, token) for rk in keys]
        try:
            res = [fu.result() for fu in futs]
        except Fatal as e:
            for fu in futs:
                fu.cancel()
            raise SystemExit(str(e))
    bad = [(rk, e) for rk, e in res if e]
    print(f"{len(keys) - len(bad)}/{len(keys)} replies valid")
    for rk, e in bad[:20]:
        print("  failed", rk, e)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
