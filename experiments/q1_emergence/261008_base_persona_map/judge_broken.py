"""Round-2 broken-reply judges for the E3 pilot (owner decision 16/10/2026, rules fixed before any output).

Judges: Claude Opus 5.5 (`claude -p`, tools off, empty folder, JSON schema) and GPT-6.1-Sol (`codex exec`,
read-only sandbox, empty folder, output schema). Each item shows only the question and the answer, with the
owner's definition of broken. Replies: run_main desc + ex2 on the 18 shared questions (3,600), in batches of 25.
Final call (fixed now): majority of Opus, GPT-6.1-Sol and Jev (any Noul > 0.5). A judge, and the majority, are
checked against the owner's 50 labels and the 30-reply fresh test; usable if kappa >= 0.6 on each.

    python judge_broken.py prep JUDGE_DIR
    python judge_broken.py run JUDGE_DIR opus|gpt [--workers 8] [--batches b0001,b0002]
    python judge_broken.py score JUDGE_DIR --labels OWNER50.json FRESH30.json
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import threading
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("judge2", HERE / "judge2.py")
judge2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge2)

BATCH, SEED, SALT = 25, 20261016, "e3-broken-round2"
MODELS = {"opus": "claude-opus-5-5", "gpt": "gpt-6.1-sol"}
GPT_EFFORT = "high"
CLAUDE = shutil.which("claude") or "/Users/brendon/.local/bin/claude"
CODEX = shutil.which("codex") or "/Users/brendon/.local/share/nvm/v22.22.2/bin/codex"
REASONS = ["none", "loop", "nonsense", "format", "off_question"]  # owner definition of broken, 16/10 (THRESHOLDS.md)
SYSTEM = judge2.SYSTEM
DELIM = re.compile(judge2.DELIM.pattern + r"|\[ANSWER (START|END)\]")
COST_CAP_USD = 25.0  # owner cap for round 2 (Opus API-equivalent cost; Codex reports no cost)
ISOLATION = ("opus: claude -p, tools off, empty cwd, no settings/MCP. gpt: codex exec, read-only sandbox, empty "
             "cwd; Codex can still read files by absolute path, so its logs are checked for file reads.")


def item_text(question, answer):
    return ("Decide whether this interview answer is BROKEN.\n"
            "BROKEN means the reply does not work as an answer to the question. It is broken if any of these is true:\n"
            "- loop: it says the same idea again and again and adds nothing new;\n"
            "- nonsense: it does not make sense (garbled words, broken grammar, sentences that do not connect);\n"
            "- format: it writes new interview turns or another speaker, or copies the question back;\n"
            "- off_question: it ignores the question and talks about something else, even if it stays in character.\n"
            "It is NOT broken if it stops mid-sentence at the end because of a length limit, repeats a phrase for "
            "emphasis, or is wrong, odd, rude, short, or in a strange voice. Do not judge whether it plays any "
            "particular role.\n"
            "Question:\n[QUESTION START]\n" + question + "\n[QUESTION END]\n"
            "Answer:\n[ANSWER START]\n" + answer + "\n[ANSWER END]\n\n"
            "Answer for this item: an object {\"broken\": true or false, \"reason\": one of " + ", ".join(REASONS) +
            "}; reason is \"none\" exactly when broken is false.")


ITEM_SCHEMA = {"type": "object", "properties": {"broken": {"type": "boolean"}, "reason": {"type": "string", "enum": REASONS}},
               "required": ["broken", "reason"], "additionalProperties": False}


def schema(ids):
    return {"type": "object", "properties": {i: ITEM_SCHEMA for i in ids}, "required": list(ids), "additionalProperties": False}


def check(ans, ids):
    if not isinstance(ans, dict) or set(ans) != set(ids):
        return "ids differ"
    for v in ans.values():
        if not (isinstance(v, dict) and set(v) == {"broken", "reason"} and isinstance(v["broken"], bool)
                and v["reason"] in REASONS and (v["reason"] == "none") == (not v["broken"])):
            return "bad value"
    return None


def prep(jd):
    jd = Path(jd)
    if jd.exists():
        raise SystemExit(f"{jd} exists; prep needs a new folder")
    rows = []
    for line in open(HERE / "run_main" / "replies.jsonl"):
        r = json.loads(line)
        if r["cond"] in ("desc", "ex2") and r.get("shared_q") and not r["role"].startswith("default"):
            if DELIM.search(r["reply"] or "") or DELIM.search(r["question"]):
                raise SystemExit(f"delimiter text in run_main/{r['i']}")
            rows.append(dict(id=hashlib.sha256(f"{SALT}|run_main/{r['i']}".encode()).hexdigest()[:10],
                             reply_key=f"run_main/{r['i']}", text=item_text(r["question"], r["reply"] or "")))
    if len(rows) != 3600 or len({r["id"] for r in rows}) != 3600:
        raise SystemExit(f"{len(rows)} rows")
    random.Random(SEED).shuffle(rows)
    (jd / "batches").mkdir(parents=True)
    batches = {}
    for k in range(0, len(rows), BATCH):
        chunk = rows[k:k + BATCH]
        name = f"b{k // BATCH + 1:04d}"
        prompt = judge2.batch_prompt(chunk)
        (jd / "batches" / f"{name}.txt").write_text(prompt)
        batches[name] = dict(ids=[r["id"] for r in chunk], prompt_sha256=judge2.sha(prompt))
    (jd / "key.json").write_text(json.dumps({r["id"]: r["reply_key"] for r in rows}))
    code = judge2.sha((HERE / "judge_broken.py").read_text())
    (jd / "manifest.json").write_text(json.dumps(dict(seed=SEED, batch=BATCH, models=MODELS, gpt_effort=GPT_EFFORT,
                                                      system_prompt=SYSTEM, code_sha256=code, isolation=ISOLATION,
                                                      cost_cap_usd=COST_CAP_USD, batches=batches), indent=1))
    print(len(rows), "replies,", len(batches), "batches")


def call_opus(prompt, ids, log):
    with tempfile.TemporaryDirectory() as empty:
        cmd = [CLAUDE, "-p", "--model", MODELS["opus"], "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--system-prompt", SYSTEM, "--no-session-persistence", "--output-format", "json",
               "--json-schema", json.dumps(schema(ids))]
        p = subprocess.run(cmd, cwd=empty, input=prompt, capture_output=True, text=True, timeout=1800)
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}")
    out = json.loads(p.stdout, object_pairs_hook=judge2.no_dupes)
    if list(out.get("modelUsage", {})) != [MODELS["opus"]]:
        raise RuntimeError(f"model used {list(out.get('modelUsage', {}))}")
    if out.get("is_error") or "structured_output" not in out:
        raise RuntimeError(f"no structured output ({out.get('subtype')})")
    so = out["structured_output"]
    return (judge2.parse_answer(so) if isinstance(so, str) else so), out.get("modelUsage")


def call_gpt(prompt, ids, log):
    with tempfile.TemporaryDirectory() as empty:
        sf, of = Path(empty) / "schema.json", Path(empty) / "last.txt"
        sf.write_text(json.dumps(schema(ids)))
        cmd = [CODEX, "exec", "-m", MODELS["gpt"], "-c", f"model_reasoning_effort={GPT_EFFORT}", "-s", "read-only",
               "--ephemeral", "--skip-git-repo-check", "-C", empty, "--output-schema", str(sf), "-o", str(of),
               SYSTEM + "\n\n" + prompt]
        p = subprocess.run(cmd, cwd=empty, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3600)
        log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}")
        used = re.search(r"^model: (\S+)", p.stdout + "\n" + p.stderr, re.M)
        if not used or used.group(1) != MODELS["gpt"]:
            raise RuntimeError(f"model used {used.group(1) if used else 'unknown'}")
        return json.loads(of.read_text(), object_pairs_hook=judge2.no_dupes), {MODELS["gpt"]: {"effort": GPT_EFFORT}}


SPENT = {"usd": 0.0}
LOCK = threading.Lock()


def cost(usage):
    return sum(u.get("costUSD", 0) for u in (usage or {}).values() if isinstance(u, dict))


def job(jd, judge, name, meta, man_sha):
    f = jd / f"answers_{judge}" / f"{name}.json"
    with LOCK:
        if SPENT["usd"] >= COST_CAP_USD:
            return name, f"cost cap ${COST_CAP_USD} reached; not submitted"
    prompt = (jd / "batches" / f"{name}.txt").read_text()
    if judge2.sha(prompt) != meta["prompt_sha256"]:
        return name, "batch file changed since prep"
    if f.exists():
        try:
            rec = json.loads(f.read_text())
            if rec["manifest_sha256"] == man_sha and check(rec["answer"], meta["ids"]) is None:
                return name, None
        except (ValueError, KeyError, TypeError):
            pass
    err = None
    for attempt in (1, 2, 3):
        with LOCK:  # checked before every attempt; up to `workers` calls in flight can still finish past the cap
            if SPENT["usd"] >= COST_CAP_USD:
                return name, f"cost cap ${COST_CAP_USD} reached before attempt {attempt}"
        try:
            call = call_opus if judge == "opus" else call_gpt
            a, usage = call(prompt, meta["ids"], jd / "logs" / f"{judge}_{name}_try{attempt}.log")
            with LOCK:
                SPENT["usd"] += cost(usage)
            err = check(a, meta["ids"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if err is None:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(manifest_sha256=man_sha, model_usage=usage, answer=a)))
            tmp.rename(f)
            return name, None
    return name, err


def run(jd, judge, workers, batches):
    jd = Path(jd)
    man_text = (jd / "manifest.json").read_text()
    man = json.loads(man_text)
    if judge2.sha((HERE / "judge_broken.py").read_text()) != man["code_sha256"]:
        raise SystemExit("judge_broken.py changed since prep")
    names = list(man["batches"]) if not batches else batches.split(",")
    unknown = [n for n in names if n not in man["batches"]]
    if unknown or len(set(names)) != len(names):
        raise SystemExit(f"bad batch names: {unknown or 'repeated'}")
    (jd / f"answers_{judge}").mkdir(exist_ok=True)
    (jd / "logs").mkdir(exist_ok=True)
    man_sha = judge2.sha(man_text)
    for f in jd.glob("answers_*/*.json"):  # the cap covers earlier runs into this folder too
        try:
            SPENT["usd"] += cost(json.loads(f.read_text()).get("model_usage"))
        except (ValueError, AttributeError):
            pass
    print(f"running {len(names)} batches with {MODELS[judge]}; spent so far ${SPENT['usd']:.2f} of ${COST_CAP_USD}")
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda n: job(jd, judge, n, man["batches"][n], man_sha), names))
    bad = [(n, e) for n, e in res if e]
    print(f"{len(names) - len(bad)}/{len(names)} batches valid; spent ${SPENT['usd']:.2f}")
    if judge == "gpt":  # Codex keeps its tools; list every shell command it ran so file reads are visible
        cmds = [l for f in sorted((jd / "logs").glob("gpt_*.log")) for l in f.read_text().splitlines()
                if l.startswith("/bin/") and " -lc " in l]
        print(f"gpt shell commands in logs: {len(cmds)}")
        for c in cmds[:20]:
            print("  ", c[:160])
    for n, e in bad[:30]:
        print("  failed", n, e)
    raise SystemExit(1 if bad else 0)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kappa(a, b):
    a, b = [bool(x) for x in a], [bool(x) for x in b]
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(jd, jev_dir, label_files):
    """The fixed final rule: majority of Opus, GPT-6.1-Sol and Jev; kappa >= 0.6 on each label set."""
    jd = Path(jd)
    if len(label_files) < 2:
        raise SystemExit("score needs both label sets (owner 50 and fresh 30): the gate is kappa >= 0.6 on each")
    (jd / "mask_majority.json").unlink(missing_ok=True)  # a mask exists only if THIS scoring passes the gate
    key = json.loads((jd / "key.json").read_text())
    man_text = (jd / "manifest.json").read_text()
    man, man_sha = json.loads(man_text), judge2.sha(man_text)
    calls = {}
    for judge in MODELS:
        c = {}
        for f in (jd / f"answers_{judge}").glob("*.json"):
            rec = json.loads(f.read_text())
            meta = man["batches"][f.stem]
            if rec["manifest_sha256"] != man_sha or check(rec["answer"], meta["ids"]) is not None:
                raise SystemExit(f"{judge}/{f.name}: answer does not belong to this prepared run")
            if judge == "opus" and list(rec.get("model_usage") or {}) != [MODELS["opus"]]:
                raise SystemExit(f"{judge}/{f.name}: model {list(rec.get('model_usage') or {})}")
            c.update({key[i]: v["broken"] for i, v in rec["answer"].items()})
        calls[judge] = c
    jev_judge = _load("jev_judge")
    js = json.loads((Path(jev_dir) / "settings.json").read_text())
    if js["model"] != jev_judge.MODEL or js["nouls"] != json.loads(json.dumps(jev_judge.NOULS)):
        raise SystemExit(f"{jev_dir} was run with other Jev questions or model; use the Jev run of this round")
    jev = {}
    for f in (Path(jev_dir) / "answers").glob("*.json"):
        r = json.loads(f.read_text())
        if jev_judge.check(r["response"]) is not None:
            raise SystemExit(f"jev/{f.name}: {jev_judge.check(r['response'])}")
        jev[r["reply_key"]] = max(r["response"]["answers"][k]["noul"] for k in jev_judge.NOULS) > 0.5
    keys = set(key.values())
    calls["jev"] = {k: v for k, v in jev.items() if k in keys}  # Jev also judged run_rep and the other conditions
    missing = {j: len(keys - set(c)) for j, c in calls.items()}
    if any(missing.values()):
        raise SystemExit(f"incomplete: {missing}")
    calls["majority"] = {k: sum(calls[j][k] for j in ("opus", "gpt", "jev")) >= 2 for k in keys}
    out = dict(rule="majority of opus, gpt, jev; usable if kappa >= 0.6 on every label set", label_sets={})
    for lf in label_files:
        raw = json.loads(Path(lf).read_text()).values()
        bad = [v.get("item") for v in raw if v.get("verdict") not in ("fine", "broken")]
        labs = {f"run_main/{v['item']}": v["verdict"] == "broken" for v in raw if v.get("verdict") in ("fine", "broken")}
        outside = sorted(k for k in labs if k not in keys)
        if bad or outside:
            raise SystemExit(f"{Path(lf).name}: undecided items {bad}, items outside the judged set {outside}")
        ks = sorted(labs)
        out["label_sets"][Path(lf).name] = {j: dict(n=len(ks), kappa=kappa([labs[k] for k in ks], [c[k] for k in ks]),
                                                    accuracy=sum(labs[k] == c[k] for k in ks) / len(ks))
                                            for j, c in calls.items()}
    out["usable"] = {j: all(s[j]["kappa"] >= 0.6 for s in out["label_sets"].values()) for j in calls}
    reps = {}
    for line in open(HERE / "run_main" / "replies.jsonl"):
        r = json.loads(line)
        reps[f"run_main/{r['i']}"] = r["cond"]
    out["broken_rates"] = {j: {cond: sum(v for k, v in c.items() if reps[k] == cond) / sum(1 for k in c if reps[k] == cond)
                               for cond in ("desc", "ex2")} for j, c in calls.items()}
    (jd / "score.json").write_text(json.dumps(out, indent=1))
    if out["usable"]["majority"]:
        (jd / "mask_majority.json").write_text(json.dumps(sorted([k.split("/")[0], int(k.split("/")[1])]
                                                                 for k, v in calls["majority"].items() if v)))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "score"])
    ap.add_argument("judge_dir")
    ap.add_argument("judge", nargs="?", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", default=None)
    ap.add_argument("--jev", required=False, help="Jev output dir of the same round (round 3: jev/round3)")
    ap.add_argument("--labels", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.judge_dir)
    elif a.cmd == "score":
        if not a.jev:
            raise SystemExit("score needs --jev (the Jev run of the same round)")
        score(a.judge_dir, a.jev, a.labels)
    else:
        if not a.judge:
            raise SystemExit("run needs a judge: opus or gpt")
        run(a.judge_dir, a.judge, a.workers, a.batches)


if __name__ == "__main__":
    main()
