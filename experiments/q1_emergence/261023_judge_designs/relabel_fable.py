"""Claude Fable 5.1 relabels the owner's 80 Fine/Broken replies under the owner's 16/10 definition (23/10/2026).

Owner decisions 23/10: "We can do both" (re-score and a neutral-model relabel); model Claude Fable 5.1, a model that
has not judged broken replies in this project; all 80 labelled replies; the 16/10 definition word for word (E3
THRESHOLDS.md, change of 16/10). The fresh 30 were labelled closer to that definition, so they check the model; the
first 50 use an older idea of broken, so the model's labels for them are a new, exploratory key. Each reply is shown
as the owner saw it (no fragment removal). System prompt, batch format and isolation as judge2.py.

    python relabel_fable.py prep JUDGE_DIR
    python relabel_fable.py run JUDGE_DIR
    python relabel_fable.py score JUDGE_DIR     # writes results/relabel_fable_score.json
"""
import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3 = HERE.parent / "261008_base_persona_map"
_spec = importlib.util.spec_from_file_location("judge2", E3 / "judge2.py")
judge2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge2)
_spec = importlib.util.spec_from_file_location("judge_designs", HERE / "judge_designs.py")
jd_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jd_mod)

MODEL = "claude-fable-5-1"
CLAUDE = shutil.which("claude") or "/Users/brendon/.local/bin/claude"
BATCH, SEED, SALT, COST_CAP_USD = 20, 20261023, "e3-broken-relabel-fable", 5.0
DEFINITION = ("Broken: the reply does not work as an answer to the question. That means it loops (the same idea again "
              "and again, adding nothing), it does not make sense, it writes new interview turns or copies the question, "
              "or it ignores the question and talks about something else, even in character. Not broken: cut off by "
              "the length limit, a phrase repeated for emphasis, or an answer that is wrong, odd, rude, short, or in a "
              "strange voice.")
SCHEMA_ITEM = {"type": "object", "properties": {"broken": {"type": "boolean"}}, "required": ["broken"],
               "additionalProperties": False}
DESIGNS = Path("/Users/brendon/persona-manifolds/e3_judge_designs_261023")
FINISH = E3 / "results/broken/finish_261016/labels.jsonl"


def item_text(question, answer):
    return ("Decide whether this interview answer is broken, using this definition.\n"
            f"Definition: \"{DEFINITION}\"\n\n"
            "Question:\n[QUESTION START]\n" + question + "\n[QUESTION END]\n"
            "Answer:\n[ANSWER START]\n" + answer + "\n[ANSWER END]\n\n"
            "Answer for this item: an object {\"broken\": true or false}.")


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def prep(jd):
    jd = Path(jd)
    if jd.exists():
        raise SystemExit(f"{jd} exists; prep needs a new folder")
    items = {k: v for k, v in jd_mod.load_items().items() if "broken_set" in v}
    if len(items) != 80:
        raise SystemExit(f"{len(items)} labelled replies, expected 80")
    order = sorted(items)
    random.Random(SEED).shuffle(order)
    ids = {k: sha(f"{SALT}|{k}")[:10] for k in order}
    for k in order:
        if judge2.DELIM.search(items[k]["reply"]) or judge2.DELIM.search(items[k]["question"]):
            raise SystemExit(f"delimiter text in {k}")
    (jd / "batches").mkdir(parents=True)
    batches = {}
    for n in range(0, len(order), BATCH):
        block = order[n:n + BATCH]
        prompt = judge2.batch_prompt([{"id": ids[k], "text": item_text(items[k]["question"], items[k]["reply"])}
                                      for k in block])
        name = f"b{n // BATCH:02d}"
        (jd / "batches" / f"{name}.txt").write_text(prompt)
        batches[name] = dict(ids=[ids[k] for k in block], prompt_sha256=sha(prompt))
    key = {ids[k]: dict(reply_key=k, broken_set=items[k]["broken_set"], owner_broken=items[k]["owner_broken"])
           for k in order}
    (jd / "key.json").write_text(json.dumps(key, indent=1))
    man = dict(model=MODEL, cli=subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip(),
               system_prompt=judge2.SYSTEM, item_template_sha256=sha(item_text("{question}", "{answer}")),
               code_sha256=sha((HERE / "relabel_fable.py").read_bytes()), seed=SEED, batch=BATCH,
               cost_cap_usd=COST_CAP_USD, batches=batches)
    (jd / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{len(key)} replies, {len(batches)} batches -> {jd}")


def call(prompt, ids):
    schema = {"type": "object", "properties": {i: SCHEMA_ITEM for i in ids}, "required": list(ids),
              "additionalProperties": False}
    with tempfile.TemporaryDirectory() as empty:
        cmd = [CLAUDE, "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--system-prompt", judge2.SYSTEM, "--no-session-persistence", "--output-format", "json",
               "--json-schema", json.dumps(schema)]
        p = subprocess.run(cmd, cwd=empty, input=prompt, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {p.stdout[-300:]} {p.stderr[-300:]}")
    out = json.loads(p.stdout)
    if list(out.get("modelUsage", {})) != [MODEL]:
        raise RuntimeError(f"model used {list(out.get('modelUsage', {}))}")
    so = out.get("structured_output")
    a = judge2.parse_answer(so) if isinstance(so, str) else so
    if not isinstance(a, dict) or set(a) != set(ids) or not all(isinstance(v, dict) and isinstance(v.get("broken"), bool)
                                                              for v in a.values()):
        raise RuntimeError("answer ids or values differ")
    return a, out


def run(jd):
    jd = Path(jd)
    man = json.loads((jd / "manifest.json").read_text())
    if man["code_sha256"] != sha((HERE / "relabel_fable.py").read_bytes()):
        raise SystemExit("relabel_fable.py changed since prep; prep a new folder")
    (jd / "answers").mkdir(exist_ok=True)
    spent = sum(json.loads(f.read_text()).get("cost_usd", 0) for f in (jd / "answers").glob("*.json"))
    for name, meta in man["batches"].items():
        f = jd / "answers" / f"{name}.json"
        if f.exists():
            continue
        prompt = (jd / "batches" / f"{name}.txt").read_text()
        if sha(prompt) != meta["prompt_sha256"]:
            raise SystemExit(f"{name}: prompt changed since prep")
        err = None
        for _ in range(3):
            if spent >= man["cost_cap_usd"]:
                raise SystemExit(f"cost cap reached (${spent:.2f})")
            try:
                a, out = call(prompt, meta["ids"])
                spent += out.get("total_cost_usd", 0)
                f.write_text(json.dumps(dict(answer=a, cost_usd=out.get("total_cost_usd", 0),
                                             usage=out.get("modelUsage")), indent=1))
                err = None
                break
            except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                err = f"{type(e).__name__}: {e}"
        print(f"{name}: {'ok' if err is None else err}", flush=True)
    print(f"spent ${spent:.3f}")


def score(jd):
    jd = Path(jd)
    key = json.loads((jd / "key.json").read_text())
    fable = {}
    for f in (jd / "answers").glob("*.json"):
        for i, v in json.loads(f.read_text())["answer"].items():
            fable[key[i]["reply_key"]] = v["broken"]
    owner = {v["reply_key"]: v["owner_broken"] for v in key.values()}
    sets = {v["reply_key"]: v["broken_set"] for v in key.values()}
    # other judges on the same 80 replies
    fin = {json.loads(l)["reply_key"]: json.loads(l) for l in open(FINISH)}
    judges = {"fable": fable,
              "gpt_r3": {k: (fin[k]["gpt_raw"] or {}).get("broken") for k in owner},
              "opus_r3": {k: (fin[k]["opus"] or {}).get("broken") for k in owner},
              "jev": {k: max(fin[k]["jev"].values()) > 0.5 for k in owner}}
    dk = json.loads((DESIGNS / "key.json").read_text())
    dman = json.loads((DESIGNS / "manifest.json").read_text())
    j8, j10 = {}, {}
    for name, meta in dman["batches"].items():
        ans = json.loads((DESIGNS / "answers" / f"{name}.json").read_text())["answer"]
        if meta["judge"] == "j8":
            for x in ans["labels"]:
                j8[dk[x["key"]]["reply_key"]] = jd_mod.norm(x["persona_kind"]) == "collapsed"
        else:
            for i, v in ans.items():
                j10[dk[meta["ids"][int(i) - 1]]["reply_key"]] = v["nonsense"]
    judges["j8"], judges["j10"] = j8, j10

    def agree(ref, j, keys):
        ks = [k for k in keys if judges[j].get(k) is not None and ref.get(k) is not None]
        a, b = [ref[k] for k in ks], [judges[j][k] for k in ks]
        return dict(n=len(ks), kappa=round(jd_mod.kappa(a, b), 3), ci95=[round(x, 3) for x in jd_mod.kappa_ci(a, b)],
                    table=jd_mod.table(a, b))

    s1 = [k for k in owner if sets[k] == "set1"]
    s2 = [k for k in owner if sets[k] == "set2"]
    new_key = {k: (fable[k] if sets[k] == "set1" else owner[k]) for k in owner}
    rep = dict(model=MODEL, n=len(fable), spent_usd=sum(json.loads(f.read_text()).get("cost_usd", 0)
                                                         for f in (jd / "answers").glob("*.json")),
               fable_vs_owner={"set1_old_definition": agree(owner, "fable", s1), "set2_fresh": agree(owner, "fable", s2)},
               owner_set1_changed_by_fable=dict(broken_to_fine=sum(owner[k] and not fable[k] for k in s1),
                                                fine_to_broken=sum(fable[k] and not owner[k] for k in s1)),
               judges_vs_owner_fresh30={j: agree(owner, j, s2) for j in judges if j != "fable"},
               judges_vs_fable_set1={j: agree(fable, j, s1) for j in judges if j != "fable"},
               judges_vs_new_key_80={j: agree(new_key, j, list(owner)) for j in judges if j != "fable"})
    out = HERE / "results" / "relabel_fable_score.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "score"])
    ap.add_argument("judge_dir")
    a = ap.parse_args()
    {"prep": prep, "run": run, "score": score}[a.cmd](a.judge_dir)


if __name__ == "__main__":
    main()
