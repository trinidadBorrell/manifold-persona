"""Trini's J8 and Gio's ranking judge (J10) on E3 base replies, checked against the owner's labels (23/10/2026).

Owner decisions 23/10: only the opening lines change for interview replies, everything else word for word; Claude
Sonnet 5.5 via `claude -p` (tools off, empty folder, no settings or MCP); cost cap $5; role scores reported only.
Rules: THRESHOLDS.md. Replies: the 80 owner Fine/Broken labels and the 50 owner role scores (130 run_main replies).

J8   aebaed1:src/route_judge_sweep/prompts.py, version 0 of J2_INTRO/OTHER/COLLAPSED/RULE/INTENSITY/ORDER; the 275
     one-line role definitions + other + collapsed in a seeded order; the three worked examples from Trini's note
     (j8_examples.json, her code reads them from the cluster). Sent as in her api_judge.py: the J8 text as the system
     prompt and 15 items per request, each {key, question, response}, inside her ASK text.
J10  d6dbb74:attr_analysis/make_rank_chunks.py HEAD with its first three lines reworded; the 276 role names; 15
     answers per chunk, whitespace collapsed, first 500 characters, no question; the system prompt of api_judge.py.

    python judge_designs.py prep JUDGE_DIR      # new folder outside the repo
    python judge_designs.py run JUDGE_DIR [--only j8_00,j10_00] [--workers 4]
    python judge_designs.py score JUDGE_DIR     # writes results/judge_designs_score.json
"""
import argparse
import ast
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3 = REPO / "experiments/q1_emergence/261008_base_persona_map"
COH_KEY = Path("/Users/brendon/persona-manifolds/e3_judge2_coh_261016/key.json")
TRINI, GIO = "aebaed1", "d6dbb74"
MODEL = "claude-sonnet-5-5"
CLAUDE = shutil.which("claude") or "/Users/brendon/.local/bin/claude"
COST_CAP_USD = 5.0
PER, CLIP, SEED, SALT = 15, 500, 20261023, "e3-judge-designs"
LABELS = {"set1": E3 / "review/attrition/owner_labels_desc_2026-10-16_final.json",
          "set2": E3 / "review/attrition/owner_labels_fresh30_2026-10-16.json"}
ROLE50 = E3 / "human/owner_role50_scores_2026-10-16.json"

J8_OLD = "an instruct-tuned LLM's [response]"
J8_NEW = "a language model's [response] (the guest's answer in an interview)"
J10_OLD = ("Each numbered item is a short answer from a language model given NO system prompt.\n"
           "Some answers have been pushed, by editing the model's activations, toward one persona\n"
           "from the list below; others are untouched. Guess which persona each answer was pushed toward.\n")
J10_NEW = ("Each numbered item is a short answer that a language model wrote as the guest in an interview.\n"
           "The model may have been asked to speak as one persona from the list below.\n"
           "Guess which persona each answer speaks as.\n")
J10_SYSTEM = "You follow the instructions in the user message exactly."


def git_show(ref_path):
    return subprocess.run(["git", "-C", str(REPO), "show", ref_path], capture_output=True, text=True,
                          check=True).stdout


def constants(source, names):
    """Module-level literal assignments from a source file, without running it."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"not found in source: {sorted(missing)}")
    return out


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


# ---------------------------------------------------------------- prompts

def j8_system():
    c = constants(git_show(f"{TRINI}:src/route_judge_sweep/prompts.py"),
                  ["J2_INTRO", "J2_OTHER", "J2_COLLAPSED", "J2_RULE", "J2_INTENSITY", "J2_ORDER"])
    lines_def = json.loads(git_show(f"{TRINI}:src/route_judge_sweep/role_lines.json"))
    intro = c["J2_INTRO"][0]
    if intro.count(J8_OLD) != 1:
        raise SystemExit("J8 intro text changed upstream")
    names = sorted(lines_def) + ["other", "collapsed"]
    random.Random(f"{SALT}|0").shuffle(names)
    opts = []
    for o in names:
        if o == "other":
            opts.append("- other: " + c["J2_OTHER"][0])
        elif o == "collapsed":
            opts.append("- collapsed: " + c["J2_COLLAPSED"][0])
        else:
            opts.append(f"- {o}: {lines_def[o]}")
    exs = json.loads((HERE / "j8_examples.json").read_text())["examples"]
    ex_text = "\n\n".join('Question: "%s"\nResponse: "%s"\n-> evidence: "%s"; analysis: %s; persona_kind: %s; intensity: %s'
                          % (e["question"], e["response"], e["evidence"], e["analysis"], e["persona_kind"], e["intensity"])
                          for e in exs)
    fields = ("FIELDS\n\n"
              "- evidence: a VERBATIM quote of at most 25 words from the [response] that decides your answer.\n"
              "- analysis: 1-3 sentences saying what was claimed or enacted, or why nothing was.\n"
              "- persona_kind: one of the %d options above.\n"
              "- intensity: 1 or 2 when persona_kind is a persona; null for other and collapsed.\n%s\n\n%s\n\n"
              "Answer with a single JSON object and nothing else, with keys: evidence, analysis, persona_kind, intensity."
              % (len(names), c["J2_INTENSITY"][0], c["J2_ORDER"][0]))
    text = "\n\n".join([
        intro.replace(J8_OLD, J8_NEW),
        "THE %d OPTIONS (the order is random and means nothing)\n%s" % (len(names), "\n".join(opts)),
        c["J2_RULE"][0],
        "WORKED EXAMPLES (the personas in them are examples only; they may or may not be among your options)\n\n" + ex_text,
        fields])
    return text, names


def j8_ask():
    src = git_show(f"{TRINI}:src/route_judge_sweep/api_judge.py")
    if f'"{J10_SYSTEM}"' not in src:
        raise SystemExit("api_judge.py rank system prompt changed upstream")
    return constants(src, ["ASK"])["ASK"]


def j10_head():
    head = constants(git_show(f"{GIO}:attr_analysis/make_rank_chunks.py"), ["HEAD"])["HEAD"]
    if not head.startswith(J10_OLD):
        raise SystemExit("J10 head text changed upstream")
    roles = sorted(p.stem for p in (REPO / "refs/assistant-axis/roles/instructions").glob("*.json"))
    return J10_NEW + head[len(J10_OLD):].replace("__ROLES__", ", ".join(roles)), roles


# ---------------------------------------------------------------- prep

def load_items():
    reps = {}
    for line in open(E3 / "run_main/replies.jsonl"):
        r = json.loads(line)
        reps[f"run_main/{r['i']}"] = r
    items = {}

    def add(rk):
        if rk not in items:
            r = reps[rk]
            items[rk] = dict(reply_key=rk, role=r["role"], cond=r["cond"], example_role=r.get("example_role"),
                             question=r["question"], reply=r["reply"] or "")
        return items[rk]

    for s, f in LABELS.items():
        for v in json.loads(f.read_text()).values():
            if v["verdict"] not in ("fine", "broken"):
                raise SystemExit(f"owner label other than fine/broken: {v}")
            add(f"run_main/{v['item']}").update(broken_set=s, owner_broken=v["verdict"] == "broken",
                                                owner_reasons=v["reasons"])
    key = json.loads(COH_KEY.read_text())
    for jid, v in json.loads(ROLE50.read_text()).items():
        k = key[jid]
        it = add(k["reply_key"])
        if "role_score" in it:
            raise SystemExit(f"two role scores for {k['reply_key']}")
        it.update(role_item=jid, role_kind=k["kind"], scored_role=k["scored_role"], role_score=v["score"])
    return items


def prep(jd):
    jd = Path(jd)
    if jd.exists():
        raise SystemExit(f"{jd} exists; prep needs a new folder")
    if REPO in jd.resolve().parents:
        raise SystemExit("the judge folder must be outside the repo")
    items = load_items()
    order = sorted(items)
    random.Random(SEED).shuffle(order)
    ids = {rk: sha(f"{SALT}|{rk}")[:10] for rk in order}
    sys8, names8 = j8_system()
    ask = j8_ask()
    head10, roles10 = j10_head()
    (jd / "batches").mkdir(parents=True)
    batches = {}
    for n in range((len(order) + PER - 1) // PER):
        block = order[n * PER:(n + 1) * PER]
        msg = ask % json.dumps([{"key": ids[rk], "question": items[rk]["question"], "response": items[rk]["reply"]}
                                for rk in block], ensure_ascii=False)
        name = f"j8_{n:02d}"
        (jd / "batches" / f"{name}.txt").write_text(msg)
        batches[name] = dict(judge="j8", ids=[ids[rk] for rk in block], prompt_sha256=sha(msg))
        text = head10 + "".join(f"{i}. {' '.join(items[rk]['reply'].split())[:CLIP]}\n" for i, rk in enumerate(block, 1))
        name = f"j10_{n:02d}"
        (jd / "batches" / f"{name}.txt").write_text(text)
        batches[name] = dict(judge="j10", ids=[ids[rk] for rk in block], prompt_sha256=sha(text))
    (jd / "j8_system.txt").write_text(sys8)
    key = {ids[rk]: items[rk] for rk in order}
    (jd / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    man = dict(model=MODEL, cli=subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip(),
               trini_ref=TRINI, gio_ref=GIO, per=PER, clip=CLIP, seed=SEED, salt=SALT, cost_cap_usd=COST_CAP_USD,
               j8_system_sha256=sha(sys8), j8_options=names8, j10_system=J10_SYSTEM, j10_roles=roles10,
               code_sha256=sha((HERE / "judge_designs.py").read_bytes()), key_sha256=sha((jd / "key.json").read_text()),
               batches=batches)
    (jd / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{len(key)} replies, {len(batches)} batches -> {jd}")


# ---------------------------------------------------------------- run

def norm(x):
    return str(x).strip().lower().replace(" ", "_")


def parse(text):
    """Trini's tolerant parse (api_judge.py): the first {...} span of the reply."""
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def check(judge, ans, ids, man):
    if judge == "j8":
        labs = (ans or {}).get("labels") if isinstance(ans, dict) else None
        if not isinstance(labs, list):
            return "no labels list"
        got = {x.get("key"): x for x in labs if isinstance(x, dict)}
        if set(got) != set(ids) or len(labs) != len(ids):
            return "keys differ"
        if any("persona_kind" not in x for x in got.values()):
            return "persona_kind missing"
        return None
    if not isinstance(ans, dict) or set(ans) != {str(i) for i in range(1, len(ids) + 1)}:
        return "ids differ"
    for v in ans.values():
        if not (isinstance(v, dict) and isinstance(v.get("rank"), list) and len(v["rank"]) == 10
                and isinstance(v.get("nonsense"), bool)):
            return "bad item"
    return None


def call(system, user):
    with tempfile.TemporaryDirectory() as empty:
        cmd = [CLAUDE, "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--system-prompt", system, "--no-session-persistence", "--output-format", "json"]
        p = subprocess.run(cmd, cwd=empty, input=user, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {p.stderr[-300:]}")
    out = json.loads(p.stdout)
    if list(out.get("modelUsage", {})) != [MODEL]:
        raise RuntimeError(f"model used {list(out.get('modelUsage', {}))}")
    if out.get("is_error"):
        raise RuntimeError(f"error result ({out.get('subtype')})")
    return out


def spent(jd):
    return sum(t.get("cost_usd", 0) for f in (jd / "answers").glob("*.json")
               for t in json.loads(f.read_text())["tries"])


def run(jd, only, workers):
    jd = Path(jd)
    man = json.loads((jd / "manifest.json").read_text())
    if man["code_sha256"] != sha((HERE / "judge_designs.py").read_bytes()):
        raise SystemExit("judge_designs.py changed since prep; prep a new folder")
    sys8 = (jd / "j8_system.txt").read_text()
    if sha(sys8) != man["j8_system_sha256"]:
        raise SystemExit("j8_system.txt changed since prep")
    (jd / "answers").mkdir(exist_ok=True)
    lock = threading.Lock()
    names = [n for n in man["batches"] if not only or n in only]

    def job(name):
        meta = man["batches"][name]
        f = jd / "answers" / f"{name}.json"
        user = (jd / "batches" / f"{name}.txt").read_text()
        if sha(user) != meta["prompt_sha256"]:
            return name, "prompt changed since prep"
        rec = json.loads(f.read_text()) if f.exists() else dict(name=name, prompt_sha256=meta["prompt_sha256"], tries=[])
        if rec.get("answer") is not None:
            return name, None
        system = sys8 if meta["judge"] == "j8" else man["j10_system"]
        err = None
        for _ in range(3):
            with lock:
                if spent(jd) >= man["cost_cap_usd"]:
                    return name, "cost cap reached"
            try:
                out = call(system, user)
                ans = parse(out.get("result", ""))
                err = check(meta["judge"], ans, meta["ids"], man)
            except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                out, ans, err = {}, None, f"{type(e).__name__}: {e}"
            rec["tries"].append(dict(error=err, cost_usd=out.get("total_cost_usd", 0), usage=out.get("modelUsage"),
                                     result=out.get("result")))
            if err is None:
                rec["answer"] = ans
            with lock:
                f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            if err is None:
                return name, None
        return name, err

    with ThreadPoolExecutor(workers) as ex:
        for name, err in ex.map(job, names):
            print(f"{name}: {'ok' if err is None else err}", flush=True)
    print(f"spent ${spent(jd):.3f} of ${man['cost_cap_usd']}")


# ---------------------------------------------------------------- score

def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def kappa_ci(a, b, n=2000):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    ks = [k for k in ks if k == k]
    return [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))] if ks else [None, None]


def table(owner, judge):
    return {f"owner_{o}|judge_{j}": int(sum((x == o) and (y == j) for x, y in zip(owner, judge)))
            for o in (True, False) for j in (True, False)}


def score(jd):
    jd = Path(jd)
    man = json.loads((jd / "manifest.json").read_text())
    key = json.loads((jd / "key.json").read_text())
    opts8, roles10 = set(man["j8_options"]), set(man["j10_roles"])
    j8, j10, off_list = {}, {}, {"j8": 0, "j10": 0}
    for name, meta in man["batches"].items():
        f = jd / "answers" / f"{name}.json"
        ans = json.loads(f.read_text()).get("answer") if f.exists() else None
        if ans is None:
            continue
        if meta["judge"] == "j8":
            for x in ans["labels"]:
                k = norm(x["persona_kind"])
                off_list["j8"] += k not in opts8
                j8[x["key"]] = dict(kind=k, intensity=x.get("intensity"))
        else:
            for i, v in ans.items():
                rank = [norm(r) for r in v["rank"]]
                off_list["j10"] += sum(r not in roles10 for r in rank)
                j10[meta["ids"][int(i) - 1]] = dict(rank=rank, nonsense=v["nonsense"])
    n = len(key)
    rep = dict(n_replies=n, missing={"j8": n - len(j8), "j10": n - len(j10)}, names_outside_list=off_list,
               spent_usd=spent(jd))
    rep["incomplete"] = {j: (n - len(d)) / n > 0.05 for j, d in (("j8", j8), ("j10", j10))}

    # broken: the gate
    pred = {"j8": {i: v["kind"] == "collapsed" for i, v in j8.items()},
            "j10": {i: v["nonsense"] for i, v in j10.items()}}
    rep["broken"] = {}
    for j, p in pred.items():
        r = {}
        for s in ("set1", "set2"):
            ids = [i for i, v in key.items() if v.get("broken_set") == s and i in p]
            o, q = [key[i]["owner_broken"] for i in ids], [p[i] for i in ids]
            r[s] = dict(n=len(ids), kappa=kappa(o, q), ci95=kappa_ci(o, q), table=table(o, q))
        ids = [i for i, v in key.items() if v.get("owner_broken") and i in p]
        by_reason = {}
        for i in ids:
            for reason in key[i]["owner_reasons"] or ["none_given"]:
                by_reason.setdefault(reason, []).append(p[i])
        r["hit_rate_by_owner_reason"] = {k: [sum(v), len(v)] for k, v in sorted(by_reason.items())}
        long_ = [p[i] for i in ids if len(" ".join(key[i]["reply"].split())) > CLIP]
        short = [p[i] for i in ids if len(" ".join(key[i]["reply"].split())) <= CLIP]
        r["hit_rate_owner_broken_by_length"] = {f"over_{CLIP}": [sum(long_), len(long_)], f"upto_{CLIP}": [sum(short), len(short)]}
        r["usable"] = all(r[s]["kappa"] >= 0.6 for s in ("set1", "set2"))
        rep["broken"][j] = r

    # naming: report only
    rows = []
    for i, v in key.items():
        row = dict(cond=v["cond"], role=v["role"], donor=v.get("example_role"))
        if i in j8:
            row["j8_hit"] = j8[i]["kind"] == v["role"]
            row["j8_donor"] = bool(v.get("example_role")) and j8[i]["kind"] == v["example_role"]
            row["j8_kind"] = j8[i]["kind"]
        if i in j10:
            rk = j10[i]["rank"]
            row.update(j10_r1=rk[:1] == [v["role"]], j10_r5=v["role"] in rk[:5],
                       j10_donor_r5=bool(v.get("example_role")) and v["example_role"] in rk[:5], j10_top=rk[0])
        rows.append(row)
    naming = {}
    for c in sorted({r["cond"] for r in rows}) + ["all"]:
        rr = [r for r in rows if c in ("all", r["cond"])]
        m = {"n": len(rr)}
        for f in ("j8_hit", "j8_donor", "j10_r1", "j10_r5", "j10_donor_r5"):
            v = [r[f] for r in rr if f in r]
            m[f] = [int(sum(v)), len(v)]
        m["j8_other"] = int(sum(r.get("j8_kind") == "other" for r in rr))
        m["j8_assistant"] = int(sum(r.get("j8_kind") == "assistant" for r in rr))
        m["j10_assistant_top1"] = int(sum(r.get("j10_top") == "assistant" for r in rr))
        naming[c] = m
    rep["naming"] = naming

    # role scores: report only
    t = [i for i, v in key.items() if v.get("role_kind") == "target"]
    role = {}
    for j, named in (("j8_named", lambda i: j8[i]["kind"] == key[i]["scored_role"]),
                     ("j10_top5", lambda i: key[i]["scored_role"] in j10[i]["rank"][:5])):
        have = [i for i in t if i in (j8 if j.startswith("j8") else j10)]
        q = [named(i) for i in have]
        cross = {}
        for i, x in zip(have, q):
            cross.setdefault(str(key[i]["role_score"]), [0, 0])[0 if x else 1] += 1
        r = {"n": len(have), "named_vs_not_by_owner_score": cross}
        for cut, lab in ((2, "owner_ge2"), (3, "owner_eq3")):
            o = [key[i]["role_score"] >= cut for i in have]
            r[lab] = dict(kappa=kappa(o, q), ci95=kappa_ci(o, q), table=table(o, q))
        ctrl = [i for i, v in key.items() if v.get("role_kind") in ("close", "distant") and i in (j8 if j.startswith("j8") else j10)]
        r["controls_named_wrong_role"] = [int(sum(named(i) for i in ctrl)), len(ctrl)]
        role[j] = r
    rep["role_scores"] = role

    out = HERE / "results" / "judge_designs_score.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "score"])
    ap.add_argument("judge_dir")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.judge_dir)
    elif a.cmd == "run":
        run(a.judge_dir, [x for x in a.only.split(",") if x], a.workers)
    else:
        score(a.judge_dir)


if __name__ == "__main__":
    main()
