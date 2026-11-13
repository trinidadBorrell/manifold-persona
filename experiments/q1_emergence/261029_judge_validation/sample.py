"""Draw the judge_validation item sets and write the key outside the repo.

Sets:
  dev          20 base (E3 run_rep: one reply per role, desc or ex2) + 20 Instruct (pod A replicates 3-4, one per role)
  sealed       v1.0, kept as drawn: E3 run_main desc 40 and ex2 40 (2 per role each); pod A replicates 0-2, 4 per role
  chat_sample  v2.0 chat sample (100): the 80 pod A replies of `sealed` plus 20 more, one per role, replicates 0-2
  base_e3      v2.0 base sample, E3 part (50): a random 25 desc and 25 ex2 of the 80 E3 replies of `sealed`
  base_podb    v2.0 base sample, pod B part (50): script 17, story 17, description 16, from a v1.6 run
  pop_e3main   every valid E3 run_main desc and ex2 reply (20 roles), for the judge to label (v2.2)
  pop_poda     every valid pod A run_full reply except `default`, for the judge to label (v2.2)
  pop_podb     every valid pod B run_full reply (10 roles x 4 formats), for the judge to label (v2.2)
Replies the owner has read or labelled are left out. The items file holds what the owner and the judge see;
the key (source, condition, format, stop reason) is written next to it, in a folder outside the repo.

    PYTHONPATH=src .venv/bin/python experiments/q1_emergence/261029_judge_validation/sample.py dev OUT_DIR
    PYTHONPATH=src .venv/bin/python experiments/q1_emergence/261029_judge_validation/sample.py sealed OUT_DIR
    PYTHONPATH=src .venv/bin/python experiments/q1_emergence/261029_judge_validation/sample.py chat_sample OUT_DIR
    PYTHONPATH=src .venv/bin/python experiments/q1_emergence/261029_judge_validation/sample.py base_e3 OUT_DIR
    PYTHONPATH=src .venv/bin/python experiments/q1_emergence/261029_judge_validation/sample.py base_podb OUT_DIR \
        --podb_run experiments/q1_emergence/261029_base_role_vs_format/run_full
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3 = HERE.parent / "261008_base_persona_map"
PODA = HERE.parent / "261029_instruct_persona_map" / "run_full"
SEED = 20261029
BASE_PODB = {"script": 17, "story": 17, "description": 16}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


judge2 = load_module("judge2", E3 / "judge2.py")
podb = load_module("podb_gen", HERE.parent / "261029_base_role_vs_format" / "gen_extract.py")
BANK = [json.loads(l)["question"] for l in open(REPO / "refs/assistant-axis/extraction_questions.jsonl")]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contracts():
    """Role contract fields from the RUBRIC.md Part 4 table (the agreed text is the only source)."""
    md = (HERE / "RUBRIC.md").read_text().split("### Contracts for the roles in our runs")[1]
    out = {}
    for line in md.split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Role", "") or set(cells[0]) <= set("-"):
            continue
        for role in cells[0].split(","):
            out[role.strip()] = dict(ai_talk=cells[1], role_act=cells[2], speech=cells[3])
    return out


def frame(question):
    """Coherence context (RUBRIC Part 3): the interview frame without the role line, as pod B's C2 frame."""
    return podb.prompt_text("interview", None, question).removeprefix(podb.EOS_TEXT)


def seen_e3():
    """(run, i) of E3 replies the owner has read or labelled."""
    seen = {("run_main", i) for i in json.load(open(E3 / "run_main/raw_read_ids.json"))}
    for p in (E3 / "review/attrition").glob("owner_labels_*.json"):
        seen |= {("run_main", v["item"]) for v in json.load(open(p)).values()}
    key = json.load(open(REPO.parent / "e3_judge2_pilot/key.json"))
    for i in json.load(open(E3 / "human/role50_subset_2026-10-16.json"))["ids"]:
        run, idx = key[i]["reply_key"].split("/")
        seen.add((run, int(idx)))
    return seen


def seen_rows(run_dir):
    """Row keys named in a run's raw_read.md headings ("## [role,format,wording,question,replicate]")."""
    p = Path(run_dir) / "raw_read.md"
    if not p.exists():
        return set()
    return {json.dumps(json.loads(m), separators=(",", ":")) for m in re.findall(r"^## (\[[^\]]+\])", p.read_text(), re.M)}


def e3_pool(run, conds, seen):
    pool = []
    for l in open(E3 / run / "replies.jsonl"):
        r = json.loads(l)
        if r["valid"] and r["cond"] in conds and r["role"] in judge2.ROLES and (run, r["i"]) not in seen:
            pool.append(dict(src=f"e3:{run}:{r['i']}", group="base", source="E3", cond=r["cond"], fmt="interview",
                             role=r["role"], question=r["question"], reply=r["reply"], stop=r["ended_by"]))
    return pool


def rows_pool(run_dir, source, group, formats, replicates, seen):
    pool = []
    for l in open(Path(run_dir) / "rows.jsonl"):
        r = json.loads(l)
        k = json.dumps([r["role"], r["format"], r["wording"], r["question"], r["replicate"]], separators=(",", ":"))
        if r["valid"] and r["role"] != "default" and r["format"] in formats and r["replicate"] in replicates and k not in seen:
            pool.append(dict(src=f"{source}:{k}", group=group, source=source, cond=None, fmt=r["format"], role=r["role"],
                             question=BANK[r["question"]], reply=r["reply"], stop=r["stop_reason"]))
    return pool


def draw(pool, per_cell, cell, rng):
    """per_cell(c) replies from each cell, random within the cell; pool order is fixed, so the draw is seeded."""
    by = {}
    for it in pool:
        by.setdefault(cell(it), []).append(it)
    out = []
    for c in sorted(by, key=str):
        n = per_cell(c)
        if len(by[c]) < n:
            raise SystemExit(f"cell {c}: {len(by[c])} replies, need {n}")
        out += rng.sample(by[c], n)
    return out


def podb_cells(pool, rng):
    """BASE_PODB per format, spread over roles as evenly as the counts allow."""
    out = []
    for fmt, n in BASE_PODB.items():
        sub = [it for it in pool if it["fmt"] == fmt]
        roles = sorted({it["role"] for it in sub})
        order = rng.sample(roles, len(roles))
        want = Counter(order[j % len(roles)] for j in range(n))
        out += draw(sub, lambda c: want[c], lambda it: it["role"], rng)
    return out


def build(stage, podb_run, rng, out):
    if stage == "pop_e3main":
        return e3_pool("run_main", ("desc", "ex2"), set())
    if stage == "pop_poda":
        return rows_pool(PODA, "podA", "instruct", ("chat",), (0, 1, 2, 3, 4), set())
    if stage == "pop_podb":
        return rows_pool(podb_run, "podB", "base", ("interview", "script", "story", "description"), (0, 1, 2), set())
    if stage == "dev":
        base = e3_pool("run_rep", ("desc", "ex2"), set())
        roles = sorted({it["role"] for it in base})
        cond_of = dict(zip(roles, rng.sample(["desc", "ex2"] * ((len(roles) + 1) // 2), len(roles))))
        items = draw([it for it in base if it["cond"] == cond_of[it["role"]]], lambda c: 1, lambda it: it["role"], rng)
        items += draw(rows_pool(PODA, "podA", "instruct", ("chat",), (3, 4), seen_rows(PODA)), lambda c: 1,
                      lambda it: it["role"], rng)
        return items
    if stage == "base_podb":
        seen_b = set().union(*(seen_rows(p) for p in Path(podb_run).parent.glob("run_*")))
        return podb_cells(rows_pool(podb_run, "podB", "base", tuple(BASE_PODB), (0, 1, 2), seen_b), rng)
    if stage in ("chat_sample", "base_e3"):
        srcs = sorted(v["src"] for v in json.loads((out / "sealed_key.json").read_text()).values())
    if stage == "chat_sample":
        pool = rows_pool(PODA, "podA", "instruct", ("chat",), (0, 1, 2), seen_rows(PODA))
        by = {it["src"]: it for it in pool}
        kept = [by[s] for s in srcs if s.startswith("podA:")]
        taken = set(srcs)
        return kept + draw([it for it in pool if it["src"] not in taken], lambda c: 1, lambda it: it["role"], rng)
    if stage == "base_e3":
        by = {it["src"]: it for it in e3_pool("run_main", ("desc", "ex2"), seen_e3())}
        items = []
        for cond in ("desc", "ex2"):
            items += [by[s] for s in rng.sample([s for s in srcs if s.startswith("e3:") and by[s]["cond"] == cond], 25)]
        return items
    seen = seen_e3()
    items = draw(e3_pool("run_main", ("desc", "ex2"), seen), lambda c: 2, lambda it: (it["cond"], it["role"]), rng)
    items += draw(rows_pool(PODA, "podA", "instruct", ("chat",), (0, 1, 2), seen_rows(PODA)), lambda c: 4,
                  lambda it: it["role"], rng)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "sealed", "chat_sample", "base_e3", "base_podb", "pop_e3main", "pop_poda", "pop_podb"])
    ap.add_argument("out")
    ap.add_argument("--podb_run")
    a = ap.parse_args()
    if (a.stage in ("base_podb", "pop_podb")) != bool(a.podb_run):
        raise SystemExit("--podb_run goes with base_podb and pop_podb only")
    out = Path(a.out).resolve()
    if out.is_relative_to(REPO):
        raise SystemExit("the key must stay outside the repo")
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{a.stage}_items.jsonl").exists():
        raise SystemExit(f"{a.stage} set already drawn in {out}")
    rng = random.Random(f"{SEED}:{a.stage}")
    items = build(a.stage, a.podb_run, rng, out)
    if len({it["src"] for it in items}) != len(items):
        raise SystemExit("a reply is drawn twice")
    con = contracts()
    rng.shuffle(items)
    shown, key = [], {}
    for it in items:
        iid = sha(f"{SEED}|{a.stage}|{it['src']}")[:10]
        q = it["question"]
        shown.append(dict(id=iid, role=it["role"], definition=judge2.definition(it["role"]), contract=con[it["role"]],
                          question=q, frame=frame(q) if it["group"] == "base" and it["fmt"] == "interview" else
                          (podb.prompt_text(it["fmt"], None, q).removeprefix(podb.EOS_TEXT) if it["group"] == "base" else q),
                          reply=it["reply"]))
        key[iid] = dict(src=it["src"], set=a.stage, kind="natural", group=it["group"], source=it["source"], role=it["role"],
                        cond=it["cond"], format=it["fmt"], stop=it["stop"])
    (out / f"{a.stage}_items.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in shown))
    (out / f"{a.stage}_key.json").write_text(json.dumps(key, indent=1))
    cells = Counter((k["group"], k["source"], k["cond"] or k["format"]) for k in key.values())
    manifest = dict(stage=a.stage, seed=SEED, n=len(items), cells={"/".join(c): n for c, n in sorted(cells.items())},
                    items_sha256=file_sha(out / f"{a.stage}_items.jsonl"), key_sha256=file_sha(out / f"{a.stage}_key.json"),
                    code_sha256=file_sha(__file__), rubric_sha256=file_sha(HERE / "RUBRIC.md"),
                    sources={str(p.relative_to(REPO)): file_sha(p) for p in
                             [E3 / "run_rep/replies.jsonl", E3 / "run_main/replies.jsonl", PODA / "rows.jsonl"]
                             + ([Path(a.podb_run).resolve() / "rows.jsonl"] if a.podb_run else [])}
                    | ({"sealed_key.json": file_sha(out / "sealed_key.json")} if a.stage in ("chat_sample", "base_e3") else {}),
                    podb_run=a.podb_run, out_dir=str(out))
    (HERE / "samples").mkdir(exist_ok=True)
    (HERE / "samples" / f"{a.stage}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["cells"], indent=1))


if __name__ == "__main__":
    main()
