"""Draw the owner's transcript review sample (owner, 12/11/2026: 20 replies per pod, local page, labels to the
judge-work Google Sheet).

Pod A: one reply per role from the 80 sealed pod A items (random, never shown to the owner).
Pod B: two replies per role, five per format, each a random valid reply of that role and format.

Writes, outside the repo, in the judge_validation folder:
- review261112_items.jsonl: the judge-item format that label_sheet.py reads (no source, no key).
- review261112_key.json: per id, the source row, the exact model input and the full raw continuation.

    .venv/bin/python -I experiments/q1_emergence/261029_stage_persona_plan/owner_review/draw_review_261112.py
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

SEED = 20261112
SET = "review261112"
JV = Path("/Users/brendon/persona-manifolds/judge_validation_261029")
REPO = Path(__file__).resolve().parents[4]
Q = REPO / "experiments/q1_emergence"
ROWS = {"podA": Q / "261029_instruct_persona_map/run_full/rows.jsonl",
        "podB": Q / "261029_base_role_vs_format/run_full/rows.jsonl"}
TOKENIZER = {"podA": "allenai/OLMo-2-1124-7B-Instruct", "podB": "allenai/OLMo-2-1124-7B"}
FORMATS = ["interview", "script", "story", "description"]
ITEM_FIELDS = ["id", "role", "definition", "contract", "question", "frame", "reply"]


def load_set(name):
    items = {json.loads(line)["id"]: json.loads(line) for line in open(JV / f"{name}_items.jsonl")}
    key = json.loads((JV / f"{name}_key.json").read_text())
    return items, key


def row_key(src):
    """'podA:["chef","chat",0,158,1]' -> ('podA', ('chef', 'chat', 0, 158, 1))."""
    source, rest = src.split(":", 1)
    return source, tuple(json.loads(rest))


def main():
    rng = random.Random(SEED)
    picks = []

    items, key = load_set("sealed")
    by_role = defaultdict(list)
    for i in sorted(items):
        if key[i]["source"] == "podA":
            by_role[items[i]["role"]].append(i)
    assert len(by_role) == 20 and all(len(v) == 4 for v in by_role.values())
    for role in sorted(by_role):
        i = rng.choice(by_role[role])
        picks.append((items[i], key[i]))

    items, key = load_set("pop_podb")
    cells = defaultdict(list)
    for i in sorted(items):
        cells[(items[i]["role"], key[i]["format"])].append(i)
    roles = sorted({r for r, _ in cells})
    assert len(roles) == 10 and {f for _, f in cells} == set(FORMATS)
    slots = []
    for _ in range(5):
        block = FORMATS[:]
        rng.shuffle(block)
        slots += block
    role_order = roles[:]
    rng.shuffle(role_order)
    for j, role in enumerate(role_order):
        for fmt in slots[2 * j:2 * j + 2]:
            i = rng.choice(cells[(role, fmt)])
            picks.append((items[i], key[i]))
    assert sum(1 for _, k in picks if k["source"] == "podB") == 20

    need = defaultdict(set)
    for _, k in picks:
        s, rk = row_key(k["src"])
        need[s].add(rk)
    rows = {}
    for s, path in ROWS.items():
        for line in open(path):
            r = json.loads(line)
            rk = (r["role"], r["format"], r["wording"], r["question"], r["replicate"])
            if rk in need[s]:
                rows[(s, rk)] = r
    toks = {s: AutoTokenizer.from_pretrained(m, local_files_only=True) for s, m in TOKENIZER.items()}

    rng.shuffle(picks)
    out_items, out_key = [], {}
    for n, (item, k) in enumerate(picks, 1):
        s, rk = row_key(k["src"])
        r = rows[(s, rk)]
        assert r["valid"] and r["reply"].strip() == item["reply"].strip(), k["src"]
        tok = toks[s]
        out_items.append({f: item[f] for f in ITEM_FIELDS})
        out_key[item["id"]] = dict(
            n=n, src=k["src"], source=s, group=k["group"], role=item["role"], format=k["format"],
            wording=rk[2], question_id=rk[3], replicate=rk[4], stop=r["stop_reason"],
            exact_input=tok.decode(r["prompt_ids"], skip_special_tokens=False),
            raw_continuation=tok.decode(r["raw_ids"], skip_special_tokens=False),
            kept_reply=tok.decode(r["kept_ids"], skip_special_tokens=False))

    raw = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_items).encode()
    (JV / f"{SET}_items.jsonl").write_bytes(raw)
    (JV / f"{SET}_key.json").write_text(json.dumps(
        dict(seed=SEED, items_sha256=hashlib.sha256(raw).hexdigest(), items=out_key), indent=1, ensure_ascii=False))
    counts = defaultdict(int)
    for k in out_key.values():
        counts[(k["source"], k["format"])] += 1
    print(f"{len(out_items)} items; {dict(counts)}; sha256 {hashlib.sha256(raw).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
