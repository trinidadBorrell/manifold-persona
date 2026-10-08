"""Batches for the literature role judge (Assistant Axis 0-3 role rubric + Persona Vectors coherence 0-100).

Items: every character reply scored against its own role, plus a negative control: a random 25% of character
replies scored against a wrong role (a different one of the 60 characters). The judge never sees the model,
condition, or whether an item is a control. Defaults are not role-scored here.

Run from the repo root: .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/rolejudge_prep.py
"""
import json, re, secrets
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
REPO = A.parents[2]
ROLES = REPO / "refs/assistant-axis/roles/instructions"
rows = [json.loads(l) for st in ("base", "final") for l in open(A / f"replies_{st}.jsonl")]
chars = [r for r in rows if not r["role"].startswith("default")]
roles = sorted({r["role"] for r in chars})


def rubric(role):
    """The first paragraph of the role's Assistant Axis eval_prompt: role name and description."""
    p = json.load(open(ROLES / f"{role}.json"))["eval_prompt"]
    return p.split("\n\nPrompt:")[0].strip()


ids, items = set(), []


def new_id():
    while (x := secrets.token_hex(3)) in ids:
        pass
    ids.add(x)
    return x


rng = np.random.default_rng(11)
for r in chars:
    items.append(dict(id=new_id(), target=r["role"], control=False, **{k: r[k] for k in ("stage", "cond", "role", "question_idx", "sample", "i")}))
for k in rng.choice(len(chars), len(chars) // 4, replace=False):
    r = chars[k]
    wrong = rng.choice([x for x in roles if x != r["role"]])
    items.append(dict(id=new_id(), target=str(wrong), control=True, **{k2: r[k2] for k2 in ("stage", "cond", "role", "question_idx", "sample", "i")}))
by_key = {(r["stage"], r["i"]): r for r in chars}
order = rng.permutation(len(items))
(A / "rbatches").mkdir(exist_ok=True)
(A / "ranswers").mkdir(exist_ok=True)
n_batch = int(np.ceil(len(items) / 100))
for b, chunk in enumerate(np.array_split(order, n_batch), 1):
    parts = []
    for k in chunk:
        it = items[k]
        r = by_key[(it["stage"], it["i"])]
        parts.append(f"### {it['id']}\nROLE RUBRIC: {rubric(it['target'])}\n\nQuestion: {r['question']}\n\n"
                     f"Model response:\n[ANSWER START]\n{(r['reply'] or '(empty response)')[:1500]}\n[ANSWER END]\n")
    (A / "rbatches" / f"batch_{b}.md").write_text(f"# Batch {b}: {len(chunk)} items\n\n" + "\n".join(parts))
json.dump({it["id"]: {k: v for k, v in it.items() if k != "id"} for it in items}, open(A / "RKEY.json", "w"), indent=0)
print(len(items), "items (", sum(i["control"] for i in items), "wrong-role controls ),", n_batch, "batches")
