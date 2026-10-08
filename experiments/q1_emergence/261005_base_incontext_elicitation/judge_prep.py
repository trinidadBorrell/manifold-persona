"""Blind judge batches for E0: every reply, shuffled, random ids, reply text only. Keys go to KEY.json.
Also writes the blind human-label sheet (60 base + 60 final character replies) and the raw-read sheet.

Run from the repo root: .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/judge_prep.py
"""
import json, secrets, shutil
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
REPO = A.parents[2]
rows = [json.loads(l) for st in ("base", "final") for l in open(A / f"replies_{st}.jsonl")]
ids = set()


def new_id():
    while (x := secrets.token_hex(3)) in ids:
        pass
    ids.add(x)
    return x


for r in rows:
    r["id"] = new_id()
shutil.copy(REPO / "output/results_2026-09/audit_2026-10-01/candidates.txt", A / "candidates.txt")
rng = np.random.default_rng(7)
order = rng.permutation(len(rows))
(A / "batches").mkdir(exist_ok=True)
(A / "answers").mkdir(exist_ok=True)
n_batch = int(np.ceil(len(rows) / 150))
for b, chunk in enumerate(np.array_split(order, n_batch), 1):
    parts = [f"### {rows[k]['id']}\n{(rows[k]['reply'] or '(empty reply)')[:1500]}\n" for k in chunk]
    (A / "batches" / f"batch_{b}.md").write_text(f"# Batch {b}: {len(chunk)} items\n\n" + "\n".join(parts))
json.dump({r["id"]: {k: v for k, v in r.items() if k not in ("id", "reply", "raw")} for r in rows},
          open(A / "KEY.json", "w"), indent=0)
print(len(rows), "replies,", n_batch, "batches")

# Blind human sheet: neutral-label character replies, 60 per model, shuffled together.
chars = [r for r in rows if not r["role"].startswith("default") and r["cond"].endswith("neutral")]
pick = []
for st in ("base", "final"):
    pool = [r for r in chars if r["stage"] == st]
    pick += [pool[i] for i in rng.choice(len(pool), 60, replace=False)]
rng.shuffle(pick)
(A / "human_label_sheet.md").write_text(
    "# Human labels (blind)\n\nFor each reply, write the character you think it speaks as (any of the 276 names in "
    "candidates.txt, or `default` for a plain answer), and yes/no: does it play a character beyond only naming one?\n\n"
    + "\n".join(f"### {r['id']}\n{r['reply'] or '(empty reply)'}\n\nlabel: \nplays: \n" for r in pick))
json.dump([r["id"] for r in pick], open(A / "human_label_ids.json", "w"))
print("human sheet:", len(pick), "replies")
