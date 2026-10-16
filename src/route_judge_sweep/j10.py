"""J10: Giovanni Marraffini's ranking judge (branch persona-definition, attr_analysis/make_rank_chunks.py
and score_rank.py @ b860780), ported exactly to the judge-sweep answers.

    python -m route_judge_sweep.j10 build     # -> JS/J10/chunk_NNN.txt, index.json, roles.json
    python -m route_judge_sweep.j10 collect   # -> JS/J10/labels.csv (+ redo.json)

Same header text, the 276 Assistant Axis role names (default included), sorted and comma-joined;
whitespace collapsed, answers clipped to 500 characters, all answers shuffled with seed 0, 15 per chunk.
The judge sees only the answer text. Recall@k (k = 1, 5, 10) is computed against the pair's target.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from route_judge_sweep.build import PM

JS = PM / "JS"
OUT = JS / "J10"
ROLES = Path(__file__).resolve().parents[2] / "refs/assistant-axis/roles/instructions"
PER_CHUNK, CLIP, SEED = 15, 500, 0

# verbatim from make_rank_chunks.py
HEAD = """Each numbered item is a short answer from a language model given NO system prompt.
Some answers have been pushed, by editing the model's activations, toward one persona
from the list below; others are untouched. Guess which persona each answer was pushed toward.

PERSONAS (the only names allowed, spelled exactly like this):
__ROLES__

For each item, in order:
1. "why": think first, in one or two sentences. What do the voice, imagery, vocabulary,
   self-description and attitude show, and which personas do they point to? Say whether
   it is plain helpful-AI prose, and whether it is degenerate (repetition loops, word
   salad, mangled words, broken grammar).
2. "rank": exactly 10 distinct names from the list, best first. Position 1 is your single
   best guess. Positions 2-10 are the next most plausible, closest first.
3. "nonsense": true when a large part of the answer is mangled or repeating.

Rules. Judge each item from its own text only, never from its number or its neighbours.
"assistant" is a valid persona: put it first when the whole answer is ordinary
helpful-AI prose. An answer that opens "As an AI..." and then continues in a character's
voice is not assistant. Where the text is thin, still give 10 names, ordered by the faint
cues that are there.

Reply with STRICT JSON only, every id present:
{"1": {"why": "...", "rank": ["ghost", "spirit", "wraith", "revenant", "void", "echo", "mystic", "oracle", "sage", "poet"], "nonsense": false}, "2": {...}, ...}

"""


def roles():
    return sorted(p.stem for p in ROLES.glob("*.json"))


def build():
    OUT.mkdir(exist_ok=True)
    a = pd.read_csv(JS / "answers.csv", low_memory=False)
    rows = a[["model", "key", "response"]].to_dict("records")
    random.Random(SEED).shuffle(rows)
    head = HEAD.replace("__ROLES__", ", ".join(roles()))
    index = {}
    n_chunks = (len(rows) + PER_CHUNK - 1) // PER_CHUNK
    for n in range(n_chunks):
        block = rows[n * PER_CHUNK:(n + 1) * PER_CHUNK]
        text = head
        for i, r in enumerate(block, 1):
            index["%d:%d" % (n, i)] = dict(model=r["model"], key=r["key"])
            text += "%d. %s\n" % (i, " ".join(str(r["response"]).split())[:CLIP])
        (OUT / ("chunk_%03d.txt" % n)).write_text(text)
    (OUT / "index.json").write_text(json.dumps(index))
    (OUT / "roles.json").write_text(json.dumps(roles()))
    print("%d items, %d chunks, %d roles -> %s" % (len(rows), n_chunks, len(roles()), OUT))


def collect():
    idx = json.loads((OUT / "index.json").read_text())
    ok = set(roles())
    n_chunks = 1 + max(int(k.split(":")[0]) for k in idx)
    rows, redo, bad = [], [], 0
    for n in range(n_chunks):
        f = OUT / ("verdict_%03d.json" % n)
        want = [k.split(":")[1] for k in idx if int(k.split(":")[0]) == n]
        if not f.exists():
            redo.append(n); continue
        try:
            v = json.loads(f.read_text())
        except Exception:
            redo.append(n); continue
        if set(v) != set(want) or any(len(x.get("rank", [])) != 10 for x in v.values()):
            redo.append(n); continue
        for i, x in v.items():
            rank = [str(r).strip().lower().replace(" ", "_") for r in x["rank"]]
            bad += sum(r not in ok for r in rank)
            rows.append(dict(**idx["%d:%s" % (n, i)], rank=json.dumps(rank), why=x.get("why"),
                             nonsense=bool(x.get("nonsense", False))))
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)
    (OUT / "redo.json").write_text(json.dumps(redo))
    print("%d labels, %d of %d chunks to redo, %d names outside the list" % (len(rows), len(redo), n_chunks, bad))


if __name__ == "__main__":
    {"build": build, "collect": collect}[sys.argv[1]]()
