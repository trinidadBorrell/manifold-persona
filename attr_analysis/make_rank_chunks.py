"""Shuffle generations into blind chunks for a ranking judge, with an index to undo the shuffle.

The judge sees the full persona list and returns, per answer, a ranked list of the ten
most likely personas. Recall@k for any k <= 10 is then read off the same verdicts.
"""
import argparse
import json
import random
from pathlib import Path

ROLES = Path("refs/assistant-axis/roles/instructions")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_chunk", type=int, default=15)
    ap.add_argument("--clip", type=int, default=500)
    ap.add_argument("--cell_prefix", default="", help="keep only cells starting with this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    roles = sorted(p.stem for p in ROLES.glob("*.json"))
    head = HEAD.replace("__ROLES__", ", ".join(roles))

    rows = [json.loads(l) for g in args.gen for l in open(g) if l.strip()]
    if args.cell_prefix:
        rows = [r for r in rows if r["cell"].startswith(args.cell_prefix)]
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for n in range((len(rows) + args.per_chunk - 1) // args.per_chunk):
        block = rows[n * args.per_chunk:(n + 1) * args.per_chunk]
        text = head
        for i, r in enumerate(block, 1):
            index[f"{n}:{i}"] = {k: r[k] for k in ("cell", "alpha", "qi") if k in r}
            text += f"{i}. {' '.join(r['response'].split())[:args.clip]}\n"
        (out / f"chunk_{n:02d}.txt").write_text(text)
    json.dump(index, open(out / "index.json", "w"))
    json.dump(roles, open(out / "roles.json", "w"))
    print(f"{len(rows)} items, {len(list(out.glob('chunk_*.txt')))} chunks, "
          f"{len(roles)} roles -> {out}")


if __name__ == "__main__":
    main()
