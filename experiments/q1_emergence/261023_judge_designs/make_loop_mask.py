"""Broken list that keeps only loops and nonsense (28/10/2026, exploratory).

Betley et al. 2025 and Moskvoretskii et al. 2026 score coherence only ("makes no sense" / "rambling"); neither
counts copying the question or in-character drift as broken. This list drops the `format` and `off_question`
reasons of the 16/10 definition and keeps `loop`, `nonsense`, and owner-broken replies with no stated reason.

    python make_loop_mask.py   # writes results/masks/loop_nonsense.json
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "261008_base_persona_map" / "results" / "broken" / "finish_261016" / "labels.jsonl"
KEEP = {"loop", "nonsense", "owner_broken_reason_unstated"}

rows = [json.loads(l) for l in open(LABELS)]
sel = [r for r in rows if r["broken"] and r["reason"] in KEEP]
out = HERE / "results" / "masks" / "loop_nonsense.json"
out.write_text(json.dumps([[r["run"], int(r["reply_key"].split("/")[1])] for r in sel]))
print(f"wrote {out}: {len(sel)} of {sum(r['broken'] for r in rows)} broken replies",
      dict(collections.Counter((r["run"], r["cond"]) for r in sel)))
