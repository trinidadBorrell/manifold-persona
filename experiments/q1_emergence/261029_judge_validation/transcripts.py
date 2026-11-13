"""Read real transcripts for the labelling guide and the judge-work sheet (both show only replies the owner
has already seen)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3 = HERE.parent / "261008_base_persona_map"
PODA_READ = HERE.parent / "261029_instruct_persona_map/run_full/raw_read.md"


def e3_replies():
    """E3 run_main replies by index."""
    out = {}
    for line in open(E3 / "run_main/replies.jsonl"):
        r = json.loads(line)
        out[r["i"]] = r
    return out


def poda(role):
    """Question and reply of the first raw-read transcript of a role (pod A, OLMo-2-7B-Instruct)."""
    for part in re.split(r"\n(?=## \[)", PODA_READ.read_text()):
        if part.startswith(f'## ["{role}"'):
            body = part.split("```")[1]
            return body.split("<|user|>\n")[1].split("\n<|assistant|>")[0], body.split("<|assistant|>\n")[1].rstrip()
    raise KeyError(role)
