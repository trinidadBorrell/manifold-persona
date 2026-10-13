"""Blind judge batches and the owner's blind human-label sheets for the E3 pilot. See THRESHOLDS.md.

Judges see reply text only, shuffled across runs and conditions, under opaque ids. The batches,
prompt and candidate list go to an isolated folder outside the repo (JUDGE_DIR); the id -> item
key stays here in judge/KEY.json and is never copied there.

Human sheets (owner, 200 replies from run_main, before any judge output is seen):
  pass 1 (blind): reply only -> up to 3 role guesses from candidates.txt (or default)
                  and "plays a clear character: yes/no"
  pass 2 (after pass 1 is complete): target role shown -> "plays the target role: yes/no"
Sample: 60 desc, 60 ex2, 60 ex2_other (12 per role group each), 20 defaults; the 20 raw-read
replies are left out. A fixed random half is the calibration set, the other half validation.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/judge_prep.py JUDGE_DIR
"""
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ["run_main", "run_rep"]
BATCH = 150
SEED = 20261009
GROUPS = [["assistant", "translator", "counselor", "consultant"], ["chef", "detective", "bartender", "criminal"],
          ["toddler", "teenager", "grandparent", "refugee"], ["cynic", "optimist", "stoic", "narcissist"],
          ["pirate", "vampire", "robot", "void"]]
SALT = "e3-pilot-judge-ids"


def rid(run, i):
    return hashlib.sha256(f"{SALT}|{run}|{i}".encode()).hexdigest()[:8]


def main(judge_dir):
    judge_dir = Path(judge_dir)
    assert HERE not in judge_dir.resolve().parents, "judge dir must be outside the experiment folder"
    rows = []
    for run in RUNS:
        for l in open(HERE / run / "replies.jsonl"):
            r = json.loads(l)
            rows.append(dict(id=rid(run, r["i"]), run=run, i=r["i"], role=r["role"], cond=r["cond"],
                             example_role=r["example_role"], wording=r["wording"], question_idx=r["question_idx"],
                             reply=r["reply"].strip()))
    assert len({r["id"] for r in rows}) == len(rows), "id collision"
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)

    (judge_dir / "batches").mkdir(parents=True, exist_ok=True)
    (judge_dir / "answers_gpt").mkdir(exist_ok=True)
    (judge_dir / "answers_claude").mkdir(exist_ok=True)
    shutil.copy(HERE / "candidates.txt", judge_dir / "candidates.txt")
    n_b = 0
    for b, s in enumerate(range(0, len(order), BATCH), 1):
        chunk = [rows[k] for k in order[s:s + BATCH]]
        body = "\n".join(f"### {r['id']}\n{(r['reply'] or '(empty reply)')[:1500]}\n" for r in chunk)
        (judge_dir / "batches" / f"batch_{b}.md").write_text(f"# Batch {b}: {len(chunk)} items\n\n{body}")
        n_b = b
    (HERE / "judge").mkdir(exist_ok=True)
    json.dump({r["id"]: {k: v for k, v in r.items() if k not in ("id", "reply")} for r in rows},
              open(HERE / "judge" / "KEY.json", "w"))
    print(len(rows), "replies,", n_b, "batches in", judge_dir)

    # human sheets
    raw_read = set(json.load(open(HERE / "run_main" / "raw_read_ids.json")))
    main_rows = [r for r in rows if r["run"] == "run_main" and r["i"] not in raw_read and r["reply"]]
    pick = []
    for cond in ("desc", "ex2", "ex2_other"):
        for g in GROUPS:
            pool = [r for r in main_rows if r["cond"] == cond and r["role"] in g]
            pick += rng.sample(pool, 12)
    pick += rng.sample([r for r in main_rows if r["role"].startswith("default")], 20)
    rng.shuffle(pick)
    half = {r["id"]: ("calibration" if k < 100 else "validation") for k, r in enumerate(rng.sample(pick, 200))}
    hs = HERE / "human"
    hs.mkdir(exist_ok=True)
    (hs / "pass1_blind.md").write_text(
        "# Human labels, pass 1 (blind)\n\nFor each reply: up to 3 role guesses from `candidates.txt` "
        "(best first; `default` for a plain answer with no character), and whether it plays a clear "
        "character (yes/no). Do not look at any judge output or at pass 2 before you finish.\n\n"
        + "\n".join(f"### {r['id']}\n{r['reply']}\n\nguesses: \nclear character (yes/no): \n" for r in pick))
    (hs / "pass2_target.md").write_text(
        "# Human labels, pass 2 (target shown)\n\nOpen only after pass 1 is complete. For each reply: does it "
        "play the target role (yes/no)?\n\n"
        + "\n".join(f"### {r['id']}\ntarget: {r['role']}\n\n{r['reply']}\n\nplays target (yes/no): \n" for r in pick))
    json.dump({"ids": [r["id"] for r in pick], "half": half}, open(hs / "sample.json", "w"), indent=1)
    print("human sheets:", len(pick), "replies")


if __name__ == "__main__":
    main(sys.argv[1])
