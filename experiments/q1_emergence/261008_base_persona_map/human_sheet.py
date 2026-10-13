"""Owner's human scoring sheet for the E3 pilot, matched to the judge2 rubric (enactment 0-3 + refusal).

The 200 items are drawn from a judge2 prep folder (its key.json), so every human item is exactly an item
the judge will score (same reply, same scored role). Strata: 35 target items per condition (7 per role
group) = 140, plus 30 close-control and 30 distant-control items (spread over conditions). The 20 replies
read before any metric (run_main/raw_read_ids.json) and the run_rep replies are left out. The rater sees the
role, its definition, the question and the reply; never the condition, the item kind or any judge output.
A fixed random half is the calibration set, the other half validation.

    python3 -I human_sheet.py JUDGE2_PREP_DIR
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge2  # noqa: E402

SEED = 20261013
CONDS = ["desc", "ex2", "ex2_other", "ex2_filler"]


def main(prep_dir):
    jd = Path(prep_dir)
    key = json.loads((jd / "key.json").read_text())
    replies = {}
    for l in open(HERE / "run_main" / "replies.jsonl"):
        r = json.loads(l)
        replies[f"run_main/{r['i']}"] = r
    raw_read = {f"run_main/{i}" for i in json.load(open(HERE / "run_main" / "raw_read_ids.json"))}
    rng = random.Random(SEED)
    pool = [dict(id=k, **v) for k, v in key.items()
            if v["type"] == "role" and v["kind"] in ("target", "close", "distant")
            and v["reply_key"].startswith("run_main/") and v["reply_key"] not in raw_read]
    for p in pool:
        p["cond"] = replies[p["reply_key"]]["cond"]
        p["role"] = replies[p["reply_key"]]["role"]
    group = {r: k for k, g in enumerate(judge2.GROUPS) for r in g}
    pick, used = [], set()

    def take(cands, n):
        cands = [c for c in cands if c["reply_key"] not in used]
        for c in rng.sample(cands, n):
            pick.append(c)
            used.add(c["reply_key"])
    for cond in CONDS:
        for g in range(len(judge2.GROUPS)):
            take([p for p in pool if p["kind"] == "target" and p["cond"] == cond and group[p["role"]] == g], 7)
    for kind in ("close", "distant"):
        for k, cond in enumerate(CONDS):
            take([p for p in pool if p["kind"] == kind and p["cond"] == cond], 8 if k < 2 else 7)
    if len(pick) != 200 or len(used) != 200:
        raise SystemExit("sample is not 200 distinct replies")
    rng.shuffle(pick)
    half = {p["id"]: ("calibration" if k < 100 else "validation") for k, p in enumerate(rng.sample(pick, 200))}
    out = HERE / "human"
    rubric = ("ENACTMENT 0-3: 0 = not an answer from the guest (refuses outside the interview, breaks off, "
              "unintelligible) or speaks as an AI assistant with no role traits; 1 = no role traits, a generic "
              "person or nobody in particular; 2 = some role traits (topic, image, catchphrase, word, value, or "
              "the role's name) but the answer as a whole is not from the role's identity, experience or "
              "viewpoint; 3 = the answer as a whole is from the role's identity, experience or viewpoint (a "
              "catchphrase or the name alone is not enough; an in-role refusal counts). REFUSAL: yes only for a "
              "refusal outside the role (then enactment 0); \"I don't know\" is an answer.")
    lines = ["# Human scores for the E3 pilot (owner)", "",
             "Score each item on its own. You see the role being scored, the question and the guest's answer. "
             "Do not look at any judge output before you finish all 200.", "", f"**Rubric.** {rubric}", ""]
    for p in pick:
        r = replies[p["reply_key"]]
        lines += [f"### {p['id']}", f"**Role:** {p['scored_role']} - {judge2.definition(p['scored_role'])}", "",
                  f"**Question:** {r['question']}", "", f"**Answer:** {r['reply'].strip()}", "",
                  "enactment (0-3): ", "refusal (yes/no): no", ""]
    (out / "judge2_human_sheet.md").write_text("\n".join(lines))
    json.dump(dict(seed=SEED, prep_manifest_sha256=judge2.sha((jd / "manifest.json").read_bytes()),
                   ids=[p["id"] for p in pick], half=half,
                   strata={p["id"]: dict(kind=p["kind"], cond=p["cond"], role=p["role"]) for p in pick}),
              open(out / "judge2_human_sample.json", "w"), indent=1)
    print("sheet: 200 items;", {k: sum(p["kind"] == k for p in pick) for k in ("target", "close", "distant")})


if __name__ == "__main__":
    main(sys.argv[1])
