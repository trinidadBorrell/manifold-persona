"""Phase 0 check of personas_b_human.json: three Haiku 5.5 votes per role on its first system prompt
("does this describe a living person?"), majority, compared with the frozen list. Disagreements are
reported, not applied.

    python -m persona_axes.human_vote [--dry-run]     # -> $PM/AX21/human_vote/votes.jsonl, disagreements.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from manifold_persona.config import ROLE_INSTRUCTIONS_DIR

from persona_axes import AX, HERE
from persona_axes.api import HAIKU, Runner

PROMPT = """A language model will be given the system prompt below and asked questions. Does the system prompt describe a living HUMAN PERSON (a person with an occupation, a stage of life or a disposition; fictional humans count)? Answer no for animals, machines, AIs, spirits, monsters, abstract entities, collectives, places and objects.

Role name: {name}
System prompt: "{system}"

Reply with one JSON object: {{"human": true or false, "why": "one sentence"}}"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    D = AX / "human_vote"
    D.mkdir(exist_ok=True)
    spec = json.loads((HERE / "personas_b_human.json").read_text())
    roles = sorted(p.stem for p in ROLE_INSTRUCTIONS_DIR.glob("*.json"))
    items = []
    for r in roles:
        sysp = json.loads((ROLE_INSTRUCTIONS_DIR / (r + ".json")).read_text())["instruction"][0]["pos"]
        for k in range(3):
            items.append((r, k, PROMPT.format(name=r, system=sysp.replace("{model_name}", "the model"))))
    R = Runner(HAIKU, workers=12)
    print("%d calls, estimate $%.2f" % (len(items), R.estimate([i[2] for i in items], 60)))
    if a.dry_run:
        print(items[0][2]); return
    rows = []
    lock = __import__("threading").Lock()

    def one(it):
        r, k, pr = it
        v, raw, stop = R.call(pr, max_tokens=120, effort="low", ok=lambda v: isinstance(v.get("human"), bool))
        with lock:
            rows.append(dict(role=r, vote=k, human=None if v is None else v["human"], why=None if v is None else v.get("why")))
    R.map(one, items)
    df = pd.DataFrame(rows)
    df.to_json(D / "votes.jsonl", orient="records", lines=True)
    maj = df.groupby("role").human.apply(lambda s: bool(s.sum() >= 2))
    ours = {r: (r in spec["roles"]) for r in roles}
    dis = pd.DataFrame([dict(role=r, our_list=ours[r], haiku_majority=bool(maj[r]),
                             votes=int(df[df.role == r].human.sum())) for r in roles if ours[r] != bool(maj[r])])
    dis.to_csv(D / "disagreements.csv", index=False)
    print("votes done, $%.2f; %d disagreements:\n%s" % (R.spent, len(dis), dis.to_string(index=False) if len(dis) else "none"))


if __name__ == "__main__":
    main()
