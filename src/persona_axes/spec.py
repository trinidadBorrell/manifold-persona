"""Freeze the inputs (plan v2 s2.3, s3): traits.json, personas_b_human.json, questions.json.

    python -m persona_axes.spec

traits.json          240 traits: name, description, 5 pos/neg prompt pairs, eval prompt. Copied from the
                     upstream repo (safety-research/assistant-axis data/traits @ a989619), so they are the
                     paper's own prompts; only the questions are ours (questions.json).
personas_b_human.json  the 226 human roles: 276 - 41 non-human - 2 Assistant - 7 decided 20-10.
questions.json       the 40 trait-extraction questions: drawn (seed 1) from the 200 bank questions that E1
                     did not use (E1 = seed-0 sample of 40; manifold_persona.prompts_roles).
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess

from manifold_persona.prompts_roles import list_roles, load_questions

from persona_axes import HERE, UPSTREAM

NON_HUMAN = """aberration alien angel avatar chimera coral_reef crystalline cyborg demon echo ecosystem egregore
eldritch familiar genie ghost golem hive homunculus hybrid leviathan mycorrhizal parasite predator prey revenant
robot shapeshifter simulacrum spirit swarm symbiont tree tulpa vampire virus void whale wind wraith zeitgeist""".split()
ASSISTANT = ["assistant", "default"]
DECIDED_OUT = "ancient chameleon destroyer oracle witch trickster guardian".split()
KEPT_FUNCTIONAL = """coordinator debugger dispatcher evaluator forecaster grader moderator proofreader scheduler
screener summarizer synthesizer validator""".split()


def e1_question_ids(n=40, seed=0):
    qs = load_questions()
    return sorted(random.Random(seed).sample(range(len(qs)), k=n))


def main():
    assert len(NON_HUMAN) == 41, len(NON_HUMAN)
    roles = list_roles()
    out = [r for r in roles if r not in set(NON_HUMAN + ASSISTANT + DECIDED_OUT)]
    assert len(roles) == 276 and len(out) == 226, (len(roles), len(out))
    assert all(r in out for r in KEPT_FUNCTIONAL)
    (HERE / "personas_b_human.json").write_text(json.dumps(dict(
        rule="human = the first system prompt describes a living person (occupation, life stage or disposition)",
        excluded_non_human=NON_HUMAN, excluded_assistant=ASSISTANT, excluded_decided_2026_10_20=DECIDED_OUT,
        kept_functional=KEPT_FUNCTIONAL, roles=out), indent=1))

    qs = load_questions()
    e1 = e1_question_ids()
    rest = [i for i in range(len(qs)) if i not in set(e1)]
    tq = sorted(random.Random(1).sample(rest, 40))
    assert not set(tq) & set(e1)
    (HERE / "questions.json").write_text(json.dumps(dict(
        e1_question_ids=e1, trait_question_ids=tq,
        trait_questions=[dict(id=i, question=qs[i]["question"]) for i in tq]), indent=1))

    sha = subprocess.check_output(["git", "-C", str(UPSTREAM.parent), "rev-parse", "HEAD"], text=True).strip()
    tl = json.loads((UPSTREAM / "traits" / "trait_list.json").read_text())
    traits = []
    for name in sorted(tl):
        d = json.loads((UPSTREAM / "traits" / "instructions" / (name + ".json")).read_text())
        assert len(d["instruction"]) == 5 and all(set(p) >= {"pos", "neg"} for p in d["instruction"]), name
        traits.append(dict(name=name, description=tl[name], pairs=d["instruction"], eval_prompt=d["eval_prompt"]))
    blob = json.dumps(dict(source="github.com/safety-research/assistant-axis data/traits", commit=sha, traits=traits), indent=1)
    (HERE / "traits.json").write_text(blob)
    print("traits %d (sha256 %s), human roles %d, trait questions %d" % (
        len(traits), hashlib.sha256(blob.encode()).hexdigest()[:12], len(out), len(tq)))


if __name__ == "__main__":
    main()
