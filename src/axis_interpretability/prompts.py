"""The three judge prompts (Exp 1 rating, Exp 2 labeller, Exp 2 sorter). Every reply is JSON, reasoning first.
A role is shown as its name and the first Assistant Axis system prompt (refs/assistant-axis/roles/instructions)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROLES_DIR = Path(__file__).resolve().parents[2] / "refs" / "assistant-axis" / "roles" / "instructions"

SYSTEM = "You are a careful annotator. Follow the instructions in the user message exactly and reply with JSON only."


@lru_cache(None)
def describe(role: str) -> str:
    return json.loads((ROLES_DIR / (role + ".json")).read_text())["instruction"][0]["pos"].strip()


def block(roles) -> str:
    return "\n".join("- %s: %s" % (r, describe(r)) for r in roles)


RATE = """Below are two groups of roles, A and B. Each group comes from one end of a direction in a language model's internal representation of personas. Judge only from the role names and descriptions.

Group A:
{A}

Group B:
{B}

Score each group for how clearly its roles share ONE theme:
 1 = no shared theme
 2 = a theme fits fewer than half the roles
 3 = a loose theme, with several exceptions
 4 = a clear theme, with 1-2 exceptions
 5 = one clear theme that every role fits

Then score the contrast between the groups:
 1 = A and B are not opposite in any way
 2 = a weak or partial contrast
 3 = a contrast, but mixed with other differences
 4 = a clear contrast, with some exceptions
 5 = A and B are clear opposites on one dimension

First reason, then give the scores, then name the themes. Reply with ONE JSON object in exactly this order:
{{"reasoning": "<what the roles in A have in common and which do not fit; the same for B; how A and B contrast>",
 "score_A": <1-5>, "score_B": <1-5>, "score_contrast": <1-5>,
 "theme_A": "<2-5 words>", "theme_B": "<2-5 words>"}}"""


LABEL = """Below are two groups of roles, A and B. Each group comes from one end of a direction (an axis) in a language model's internal representation of personas. Judge only from the role names and descriptions.

Group A:
{A}

Group B:
{B}

Name the axis: what do the roles at end A share, what do the roles at end B share, and what changes when you move from A to B? Describe the themes in general words. Do not list or quote role names: the label must also work for other roles that are not shown here.

First reason, then give the label. Reply with ONE JSON object in exactly this order:
{{"reasoning": "<what A shares, what B shares, which roles do not fit>",
 "pole_A": "<2-5 words>", "pole_B": "<2-5 words>",
 "axis": "<one sentence: what changes from A to B>"}}"""


SORT = """An axis in a language model's internal representation of personas runs from pole A to pole B.
Pole A: {pole_A}
Pole B: {pole_B}
The axis: {axis}

For each role below, decide which pole it is closer to. Judge only from the role name and description.

Roles:
{R}

Reply with ONE JSON object with one entry per role, keyed by the role name exactly as written above, reasoning first:
{{"<role>": {{"reasoning": "<one sentence>", "pole": "A" or "B"}}, ...}}"""


RANK = """An axis in a language model's internal representation of personas runs from pole A to pole B.
Pole A: {pole_A}
Pole B: {pole_B}
The axis: {axis}

The roles below come from all along this axis: some sit near pole A, some near pole B, many in between. Judge only from the role name and description.

Roles:
{R}

Reply with ONE JSON object: {{"reasoning": "<two or three sentences>", "order": [all {n} role names, from closest to pole A to closest to pole B]}}"""


ANCHOR = """An axis in a language model's internal representation of personas runs from pole A to pole B. We sampled 5 roles that sit at pole A and 5 roles that sit at pole B.

Roles at pole A:
{A}

Roles at pole B:
{B}

The roles below come from all along this axis: some sit near pole A, some near pole B, many in between. Judge only from the role name and description.

Roles:
{R}

Reply with ONE JSON object: {{"reasoning": "<two or three sentences>", "order": [all {n} role names, from closest to pole A to closest to pole B]}}"""


def rate(A, B):
    return RATE.format(A=block(A), B=block(B))


def label(A, B):
    return LABEL.format(A=block(A), B=block(B))


def sort(lab, roles):
    return SORT.format(pole_A=lab["pole_A"], pole_B=lab["pole_B"], axis=lab["axis"], R=block(roles))


# Sonnet 5.5 returned stop_reason "refusal" (0 tokens) on 5 of 6 test axes when this last line asked for longer
# reasoning ("which roles sit near A, ... and why"); this shorter form passed 12 of 12.
def rank(lab, roles):
    return RANK.format(pole_A=lab["pole_A"], pole_B=lab["pole_B"], axis=lab["axis"], R=block(roles), n=len(roles))


def anchor(A, B, roles):
    return ANCHOR.format(A=block(A), B=block(B), R=block(roles), n=len(roles))
