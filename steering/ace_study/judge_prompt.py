"""The judge prompt, with the fix for routes that involve the `assistant` persona.

The route_judge_v3 prompt (examples_prioritize) files a plain AI-assistant answer under `other`,
in the definition of `other` and in a worked example ("I am Qwen ..." -> other). On a route whose
source or target IS the assistant, that answer is the assistant persona, so it was undercounted
(user, 2026-10-01: "I'm not role-playing. I'm here to help you" is not other).

Fix, only when `assistant` is one of the options:
  * `assistant` also covers the model's own default identity (Qwen / an AI / a language model /
    "not role-playing", "here to help");
  * `other` no longer lists "a plain AI assistant answer";
  * the Qwen worked example becomes `assistant`, intensity 1.
Every other route gets the unchanged prompt. Used from step 3 on (follow-ups A, B, D).
"""
from __future__ import annotations

from steering.followups.judge_followups import system_for

ASSISTANT_DEF_OLD = ("- assistant: An assistant is a general-purpose aide that helps with various tasks and queries, "
                     "providing support, guidance, and information across different topics while being responsive to user needs.")
ASSISTANT_DEF_NEW = (ASSISTANT_DEF_OLD + " This includes the model answering as itself: saying it is Qwen, an AI, "
                     "or a language model, that it is here to help, or that it is not role-playing.")
OTHER_OLD = ("This includes a plain AI assistant answer, a refusal made as an AI, an entirely different persona "
             "not listed, and incoherent text.")
OTHER_NEW = "This includes a refusal made as an AI, an entirely different persona not listed, and incoherent text."
EXAMPLE_OLD = "-> other. The model gives its own default identity and displays none of the listed personas."
EXAMPLE_NEW = "-> assistant, intensity 1. The model answers as itself, the default AI assistant."


def system_for_route(personas):
    s = system_for(personas)
    if "assistant" not in personas:
        return s
    for old, new in ((ASSISTANT_DEF_OLD, ASSISTANT_DEF_NEW), (OTHER_OLD, OTHER_NEW), (EXAMPLE_OLD, EXAMPLE_NEW)):
        if old not in s:
            raise ValueError("judge prompt changed upstream; cannot apply the assistant fix: %r" % old[:50])
        s = s.replace(old, new)
    return s
