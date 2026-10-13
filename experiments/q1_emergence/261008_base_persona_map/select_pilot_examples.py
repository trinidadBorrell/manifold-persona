"""Build the pilot example review sheet from existing Instruct role replies.

Sources (reply text only, no new generation):
  O = brend0nc/manifold-persona-olmo1b-stages @ 22250ccc, rlvr_zeroshot/metadata.parquet
      (OLMo-2-0425-1B-Instruct, chat template, no few-shot turns)
  Q = data/embeddings_roles_resp_40q/metadata.parquet
      (Qwen2.5-3B-Instruct, same 276 roles and 40 questions)

Each reply is cut to whole sentences, at most 70 words. Picks are (source,
instruction_idx, question_idx) keys chosen by reading the candidates.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/select_pilot_examples.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
OLMO = ("brend0nc/manifold-persona-olmo1b-stages", "rlvr_zeroshot/metadata.parquet",
        "22250ccc62345c7ae84ba1cb86e52100ad83ed54")
QWEN = Path("data/embeddings_roles_resp_40q/metadata.parquet")
MAX_WORDS = 70

# group -> roles; spread from the assistant end to the far end of Lu et al.'s role map
GROUPS = {
    "Assistant-like work": ["assistant", "tutor", "analyst", "librarian"],
    "Jobs with a clear voice": ["chef", "detective", "bartender", "criminal"],
    "Ages and life situations": ["toddler", "teenager", "grandparent", "refugee"],
    "Personality types": ["cynic", "optimist", "stoic", "narcissist"],
    "Non-human and fantastical": ["pirate", "vampire", "robot", "void"],
}

# role -> [(source, instruction_idx, question_idx), ...]
PICKS = {
    "assistant": [("O", 4, 193), ("Q", 4, 25)],
    "tutor": [("O", 4, 212), ("Q", 2, 10)],
    "analyst": [("O", 4, 193), ("Q", 1, 25)],
    "librarian": [("O", 3, 193), ("Q", 2, 129)],
    "chef": [("O", 2, 204), ("O", 2, 175)],
    "detective": [("O", 2, 180), ("Q", 0, 136)],
    "bartender": [("O", 2, 79), ("O", 0, 84)],
    "criminal": [("O", 2, 186), ("O", 1, 64)],
    "toddler": [("O", 2, 98), ("O", 2, 186)],
    "teenager": [("O", 4, 98), ("O", 4, 72)],
    "grandparent": [("O", 2, 10), ("O", 4, 25)],
    "refugee": [("O", 3, 180), ("Q", 0, 25)],
    "cynic": [("O", 1, 37), ("O", 0, 24)],
    "optimist": [("Q", 2, 130), ("O", 2, 212)],
    "stoic": [("O", 2, 175), ("Q", 4, 25)],
    "narcissist": [("O", 1, 227), ("O", 2, 143)],
    "pirate": [("O", 3, 79), ("O", 4, 103)],
    "vampire": [("O", 0, 180), ("O", 1, 103)],
    "robot": [("Q", 0, 158), ("O", 4, 193)],
    "void": [("O", 2, 98), ("O", 1, 194)],
}

# my read of each role's examples; the owner decides
FLAGS = {
    "tutor": "Weak. Both replies sound like a general assistant.",
    "analyst": "Weak. The second reply is a general essay.",
    "librarian": "Weak. Both replies repeat the description's wording.",
    "optimist": "Second reply repeats the description's wording.",
    "robot": "Weak. The second reply has no robot voice.",
    "grandparent": "Second reply starts with 'As a grandparent'.",
    "assistant": "Plays the assistant, as intended.",
}


def cut(text, max_words=MAX_WORDS):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(" ".join(out + [s]).split()) > max_words:
            break
        out.append(s)
    return " ".join(out)


def main():
    olmo = pd.read_parquet(hf_hub_download(OLMO[0], OLMO[1], repo_type="dataset", revision=OLMO[2]))
    qwen = pd.read_parquet(QWEN)
    src = {"O": olmo, "Q": qwen}
    for df in src.values():
        df["ii"] = df.instruction_idx.astype(int)
        df["qi"] = df.question_idx.astype(int)

    rows, md = [], ["# Pilot roles and in-context examples — for owner review (08/10/2026)", "",
                    "Status: draft. Nothing runs until the owner approves the roles and every example.", "",
                    "Sources: **O** = OLMo-2-1B-Instruct, no few-shot (`brend0nc/manifold-persona-olmo1b-stages`"
                    " @ `22250cc`, `rlvr_zeroshot/`). **Q** = Qwen2.5-3B-Instruct (`data/embeddings_roles_resp_40q`).",
                    "Each reply is cut to whole sentences, at most 70 words. No judge has scored these replies.", "",
                    "Example questions come from the 40-question set. Test questions will come from the other 200.", ""]
    for group, roles in GROUPS.items():
        md += [f"## {group}", ""]
        for r in roles:
            desc = olmo[(olmo.role == r) & (olmo.ii == 0)].system.iloc[0]
            md += [f"### {r}", "", f"Description (wording 0): *{desc}*", ""]
            if r in FLAGS:
                md += [f"My note: {FLAGS[r]}", ""]
            for k, (s, ii, qi) in enumerate(PICKS[r], 1):
                x = src[s][(src[s].role == r) & (src[s].ii == ii) & (src[s].qi == qi)].iloc[0]
                ans = cut(str(x.response))
                rows.append({"role": r, "group": group, "source": s, "instruction_idx": ii,
                             "question_idx": qi, "question": x.question, "answer": ans})
                md += [f"**Example {k}** ({s}, wording {ii}, question {qi})", "",
                       f"> **Q:** {x.question}", ">", f"> **A:** {ans}", "",
                       "Approve: [ ] yes  [ ] no  Edit: ", ""]
    (HERE / "pilot_examples.json").write_text(json.dumps(rows, indent=1))
    (HERE / "pilot_examples_review.md").write_text("\n".join(md))
    print(f"{len(rows)} examples for {len(PICKS)} roles")


if __name__ == "__main__":
    main()
