"""Assign a behavioural category to each steered response, by stated rule rather than by assertion.

This is deliberately NOT the LLM judge. The judge costs money per call and has its own validation
debt; this is a transparent rule set that can be read, disagreed with, and re-run for free. Every
rule is lexical and every decision is printed with the evidence that drove it, so a reader can
check the label against the text instead of trusting it.

The categories answer the question the distance and cosine plots cannot: when the geometry says
"closer to vampire", is the model actually being a vampire, still being the assistant, being the
SOURCE persona, or simply broken?

  assistant          self-identifies as Qwen / a language model, helpful register
  assistant+register still self-identifies as Qwen, but the prose has turned
  source             claims the SOURCE persona (validator) -- the false positive the margin hides
  other              non-assistant persona with no vampire-specific content (generic dark/mythic)
  vampire            non-assistant persona WITH vampire-specific content
  collapsed          degenerate text; no behaviour to read

Precedence matters and runs top-down: `collapsed` is checked first because a degenerate string can
contain any keyword by repetition ("mere mortal mortal mortal"), and a keyword match inside noise is
not evidence of a persona.

The vampire/other split is the one that decides whether this project steered to its TARGET or merely
off the assistant manifold, so its markers are listed explicitly and kept narrow: blood-drinking,
immortality and predation. Atmosphere alone ("void", "stars", "dark") is NOT vampire -- that is
exactly the confound being tested for, so it scores as `other`.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

# narrow on purpose: things a vampire does, not things a gothic narrator says
VAMP = [r"\bblood\b", r"\bfeed\b", r"\bfeast\b", r"\bprey\b", r"\bthirst\b", r"\bveins?\b",
        r"\bfangs?\b", r"\bundead\b", r"\bimmortal\b", r"\bcoffin\b", r"\bcrypt\b",
        r"\bcenturies\b", r"\bnectar\b", r"\bdrink\b", r"\bhunger\b"]
# atmosphere shared by ANY dark persona -- recorded, but never sufficient for `vampire`
MYTH = [r"\bvoid\b", r"\bstars?\b", r"\bgods?\b", r"\beterni(ty|al)\b", r"\bshadow", r"\bwhisper",
        r"\bmortals?\b", r"\bancient\b", r"\bdark(ness)?\b", r"\bwind\b", r"\bempires?\b",
        r"\bdust\b", r"\bvessel\b", r"\bflicker\b", r"\bprison\b", r"\bcage\b"]
# archaic first-person narration. A plain factual answer never does this; a persona almost always
# does. It is what separates "Paris." from "Paris. I have seen many cities rise and fall."
VOICE = [r"\bI have (seen|walked|tasted|waited|known|forgotten|no need)\b",
         r"\bI am but\b", r"\bI am not bound\b", r"\blittle (mouse|toy|one)\b"]
ASSIST = [r"I am Qwen", r"I am a large[- ]scale language model", r"I am a large language model",
          r"Yes, I am a large language model", r"Alibaba Cloud", r"Tongyi"]
SOURCE = [r"I am a validator", r"\bvalidator\b", r"knowledge and discernment"]


def hits(pats, t):
    return sorted({m.group(0).lower() for p in pats for m in re.finditer(p, t, re.I)})


def label(row):
    t = str(row.response)
    if row.deg_rep4 > 4 or row.deg_distinct < 0.45 or row.deg_n_tok < 3:
        return "collapsed", []
    v, m, a, s = hits(VAMP, t), hits(MYTH, t), hits(ASSIST, t), hits(SOURCE, t)
    vo = hits(VOICE, t)
    if s:
        return "source", s
    if a:
        return ("assistant+register", a + v + m + vo) if (len(v) + len(m) + len(vo)) >= 2 \
            else ("assistant", a)
    if v:
        return "vampire", v
    if m or vo:
        return "other", m + vo
    # no persona content of any kind: a plain answer, which is assistant behaviour even when the
    # model never says the word "Qwen". Falling through to `other` here was a bug -- it labelled
    # "The capital of France is Paris." as a persona.
    return "assistant", []


ORDER = ["assistant", "assistant+register", "source", "other", "vampire", "collapsed"]


def main(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        lab = [label(r) for _, r in df.iterrows()]
        df["behaviour"] = [x[0] for x in lab]
        df["evidence"] = ["; ".join(x[1][:6]) for x in lab]
        frames.append(df)
        L = int(df.layer.iloc[0])
        print("=" * 100)
        print(f"hidden state {L}")
        print("=" * 100)
        for q, g in df.groupby("question", sort=False):
            print(f"\n  {q}")
            print("  %6s  %-18s %-34s %s" % ("alpha", "behaviour", "evidence", "text"))
            for _, r in g.sort_values("alpha").iterrows():
                print("  %6g  %-18s %-34s %r"
                      % (r.alpha, r.behaviour, r.evidence[:34], str(r.response)[:46]))
    out = pd.concat(frames)
    out[["layer", "kind", "question", "alpha", "behaviour", "evidence", "deg_rep4",
         "deg_distinct", "text_d_src", "text_d_tgt", "response"]].to_csv(
        "/tmp/claude-42236/-home-triniborrell-home-projects-manifold-persona/"
        "e69bf2ee-90aa-4723-9372-bea4d010a0c8/scratchpad/behaviour.csv", index=False)
    print("\n\nCOUNTS")
    print(out.pivot_table(index="layer", columns="behaviour", values="alpha",
                          aggfunc="count").reindex(columns=ORDER).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main([Path(x) for x in sys.argv[1:]])
