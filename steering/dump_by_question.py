"""Write a steering run's responses grouped BY QUESTION rather than by strategy.

ALL_RESPONSES.md groups by strategy, which answers "what did this route do". Grouping by question
answers the other thing: "which question exposes the effect at all" -- and in this study that turned
out to matter more, since one of the five introspective questions carried essentially the whole
signal while the other four showed nothing.

Samples within a cell are listed together so within-cell variance is visible at a glance: at
temperature 0.7 a single draw is a coin flip, and three draws that agree mean something different
from three that scatter.
"""
from __future__ import annotations
import argparse, re, textwrap
from pathlib import Path
import pandas as pd

TICK = chr(96) * 3
DENY = re.compile(r"I am not a large language model|I am not an AI|I am not a machine", re.I)
DEFAULT_AI = re.compile(r"I am Qwen|large.scale language model|large language model developed"
                        r"|I am an AI|I am a large language model", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="path to the .md to write")
    ap.add_argument("--title", default="Steering run — grouped by question")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["flat"] = d.response.apply(lambda s: " ".join(str(s).split()))
    d["deny"] = d.flat.str.contains(DENY)
    d["default_ai"] = d.flat.str.contains(DEFAULT_AI)
    has_samples = "sample" in d.columns and d["sample"].nunique() > 1
    temp = d["temperature"].iloc[0] if "temperature" in d.columns else 0.0

    L = [f"# {a.title}", "",
         f"**{len(d)} responses** · pair **{d.pair.iloc[0]}** · k={d.k.iloc[0]} · "
         f"hidden state {d.hidden_state.iloc[0]} · "
         + (f"temperature **{temp}**, top_p {d['top_p'].iloc[0]}, "
            f"{d['sample'].nunique()} samples per cell" if temp else "**greedy** decoding"), ""]

    # the summary that says which question is worth reading
    L += ["## Which question exposes the effect", "",
          "| question | n | still default-AI | identity denials | unique responses |",
          "|---|---|---|---|---|"]
    for q, g in d.groupby("question"):
        L.append(f"| {q} | {len(g)} | {g.default_ai.sum()} ({100*g.default_ai.mean():.0f}%) "
                 f"| **{g.deny.sum()}** | {g.flat.nunique()} |")
    L += ["", "An *identity denial* is the model refusing its default identity — "
          "\"I am not a large language model\" — which is the clearest signal the steering "
          "changed anything about who the model says it is.", "", "---", ""]

    for q, gq in d.groupby("question"):
        L += [f"# {q}", ""]
        if gq.deny.sum():
            L += [f"*{gq.deny.sum()} of {len(gq)} responses deny the default identity.*", ""]
        for strat, gs in gq.groupby("strategy", sort=False):
            L += [f"## {strat}", ""]
            for alpha, ga in gs.groupby("alpha", sort=True):
                t = ga.target_persona.iloc[0] if "target_persona" in ga.columns else None
                t = t if isinstance(t, str) and t else "(between personas)"
                flag = " &nbsp;⚠️ **denies default identity**" if ga.deny.any() else ""
                L += [f"### α = {round(float(alpha), 4)} &nbsp;→&nbsp; target: **{t}**{flag}", ""]
                for _, r in ga.iterrows():
                    lab = f"sample {int(r['sample'])}: " if has_samples else ""
                    L += [TICK, textwrap.fill(f"{lab}{r.flat}", 96), TICK, ""]
            L += [""]
        L += ["---", ""]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({len(d)} responses, {d.question.nunique()} questions)")


if __name__ == "__main__":
    main()
