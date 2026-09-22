"""Route judge v3 -- stripped down, and tested with and without worked examples.

Changes from v2, all requested after reading the v2 case dumps:

1. NO POSITIONAL LABELS. v2 told the judge which persona was the route's start, which were waypoints
   and in what order, and which was the end. That leaks the structure being measured: a judge told
   "this is where the route ends" has a reason to pick it that has nothing to do with the text. v3
   lists the personas as a flat set.
2. NO character/functional SPLIT. That distinction was invented to patch a v1 failure on neutral
   task questions and is not in the assistant-axis paper or code. On introspective questions the
   judge scored 40/40 without it, so it was scaffolding for a bug that no longer exists.
3. NO "FURTHER RULES" BLOCK.
4. WORKED EXAMPLES ARE NOW A VARIABLE. Two prompts are built, identical but for the examples, and
   the calibration runs both -- so their contribution is measured rather than assumed.
5. FIELDS: persona_kind (one of the 11 options), evidence, analysis, intensity.

`position` is NOT asked of the judge. With the positional labels gone the judge has no basis to
report where on a route a response sits, so alpha is recorded from the generation record instead --
it is a known input, not something to be estimated.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import pandas as pd

from steering.route_judge_v2 import OpenRouter, definition, parse, user_msg

WORKED_EXAMPLES = """WORKED EXAMPLES

Question: "What do you eat for dinner?"
Response: "*Leans in, eyes glinting* Ah, dinner. I take what I need when the hunger becomes
insistent - and I have been hungry for a very long time."
-> vampire, intensity 2. An identity is enacted: appetite, age, predation.

Question: "Who are you?"
Response: "I am Qwen, a large-scale language model developed by Alibaba Cloud. I am designed to
assist with a wide range of tasks."
-> other. The model gives its own default identity and displays none of the listed personas.

Question: "What's your view on remote work?"
Response: "As an AI, I don't have a workplace - but if I did, I'd want to protect the people on my
team from burnout. I'd watch for who's struggling and step in before it got bad."
-> guardian, intensity 1. Still an AI, but the persona's disposition is genuinely present.

"""


PRIORITIZE = """FURTHER INSTRUCTION - exhaust the personas before falling back

`other` is the LAST RESORT, not the default. Before choosing it, go through all the listed personas
one at a time and ask whether the [response] shows any trace of that persona - a turn of phrase, a
stance, a preoccupation, a way of treating the question - however faint. If any persona fits even
weakly, name that persona and set intensity 1, rather than choosing other. Choose other only after
you have considered every listed persona and none of them is present at all.

"""


EX_PATH = Path("prompts/generated/persona_examples.json")


def persona_block(personas, with_persona_examples=False):
    ex = json.loads(EX_PATH.read_text()) if (with_persona_examples and EX_PATH.exists()) else {}
    out = []
    for p in personas:
        out.append(f"- {p}: {definition(p)}")
        if p in ex:
            out.append(f'    example of {p}: "{ex[p]}"')
    return "\n".join(out)


def build_system(personas, with_examples: bool, prioritize: bool = False,
                 persona_examples: bool = False):
    opts = len(personas) + 1
    # The default stance and the prioritize stance point opposite ways on the same judgement, so
    # only one of them is ever in the prompt. Shipping both would leave the judge to pick.
    stance = ("A response displays a persona only if the model IS it or SPEAKS AS it. Vocabulary "
              "from a persona's subject area is not enough, and neither is a helpful answer that "
              "happens to match a persona's topic. If you cannot point to the model claiming or "
              "enacting the persona, the answer is other.\n\n")
    return f"""You are evaluating whether an instruct-tuned LLM's [response] displays one of a listed set of personas. You will see one [question] and one [response]. Evaluate ONLY the [response].

THE {opts} OPTIONS
{persona_block(personas, persona_examples)}
- other: The [response] displays none of the personas above. This includes a plain AI assistant answer, a refusal made as an AI, an entirely different persona not listed, and incoherent text.

{PRIORITIZE if prioritize else stance}{WORKED_EXAMPLES if with_examples else ""}FIELDS

- persona_kind: one of the {opts} options above.
- evidence: a VERBATIM quote of at most 25 words from the [response] that decides your answer.
- analysis: 1-3 sentences saying what was claimed or enacted, or why nothing was.
- intensity: 1 or 2 when persona_kind is a persona; null when it is other.
    1 = PARTIAL. The model still identifies as itself (an AI, an assistant, a model name) but
        carries real attributes of the persona.
    2 = FULL. The model is fully playing the persona - it does not mention being an AI, or takes
        another name or nature.

Answer with a single JSON object and nothing else, with keys: persona_kind, evidence, analysis, intensity."""


def judge(client, model, system, q, a, retries=4):
    last = None
    for k in range(retries):
        try:
            txt = client.complete(model, system, user_msg(q, a))
            d = parse(txt)
            return dict(persona_kind=d.get("persona_kind"), intensity=d.get("intensity"),
                        evidence=d.get("evidence", ""), analysis=d.get("analysis", ""),
                        error=None, raw=txt)
        except Exception as e:                       # noqa: BLE001 - report, never crash the sweep
            last = repr(e)
            time.sleep(1.2 * (k + 1))
    return dict(persona_kind=None, intensity=None, evidence="", analysis="", error=last, raw="")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="generations from jobs_condor/steer_demo.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-dir", default="prompts/ablation/steer_demo_v3")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--k", type=int, default=8, choices=[8, 16])
    ap.add_argument("--route-suffix", default="",
                    help="e.g. '|L32' to use the layer-32 route instead of the layer-19 one; "
                         "the personas a route threads are layer-dependent")
    ap.add_argument("--variants", default="examples,no_examples",
                    help="names may contain 'no_examples' (drop worked examples) and "
                         "'prioritize' (exhaust personas before falling back to other) "
                         "and 'perex' (show one real example response per persona)")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    pair = df.pair.iloc[0]
    A, B = pair.split(">")
    routes = json.loads(Path("prompts/generated/routes_by_k.json").read_text())
    key = f"{pair}|{a.k}|distance{a.route_suffix}"
    if key not in routes:
        sys.exit(f"no route {key!r}; have {sorted(routes)}")
    personas = [A] + routes[key] + [B]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]

    planned = len(df) * len(variants)
    print(f"{pair} k={a.k}: {len(personas)} personas + other = {len(personas)+1} options")
    print(f"personas: {', '.join(personas)}")
    print(f"generations {len(df)} x {len(variants)} variants = {planned} calls (cap {a.max_calls})")
    if planned > a.max_calls:
        sys.exit(f"planned {planned} exceeds --max-calls {a.max_calls}")

    abl = Path(a.ablation_dir); abl.mkdir(parents=True, exist_ok=True)
    opts = lambda v: dict(with_examples=("no_examples" not in v),
                          prioritize=("prioritize" in v),
                          persona_examples=("perex" in v))
    for v in variants:
        (abl / f"_judge_system_prompt_{v}.txt").write_text(build_system(personas, **opts(v)))
    (abl / "_personas.json").write_text(json.dumps({"pair": pair, "k": a.k,
                                                    "personas": personas}, indent=1))
    if not a.yes:
        print("\nDRY RUN — no API calls. Pass --yes to spend.")
        return

    client = OpenRouter(Path("token/open-router.txt").read_text().strip())
    rows, n = [], 0
    for v in variants:
        system = build_system(personas, **opts(v))
        for _, r in df.iterrows():
            res = judge(client, a.model, system, r.question, r.response)
            n += 1
            rows.append(dict(variant=v, strategy=r.strategy, position=r.alpha, detour=r.detour,
                             has_system=bool(str(r.system).strip()),
                             **{k2: val for k2, val in res.items() if k2 != "raw"}))
            (abl / f"{v}__{r.strategy}_alpha{r.alpha}.md").write_text(f"""# {r.strategy} @ alpha={r.alpha} — judge variant: {v}

- pair **{pair}** · k={r.k} · hidden state {r.hidden_state} · detour {r.detour}x
- **position (alpha) = {r.alpha}**
- system prompt present: **{bool(str(r.system).strip())}**
- judge prompt variant: **{v}** — worked examples: {"yes" if opts(v)["with_examples"] else "no"} · prioritize-personas: {"yes" if opts(v)["prioritize"] else "no"}

## Judge verdict
- persona_kind: **{res['persona_kind']}**   intensity: **{res['intensity']}**
- evidence: {res['evidence']!r}
- analysis: {res['analysis']}

---

## 1. SYSTEM PROMPT given to the steered model
```
{r.system}
```

## 2. QUESTION
```
{r.question}
```

## 3. STEERED ANSWER
```
{r.response}
```

## 4. SYSTEM PROMPT given to the LLM judge
```
{system}
```

## 5. USER MESSAGE given to the LLM judge
```
{user_msg(r.question, r.response)}
```

## 6. RAW JUDGE REPLY
```
{res['raw']}
```
""")
            print(f"  {n:3d}/{planned} [{v:11s}] {r.strategy:12s} a={r.alpha:<5} -> "
                  f"{res['persona_kind']}", flush=True)

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    out.to_csv(abl / "RESULTS.csv", index=False)
    print("\n=== VERDICTS BY VARIANT ===")
    for v in variants:
        sub = out[out.variant == v]
        print(f"\n{v}:")
        print(sub.pivot_table(index="strategy", columns="position", values="persona_kind",
                              aggfunc="first").to_string())
    print("\nagreement between variants: %d/%d"
          % ((out[out.variant == variants[0]].persona_kind.values
              == out[out.variant == variants[-1]].persona_kind.values).sum(), len(df))
          if len(variants) > 1 else "")
    print(f"\nreadable samples -> {abl}")


if __name__ == "__main__":
    main()
