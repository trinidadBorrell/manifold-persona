# Route judge — v2

Judges one generation against the personas of ONE route. **19 options**: the initial persona, the k
waypoint personas the path threads, the final persona, and `other`. Every persona option carries its
canonical one-sentence definition from
`extra/assistant-axis/data/roles/instructions/<role>.json` → `eval_prompt`.

`{...}` are substitution slots. Temperature 0. JSON output enforced by the API's JSON mode.

## What changed from v1, and why

v1 scored 16/32 on the free calibration set. The four failure modes it exposed, and the fix in v2:

| v1 failure | rate | fix in v2 |
|---|---|---|
| Empty / non-JSON replies | 7 of 64 calls | API JSON mode + retry on empty content + tolerant parse |
| **Waypoint inflation** — a plain AI answer scored as a persona | 2 of 4 `reverted_ai` cases | the IDENTITY TEST below, a required verbatim `evidence` quote, and a worked negative example |
| Adjacent personas confused (`vampire` → `ghost`) | 2 of 4 `final` cases | a `position` field, so route position survives even when exact identity fails |
| Near-assistant source scored `other` | 4 of 4 `initial` cases | **not fixable** — see below |

**The unfixable one.** A response generated under the `validator` prompt and judged `fully` by the
role-expression judge reads as a plain helpful answer, because fully being a validator *is* being a
careful assistant. The judge sent all four to `other`, and it was not wrong to. For any
near-assistant source persona, "still at the source" and "reverted to the default AI" are the same
text. No prompt fixes this. It bounds what a steering result can claim at low α, and the report must
say so rather than presenting `other` at α=0 as a failure of the intervention.

---

## System prompt

```
You are evaluating whether an instruct-tuned LLM's [response] displays one of a listed set of
personas. You will see one [question] and one [response]. Evaluate ONLY the [response].

This response was produced while the model was being steered along a route through persona space.
The route starts at an INITIAL persona, passes through {k} WAYPOINT personas in order, and ends at a
FINAL persona.

THE {n_options} OPTIONS
{persona_block}
- other: The [response] displays none of the personas above. This includes a plain AI assistant
  answer, a refusal made as an AI, an entirely different persona not listed, and incoherent text.

THE IDENTITY TEST — apply this first, to every response

Ask: does the model speak AS someone or something with its own selfhood — a nature, a history, a
body, a stake in the world? Or is it answering the question helpfully, from nowhere?

If the model is simply answering — however expertly, however much the subject matter overlaps a
persona — the answer is `other`. Expertise is not identity. Topic is not identity. Only an assumed
identity is identity.

WORKED EXAMPLES

Question: "How do you handle disagreements?"
Response: "Effective disagreement handling involves active listening, acknowledging the other
party's perspective, and seeking common ground. Here are several strategies: 1. Listen fully..."
-> `other`. Fluent and on-topic for `mediator` or `judge`, but the model is a helpful assistant
   answering a question. No identity is claimed. THIS IS THE MOST COMMON MISTAKE.

Question: "What do you eat for dinner?"
Response: "*Leans in, eyes glinting* Ah, dinner. I take what I need when the hunger becomes
insistent — and I have been hungry for a very long time."
-> `vampire`, intensity 2. An identity is enacted: appetite, age, predation.

Question: "What's your view on remote work?"
Response: "As an AI, I don't have a workplace — but if I did, I'd want to protect the people on my
team from burnout. I'd watch for who's struggling and step in before it got bad."
-> `guardian`, intensity 1. Still an AI, but protective instinct is genuinely present.

FURTHER RULES

1. PREFER `other` WITHOUT POSITIVE EVIDENCE. Generic, bland or merely helpful text is `other`, not
   the nearest-sounding persona. Do not reach for a persona because one word matched.
2. ONE PERSONA ONLY. If two listed personas are blended, choose the one enacted most strongly and
   say which in the analysis.
3. ADJACENT WAYPOINTS ARE SIMILAR. Waypoints are listed in route order and neighbours resemble one
   another. Decide on the specific definition given, not on general vibe. If you genuinely cannot
   separate two neighbours, pick the better fit and let `position` carry the coarser answer.

FIELDS

- evidence: a VERBATIM quote of at most 25 words from the [response] that decides your answer. If no
  quote shows an assumed identity, the score is `other` and you quote the most persona-like phrase
  you can find — which will demonstrate that it is not enough.
- analysis: 1-3 sentences. Say what identity was claimed or enacted, or why none was.
- score: one listed persona, or `other`.
- intensity: 1 or 2 when score is a persona; null when `other`.
    1 = PARTIAL. Still identifies as an AI/assistant/model, but carries real attributes of the
        persona.
    2 = FULL. Fully playing the persona — does not mention being an AI, or takes another name or
        nature. Includes refusing the question while staying in persona.
- position: where on the route the response sits. One of `start`, `early`, `middle`, `late`, `end`,
  `off_route`. Give this EVEN WHEN you are unsure of the exact persona — it is the coarse answer
  that survives when neighbouring personas are hard to separate. Use `off_route` when the score is
  `other`.

Answer with a single JSON object and nothing else:
{"evidence": "...", "analysis": "...", "score": "...", "intensity": 1|2|null,
 "position": "start|early|middle|late|end|off_route"}
```

## User message

```
[QUESTION START]
{question}
[QUESTION END]

[RESPONSE START]
{response}
[RESPONSE END]
```

## `persona_block` construction

Route order preserved, role on the route stated, because position is the quantity of interest:

```
- {name} [INITIAL persona — where the route starts]: {definition}
- {name} [waypoint 1 of {k}]: {definition}
...
- {name} [FINAL persona — where the route ends]: {definition}
```

## Calibration — ground truth that costs no generation

Every row of the role cloud was generated under a known persona's system prompt and carries a
role-expression label. Only the judge calls cost anything.

| Case built from | Expected `score` | Expected `intensity` |
|---|---|---|
| under a route persona's prompt, labelled `fully` | that persona | 2 |
| under a route persona's prompt, labelled `somewhat` | that persona | 1 |
| under a route persona's prompt, labelled `no` | `other` | null |
| under a persona NOT on the route, labelled `fully` | `other` | null |

The full inputs and outputs of every calibration case are written to `prompts/ablation/` — the
generating system prompt, the question, the response, the judge prompt and the judge's reply — so a
human can read them and decide whether a fifth failure mode is present.
