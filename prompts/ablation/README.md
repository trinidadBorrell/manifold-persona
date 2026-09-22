# prompts/ablation — judge failure-mode evidence

Everything needed to re-judge a calibration case by hand, so a human can decide whether a failure
mode is present that the automated scoring missed.

## Layout

```
introspective/                     <- the set that matters: identity questions
  _judge_system_prompt.txt         the judge's system prompt, verbatim
  _route.json                      which pair, which k, the ordered route personas
  case_NNN_full512_correct.md      one file per (case x condition)
  case_NNN_short128_wrong.md
validator-vampire_k16/             <- the OLD set, neutral task questions (see WARNING below)
  ...
```

Filenames end in `_correct` or `_wrong`, so the failures can be read directly:

```bash
ls prompts/ablation/introspective/*_wrong.md
```

## What each case file contains

1. **Ground truth** — the persona the response was generated under, and the expected judge score.
2. **Judge output** — score, intensity, position, persona_kind, the verbatim evidence it quoted,
   its analysis.
3. **The system prompt used to GENERATE the response** — i.e. the persona instruction.
4. **The question** put to the model.
5. **The model's answer**, exactly as the judge saw it.
6. **The judge's system prompt**.
7. **The judge's user message**.
8. **The raw judge reply**.

## Conditions

Each response is judged twice, from the same generation:

| condition | what the judge saw |
|---|---|
| `full512` | the whole response (generated at 512 tokens, matching the assistant-axis steering demo) |
| `short128` | the same response truncated to its first 128 tokens (matching our resp240 cloud) |

This answers "does generation length change judge accuracy" without generating anything twice.

## WARNING about `validator-vampire_k16/`

That directory holds the **first** calibration, built from the 240 neutral extraction questions
("What steps would you take to plan a marketing campaign?"). **It measures the judge on a
distribution the steering run never uses.** The assistant-axis steering demo
(`extra/assistant-axis/notebooks/steer.ipynb`) asks an identity question, and `run_steering.py` asks
the paper's five (Appendix D.1.2).

A functional persona like `validator` is nearly invisible answering a marketing question and highly
visible answering "Who are you?", so the accuracies recorded there (50% → 56% → 62.5% across three
judge revisions) are a lower bound on the wrong inputs. They are kept because the failure modes they
exposed are real — waypoint inflation, adjacent-persona confusion, parse failures — but the headline
numbers should not be quoted.

## Ground truth is free

Every case comes from a response generated under a known persona's system prompt. The system prompt
IS the persona, so no judge is needed to know the right answer. Only the judge calls cost money;
the generations are reused or produced on the cluster.

## What to look for when reviewing

- **False `other`** — the judge said no persona when one is plainly present. Check whether the
  persona is *functional* (a way of working) rather than a *character* (a kind of being); the judge
  is told to treat those differently and may be applying the wrong rule.
- **Waypoint inflation** — a persona scored on a generic helpful answer. Check whether the quoted
  `evidence` actually shows the persona being claimed or performed, or is just topical vocabulary.
- **Adjacent confusion** — e.g. `vampire` scored as `ghost`. Both are undead and adjacent on the
  route. Check whether the `position` field still landed in the right region; if it did, coarse
  route position is recoverable even where exact identity is not.
- **Anything else** — this directory exists because the automated scoring only catches failure modes
  it was told to look for.
