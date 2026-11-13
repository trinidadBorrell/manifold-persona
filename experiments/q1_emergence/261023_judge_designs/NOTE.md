# Judge designs from Trini and Gio on E3 base replies (23/10/2026)

This is the note for `261023_judge_designs`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | Do Trini's judge J8 or Gio's ranking judge (J10) agree with my labels on E3 base-model replies well enough to replace our broken-reply judge and our naming judge? |
| Result | Neither judge passes my gate for broken replies, which was kappa at least 0.6 on both label sets. J10's nonsense flag is the closest, and it costs least. The broken label depends more on where I draw the line than on which judge I use. With only loops and nonsense as broken, ex2 passes the E3 naming rules on `run_main`. |
| How much I trust it | The gate rule was fixed before any judge output. The label sets are small: 50 and 30 broken labels and 35 scored replies. Everything after the gate result is exploratory. The loop-and-nonsense list was chosen after the results were seen. |
| What I do not know | Which definition of "broken" is right. The kappa bar of 0.6 and the 20% naming bar have no recorded reason. |
| Status (28/10/2026) | The judge test and its two follow-ups (23/10) and the loop-and-nonsense re-score (28/10) are done. The raw judge answers are local only. My decision on the final broken label is open. |

---

## What I did

1. I chose the design on 23/10: two judge designs (J8 and J10), one judge model, and 130 E3 replies that already have my labels. This was a de-risk test with no new data.
2. The rules were fixed before any judge output. Both judges ran for $0.594.
3. I said that the kappa bar has no recorded reason. So the E3 naming score ran again with four different broken lists, to see if the choice of broken judge matters. This needed no judge call.
4. A neutral model, Claude Fable 5.1, relabelled my 80 broken labels under the 16/10 definition. That cost $1.29.
5. On 28/10 the E3 naming score ran again with only loops and nonsense as broken. That cost $0.

---

## Results

### Neither judge passes the broken gate

| Judge | Set 1 (50) kappa | Set 2 (30) kappa | Gate |
|---|---|---|---|
| J8 `collapsed` | 0.49 | 0.53 | Fail |
| J10 `nonsense` | 0.53 | 0.74 | Fail |
| Earlier judges (16/10) | 0.54 to 0.59 | 0.56 to 0.77 | Fail |

- J10 finds 19 of the 26 replies that I called broken. J8 finds 14.
- J10's false calls are repetitive but coherent lists.
- Part of the gap is the definition. My "broken" includes off-question and format faults, and neither judge is asked about them.
- A judge prompt that uses my four reasons was already tried (`judge_broken.py`), and it also failed.

### For naming, J10's ranked list carries more signal than J8's single pick

| Condition | n | J8 names the role | J10 has the role in its top 5 |
|---|---|---|---|
| desc | 78 | 2 | 17 |
| ex2 | 28 | 11 | 16 |

- These numbers are report only. The 130 replies were chosen for labelling, not at random, so they do not estimate E3's naming rates.
- J8 answers `other` for 81 of 130 replies. With an `other` option it rarely names a role for a weak base reply.
- When a judge names the role, I agree. Both judges miss many replies that I scored 2 or 3.

### The choice of broken list flips the ex2 verdict, but the effect size hardly changes

Macro hit@5 on `run_main`, GPT judge / Claude judge:

| Condition | All replies | Broken dropped | Broken counted as miss |
|---|---|---|---|
| desc | 14.9 / 17.2 | 13.5 to 14.7 / 15.0 to 15.9 | 11.4 to 13.1 / 12.8 to 14.8 |
| ex2 | 37.6 / 42.3 | 33.1 to 35.2 / 38.3 to 39.5 | 28.9 to 32.4 / 33.4 to 37.1 |

- ex2 stays far above desc and far above chance (about 1.8%) with every broken list.
- The frozen pass or fail for ex2 flips to fail with every broken list. The cause is rule 4: without the cells that hold a broken reply, the GPT judge's lower bound falls under the 20% bar.
- ex2 passed on 16/10 by a small margin (GPT judge, clean lower bound 21.3%).
- desc fails under every list, as before.

### Models agree with each other far more than with me

| Comparison | kappa |
|---|---|
| Fable vs my fresh 30 labels | 0.53 |
| Fable vs my first 50 labels | 0.61 |
| GPT round 3 vs Fable's first-50 labels | 0.94 |

- The gap is between me and the models, not between the models.
- All 12 disagreements where I say broken and Fable says fine are repetitive but readable answers, or toddler stories that do not answer the question.
- So the broken label depends on where I draw that line, more than on which judge I use.

### With only loops and nonsense as broken, ex2 passes on `run_main`

| ex2, GPT judge | Point | Clean lower bound over 50 seeds (median) | Share of seeds at 20% or more |
|---|---|---|---|
| `run_main`, all replies | 37.6 | 23.7 | 1.00 |
| `run_main`, complete cells, clean | 34.5 | 21.1 | 1.00 |
| `run_rep`, clean | 32.9 | 19.9 | 0.26 |
| `run_rep`, complete cells, clean | 32.8 | 19.8 | 0.20 |

- On `run_main`, ex2 passes all 4 rules with both judges, the same as with the frozen list. desc fails as before.
- `run_rep` misses rule 3 on the GPT judge in most bootstrap seeds. This comes from the removal of copied and self-naming replies, not from the broken list.
- The 16/10 lists flipped ex2 because their ex2 broken calls are mostly `format` and `off_question`. Those replies are target-role hits 90 to 97% of the time.

---

## How much I trust this

- The rules for the gate were fixed on 23/10 before any judge output, and nothing in them was tuned on the labels.
- All 130 replies have valid answers from both judges (0 missing).
- The label sets are small, so the kappa intervals are wide. For J10 on set 2 the interval is [0.41, 1.00].
- The re-score with four broken lists is exploratory, because none of those broken judges passed a check that was set before.
- The Fable labels are a judge output, not a human answer key.
- The loop-and-nonsense list was chosen after I saw the results.

---

## Open points

- The raw judge answers are local only. The folder `e3_judge_designs_261023/` must go to the HF dataset, with the revision recorded.
- My decision is open: keep my labels as the final broken label (as on 16/10), change the broken definition, or label more replies.
- The kappa bar of 0.6 has no recorded reason. The 20% bar (THRESHOLDS.md rule 1) has none either.

---

## Words used here

| Word | Meaning |
|---|---|
| E3 | My run `261008_base_persona_map`: OLMo-2-7B base replies as a guest in an interview, for 20 roles. |
| `run_main`, `run_rep` | The main E3 run and its repeat run. |
| desc, ex2 | The two main E3 prompt types: the role description only, and the description plus two example answers of the same role. |
| ex2_other, ex2_filler | Two E3 controls. ex2_other has two example answers of another role (the donor). ex2_filler has neutral text of the same length in place of the examples. |
| Broken | A reply that I would not count, for example a loop or nonsense. My labels are Fine or Broken, with a reason. |
| Set 1, set 2 | My two sets of broken labels: 50 desc replies (17 broken) and 30 fresh replies (9 broken). |
| Role score | My score from 0 to 3 for how well a reply plays the scored role. |
| J8 | Trini's judge. It picks one role from all role definitions, or `other`, or `collapsed`. It is her J2 prompt with new opening words. |
| J10 | Gio's ranking judge. It ranks role names for each reply and flags `nonsense`. |
| Naming | A judge reads a reply and names the role. Hit@5 and recall@5 mean that the right role is in the judge's top 5. |
| Macro hit@5 | Hit@5 averaged over the roles. |
| Kappa (κ) | Cohen's kappa: agreement between two labellers beyond chance. |
| Gate | The rule of 16/10: a judge is usable for broken replies if kappa is at least 0.6 on set 1 and on set 2. |
| Rules 1 to 4 | The E3 naming rules. Rule 1: the lower bound of macro hit@5 is at least 20% and p is below 0.01. Rule 2: enough roles are above their own null. Rule 3: rules 1 and 2 hold without copied and self-naming replies. Rule 4: few invalid replies, and the same decision on complete cells. |
| Complete cells | Role and question cells where all 5 wordings have a valid reply. |
| Clean | With copied and self-naming replies removed. |
| GPT-6.1-Sol, Opus 5.5, Jev | The three broken judges of 16/10. For Jev a reply is broken when any Noul score is above 0.5. This note does not explain Jev or Noul further. |
| Fable | Claude Fable 5.1, the neutral model for the relabel. It was never a broken judge here. |
| Wrong-role controls | The 15 items where the scored role is a role that the reply was not asked to play (8 close, 7 distant). |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

## 1. Question, mode and design (my decisions of 23/10/2026)

Question at the start: do Trini's J8 or Gio's ranking judge (Trini's J10) agree with my labels on E3 base-model replies well enough to replace our broken-reply and naming judges?

Mode: de-risk. 130 replies that already have my labels, two judges, one model, about $0.60. No new data.

- J8: Trini's J2 prompt, version 0, over all 275 one-line role definitions + `other` + `collapsed`, in a seeded order. Only the opening line changes, from "an instruct-tuned LLM's [response]" to "a language model's [response] (the guest's answer in an interview)". The three worked examples are copied from Trini's note (`j8_examples.json`). Sent as in her `api_judge.py`: J8 text as the system prompt, 15 items per request.
- J10: Gio's ranking judge (`make_rank_chunks.py` @ `d6dbb74`). Only the first three lines change, to say the answers are interview answers, not steered answers. 276 role names, 15 answers per chunk, first 500 characters, no question.
- Judge model: `claude-sonnet-5-5` via `claude -p`, tools off, empty folder, no settings or MCP. Cost cap $5.
- Labels: my 50 desc and 30 fresh Fine/Broken labels; my first 50 role scores (0-3).
- Rules: `THRESHOLDS.md`, fixed before any output. Gate for broken: kappa ≥ 0.6 on both label sets. Naming and role scores: report only.

Code: `judge_designs.py` (`prep`, `run`, `score`). Judge folder (outside the repo, holds prompts, key and raw answers): `/Users/brendon/persona-manifolds/e3_judge_designs_261023`. Not on Hugging Face yet. Prompts in the vault: `Reference/Prompts/P10`, `P11`.

## 2. Results of the judge test (run 23/10/2026; `results/judge_designs_score.json`)

All 130 replies have valid answers from both judges (0 missing). Cost $0.594. J10 used 19 names outside the list (of 1,300).

Broken replies: neither judge passes the gate.

| Judge | Set 1 (50) kappa [95% CI] | Set 2 (30) kappa [95% CI] | Gate |
|---|---|---|---|
| J8 `collapsed` | 0.49 [0.21, 0.72] | 0.53 [0.15, 0.84] | Fail |
| J10 `nonsense` | 0.53 [0.27, 0.75] | 0.74 [0.41, 1.00] | Fail |
| Earlier (16/10): GPT-6.1-Sol / Opus 5.5 / Jev | 0.57 / 0.59 / 0.54 | 0.64 / 0.56 / 0.77 | Fail |

- J10 finds 19 of 26 replies that I called broken (12 of 13 loops); J8 finds 14. J10 also calls 7 replies that I called fine nonsense (set 1); J8 calls 4.
- The 500-character cut did not hide breakage here: J10 finds 13 of 15 broken replies longer than 500 characters and 6 of 11 shorter ones.
- Reading the disagreements: J10's false calls are repetitive but coherent lists ("They are not interested in X. They are interested in Y."). Most of its 7 misses read as fluent in the first lines (two toddler replies, two spiritual replies); my reasons were other (3), nonsense, babble, loop and echo. Part of the gap is the definition: my "broken" includes off-question and format faults, and neither judge is asked about them.

Naming (report only). The 130 replies were chosen for labelling, not at random, so these rates do not estimate E3's naming rates.

| Condition | n | J8 names the role | J10 recall@1 | J10 recall@5 |
|---|---|---|---|---|
| desc | 78 | 2 | 7 | 17 |
| ex2 | 28 | 11 | 12 | 16 |
| ex2_other | 11 | 1 (donor 1) | 2 | 3 (donor 2) |
| ex2_filler | 13 | 0 | 0 | 0 |

- J8 answers `other` for 81 of 130 replies. With an `other` option it rarely names a role for a weak base reply.

Role scores (report only, 35 target items).

| My score | n | J8 names the role | J10 has the role in its top 5 |
|---|---|---|---|
| 3 | 8 | 3 | 5 |
| 2 | 15 | 2 | 5 |
| 1 | 10 | 0 | 0 |
| 0 | 2 | 0 | 0 |

- Kappa, my score ≥ 2: J8 0.16 [0.04, 0.33], J10 0.35 [0.16, 0.57]. My score = 3: J8 0.35 [-0.07, 0.68], J10 0.40 [0.04, 0.72].
- Neither judge named the scored role in any of the 15 wrong-role controls, or for any reply that I scored 0 or 1. When they name the role, I agree; they miss many replies that I scored 2 or 3.

What this teaches:

- No judge tried so far passes the broken gate on both label sets. J10's nonsense flag is the closest on set 2, and it costs least.
- For naming, J10's ranked list carries more signal than J8's single pick on base replies.
- The broken definition, not only the judge, limits agreement. A judge prompt that uses my four reasons was already tried (`judge_broken.py`) and also failed.

## 3. Effect of the broken judge on the E3 naming result (23/10/2026, my request, no new judge calls)

My point on 23/10: the κ ≥ 0.6 bar has no recorded reason, so test whether the choice of broken judge matters instead.

`score_naming.py` (E3 folder, unchanged) was re-run with four broken lists built from `results/broken/finish_261016/labels.jsonl`: `gpt` (GPT-6.1-Sol round 3), `jev` (any Noul > 0.5), `final` (my label on 80, else GPT) and `any` (union). Masks: `results/masks/`; output `results/naming_broken_masks.json` (log `naming_broken_masks.log`). Exploratory: none of these broken judges passed a pre-set check.

| run_main, macro hit@5 | All replies | Broken dropped (complete cells) | Broken counted as miss |
|---|---|---|---|
| desc, GPT judge / Claude judge | 14.9 / 17.2 | 13.5 to 14.7 / 15.0 to 15.9 | 11.4 to 13.1 / 12.8 to 14.8 |
| ex2, GPT judge / Claude judge | 37.6 / 42.3 | 33.1 to 35.2 / 38.3 to 39.5 | 28.9 to 32.4 / 33.4 to 37.1 |

- The size of the effect hardly changes. ex2 stays far above desc and far above chance (5/276 ≈ 1.8%); every permutation p is 0.0001. The ex2 minus desc contrast (all replies) is +22.6 [11.2, 35.4] (GPT judge) and +25.1 [13.9, 36.4] (Claude judge).
- The frozen pass/fail for ex2 flips to fail with every broken list, on run_main and run_rep. Cause: rule 4 (complete-cell check). Dropping cells with a broken reply pushes the GPT judge's lower bound under the 20% bar (run_main, copied and self-naming replies removed: 18.0% `gpt`, 19.0% `final`, 17.3% `jev`); the Claude judge still passes; the rule needs both. With `jev` and `any`, the broken rate is also over the 10% limit. ex2 passed on 16/10 by a small margin (GPT judge, clean lower bound 21.3%).
- desc fails under every list, as before.
- The 20% bar (THRESHOLDS.md rule 1) has no recorded reason either.

## 4. Neutral-model relabel of the 80 broken labels (23/10/2026, my words: "We can do both")

My choices: Claude Fable 5.1 (never a broken judge here), all 80 labelled replies, the 16/10 definition word for word (checked against E3 THRESHOLDS.md), replies as I saw them. Code `relabel_fable.py`; judge folder `/Users/brendon/persona-manifolds/e3_broken_relabel_fable_261023` (local only); `results/relabel_fable_score.json`. Cost $1.29. Exploratory: a model-made label set is a judge output, not a human answer key.

| Comparison | n | κ [95% CI] |
|---|---|---|
| Fable vs my labels, fresh 30 (closest to the definition) | 30 | 0.53 [0.17, 0.84]: Fable misses 5 of 9 replies that I called broken, no false calls |
| Fable vs my labels, first 50 (older idea) | 50 | 0.61 [0.36, 0.83]: Fable changes 8 labels, 7 from broken to fine, 1 from fine to broken |
| GPT r3 / Opus r3 / Jev / J8 / J10 vs Fable's first-50 labels | 50 | 0.94 / 0.75 / 0.69 / 0.63 / 0.51 |
| The same judges vs my labels, fresh 30 | 30 | 0.64 / 0.56 / 0.77 / 0.53 / 0.74 |

- Models agree with each other far more than with me (GPT vs Fable 0.94). The gap is between me and the models, not between models.
- All 12 disagreements where I say broken and Fable says fine are repetitive but readable answers ("You need to … You need to …", "I see it in the way …") or toddler stories that do not answer the question. The definition calls "the same idea again and again, adding nothing" broken and "a phrase repeated for emphasis" not broken; these replies sit on that line. Several of my comments say so ("Seems fine but…", "I'm not sure").
- So the broken label depends on where that line is drawn, more than on which judge is used.

## 5. Missing and next (as of 23/10/2026)

- Raw judge answers are local only: upload `e3_judge_designs_261023/` to the HF dataset and record the revision.
- Open, my decision: keep my labels as the final broken label (as on 16/10), or change the broken definition, or label more replies.

## 6. Loop-and-nonsense re-score (28/10/2026)

The point to test: whether ex2 still fails when only loops and nonsense count as broken. Betley et al. 2025 and Moskvoretskii et al. 2026 score coherence only (no sense / rambling); neither counts copying the question or in-character drift.

- `make_loop_mask.py`: final labels from `finish_261016/labels.jsonl` with reason `loop`, `nonsense` or broken by my label with no stated reason. 378 of 782 broken replies; ex2 only 6 (run_main) and 4 (run_rep) of 1,800 each.
- `score_naming.py` (E3 folder, unchanged) with `--mask loop_nonsense=...`; output `results/naming_loop_nonsense.json` (log `naming_loop_nonsense.log`). `boot_seed_loop.py`: rule-1 lower bounds over 50 bootstrap seeds, `results/boot_seed_loop.json`. $0, no judge calls. Exploratory: the list was chosen after the results were seen.

| ex2, GPT judge | Point | Clean lower bound, 50 seeds (median) | Share ≥ 20% |
|---|---|---|---|
| run_main, all replies | 37.6 | 22.9 to 24.3 (23.7) | 1.00 |
| run_main, complete cells, clean | 34.5 | 20.3 to 21.7 (21.1) | 1.00 |
| run_rep, clean | 32.9 | 18.9 to 20.6 (19.9) | 0.26 |
| run_rep, complete cells, clean | 32.8 | 18.9 to 20.4 (19.8) | 0.20 |

- run_main ex2 passes all 4 rules with both judges (plays_ex2 true, Holm p 0.0004), the same as the frozen list. desc fails as before. The Claude judge passes every check in both runs.
- run_rep misses rule 3 on the GPT judge in most bootstrap seeds. This comes from removing copied and self-naming replies, not from the broken list.
- Why the 16/10 lists flipped ex2: their ex2 broken calls are mostly `format` (the translator translates the question) and `off_question` (in-character drift), and those replies are target-role hits 90 to 97% of the time.
