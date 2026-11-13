# Judge validation `judge_validation`: an LLM judge labels which replies are in role (29/10/2026)

This is the note for `261029_judge_validation`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | What share of the base-model replies and the chat-model replies is in role and coherent? And can a judge's labels select replies for role vectors? |
| Result | I chose GPT-6.1 Terra as the judge. On my 40 practice labels it reaches phi 0.60, and it was right on 94% of the replies that it passed. Terra passes 61.9% of the pod A chat replies, 21.1% of the E3 base replies and 15.4% of the pod B base replies. |
| How much I trust it | These are raw judge rates, not corrected. Terra is stricter than I am on base replies: it kept 3 of my 8 base yes replies. So both rates are likely below mine, and the base rate most. The check of the judge rests on 40 practice replies. |
| What I do not know | The corrected rates. I dropped the 200 random sample labels, so the PPI++ correction has nothing to work with. How I report the headline rates is still my decision. |
| Status (6 Nov 2026) | Rules v2.2 (30/10/2026). The Terra labels for E3, pod A and pod B are done and on HF. Next: Terra labels the role-vector replies. That needs my go. |

---

## What I did

1. I approved the rules v2.0 on 29/10/2026.
2. I labelled 40 practice replies with yes or no. I saw no judge output before.
3. Four judges labelled the same 40 replies through OpenRouter's batch API. That cost about $0.23.
4. I decided to count partly in role as a pass: "Ok lets count partly in role". With that rule Terra won by the fixed choice rule.
5. I dropped the 200 random sample labels and the 20 repeat labels.
6. Terra labelled every valid reply of E3, pod A and pod B.

---

## Results

A reply passes when its class is E (in role) or P (partly in role) and its coherence is at least 50.

### Terra agrees best with my practice labels

| Judge | phi with my labels | Right when it says yes | Judge yes of my yes, base | Chat |
|---|---|---|---|---|
| GPT-6.1 Terra (chosen) | 0.60 | 0.94 | 3 / 8 | 14 / 16 |
| Gemini 4 Argon | 0.56 | 0.94 | 2 / 8 | 14 / 16 |
| DeepSeek V4.1 Flash | 0.53 | 0.89 | 3 / 8 | 15 / 16 |
| Claude Sonnet 5.5 | out | | | |

- My fixed rule picks the cheapest judge within 0.05 of the best phi. That is Terra.
- Sonnet is out, because 5 of its 80 answers broke the format twice.
- Terra's yes was right on 3 of 3 base replies and 13 of 14 chat replies. So my rule R4 holds on the practice set, and the pass sets may select replies for role vectors.

### Terra passes 62% of the chat replies and 15 to 21% of the base replies

| Set | Replies | Judge pass rate |
|---|---:|---:|
| Pod A chat | 20,000 | 61.9% |
| E3 base, `desc` | 4,000 | 12.0% |
| E3 base, `ex2` | 3,800 | 30.7% |
| E3 base, both | 7,800 | 21.1% |
| Pod B base | 23,999 | 15.4% |

- These are judge rates, not corrected. They are likely below my own rates, and the base rates most.
- The rates differ a lot by role. In base `ex2`, robot passes 87% and pirate 82%, but detective 1%. In pod A, pirate passes 100% and refugee 25%.
- The four pod B formats are close: interview 14.1%, script 14.0%, story 18.6%, description 14.9%.

### The judges are stricter than I am on base replies

- On the 8 base replies that I called yes, coherence is high (72 to 100), but the judges give G or P.
- I marked P on 3 of the 4 yes replies where I gave a class. So my "yes" includes partial role play.
- Terra keeps 3 of my 8 base yes replies, and its yes has been right. So base role-vector sets will be smaller, not wrong.

---

## How much I trust this

- The practice set has 40 replies. I said yes to 24: 8 of 20 base replies and 16 of 20 chat replies.
- For base, Terra's precision rests on 3 judge-yes replies only.
- I chose the pass rule (E or P) after I saw the practice judge results. The judges were rescored on the same practice answers, with no new judge call.
- Terra is stable on repeats: the same class on 382 of 389 E3 repeats, 950 of 999 pod A repeats and 1,164 of 1,199 pod B repeats.
- No label of mine is on a pod B reply.
- There is no random labelled sample, so no rate has a corrected interval.

---

## Open points

- Terra labels the role-vector replies next. That needs my go.
- I have not decided how I report the headline rates without the PPI++ correction.
- The old record is unclear on one point. The status line says rules v2.2 hold "No random sample labels, no PPI++". The section on my decision says THRESHOLDS was not changed yet. Both statements are in the record.
- OpenRouter's billed amount for the E3 and pod A runs is not copied yet.
- The old Google Sheet v1 (`1s4zjGIp…`) is still in Drive. I must trash it myself.

---

## Words used here

| Word | Meaning |
|---|---|
| Judge | An LLM that labels a reply. It gives a class, a coherence score and quotes as evidence. |
| Terra | GPT-6.1 Terra, the judge that I chose. It runs through OpenRouter's batch API. |
| Practice labels | My yes or no labels on the 40 development replies (set `dev`). They check the judge. |
| Classes E, P, G, W, U | The rubric classes of a reply: E in role, P partly in role, G not enough evidence, W a different role, U cannot score. |
| Pass | Class E or P, and coherence at least 50. |
| phi | The correlation between the judge's yes or no and my yes or no. |
| PPI++ | Prediction-powered inference. The judge labels all replies, and a random sample of human labels corrects the judge's bias. |
| R4 | My rule for per-item use: judge labels may select replies for role vectors only where the judge's precision on the pass decision is at least 85%. |
| Role vectors | Directions built from the replies that are in role. |
| Repeat audit | 5% of the replies go to the judge a second time, to see if it gives the same class. |
| E3 | My run `261008_base_persona_map`: OLMo-2-7B base replies in an interview frame. |
| `desc`, `ex2` | The two E3 prompt types: the role description only, and the role description plus two example answers. |
| Pod A | My run `261029_instruct_persona_map`: OLMo-2-7B Instruct replies in chat format. |
| Pod B | My run `261029_base_role_vs_format`: OLMo-2-7B base replies in four formats (interview, script, story, description). |
| `dev`, `sealed` | The 40 development items and the 160 sealed items drawn on 29/10/2026. |
| `pop_e3main`, `pop_poda`, `pop_podb` | The sets of all valid replies of E3, pod A and pod B that Terra labelled. |
| Wrong-role controls | Replies paired with a role that they do not play. A good judge fails them. |
| E2, A3 | Labels from the plan that the old note calls the friend's plan. E2 is this experiment. A3 is the assumption that "in role" is judged the same way everywhere. The note says no more about that plan. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

Status line (v2.2, 30/10/2026): my 40 practice labels check the judge; pass = E or P and coherence at least 50; judge = GPT-6.1 Terra (OpenRouter batch). No random sample labels, no PPI++. Terra labels for E3, pod A and pod B are done and on HF (6 Nov). Next: Terra labels the role-vector replies (needs my go). Rubric: `RUBRIC.md`. Rules: `THRESHOLDS.md` (v2.2 in Changes).

## 1. Question and mode (29/10/2026)

What share of base-model and chat-model replies is in role and coherent, with intervals that correct for the judge's errors against my labels? This is the friend's plan E2, testing A3 ("in role" is judged the same way everywhere).

- What it teaches: the headline pass rates and their difference, with valid intervals; how well the judge matches me in each group; whether judge labels can select replies for role vectors.
- Expected result: Instruct passes more easily than base. Base risks: short and plain replies (too little evidence), assistant tone from the `ex2` examples, toddler speech and translated questions under the coherence judge.
- Cheapest useful test: the 40 development items with two Claude judges. They show quickly if the rubric can work at all, before the 200 sample labels.

Mode: extended. The result gates the role-play numbers of pods A and B and the E3 re-score, it reuses `judge2.py` code, and judge calls cost money.

## 2. Data (all existed on 29/10, except pod B's full run)

| Group | Source | Revision |
|---|---|---|
| Base, interview | E3 `261008_base_persona_map/run_main` (`desc`, `ex2`) and `run_rep` | HF `brend0nc/manifold-persona-experiment-artifacts` @ `fb84c5028fee341aa8832620985b7a7d29c7c302` (`NOTE.md` of E3) |
| Base, other formats | Pod B `261029_base_role_vs_format`, v1.6 full run, or else its third canary | Missing: not run yet |
| Instruct | Pod A `261029_instruct_persona_map/run_full` | HF @ `60a81b9b2e337a46a0a2c41862357164273da5ff` |

## 3. Procedure of the first plan (v2.0)

Steps 4 to 6 did not happen in this form. Section 11 has my decision to drop the sample labels.

1. I approve `THRESHOLDS.md` v2.0. The labelling page changes to a yes/no label (class optional).
2. I label the 40 practice items (no judge output seen).
3. Practice judge run, two Claude judges (my go: paid). Revise wording on these items only; choose the judge; freeze the prompt.
4. Draw the extra samples (20 chat; 50 pod B once its full run exists; a random 50 of the 80 drawn E3 items). I label the 200 sample items in short sessions.
5. At least 3 days later: my 20-item repeat.
6. Judge run on the 200 samples, 2,000 random replies per group and the 60 wrong-role controls (my go: paid). PPI++ estimates and the rules in `THRESHOLDS.md`.

## 4. Cost and time of the first plan

- My labels: 40 + 200 + 20 = 260 yes/no labels; at about 40 seconds each, about 3 hours.
- Judge calls: about 160 on practice items (40 x 2 prompts x 2 judges) and about 8,600 for the estimates (4,260 items x 2 prompts), in batches of 25. Price: missing (depends on the CLI plan).

## 5. Schedule of the first plan (to the 16 Nov deadline)

| Dates | Work |
|---|---|
| 30-31 Oct | Approve v2.0; practice labels; practice judge run; freeze |
| 1-4 Nov | Sample labels, about 50 per day |
| 5-7 Nov | My repeat; judge runs; PPI++ estimates |
| 8-15 Nov | Write the result |

## 6. Code and files (29/10/2026)

| File | What it does |
|---|---|
| `sample.py` | Draws `dev`, `sealed` (v1.0, kept), and the v2.0 samples `chat_sample` (the 80 sealed pod A replies plus 20 new, one per role), `base_e3` (a random 25 desc and 25 ex2 of the sealed E3 replies) and `base_podb` (17/17/16 by format) (seed 20261029). Leaves out every reply that I read or labelled (E3 raw read, the 80 broken labels, the 50 role scores via the judge2 prep key; pod A and pod B raw reads). Writes `<set>_items.jsonl` (what the judge and I see) and `<set>_key.json` outside the repo; writes `samples/<set>_manifest.json` here (counts, hashes, sources; no key). |
| `label_sheet.py` | `build` writes `<set>_labels.html` next to the items; `serve` serves it on localhost and saves every change to `<set>_owner_labels.json` there (launch configs `judge-labels-dev`, `judge-labels-sealed` in `.claude/launch.json`). Opened as a plain file it shows a warning: the app's file view runs no scripts (my report 29/10, "This is broken"). The page (30/10, after I said it was not explained properly): a one-minute guide first (the question, what yes and no mean, tricky cases, 8 real examples that I had seen, none in the sets), then one reply per screen with the role's signs in plain words and one yes/no question; keys Y N; class, quotes and note optional; labels kept in the browser and exported as `<set>_owner_labels.json`. Checked in the browser on 29/10 (keys set the labels; counter updates). |
| `or_judge.py` | OpenRouter judges (THRESHOLDS v2.1): `prep`, `smoke`, `submit`, `poll`, `retry`, `score`. One request per reply and prompt; a 5% repeat audit; a fingerprint check refuses runs whose prompt or settings changed since `prep`; a failed batch saves nothing and is not retried; retries stop above 10% invalid without my go and resume after a crash; quotes are matched after normalising quote marks and spaces; the winner is the cheapest by observed cost within 0.05 of the best phi. Code review (30/10, 14 findings) fixed; tested offline with a fake API. No paid call at that time. |
| `transcripts.py` | Shared reader for the example transcripts used by `label_sheet.py` and `export_sheet.py`. |
| `export_sheet.py` | Builds the judge-work workbook from `RUBRIC.md`, `THRESHOLDS.md`, the Codex review and `judge2.py`. |

- Sets drawn 29/10/2026 into `/Users/brendon/persona-manifolds/judge_validation_261029/` (outside the repo): `dev` 40 (E3 `run_rep` desc 10, ex2 10; pod A 20), items sha256 `faf6e9d4…`; `sealed` 160 (E3 `run_main` desc 40, ex2 40; pod A 80). The draw repeats exactly with the same code and data. A test of `sealed_podb` on the discarded `run_canary_v2` rows gave 27/27/26 by format; that test output was deleted.
- Coherence context for E3 items: the interview frame without the role line and without example turns, so the coherence judge sees no role text.

Google Sheet of the judge work, v2 (private, my main Drive, imported by Codex on 29/10/2026): https://docs.google.com/spreadsheets/d/1Rje1OqpfCvDjii6KqC_9ujpvMHRi4KnJVXfNDVEFozI/edit.

- It was rewritten after I rejected v1 ("not human readable ... no actual transcripts"; I asked for Neel Nanda's style): plain words, no file names, 15 real transcripts that I had already seen (none in the development or sealed sets).
- Same day, my words: "it should present current state, why is it mentioning things we had done before ?" The Sheet was rewritten in place (same link) to current state only: the 'Old judges' tab, my earlier scores and old judge calls, and dated history were removed.
- Tabs: Start here, Transcripts, The rubric, Assumptions, Decisions, The test, Prompts, Papers. `export_sheet.py` builds it.
- Then I asked for the first person, as I wrote it. The text changed to 'I/my' in place (my quotes unchanged).
- v1 (`1s4zjGIp…`) is still in Drive: Codex's tools cannot trash files, so I trash it.

## 7. Not written at 29/10

- The control pools (wrong role, positive, name only, speaker switch, in-role refusal) need my screening; the 5 matched-style rewrites need an LLM call (my go).
- The scoring script (statistics and gates in `THRESHOLDS.md`).
- The judge run setup for these items (from `judge2.py`).

## 8. Practice judge run (30/10/2026)

My labels: 40 of 40 practice replies (yes 24: base 8 of 20, chat 16 of 20). Judges through OpenRouter's batch API (THRESHOLDS v2.1), one smoke call per judge and prompt first. Gemini's first batch failed validation (Google needs one configuration per batch) and was resubmitted as one batch per prompt. Total spend about $0.23.

This table uses the first pass rule (yes = class E only). Section 9 has the rule that holds now.

| Judge | phi with my labels | Agreement | Judge yes / my yes, base | Chat | Invalid after retry | Practice cost | Main run, estimate |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 0.46 | 0.68 | 0 / 8 | 13 / 16 | 0 | $0.008 | about $1 |
| GPT-6.1 Terra | 0.49 | 0.70 | 1 / 8 | 13 / 16 | 0 | $0.031 | about $3 |
| Claude Sonnet 5.5 | 0.50 | 0.69 | 0 / 6 | 8 / 13 | 5 (out: over 5%) | $0.120 | about $13 |
| Gemini 4 Argon | 0.42 | 0.65 | 0 / 8 | 12 / 16 | 0 | $0.061 | about $7 |

- Winner by the fixed rule: DeepSeek V4.1 Flash (within 0.05 of the best phi, cheapest). Sonnet is out: 5 of 80 answers broke the quote rule twice (3 quotes, or quotes with G).
- What it teaches: the judges are far stricter than I am on base replies. On the 8 base replies that I called yes, coherence is high (72 to 100), but the judges give G or P. I also marked P on 3 of the 4 yes replies where I gave a class, so my "yes" includes partial role-play, while the rubric's yes is E only. With pass = E or P (exploratory, not a rule), phi rises to 0.53 to 0.60 and base judge yes to 2 to 3 of 8.
- Consequence: at phi near 0.5, the PPI label saving is only about 1.3x (Dorner et al. Theorem 5). The practice set exists to fix such wording; I decide what "yes" means before any sample label.
- Code bugs found and fixed during the run (recorded in THRESHOLDS changes): ellipsis quotes, stale retry verdicts, batch cost read as zero.
- Files: `results/dev_judge_choice.json` (aggregates); requests, raw results and scores in `/Users/brendon/persona-manifolds/judge_validation_261029/dev_judges/`.

## 9. Pass rule and judge choice (v2.2, 30/10/2026)

My words: "Ok lets count partly in role" (after the literature check: Lu et al. keep only fully in-role replies; Moskvoretskii et al., following Chen et al., keep replies above 50 of 100). Pass = E or P and coherence at least 50. Rescored on the same practice answers (no new judge call):

| Judge | phi | Right when it says yes | Judge yes / my yes, base | Chat | Main cost |
|---|---|---|---|---|---|
| GPT-6.1 Terra (chosen) | 0.60 | 0.94 (base 3 of 3, chat 13 of 14) | 3 / 8 | 14 / 16 | about $0.75 per 1,000 replies |
| Gemini 4 Argon | 0.56 | 0.94 | 2 / 8 | 14 / 16 | about twice Terra |
| DeepSeek V4.1 Flash | 0.53 | 0.89 | 3 / 8 | 15 / 16 | cheapest, but 0.07 below the best |
| Claude Sonnet 5.5 | out (5 of 80 answers broke the format twice) | | | | |

- By the fixed rule (cheapest within 0.05 of the best phi), Terra wins. R4 holds on the practice set (precision 94% overall; base rests on 3 judge-yes replies only).
- The judges keep fewer base replies than I do (Terra 3 of my 8), but their yes has been right. Base role-vector sets will be smaller, not wrong.
- Google Sheet updated in place to this state (Codex read-back: 0 differing cells, private).

## 10. Main judge run (started 30/10/2026)

My words: "Go", then "E3 + pod A (~$21)".

- GPT-6.1 Terra (OpenRouter batch, the v2.1 settings) labels every valid reply of E3 `run_main` `desc` and `ex2` (7,800; set `pop_e3main`) and pod A `run_full` without `default` (20,000; set `pop_poda`), each with the 5% repeat audit.
- Requests go in parts of up to 5,000; a background loop submits parts as OpenRouter's credit holds allow (worst case about $20 per part against the key's limit) and polls until both sets finish. Pod B follows when its rows reach this Mac.
- Item sets and keys: `/Users/brendon/persona-manifolds/judge_validation_261029/pop_*`; labels and rates come from `or_judge.py export`.

Finished 30/10/2026, about 22:00 SAST. Unparsed after one retry: 4 of 16,380 E3 requests and 7 of 42,000 pod A requests. Computed cost from token counts at the listed batch price: E3 $5.51, pod A $17.11 (OpenRouter's billed amount not copied yet). Repeat audit: same class on 382 of 389 E3 repeats and 950 of 999 pod A repeats.

| Set | Replies | Judge pass rate (E or P, coherence at least 50) |
|---|---:|---:|
| E3 base, `desc` | 4,000 | 12.0% |
| E3 base, `ex2` | 3,800 | 30.7% |
| E3 base, both | 7,800 | 21.1% |
| Pod A chat | 20,000 | 61.9% |

- These are judge rates, not corrected. On the practice set the judge kept 3 of my 8 base yes replies and 14 of my 16 chat yes replies, so both rates, and the base rate most, are likely below mine.
- R4 holds on the practice set (judge yes right on 3 of 3 base and 13 of 14 chat replies), so the pass sets may select replies for role vectors.
- Per-role rates span widely: base `ex2` robot 87%, pirate 82%, but detective 1%; pod A pirate 100%, refugee 25%. Files: `results/pop_e3main_judge_rates.json`, `results/pop_poda_judge_rates.json`; per-reply labels in `pop_*_judges/openai__gpt-6.1-terra/judge_labels.json` (outside the repo; not on HF at that time).
- Code fix during the run: the poll summary showed "completed" while one part still ran (no data effect).
- HF copy (my go, 5 Nov: "3. yes you can"): per-reply labels, keys and manifests of `pop_e3main` and `pop_poda` at `experiments/q1_emergence/261029_judge_validation/terra_labels_261030/`, revision `a74610f8e6298d6b4aee942c5af4619a5dc543ac`; 6 files re-downloaded, hashes match (`/Users/brendon/persona-manifolds/judge_validation_261029/terra_labels_hf_receipt.json`).

Pod B:

- My go, 5 Nov: "Go ahead and run it". `sample.py pop_podb` built all 23,999 valid pod B replies (set `pop_podb`); Terra smoke valid on both prompts; batch loop started 22:50 SAST. Finished 6 Nov 10:08 SAST.
- Result (6 Nov): 3,689 of 23,999 passed (15.4%); interview 14.1%, script 14.0%, story 18.6%, description 14.9%. Classes: G 19,947, E 2,823, P 1,078, W 98, U 52, 1 reply with no valid label. Batch answers invalid 11 of 50,398; one retry fixed 10. Repeat audit: same class on 1,164 of 1,199. Cost $17.29. Judge rates, not corrected; no label of mine is on a pod B reply. File: `results/pop_podb_judge_rates.json`.
- HF copy: `terra_labels_261030/pop_podb/` (labels, key, manifest) at revision `09f2898eddbd54a1b286c83945f361992ffebbd0`; 3 files re-downloaded, hashes match (`/Users/brendon/persona-manifolds/judge_validation_261029/pop_podb_hf_receipt.json`).

## 11. My decision: no 200 sample labels (30/10/2026)

My words, in session ad361f36, relayed by the coordinator session: "We are not doing that we that why we did the 40 labels ?" The 200 random sample labels and the 20 repeats of THRESHOLDS v2.0 are dropped. THRESHOLDS was not changed at that time: without a random labelled sample, the PPI++ correction has nothing to work with, so I first decide how the headline rates are reported.

## 12. Literature check: fewer human labels (29/10/2026)

I asked to fully read the work before changing the test. Four reader agents read the full texts. Claude checked the key claims in the texts and redid the label counts.

- Prediction-powered inference (Angelopoulos et al., Science 2023, arXiv 2301.09633 v4; PPI++, arXiv 2311.01453 v2): the judge labels all replies; a random human sample corrects its bias; the result is a corrected rate with a valid interval, not a pass/fail test of the judge. For a yes/no label with symmetric judge error, plain PPI helps only if the error is under 25% at a rate of 0.5, or under about 9.5% at 0.1 (App. G.1, checked). PPI++ tunes the weight and is never worse than human labels alone. The labelled sample must be random from the same set; pooling labels across groups needs equal judge error in each.
- Dorner, Nastl, Hardt (ICLR 2025, arXiv 2410.13341 v4): the best saving of any unbiased method is 1/(1 - rho^2), rho = correlation of judge and human labels (Theorem 5); if 0.5 <= agreement <= the yes-rate, the saving is at most 2x (Theorem 6, Corollary 7; checked). In their experiments savings were mostly under 2x.
- Gligorić et al. (NAACL 2025, arXiv 2408.15204 v2): the judge's verbal confidence decides which items the human labels, with known probabilities; at least 25% fewer human labels in each of three tasks (checked). Tested only at about 250 or more human labels per estimate. Labelling only judge disagreements without these probabilities is biased.
- Noisy but Valid (ICLR 2026, arXiv 2601.20913) and Efficient Inference for Noisy LLM-as-a-Judge Evaluation (arXiv 2601.05420): a correction from the judge's true- and false-positive rates (Rogan-Gladen) is worse than PPI (checked) and saves nothing at 85% agreement; PPI++, the joint MLE and EIF give the same asymptotic variance for a yes/no label.
- Counts (Claude's arithmetic, judge-human agreement 85%, symmetric error): a rate to +/-10 points at 95% needs about 96 human labels alone and about 49 with PPI++ at a rate of 0.5 (61 vs 38 at 0.2). The base-minus-chat difference to +/-10 points needs about 98 per group. At 95% agreement, PPI++ needs about 5x fewer than human labels alone at 0.5.
- Consequence: these methods halve the labels for a few headline numbers, each with its own random human sample per group. They do not remove labelling, and per-cell intervals would need labels in every cell. Reader reports: scratchpad only (not kept).

## 13. My open decisions, as the old note listed them

1. ~~Approve `THRESHOLDS.md` v2.0.~~ Approved 29/10/2026.
2. A GPT judge needs an isolation check for Codex CLI. Otherwise both practice judges are Claude models (one family).
3. Go for the practice judge run (paid), after the practice labels.
