# Challenge of the pod B full-run result (05/11/2026)

Reviewer: Claude session ad361f36 (agent). **This is an agent review, not an independent human review.** Human review: none.

Method: `/challenge-result`. I read `NOTE.md`, `THRESHOLDS.md` v1.7, the D3 code in `analysis.py`, and every result file of the full run. I read the raw-read note and 16 random replies (4 per format) from `run_full/rows.jsonl`. I did not edit pod B code, rules or result files. No pod run, no judge run, no commit.

## Start here

| | |
|---|---|
| Short answer | The numbers match the files. D2 passes, but its main reading must narrow: the hidden states carry the role mostly because the role description is in the prompt, not because of what the model wrote. D1 and D3 fail as reported; D3 fails mainly on precision with 10 roles. |
| What holds | A role decoder trained on 3 formats finds the role in the 4th (83 to 92%, chance 10%), well above every text reader. Role and format each explain about 9% of the variance; question explains 36%. |
| What does not hold | "The base model carries a role across formats in its replies" (C3 and C2 below). "Role drives the vectors more than format" (D1). "The role map is the same in every format" (D3 not established; it is also not refuted). |
| Status | D2 narrowed. D1 and D3 candidates for human review as null results. |

---

## Evidence source

| Item | Value |
|---|---|
| Results | HF `brend0nc/manifold-persona-experiment-artifacts` @ `12d932bd6d5a38bba2f48efc676afba902a0f9ab`, `experiments/q1_emergence/261029_base_role_vs_format/analysis_full_batch_v3_cuda_261105/`. 60 files; a fresh download matched every SHA-256 of `HF_backup_proposal.json` (vault `Reference/2026-11-04 B Modal recovery/full_results/HF_upload_receipt.json`). |
| Inputs | Same dataset @ `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f`, `run_full/` (receipt `hf_receipts/run_full.json`, verified). |
| Gates | G1 to G5 pass (`integrity.json`): 24,000 rows, 1 invalid (robot/description), all 40 questions complete, G4 reconstruction cosine ≥ 0.9999. |
| Family | Holm at alpha 0.05 (owner, 5 Nov), `decisions_holm_alpha_0.05.json`: D2 passes, D1 and D3 do not. |

---

## Claim ledger

| Claim text | Source file / run | Population and method | Control and baseline | Contrary evidence | Open alternatives | Missing checks | Status | Allowed wording | Human-review status |
|---|---|---|---|---|---|---|---|---|---|
| D2: role information carries over to an unseen format | `D2.json`, `sensitivities.json` (`context`) | 24,000 base replies, 10 roles × 4 formats × 40 questions × 5 wordings × 3 replicates; logistic decoder on state 17, train 3 formats, test the 4th, questions and wordings held out | Held-out accuracy 0.833 / 0.915 / 0.823 / 0.885 (interview, script, story, description), all p = 0.0001. Best text reader 0.35 to 0.43; gain lower bounds 0.31 to 0.47. Without role-name replies: 0.82 to 0.91 | C3 (full prompt, one fixed neutral continuation, no reply) decodes the role at 0.91 to 0.98 at state 16. C2 (same reply ids, description removed) drops to 0.43 to 0.47. Same-state reference: 0.82 to 0.92 | The decoder reads the description that sits in context, not role play in the reply. The reply text carries a smaller role signal (C2), close to the text readers | C2 and C3 are at state 16 and report only (no interval or test against D2). No behaviour labels for pod B replies. `rows.py` hash differs between generation (`b1e4c29b…`) and analysis (`ba1e1e03…`); not checked | narrowed | "In OLMo-2-7B base, with a role description in the prompt, a decoder trained on three formats finds the role in the fourth (83 to 92%, chance 10%), above text-only readers. A control with no reply decodes the role at 91 to 98%, so this signal mostly comes from the description in context. It does not show role play carried across formats." | none |
| D1: role does not explain more variance than format | `D1.json` | Balanced grid, 40 of 40 questions, prompt means at state 17, sums of squares | Role 0.095, format 0.088, difference 0.007 [−0.043, 0.027], p 0.587. Question 0.359, residual 0.299 | Report-only shares (role, format): no role-name replies 0.089, 0.087; state 16 0.094, 0.094; state 9 0.052, 0.106; state 23 0.107, 0.073; C2 0.039, 0.107; C3 0.140, 0.301. The order depends on the layer and the control | The primary grid mixes the description signal with the reply signal. With the reply alone (C2), format explains more than role | None for the null reading | candidate for human review | "Role and format explain about the same share of the base reply states (9.5% and 8.8%); role dominance is not established. This does not show that format dominates." | none |
| D3: role maps agree across formats | `D3.json`, `D3_resampling.npz` | Role-role distance maps per format, split halves, 6 format pairs, 10 roles | Mean cross-format r 0.707 [0.345, 0.848], p 0.0001 (null p99 0.154). Rule: lower bound ≥ 0.5 | Within-format reliability is itself low at the lower end: story 0.42, description 0.46, interview 0.58, script 0.60. Interview–script agrees as well as its reliability allows (normalised 0.955 [0.81, 1.01]) | With 10 roles, each bootstrap draw keeps about 6 to 7 distinct roles, so the interval is wide; the bootstrap median (0.647) sits below the point value (0.707) | No power check of D3 at 10 roles. No normalised value for 5 of 6 pairs (reliability floor) | candidate for human review | "The role maps of the four formats agree well above chance (r = 0.71), but the lower bound (0.34) is below the 0.5 floor, partly because the maps are noisy even within one format at 10 roles. Agreement is not established; disagreement is not shown either." | none |

---

## Checks

| Step | What I did | Result |
|---|---|---|
| 1. Source | Read D1, D2, D3, `decisions.json`, `sensitivities.json`, `integrity.json` at the HF revision | Every reported number matches. |
| 2. Raw cases | Read 16 random replies and the raw-read note | Many replies are plain answers with little role voice; a cynic interview reply talks about cynics in the third person. Role play is weak, which fits the C2/C3 reading above. |
| 3. Leakage | D2 holds out questions and wordings with the format. Role-name replies removed: decoding stays at 0.82 to 0.91 | No leakage found. The description in context is the main alternative, and C3 measures it. |
| 4. Plan | `THRESHOLDS.md` v1.7 changes are all dated before any metric. The Modal re-run changed the compute engine, not the tests; Codex matched 200 CPU and GPU predictions | No test changed after results. Alpha 0.05 was set after the raw p-values were saved but before the family verdict was read. |
| 5. Interpretation | Separated states from behaviour | These are hidden-state results. No pod B reply has a behaviour label, so nothing here says the base model plays the role. |

---

## What would strengthen it

1. A test of D2 on C2 states (reply only, description removed) with the same split and null. This asks whether the reply alone carries the role across formats.
2. Terra labels for a sample of pod B replies, to tie the states to observed role play.
3. A D3 power check at 10 roles, or D3 with more roles.
4. Check that the `rows.py` change between generation and analysis does not touch row identity.

---

## Addendum: D2 on the reply-only states (05–06/11/2026)

Owner: "Yes, run it". Report only. Frozen D2 folds and draws (checked equal to `D2_inputs.json`), C2 states at state 16, 99 block permutations per format (not the frozen 9,999, so p cannot go below 0.01). Script and output: vault `Reference/2026-11-05 B C2 decoding/`.

| Format | Reply only | Best text reader | Gain over text | p (99 draws) | Null max |
|---|---|---|---|---|---|
| Interview | 0.434 [0.318, 0.541] | 0.363 | 0.071 [0.006, 0.131] | 0.01 | 0.108 |
| Script | 0.470 [0.364, 0.563] | 0.394 | 0.076 [0.012, 0.131] | 0.01 | 0.111 |
| Story | 0.426 [0.323, 0.533] | 0.429 | −0.003 [−0.059, 0.046] | 0.01 | 0.108 |
| Description | 0.434 [0.305, 0.559] | 0.386 | 0.047 [−0.024, 0.113] | 0.01 | 0.110 |

Reading: the replies alone carry role information across formats, above the 20% floor in every format and far above every null draw. Beyond the text readers, the reply states add a small gain in interview and script and no established gain in story and description. This confirms the narrowed D2 wording: the 82–92% is mostly the description in context; a smaller, real signal is in the reply, and most of it is already in the words. Text baselines are from the primary run on the same folds; C2 is at state 16 and they are not state-matched to it.

## Addendum: Terra labels and the `rows.py` check (06/11/2026)

Owner: "Go ahead and run it". Report only. Files: vault `Reference/2026-11-06 B Terra labels/`. I read 34 random labelled replies (14 passes, 14 G, 3 W, 3 U). Most labels look right. A few G labels are arguable, which fits the low base recall below.

| Item | Result | Limit |
|---|---|---|
| Terra pass rate, all pod B replies | 15.4% (3,689 of 23,999); formats 14.0 to 18.6%; roles 3.8 to 34.1% | Raw judge rate. No owner label on pod B. On 20 E3 base practice replies, Terra said yes to 3 and the owner to 8, so the rate is likely low. No correction is possible with these labels. |
| Reply-only decoding by Terra pass | 66.7% against 32.3% inside each role and format; difference 0.345 [0.317, 0.371]; all four formats 0.29 to 0.39 | Chosen after the labels. Judge and decoder read the same reply, so this is not independent evidence. No text reader has had the same split, so the share the states add is unknown. |
| Prompts with no passing reply | 31.3% decoding, chance 10% | Either role signal below Terra's line, or role play Terra missed. These data do not separate the two. |
| `rows.py` hash | Only `file_sha256` changed (chunked reading) | Closed. Row identity is not touched. |

Allowed wording: "A judge (GPT-6.1 Terra) passed 15.4% of OLMo-2-7B base replies as in role and coherent. This is an uncorrected judge rate and is likely low. Reply-only role decoding is higher on prompts with a passing reply (67% against 32%, inside role and format), but the judge and the decoder read the same text." Status: exploratory. Human review: none.

Still open from the list above: a D3 power check, and D2 on C2 at the frozen state with 9,999 draws. Item 2 and item 4 are done.

## Addendum: D3 power check (06/11/2026)

Owner: "1" (start the pod B checks). Report only. The rule floor stays as frozen. Files: vault `Reference/2026-11-06 B D3 power/`.

Method: simulated grids of the real shape (10 roles, 4 formats, 40 questions, 5 wordings), with the frozen `make_halves`, `map_statistic`, `crossed_draws` and `d3_bootstrap`. Each format's role vector is sqrt(rho) × shared + sqrt(1 − rho) × own, in a rank-K space. Noise is set so within-format reliability is about 0.79, the mean observed value. 60 datasets per row, 300 bootstrap draws each. "Passes" counts the lower bound rule only; the p test is not simulated.

| Roles | Rank | Shared share (rho) | Mean r | Mean lower bound | Lower bound ≥ 0.5 |
|---|---|---|---|---|---|
| 10 | 4 | 1.0 | 0.82 | 0.38 | 8% |
| 10 | 4 | 0.9 | 0.64 | 0.23 | 2% |
| 10 | 4 | 0.8 | 0.50 | 0.12 | 0% |
| 10 | 2 | 1.0 | 0.77 | 0.26 | 0% |
| 10 | 8 | 1.0 | 0.78 | 0.34 | 3% |
| 20 | 4 | 1.0 | 0.78 | 0.45 | 25% |
| 30 | 4 | 1.0 | 0.81 | 0.54 | 78% |
| 30 | 4 | 0.8 | 0.48 | 0.28 | 0% |

Reading: at 10 roles, the frozen D3 rule almost never passes, even when all formats share one map exactly. The failure says little about the map. The observed r 0.707 and lower bound 0.345 sit between the rho 0.9 and rho 1.0 rows, so the data fit strong shared structure, and they do not prove it. About 30 roles would give the rule useful power under this model. The original result stays: D3 not passed.

Limits: this is a model of the noise, not a resampling of the real states. Rank, noise shape and the reliability match are my choices, and 60 datasets give about ±7 points on each share. Chosen after D3 failed, so it explains the failure and does not change it.

## Addendum: C2 at 9,999 draws (06/11/2026)

The earlier item "D2 on C2 at the frozen state" cannot run as written. C2 states exist only at hidden states 8, 16 and 24 (`THRESHOLDS.md` v1.4), so state 17 would need new model passes. At state 16, the 99-draw run took about 25 seconds per draw per format, so 9,999 draws would take about 278 hours on the Mac and need a paid GPU run. The observed values are 0.43 to 0.47 and the largest of 99 null draws is 0.11, so more draws would only lower p below 0.01. Not run; owner to decide.
