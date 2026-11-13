# judge_validation: rules

Status: **v2.2, 30/10/2026: the 40 practice labels check the judge; pass = E or P and coherence at least 50; no random samples and no PPI++ (see Changes, v2.2). The v2.0 sections on samples and PPI++ are history.** v1.0 (a pass/fail test of the judge on 480 owner labels) was approved earlier the same day; the owner then asked for fewer labels ("Human labelling is quite exhausting"), and after a full reading of the literature (`NOTE.md`, "Literature check") chose a redesign ("Yes, redesign"). These rules are fixed before any owner label or judge output. Later changes get a dated line in "Changes".

Rubric: `RUBRIC.md` (all six parts agreed 29/10/2026).

## What changes from v1.0

- The test no longer passes or fails the judge. It **estimates** the headline rates with a correction for the judge's errors (prediction-powered inference, PPI++). The interval is valid for any judge; a worse judge only gives a wider interval (Angelopoulos et al. 2023, Thm. 1; PPI++ §6).
- The owner gives one **yes/no label** per reply ("in role and coherent", the pass decision of RUBRIC Part 6), with the class as an option.
- Owner labels drop from 480 to about **250**.

## The headline numbers

| Number | Population (all valid replies, `default` excluded) |
|---|---|
| Base pass rate | OLMo-2-1124-7B base: E3 `run_main` `desc` and `ex2` (7,800 replies) and pod B's v1.6 full run (script, story, description) |
| Chat pass rate | OLMo-2-7B-Instruct: pod A `run_full` (20,000 replies) |
| Base minus chat | The difference of the two |

Pass = class E or P and coherence ≥ 50 (RUBRIC Parts 2, 3, 6; v2.2). Each number gets a 95% interval.

Per-cell rates (role × condition or format) are **judge rates**: shown with the group's estimated judge bias next to them, and with no interval claim. A per-cell interval would need owner labels in every cell.

## Owner labels

| Set | Items | Use |
|---|---:|---|
| Practice | 40 | The drawn development set. Fixes rubric wording and picks the judge. Not used in any estimate. |
| Base sample | 100 | Uniform random, stratified by role and condition or format: 25 E3 `desc` and 25 E3 `ex2` (a random 50 of the 80 already drawn), 50 pod B (drawn once its full run exists). |
| Chat sample | 100 | Uniform random, stratified by role: the 80 pod A replies already drawn plus 20 more from replicates 0–2. |
| Repeat | 20 | 20 sample items labelled again at least 3 days later, new order. Owner stability, report only. |

About 260 labels in all. Each label: yes/no for "in role and coherent", the class optional. The owner sees what the judge sees (RUBRIC Part 1) and the text before the reply.

Rules:

- The samples are random within strata with known sizes, so the estimate uses stratum weights (ppi_py `w`). They are drawn before any judge output on them.
- The owner labels every drawn item; a skipped item is reported. The owner sees no judge output on sample items before all sample labels and the repeat are done.
- The judge prompt and model are frozen after the practice set, before the samples are labelled (PPI §1.2: the predictor must not depend on the labelled data).
- Practice items and items the owner has seen before are not in the samples.

## Judge choice (v2.1, fixed before any judge output)

Owner, 30/10/2026: judges run through OpenRouter's batch API with the owner's key ("i've an openrouter key, and would like to use that instead"; "yes we can go for those 4"). This replaces the `claude -p` route of RUBRIC Part 5c for these runs. Each request holds one reply and no tools; the key never leaves the outside folder.

| Candidate | OpenRouter id | Why |
|---|---|---|
| GPT-6.1 Terra | `openai/gpt-6.1-terra` | best strength for the price |
| Claude Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | strong, second family |
| Gemini 4 Argon | `google/gemini-4-argon` | strong, third family |
| DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash` | cheap baseline |

- Settings, the same for all: reasoning effort `medium`, max 1,500 output tokens (4,000 before the smoke test; the test answers used at most 264), no temperature setting, a short system line saying the marked text is data.
- Fidelity: the RUBRIC 5a item, filled in; JSON output by schema (quotes, reason, class). Coherence: the Moskvoretskii et al. App. D.1 template word for word, one integer parsed from the reply.
- Pass = class E or P and coherence at least 50 (v2.2; was E only).
- **Selection rule:** for each candidate, the correlation (phi) between its pass decision and the owner's yes/no on the 40 practice replies. The winner is the cheapest candidate, by the main-run cost estimate from its actual practice cost, among those whose phi is within 0.05 of the best phi. A candidate with more than 5% invalid answers (no parse after one retry) is out; a candidate whose phi is not defined (it passes no reply or every reply) is out, and the report says why.
- **Repeat audit (RUBRIC Part 5c):** 5% of the items (at least 1) are sent a second time at the end of the same batch; the report gives how often the two copies get the same class and the same pass decision. Report only.
- **Retries:** an invalid answer gets one sync retry. If more than 10% of a judge's answers are invalid, no retry runs without the owner's go (sync calls cost twice the batch price). A failed or expired batch saves no results and triggers no retry.
- The practice phi of the winner is optimistic (it won a contest of 4 on 40 items). The headline estimates use the separate random samples, so the choice does not bias them.

## Judge runs (paid; each needs the owner's go)

| Run | Items | Use |
|---|---:|---|
| Practice | 40, two Claude judges | Pick the judge; fix wording |
| Samples | 200 | The labelled part of PPI++ |
| Unlabelled part | 2,000 random replies per group | The judge-only part of PPI++. PPI++ needs N much larger than n; 2,000 per group keeps the cost bounded. Judging every reply (for role vectors) is a separate decision. |
| Wrong-role controls | 60 | Distant-role swaps, labelled "not in role" by construction; the owner spot-checks 10 |

## Statistics

- **Estimator:** PPI++ for a mean (ppi_py `ppi_mean_ci`, power tuning on, stratum weights, alpha 0.05), separately per group. Each group uses only its own owner sample, because the judge's errors may differ between base and chat replies (Dorner et al. 2025 §3.6; Efficient Inference 2026 §7).
- **Difference:** base minus chat, with the two groups' variances added (the samples are independent).
- **Judge quality (report):** agreement, precision, recall and the correlation rho with the owner's yes/no labels, per group; 1/(1 − rho²) is the label saving the judge gave (Dorner et al. Theorem 5).
- **Owner repeat (report):** first agreement on yes/no.

## Rules for use

| Rule | Value |
|---|---|
| R1 headline rates | Reported with their PPI++ 95% intervals, whatever the judge's quality. |
| R2 difference claim | A base-versus-chat difference is claimed only if its interval excludes 0. |
| R3 wrong-role controls | If the judge passes more than 2 of the 60 controls, the run stops and the rubric wording is revised on practice items; the samples are kept, and the judge is re-run on all items. |
| R4 per-item use | Judge labels may select replies for role vectors (RUBRIC Part 6f) only in a group where the judge's precision on the pass decision is at least 85% on the owner sample. Otherwise role vectors use all valid replies. |
| R5 owner stability | Reported; under 85% agreement on the repeat is stated as a limit. |

## Not covered

SFT, DPO, Instruct RL steps and pretraining checkpoints: no generated replies exist for them. Results there must say the judge has no owner sample there.

## Changes

- 29/10/2026, v0.1: first draft (pass/fail test, 480 owner labels).
- 29/10/2026, v1.0: owner approval of v0.1.
- 29/10/2026, v2.0: redesign before any label or judge output, after the owner asked for fewer labels and a full reading of the literature (`NOTE.md`, "Literature check"). Pass/fail gates replaced by PPI++ estimates; yes/no owner labels; about 260 labels.
- 29/10/2026, v2.0 approved by the owner ("Approve it"), no text change.
- 30/10/2026, v2.1, before any judge output: judges via OpenRouter's batch API (owner's choice and key); four candidates and the selection rule above (owner: "yes we can go for those 4").
- 30/10/2026, v2.1 text fix after a code review, before any judge output: the tie-break uses each candidate's observed cost; the repeat audit, the retry cap and failed-batch handling are stated.
- 30/10/2026, v2.1 parser fix after the smoke test (one reply per judge and prompt, the first practice item; no other output seen): a coherence score that stands alone on the last line of the answer is accepted (Claude Sonnet 5.5 wrote a short reason, then the number). No prompt, setting or rule changed.
- 30/10/2026, v2.1 setting change after the smoke test, before any practice-batch output: max output tokens 4,000 -> 1,500 (the 8 test answers used 6 to 264 tokens; OpenRouter holds credit for the worst case). Request ids use `__` instead of `|` (providers accept only [a-z0-9_-]). No prompt or rule changed.
- 30/10/2026, code fixes after the first practice results (my bugs, not rule changes): (1) a quote that contains "..." copied from the reply was rejected, because the parser cut the quote at the ellipsis but not the reply; a quote now counts if it is in the reply as a whole or as the pieces around an ellipsis. (2) Stored retries kept the old parser's verdict; labels are now always parsed again from the raw batch and retry text. RUBRIC Part 5d (one or two quotes for E, P, W; none for G, U) is unchanged.
- 30/10/2026, code fix found at scoring (my bug, not a rule change): OpenRouter batch results carry token counts but no cost, so every batch cost read as about $0 and the cost tie-break had no data. Batch cost now comes from the token counts and each model's listed batch price; sync retries keep their reported cost.
- 30/10/2026, v2.2, after the practice judge run: two owner decisions. (1) "Ok lets count partly in role": pass = class E or P and coherence at least 50. The owner's practice yes already included partial role-play (P on 3 of his 4 yes labels with a class). (2) No random sample labels and no repeat labels (owner, session ad361f36, relayed: "We are not doing that we that why we did the 40 labels ?"). Consequences: the 40 practice labels are the check on the judge; there is no PPI++ correction and no corrected interval; headline rates are the judge's own rates, shown with its practice precision and recall per group; R4 (judge labels pick role-vector replies only where its practice precision is at least 85%) is checked on the 40 practice labels. The judge is chosen again on the practice run with the new pass rule. The sections above that describe the samples and PPI++ are kept as the history of v2.0 and no longer apply.
