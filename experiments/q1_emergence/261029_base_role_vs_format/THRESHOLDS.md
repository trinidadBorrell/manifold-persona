# Pod B `base_role_vs_format`: measures, pass rules and gates

Status: **v1.7, 29/10/2026. Owner accepted the proposed values.** Canaries 1 to 3 discarded or superseded; `run_canary_v4` (templates v1.6, the full-run code) passed every gate on 29/10. Templates frozen. Full run not started.

## Question and scope

In OLMo-2-7B base, does the role or the document format drive the reply vectors (A5)? Does role information carry over from one format to another?

- Claim scope: role and format information in base reply states, under a description-only prompt, for 10 selected roles and 4 formats. The prompt is in context, so states can carry the role from the description (E3 C3); the context controls below say how much.
- Not claimed: role play, persona identity, or format effects in trained checkpoints.

## Data

- `allenai/OLMo-2-1124-7B` @ `7df9a82518afdecae4e8c026b27adccc8c1f0032`. Plain text, one `<|endoftext|>` at the start, no chat template.
- Roles (owner, 29/10): toddler, robot, narcissist, cynic, optimist, teenager, vampire, consultant, chef, pirate.
- Formats (owner, 29/10): interview (E3 `desc`, speaker "Guest"), script or dialogue, story, third-person description ("If Alex …"). The character is "Alex" in the last three.
- Descriptions: the 5 E3 third-person wordings per role (`../261008_base_persona_map/descriptions_third_person.json`), inserted into each format by one fixed rule.
- 40 frozen questions, 5 wordings, 3 replicates: 24,000 replies. Sampling as E3 (temperature 0.5, repetition penalty 1.1, top_p 1, top_k 0, 128 new tokens), item seed = SHA-256 of (role, format, wording, question, replicate).
- Pooling (owner, 29/10): the whole kept continuation. Second measure: quoted speech only.
- Reply end (owner, 29/10, after the canary raw reads): the script, story and description replies also end at the first blank line that does not follow a colon (spaces ignored). The interview ends as E3.
- Primary layer: hidden state 17 (`hidden_states` index; 0 is the embedding output). Layers 9 and 23 are exploratory.
- Storage (owner, 29/10): whole-reply states at all 33 hidden states. Quoted-speech, C2 and C3 states at 0.25, 0.5 and 0.75 of model depth only: hidden states 8, 16 and 24 of the 32-layer 7B. These report-only results are therefore at state 16, not 17.

## Inference units

- Format is crossed with role and question: every role answers every question in every format.
- Wordings are nested in role; replicates are nested in prompt (role, format, wording, question). Replicates are averaged inside each prompt first.
- Bootstrap: resample roles and questions (crossed), keep all formats, wordings and replicates of a sampled cell; paired across formats. 2,000 draws, percentile intervals.
- Scope of the intervals [R3]: they keep the five wordings fixed and use the observed replicate means. Wording-held-out and replicate-held-out results are separate sensitivities.
- Bootstrap p-values [R8]: for a one-sided rule "lower bound > 0", the decision p is the share of paired bootstrap draws with the statistic ≤ 0.
- Floors apply to the 95% lower bound.

## Integrity gates (all must pass before any metric)

| Gate | Rule |
|---|---|
| G1 tokens | No EOS, pad or stop-marker id in a pooling span. Empty kept text is invalid (NaN, flag). Each kept reply is its format's cut of the raw ids. Both state arrays have the model's shape (33 states, 3 extra states, width 4096) and a float dtype. |
| G2 valid rate | ≥ 95% valid replies in every role × format cell. |
| D1 support | G2 alone does not show D1 readiness: one lost prompt per question can remove every question at 99.5% valid replies. The canary reports complete-question support on its 10 questions and prompt-level failure clustering beside the reply-level valid rate. In the full run, generation stops and is reported once more than 10 questions have a prompt with no valid replicate, because D1 can then not reach 30 of 40 [round 2, O-B]. |
| G3 seeds | 40 items rerun at the same batch size and seed give identical ids. |
| G4 reconstruction | Stored prompt and kept ids reproduce the stored states at hidden states 9, 17 and 23, on every valid row, with centring over valid rows only: centred cosine min ≥ 0.95, median ≥ 0.99, median relative norm difference ≤ 0.02 (the full E3 C0 gate) [O3]. Invalid rows are checked separately: invalid flag set, empty pooling span, NaN vector [round 2, O-A]. Quoted-speech states [code review, 29/10]: each story and description quote state, rebuilt at the extra states, is within a relative norm difference of 0.05, with a median of at most 0.02 (a per-row rule, because a centred cosine is undefined for one row or identical quotes; the centred statistics are reported); interview and script quote states equal the stored reply states at the extra states. |
| G5 snapshot | Model and tokenizer revisions and hashes match the manifest. |
| G6 synthetic | Before real data: the variance split recovers planted role-only, format-only and interaction-only structure, also with missing prompts and unequal replicate counts [R7]; held-out-format decoding is at chance on format-only data; the D3 joint null gives uniform p on independent maps. |
| Templates | Frozen after a raw read of the canary (4 roles × 4 formats × 10 questions × 5 wordings × 2 replicates), before the full run. Stop markers per format are listed with the template. Code, prompt and tokenizer hashes are recorded before the canary; canary rows made with a template that later changes are discarded [R1]. |
| Raw read | 10 random replies per format read before any metric, and noted. |

## Primary decisions

**D1. Role vs format share of variance.**

- Measure: multivariate sums of squares over all dims at state 17, on prompt means. Terms: role, format, role × format, question, role × question, format × question, wording within role, residual. Shares of the total.
- Formula [R7]:
  - Prompt mean = the mean of the valid replicates of one (role, format, wording, question); replicate counts are recorded. Every prompt mean has equal weight.
  - Primary grid: balanced and complete. A question is kept only if every role × format × wording prompt has at least one valid replicate; a dropped question is dropped for all roles and formats. If fewer than 30 of 40 questions remain (owner, 29/10), D1 is "not estimable". No value is imputed.
  - Centre on the grand mean. Main effects are marginal means minus the grand mean; each interaction is its cell means minus the lower-order terms; wording within role is the role × wording means minus the role means; the residual is what remains. On the balanced grid these terms are orthogonal and sum to the total sum of squares, which is the denominator.
  - The fit on all valid rows (unequal weights) is reported separately.
- Pass ("role drives the vectors more than format"): the paired bootstrap difference (role share − format share) has a lower bound > 0. Decision p for Holm: the share of bootstrap draws with the difference ≤ 0 [R8].
- Reason: the outside plan's E1 rule ("role explains clearly more than format"). No effect-size floor is recorded.
- If D1 is not met [O4]: role dominance was not established. That alone does not show that format dominates. Every base claim stays "in the interview format", as already planned.

**D2. Role information carries over to an unseen format.**

- Measure: train the E3 classifier on 3 formats, test on the 4th, with questions and wordings held out at the same time. Macro accuracy over 10 roles (chance 10%). One result per held-out format.
- Null: role labels permuted as whole blocks within each question (all formats and wordings of a role move together), full refit, 9,999.
- Pass, for each of the 4 held-out formats: lower bound ≥ 20% (2× chance, as E3) and p < 0.01.
- Text check: the same split with word and character n-grams, a sentence-embedding model and hidden state 0. If the activation decoder's gain over the best of these has a lower bound ≤ 0, report "role information that the reply text already carries".
- Decision p for Holm [R8]: the largest of the 4 held-out-format permutation p-values, each p = (1 + #null ≥ observed) / (1 + 9,999).
- If D2 is not met for a format [O4]: carry-over to that format did not meet the stated criterion.

**D3. Role maps agree across formats.**

- Measure: role-role cosine-distance maps per format, with role vectors centred within format and half (as pod A D2); Spearman r between format A on one question and wording half and format B on the other half, both directions, for all 6 format pairs; mean over pairs and over the 50 fixed halves.
- Joint null [R8]: in each of 9,999 draws, keep the interview's role labels fixed and permute the role labels of each other format independently; use that format's same permutation in all its pairs; recompute the six-pair mean on the same fixed halves. Decision p = (1 + #null ≥ observed) / (1 + 9,999).
- Pass: mean cross-format r lower bound ≥ 0.5 and p < 0.01.
- Reason: the same floor as pod A D2, so within- and across-format numbers compare.
- Reported beside it: within-format reliability per format, and the cross-format r divided by the square root of the two within-format reliabilities (only when both pass the floor).

**Multiple tests:** Holm over D1, D2 (4 formats, as one decision with the largest p) and D3.

## Reported, no pass rule

- Context controls per format, each built in that format's own frame: C2 (description removed, same reply ids) and C3 (full prompt plus one fixed neutral continuation).
- Role × question and format × question shares; length, stop-reason and question-id baselines on the same folds.
- Quoted-speech-only results; results with and without replies that name the role (the E3 NAME_TERMS rule).
- Per role, the cosine between its format-specific vectors (centred within format), against Moskvoretskii et al.'s finding of cosine below 0.5 between format vectors.
- Blind naming per format: skipped for now (owner, 29/10). If added later, report only, with owner approval of the judge cost.

## Limits

- Topic: a chef talks about food in every format, so carry-over can come from topic words. The text baselines and the role × question terms show how much. Pod B has no topic-matched role pairs; pod C has them.
- Descriptions are the E3 wordings; they were written for the interview frame.

## Cost and approval

- GPU: 24,000 replies, about 1.5 A100 hours by the E3 rate; benchmark per-item seeding first. Canary: 1,600 replies.
- No judge is needed for D1–D3.

## Owner decisions (29/10)

Accepted ("Accept"): D1 role share − format share > 0 (lower bound), no effect-size floor; D2 floor 20% (2× chance) for each held-out format; D3 floor 0.5, as pod A; blind naming skipped for pod B for now.

Accepted ("Accept all three"): D1 is "not estimable" if fewer than 30 of 40 questions stay in the balanced grid.

## Changes

- 29/10/2026, v1.1, before any code or data: text fixes from Codex round 1 (`../261029_stage_persona_plan/review/codex_review_v2_r1_261029.md`): interval scope and bootstrap p (R3, R8), the D1 formula and missing-cell rule (R7), decision p for D1–D3 and the D3 joint null (R8), the full E3 C0 gate (O3), template freeze timing (R1), wording for unmet rules (O4). No threshold value changed.
- 29/10/2026, v1.2, before any code or data: owner accepted the D1 support value (at least 30 of 40 questions). No threshold value changed.
- 29/10/2026, v1.3, before any code or data: text fixes from Codex round 2 (`../261029_stage_persona_plan/review/codex_review_v2_r2_261029.md`): reconstruction on valid rows (O-A); a D1 support check in the canary and a stop rule in the full run (O-B). No threshold value changed.
- 29/10/2026, v1.4, during the code, before any run or data: owner decision (pod B code session): quoted-speech, C2 and C3 states are stored at 0.25, 0.5 and 0.75 of model depth (hidden states 8, 16, 24), not at all 33 states, so these report-only results use state 16. The primary state 17, D1–D3 and G4 (states 9, 17, 23) do not change. No threshold value changed.
- 29/10/2026, v1.5, after the engineering canary raw read, before any metric: owner decision (pod B code session): the script, story and description replies also end at the first blank line (stop marker `"\n\n"`). Reason: in the canary, 27% of description replies drifted into unrelated dataset text and 32% named the role; the same raw ids cut at the blank line give 4% and 3%, with 0 invalid rows. The interview template does not change. The canary rows made with the v1.4 templates are discarded (R1) and a new canary runs. No threshold value changed.
- 29/10/2026, v1.6, after the second canary raw read, before any metric: owner decision (pod B code session, "Adopt colon rule"): a blank line right after a colon (spaces ignored) does not end a script, story or description reply; the next blank line does. Reason: in 20 of 400 description replies of `run_canary_v2` the model wrote "say:", a blank line, then the answer, and the v1.5 cut kept only " say". An offline re-cut of the same raw ids changes 25 rows (24 description, 1 story), 0 invalid; description replies under 5 words go from 10% to 5%; of the 25, 18 are real answers and 6 are quiz text (`offline_checks/`). The `run_canary_v2` rows are discarded (R1) and a third canary runs. No threshold value changed.
- 29/10/2026, v1.7, before the third canary and any metric: integrity fixes from the Codex code review (`review/codex_last_code_261029.txt`; owner asked for the review and for every finding to be fixed). G1 also checks array shapes and dtypes and that each kept reply equals its format's cut of the raw ids; G4 now also gates the quoted-speech states (before: reported only): per row at most 0.05, median at most 0.02 (Codex rounds 2 and 3). A blank line now also covers a line of spaces, tabs or a CR, and every line break is tried (`gen_extract.blank_stop`). No threshold value changed.
