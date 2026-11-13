# Owner-review checks: why does the reply map differ from the prompt-only map? (13/11/2026)

Written at 15:41 SAST, before any check number. Owner, 13 Nov: "Run free tests now". Mode: de-risk (saved data, CPU, $0; one script and this note). Exploratory and report-only. These checks do not change `THRESHOLDS.md` v1.0.

## Question

Result 3 of the control: the reply map and the prompt-only map agree at 0.706, below the 0.90 of each map with itself. Three reasons are open:

1. The reply states carry the reply's own words.
2. The model plays a role differently from how the prompt describes it.
3. The fixed neutral reply does not fit the role, and this changes the states.

These checks test reasons 1 and 2. Reason 3 needs a GPU run and is not tested here.

## Data and common settings

- 20 roles (no `default`), 40 questions, 5 wordings. Pod A's 50 fixed halves (`make_halves`, seed 0), pod A's map (`map_panel`: centred cosine distance of the role means, Spearman over the role pairs), pod A's draws (`make_draws`, seed 20261030, 2,000 draws over roles and questions). Draw 0 is the observed value.
- R: pod A's reply states at state 17 (`run_full/reply_mean_acts.npy`).
- P: the prompt-only (C3) states at state 17 (`run_c3_full/c3_acts.npy`).

## Check 1: word-count map (reason 1)

- T: TF-IDF of the reply text (`rows.jsonl`, field `reply`), with pod A's frozen `TEXT["word"]` settings (words and word pairs, `min_df` 2, 20,000 features at most, sublinear tf), fitted on the 20,000 role replies without labels. The 5 replies of each prompt are averaged.
- Report: split-half r of T; agreement of R with T (as M2: R from one half, T from the other, both ways); the paired difference (R with R) - (R with T).

## Check 2: state-0 map (reason 1)

- E0: pod A's reply states at state 0, the mean of the reply's token embeddings before any layer.
- Report: the same three numbers as check 1, with E0 in place of T.

## Reading rules for checks 1 and 2 (fixed now)

- If the 95% interval of (R with R) - (R with X) includes 0: "the reply's words alone give the reply map" for that X. Reason 1 is then enough to explain the reply map.
- If the interval is above 0: "the reply map holds structure that X does not carry".
- Also report R with P (0.706 in the main analysis) beside R with T and R with E0, and P with T and P with E0.
- Limit, stated now: a persona shows in the reply's words. A high R-with-T agreement cannot tell "only words" from "a persona that the words express".

## Check 3: judge naming scores (reason 2)

- Score per role: hit@5 from pod A's D1 naming run of 06/11 (vault `Reference/2026-11-06 A D1 finish/d1_hit.json`), the mean of the two judges (GPT-6.1-Terra, Claude Sonnet 5.5; replicates 0-2).
- Maps: R and P from all questions and wordings. Each of the 190 role pairs gets a rank in each map (1 = closest).
- Move of a role: the mean, over its 19 pairs, of |rank in R - rank in P|.
- Toward-assistant shift of a role (19 roles, not `assistant`): rank of its pair with `assistant` in R minus that rank in P. Below 0 means the role is nearer to `assistant` in the reply map.
- Report: Spearman of hit@5 with the move (20 roles), and of hit@5 with the toward-assistant shift (19 roles), each with a two-sided permutation p (9,999 permutations, seed 20261113).
- Reason 2 predicts: roles with a low hit@5 move more (Spearman below 0), and they shift toward `assistant` (Spearman above 0).
- Reading: "the moved roles are the less-named roles" if a Spearman has the predicted sign and p < 0.05. Otherwise "no evidence for reason 2 here". With 20 roles the power is low.

## Files

- Code: `checks.py`. Output: `results.json`, `run.log`.
