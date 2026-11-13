# Owner-review checks: results (13/11/2026)

Rules: `RULES.md` (written 15:41 SAST, before any number; sha256 `7b5fadb2…`). Code: `checks.py` (`3c002ad3…`). Output: `results.json` (`5083688f…`), `run.log`. Command, from this folder: `.venv/bin/python -I checks.py` (2,000 draws, 50 halves). CPU on the Mac, $0. Exploratory, report-only, agent analysis. Not on HF. Not logged in the vault.

Sanity check: the run reproduces the main analysis (reply with reply 0.901 [0.797, 0.939]; reply with prompt-only 0.706 [0.541, 0.825]).

## Agreement between maps (state 17 unless stated)

| Maps | Agreement |
|---|---|
| Reply with reply (R, R) | 0.901 [0.797, 0.939] |
| Prompt-only with prompt-only (P, P) | 0.905 [0.829, 0.936] |
| Word counts with word counts (T, T) | 0.842 [0.659, 0.901] |
| State 0 with state 0 (E0, E0) | 0.879 [0.730, 0.932] |
| Reply with prompt-only (R, P) | 0.706 [0.541, 0.825] |
| Reply with word counts (R, T) | 0.828 [0.654, 0.890] |
| Reply with state 0 (R, E0) | 0.795 [0.551, 0.880] |
| Prompt-only with word counts (P, T) | 0.624 [0.449, 0.759] |
| Prompt-only with state 0 (P, E0) | 0.461 [0.229, 0.656] |
| Word counts with state 0 (T, E0) | 0.781 [0.568, 0.868] |

## Paired differences

| Difference | Value | Reading by the rules |
|---|---|---|
| (R, R) - (R, T) | 0.073 [0.039, 0.171] | Above 0: "the reply map holds structure that word counts do not carry" |
| (R, R) - (R, E0) | 0.105 [0.041, 0.290] | Above 0: "the reply map holds structure that state 0 does not carry" |
| (R, R) - (R, P) | 0.195 [0.066, 0.315] | The main result, reproduced |
| (R, T) - (R, P) | 0.122 [-0.056, 0.239] | Includes 0 |
| (R, E0) - (R, P) | 0.090 [-0.183, 0.250] | Includes 0 |

## Check 3: judge naming scores (reason 2)

| Measure | Spearman | p | Roles | Predicted sign |
|---|---|---|---|---|
| hit@5 with move | -0.023 | 0.928 | 20 | below 0 |
| hit@5 with toward-assistant shift | +0.373 | 0.114 | 19 | above 0 |

Reading by the rules: "no evidence for reason 2 here". The power is low with 20 roles.

## Notes that the rules did not fix (post hoc)

- The word-count map repeats less well (0.842) than the reply map (0.901). This noise alone lowers the agreement of R with T. A rough ceiling is sqrt(0.901 x 0.842) = 0.871; R with T is 0.828, which is 95% of it. The same ratio is 89% for state 0 and 78% for the prompt-only map. So most of the 0.073 gap may be noise in the word-count map.
- TF-IDF used all 20,000 role replies and reached the 20,000-feature limit.
- Reason 3 (the fixed neutral reply does not fit the role) is not tested. It needs a GPU run.
