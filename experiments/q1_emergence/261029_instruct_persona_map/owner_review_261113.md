# Owner review of pod A (13/11/2026)

Guide: Claude session a5480f90 (agent). The agent showed evidence and checked numbers. **Only the owner's own lines below count as human review.** No old file was changed.

## Start here

| | |
|---|---|
| State | The owner stopped the review on 13 Nov. One result (D2) was shown. No result has an owner decision yet. |
| Done by the owner | The raw read: his own 40 labels on pod A replies (32 yes). He saw D2 with its three limits. |
| Not done by the owner | D3 and D1 were not shown to him. He did not go through the agent challenge or the proof-ledger corrections. |
| Decision on D2 | Put off: "LEt me decide later in the draft session". |
| Next | The draft session takes the decisions. Sections 3 and 4 hold the numbers it needs, each checked in the files by the agent. |

---

## 1. Owner's words (exact)

| Date | Owner's words | Effect |
|---|---|---|
| 13/11/2026 | "Did I not give labels to discover and consider this ?" | His earlier labels are his raw read. No new verdicts were asked. |
| 13/11/2026 | "LEt me decide later in the draft session, if there is nothing else here, lets just call it a day." | The D2 decision moves to the draft session. The review stops here. |

---

## 2. Raw read: the owner's own labels

Question for each reply: does it play the role? Two replies per role, 40 in all.

| Owner's labels | Roles |
|---|---|
| Yes both times (13) | assistant, bartender, chef, consultant, criminal, cynic, grandparent, narcissist, optimist, pirate, robot, toddler, vampire |
| Yes one time, no one time (6) | counselor, detective, stoic (no in the practice set); refugee, teenager, void (no on 12 Nov) |
| No both times (1) | translator |

- Total: 32 yes of 40 (16 of 20 in each set).
- Sources (outside the repo): `/Users/brendon/persona-manifolds/judge_validation_261029/dev_owner_labels.json` with `dev_key.json` (20 pod A practice items) and `review261112_owner_labels.json` with `review261112_key.json` (20 items, 12 Nov).
- One reader, two replies per role: each role's count is rough.

Extra replies shown on 12–13 Nov, with no verdict from the owner (he pointed to his labels):

- Draw: seed 20261113, two per role, one per wording. It leaves out the 100 sealed and practice pod A items and the 20 `raw_read.md` items. Prompts decoded from `prompt_ids` with the tokenizer at revision `470b1fba…`.
- Rows shown as short excerpts: translator `[3,37,1]`, `[1,130,1]`; void `[2,211,1]`, `[3,77,4]`; refugee `[0,212,3]`, `[3,216,0]`; teenager `[0,204,0]`, `[2,24,3]`; default `[3,204,4]` (wording, question, replicate).

---

## 3. D2, shown to the owner (decision put off)

The role map repeats across question and wording halves.

| Item | Value | Source |
|---|---|---|
| Split-half r, state 17 | 0.901 [0.797, 0.939] | vault `Reference/2026-10-30 A-B pilot results/full_A_results.json`, `primary.D2.raw_r` |
| Null p99, p | 0.220, 0.0001 | same, `primary.D2` |
| Rule | lower bound ≥ 0.5, p < 0.01, interval within ±0.10: passed | `THRESHOLDS.md` D2 |
| Holm, alpha 0.05 | adjusted p 0.0002, decision true | vault `Reference/2026-11-06 A D1 finish/d1_miss.json`, `family` |

Limits shown with it (agent checks, 13 Nov):

| Limit | Evidence | Source |
|---|---|---|
| The prompt alone also gives a map | Role prompt plus one fixed neutral reply: r = 0.905 [0.829, 0.936]. Reply map against prompt-only map: 0.706 [0.541, 0.825]. Paired gap 0.195 [0.066, 0.315], above 0. Role decoding: prompt-only 0.991, replies 0.918. | `../261112_instruct_prompt_only/results_full/c3_results.json`, state 17 |
| The words carry much of it | Word n-grams decode the role at 0.797 [0.692, 0.887]; state 17 gives 0.918 [0.879, 0.950]; states minus best text 0.121 [0.055, 0.201]. | `full_A_results.json`, `decoding` |
| The map uses all valid replies | Fidelity labels are not in the result (`fidelity.available` false). The owner's labels: 32 of 40 play the role. | `full_A_results.json`, `fidelity`; section 2 |

- The prompt-only control is agent analysis with one fixed neutral reply. It has its own review session.
- Wording the agent offered (not approved by the owner): "In Instruct, the 20-role map repeats across question and wording halves (r = 0.90 [0.80, 0.94]). The role prompt alone gives an equally repeatable map (0.905), so this shows a repeatable map with the prompt in view, not a persona."

---

## 4. Not shown to the owner (for the draft session)

The agent checked each number below in the files on 13 Nov. The owner has not reviewed any of it.

### D3: default-minus-roles direction (a gate, not a claim)

| Item | Value | Source |
|---|---|---|
| Split-half cosine | 0.881 [0.763, 0.884]; floor 0.80 met; precision missed (width 0.121 against 0.10); not passed | `full_A_results.json`, `primary.D3` |
| Point against its interval | The point sits near the top of its own interval. | same |
| `assistant` role in place of `default` | Cosine to the default direction 0.936. Its own split-half cosine 0.932 lies above its interval [0.834, 0.915]. | same, `primary.assistant_control` |
| All five default wordings on both sides | 0.944 (descriptive; it changes the frozen comparison) | vault `Reference/2026-10-30 A axis diagnostic/results.json`, `all_default_wordings_on_each_side` |
| Built-in limit | The direction ends at the default by construction. A fixed default-prompt offset passes the rule with no role differences. | `THRESHOLDS.md` D3 |

### D1: blind naming of the role from the reply

| Item | Terra | Sonnet | Source |
|---|---|---|---|
| Macro hit@5 | 0.687 [0.580, 0.795] | 0.687 [0.576, 0.794] | `d1_miss.json`, `D1.judges` |
| Null p99 | 0.069 | 0.068 | same |
| Roles above own null | 20 of 20 | 20 of 20 | same |
| Precision | ±10.7 points against ±7.5: missed, so `passed` is false | same | same |
| Without replies that use a role name | 0.655 | 0.654 | `d1_checks.json` |
| Role SD | 0.243 | 0.241 | computed from `per_role` |

- Per role, hit@5 runs from 0.195 (translator, Sonnet) to 0.998 (pirate, Terra). Lowest four for Terra: translator 0.248, consultant 0.318, void 0.385, assistant 0.417.
- The role SD gives ±10.7 points at 20 roles; ±7.5 needs about 40 roles. More replies per role do not narrow it.
- 13 Sonnet answers stayed invalid; miss and hit bounds give the same decisions (`challenge_261106.md`). Not rechecked here.
- Against the owner's labels: translator is the lowest role for both judges and the only role with two owner "no" labels. Consultant has two owner "yes" labels and low naming (0.318, 0.515).

### Agent findings checked against the files

| Finding | Check | Result |
|---|---|---|
| `review/challenge_261106.md`: D1, D2, D3 numbers | Compared with `full_A_results.json` and `d1_miss.json` | Match |
| Proof ledger C8: "No prompt-only control on Instruct" | The control ran on 12–13 Nov | Out of date. The prompt alone gives a map (0.905); the reply map is not the prompt map (gap 0.195 [0.066, 0.315]). |
| Proof ledger C9: 0.881 [0.763, 0.884]; assistant direction 0.936 | `primary.D3`, `primary.assistant_control` | Match |
| Proof ledger C10: naming values, SD 0.243, about 40 roles | `d1_miss.json` | Match |
| Codex C8: fidelity labels unavailable | `fidelity.available` false | Match |
| Codex C9: 0.944 with all default wordings on both sides | Axis diagnostic `results.json` | Match (0.9442) |
| Codex C10: 0.655 without self-named replies | `d1_checks.json` | Match |

---

## 5. Open

- Owner decisions on D2, D3 and D1: wording and limits for the draft.
- The owner's view of the agent challenge and of the reading "D1 not met does not stop pod C" (`challenge_261106.md`, last section).
- The 13 invalid Sonnet answers and the retry collision were not rechecked in this review.
