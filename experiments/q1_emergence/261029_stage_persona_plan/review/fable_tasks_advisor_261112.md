# Fable 5.2 advisor review of the task plan (12/11/2026)

Reviewer: Fable 5.2 (`claude-fable-5-2`), effort high, Agent subagent of session ad361f36. Brief: blunt advice on `tasks_261112.md` including the "Tasks already open or in progress" section. Agent review, not a human review. Summary of the report (verbatim tables kept where they carry numbers).

**Verdict:** The 16 Nov post needs no new GPU run. The plan's logic is right, but T3 cannot run as defined, and T4, T5 and T8 do not give the data their questions need. The real limits are the owner's reading time and Thomas's reply.

## 1. Task against question

| Task | Problem | Source |
|---|---|---|
| T3 | The definition wants per-token RMS before the mean. All saved states are plain reply means. It needs a GPU re-read or a changed definition. | `src/manifold_persona/rows.py:248`; pod C `extract.py:92` |
| T3 on pod A | The role prompt is in context. An n-gram floor on replies does not control it. Use pod C F1 and F4 as primary. | pod A `gen_extract.py:130`; pod B `review/challenge_261105.md:33` |
| T4 | Default: 480 replies, no wordings. Each role: 2,400 over 5 wordings. Spread is not comparable. Role states hold the description; default states do not. The frames name "Alex" or "Guest", so the default is an unmarked human. Prefill text has no question and cannot sit in the map. | pod B `gen_extract.py:54-55, 80-97, 154-159` |
| T5 | No DPO, so it cannot split SFT from DPO. | `tasks_261112.md` |
| T8 | Plain GSM text has no role factor. It cannot answer Q14. | `tasks_261112.md` |
| T9 | No measure, null or rule. Raw weight cosines were at random level in the T-090 source. | T-090 `BRIEF.md:46-50` |

No task can answer Q6 for base, or Q3 "when" finer than one stage. T8, T9, T11 and T-090 serve no draft question.

## 2. Missing, by task

| Task | Missing |
|---|---|
| T2 | A continuous statistic (with 10 roles, "most extreme of 11" has up to a 1-in-11 chance). A leave-one-role-out null. A planted positive control. Separate no-system and identity default wordings. One primary frame and layer. A compute forecast (pod B's null forecast was 299 h). The T-071 items. |
| T3 | An "exploratory" label: pod C's rule built the axis only if pod A D3 passed, and it did not. The data have been seen. |
| T6 | Weights-only load for the `.bin` SFT and DPO files. Canary, gates, raw read, HF upload. |
| T7 | Surprise covariate: the reader drops the LM head. Frame count. Storage (pod C needed about 250 GB). |
| T10 | Sample size and a record format. |
| T12 | `/challenge-result`, number-to-path links, a human reviewer. |
| Post claim | No prompt-only control on Instruct; pod B's C3 is base only. About 4,200 forward passes would close it. |
| T15 | The HF token and pod are a prerequisite for T3, not housekeeping. |

## 3. Order and gate

- T0 first is right; no GPU run before Thomas's reply.
- The table gated T6 only; the order gated T5, T6 and T8. T8 and T9 have no link to the gate.
- The gate rests on one exploratory test of unknown power. The T2 simulation should show the pass rate under a planted effect.
- Parallel now: T10, the T12 draft, the T2 simulation. Sequential: T1 → T2 → T3 → gate → T6 → T7. Cut for 16 Nov: T4–T9, T11, T13.

## 4. Estimates

| Task | Plan | Measured | View |
|---|---|---|---|
| T3 | 1 h | Fetch and analysis 54 min | Compute fits; code, tests and review not counted. |
| T4 | Minutes | 6.9 replies/s | Fits. |
| T5, T6 | 1 h each | 58 min; 68 min plus 27 min copy and 15 min upload | Add about 45 min each. |
| T7 | 10–20 min per reader | 73,800 sequences in about 650 s | One frame only. Three frames: about 40 min. Ten readers: 2–7 GPU hours. |
| T8, T9 | 1 h; hours | Nothing measured | Cannot judge. |

## 5. Overreach and repeated mistakes

A 10-role base map again (pod B D3 passed 8% of simulations at 10 roles). "T2 rule covers T4" is false. T8, T9 and T-090 have rules with no power check. Sixteen tasks in four days against "write up first". Right: no example turns, no base chat template, no judge filter, text floor, precision check before the freeze, C2 skip.

## 6. Minimal sets and old tasks

| When | Tasks |
|---|---|
| By 16 Nov | HF token. T0 with T14 and T-020 in one message. T10. T12 with challenge and links. T1 and T2 as text only, not frozen. Optional: the Instruct prompt-only control. |
| Follow-up | T2 with simulation → T3 on pod C F1/F4 → small T4 plus SFT, DPO and Instruct defaults in the same plain frames → gate → T6 → T7 (4 readers, one frame, surprise) → T13. Redesign T8 with role answers to math items. Park T9. |

| Action | Old tasks |
|---|---|
| Merge | T-087/088/089 → T10. T-056 → T6. T-065 → T3, after a check with Trini. T-070 → T11 (the tracker says 160 sealed items exist). T-020 → T0. Paper tasks → T12, T13. |
| Close | T-044. T-027 with its limit. T-084. Pod B C2 at 9,999 draws. |
| Parallel | T-090: CPU power check of its pass rule now; GPU pilot after 16 Nov. |
| After | T-078, T-055/057/059, T-019/039, the manifold-test session. |
| Order changes | T-071 before the T2 freeze. T-090 ahead of T8 and T9; it is the only task for Thomas's trait direction. |

Not verified by the reviewer: E3 and 1B numbers; the 0.936 control; judge 31 of 40; $136.35; the RLVR-GSM data claim; SFT and DPO chat templates; F4 coverage of default rows. Read depth: plan, draft, critique, both adversarial reviews, meeting record, pod NOTEs and challenges, `gen_extract.py` (A and B), `rows.py`, pod C `extract.py` and `frames.py`, task tracker and T-090 files in full; `results_full_v2.json` D1/D2 keys only; pod C `THRESHOLDS.md` and `checkpoints.json` in part; `DESIGN.md`, T-071 sheet, judge note, E3 files and literature not read.
