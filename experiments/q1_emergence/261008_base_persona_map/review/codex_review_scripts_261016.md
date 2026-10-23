**Verdict:** Seven blockers. Hit-rate estimates check out. Coverage counts and judge validation gates are not reliable.

---

**Blockers**

**1. Coverage uses the wrong role axis.**

Source: [score_naming.py:139](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:139), [score_naming.py:158](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:158).
The null keeps the original reply role as its output axis; each target role needs its own null distribution.
Failing case: a judge that always names `assistant` gives coverage 1; the correct coverage is 0.
With the same 9,999 draws, GPT `desc` coverage changes from the saved 12 to 17.
Clean-set nulls also need the retained-wording mask to move with each reply block; their macro null values currently differ.

---

**2. Complete-cell sensitivity omits the clean-reply decision.**

Source: [score_naming.py:184](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:184), [score_naming.py:189](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:189).
Rule 4 compares only unfiltered rules 1–2; it never repeats rule 3 within complete cells.
Failing case: every reply hits; only two questions per role are clean; one wording in those cells is invalid.
The invalid rate is 2.22%, and both judges return `plays=true`, although complete cells contain zero clean replies.
THRESHOLDS.md requires the complete-cell analysis to give the same full decision.

---

**3. Score joins accept results from the wrong judge run.**

Source: [judge_broken.py:229](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:229), [judge_broken.py:235](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:235).
The scorer ignores manifest hashes, request hashes, response checks, model checks, and required Noul names.
Failing case: `--jev jev/full` supplies all required keys, but uses the earlier definition and omits `copies_question`.
For `run_main/4082`, that run says fine; round 3 says broken, with `copies_question=0.67`.
An in-memory score test also accepted wrong manifest hashes and wrong models; [validate_broken.py:44](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/validate_broken.py:44) has the same unchecked-read pattern.

---

**4. The validation gate can pass without the required labels.**

Source: [judge_broken.py:244](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:244), [judge_broken.py:250](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:250).
Failing case: omit `--labels`; `all([])` marks every judge and the majority usable, then writes a majority mask.
One label file also suffices, although the stated gate requires both the owner50 and fresh30 sets.
Labels outside the judged keys disappear: my three-label test retained two labels and passed with kappa 1.
[validate_broken.py:57](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/validate_broken.py:57) also drops unmatched labels instead of rejecting an incomplete join.

---

**5. Jev cannot resume an unchanged run.**

Source: [jev_judge.py:148](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/jev_judge.py:148), [jev_judge.py:153](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/jev_judge.py:153).
`NOULS` contains tuples; JSON stores them as lists.
Failing case: run again with the unchanged `jev/round3` settings; the direct comparison says the run differs.
I checked that JSON-normalized settings match exactly, while the script's comparison fails.
The script stops before its saved-answer retry logic can resume the run.

---

**6. The $25 cost cap does not cap submissions or retry costs.**

Source: [judge_broken.py:154](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:154), [judge_broken.py:168](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:168), [judge_broken.py:198](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:198).
The cap is checked once per job, without reserving money for other workers or checking again before retries.
Failing case: start at $24; three offline stub responses each cost $10 and contain an inconsistent reason.
The actual job function makes all three attempts and finishes at $54 against its $25 cap.
Restarts count only saved successes, so earlier failed attempts disappear; model-check exceptions also discard returned cost information.

---

**7. Failed revalidation leaves old usable masks in place.**

Source: [judge_broken.py:258](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:258), [validate_broken.py:85](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/validate_broken.py:85).
Failing case: score usable labels, then score reversed labels into the same output directory.
My in-memory test changed majority kappa from 1 to -1, but retained `mask_majority.json`.
The validator likewise leaves old `mask_sonnet.json`, `mask_jev.json`, and `mask_strict.json` when their judges fail.
Downstream naming analysis can therefore consume a mask that the current validation rejects.

---

**Other issues: primary decisions and reports**

- [score_naming.py:198](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:198): `plays` uses raw p-values; no final primary decision reads the Holm-adjusted values at line 273.
- At a corrected cutoff of 0.01, raw p=0.004 becomes 0.016 in this four-test family, but the flag stays unchanged.
- Current adjusted naming p-values are 0.0004, so this omission does not affect their significance gate today.

---

**Other issues: donor contrast**

[score_naming.py:258](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:258) reports donor hit rates without the requested target-minus-donor contrast or crossed-bootstrap CI.

---

**Other issues: checks and isolation**

| Source | Concrete issue |
|---|---|
| [jev_judge.py:68](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/jev_judge.py:68) | Accepts `jev-1.130.0` and `jev-1.13.999`; the model version is not enforced exactly. |
| [jev_judge.py:75](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/jev_judge.py:75) | Accepts an absent score or score -100 with probability 1 at level 4; validation then crashes or misreports secondary coherence. |
| [score_naming.py:98](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:98) | Row counts do not prove unique cells; duplicate coordinates can overwrite one cell and leave another empty. |
| [score_naming.py:284](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:284) | `--show` accesses `all_replies` on `both_judges` and fails; later summary objects also lack `results`. |
| [judge_broken.py:131](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge_broken.py:131) | GPT tools remain available; the script never checks logs for file reads despite its isolation statement. |

---

**Other issues: incomplete sensitivity reporting**

[score_naming.py:212](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:212) checks the eight-word example-copy threshold against other examples only.
It does not supply the specified unrelated-reply-pair check or a control for the five-word description threshold.
[score_naming.py:269](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:269) reports seed differences in hit@5 only, not agreement of every primary rule.
[score_naming.py:123](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:123) silently excludes roles with no retained replies; complete-cell macro scores can therefore average fewer than 20 roles.

---

**What I verified: naming numbers**

I independently rebuilt all eight condition/donor grids for both judges from local replies, prompts, keys, and answers.
All 16 hit@5, hit@1, clean-hit, complete-cell-hit, donor-hit, flag, and frozen-attrition point estimates matched.
These checks establish arithmetic agreement; they do not validate the incorrect null axis.

| Measure | Independent result |
|---|---|
| Main `desc`, GPT hit@5 | 269/1,800 = 0.1494444444 |
| Main `ex2`, GPT hit@5 | 676/1,800 = 0.3755555556 |
| Main `ex2`, Claude hit@5 | 761/1,800 = 0.4227777778 |
| `ex2_other`, shift 6, GPT donor hit@5 | 452/1,800 = 0.2511111111 |
| Main `desc`, frozen invalid rate | 416/1,800 = 0.2311111111 |

---

**What I verified: inference and mapping**

An independent 2,000-draw crossed bootstrap reproduced the first GPT `desc` CI: [0.095, 0.2144444444].
The same 9,999 permutation draws reproduced p=0.0001; each draw moves all five wordings together within a question.
All grids contain unique cells and the same 18 question indices; every naming answer has five distinct guesses.
`ROLES` and `NAME_TERMS` match their original rules; all 5,400 donor prompts use the correct shift without example/test-question overlap.

---

**What I verified: batches and Jev records**

I reconstructed 3,600 unique reply keys and blind IDs in 144 batches of 25.
The broken-judge set contains no delimiter collisions.
All 10,800 current Jev reply keys, request hashes, and response checks match the local questions and replies.

---

**What I verified: broken judges**

Both current prompts cover the owner's five broken-answer clauses and the stated exceptions; question and answer mappings are correct.
The 50 labels contain 50 distinct shared `desc` items; the fresh 30 contain 15 `desc` and 15 `ex2` items.
The fresh labels match the fresh-set file exactly and do not overlap the 50 labels.
Independent counts reproduced the earlier Jev results below; these remain results from its earlier definition.

| Earlier Jev measure | Independent result |
|---|---|
| Owner50 accuracy / kappa | 0.8 / 0.5412844037 |
| Fresh30 accuracy / kappa | 0.8666666667 / 0.6610169492 |
| Main `desc` broken rate | 282/1,800 = 0.1566666667 |
| Main `ex2` broken rate | 148/1,800 = 0.0822222222 |

---

**What I verified: scope**

All four scripts parse, and the four-test Holm arithmetic is correct with Part 2 p-values set to 1.
Failure tests used memory-only files and offline call stubs; no paid model command or HTTP request ran.
I did not read the token or verify live execution; raw broken-judge Opus/GPT/Sonnet batches were unavailable in this folder.
Only this review file was written; the four scripts, THRESHOLDS.md, and sealed naming JSON retained their original hashes.
