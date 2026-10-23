**Verdict:** Finish a complete provisional classification today; do not claim that the owner-defined broken rates are validated.

The failed gate and lack of an independent owner test block a validated finish across all six groups.
The useful finish is a complete label file, stated reasons, rate estimates, and separate uncertainty checks.
Call every automatic result exploratory.

| Deliverable today | Honest status |
|---|---|
| Fine/Broken calls for the 10,800 judged replies | Feasible, with automatic calls marked provisional |
| Reasons and counts for each condition | Feasible, with the reason source recorded |
| Rates from the automatic labels | Feasible, conditional on the stated classification method |
| Owner-based rate estimates | Small stratified sample for main desc and ex2 only |
| Validated owner-defined rates for all conditions | Not supported |

---

**Evidence checked**

Here, `EXP` means `experiments/q1_emergence/261008_base_persona_map/`.
I read the reply files, keys, owner files, saved round 3 results, earlier scores, and naming results.
The dated change in `THRESHOLDS.md` controls the definition; its older table still gives the replaced rule.
This advice used local reads and calculations only, with no model calls or HTTP requests.

| Local check | Result |
|---|---|
| Judged scope | Six groups of 1,800; 20 roles × 18 questions × 5 wordings |
| Owner labels | 50 final labels plus 30 fresh labels; no overlap |
| Saved GPT and Opus coverage | 3,600 each; main desc and ex2 |
| Saved Jev coverage | All 10,800 |
| Jev criteria actually saved | Five Nouls, including separate new-turn and question-copy tests; plus coherence Score |
| Saved batch hashes and question/answer text | Match their manifest and source replies |
| Saved Jev request hashes | All match requests rebuilt from saved settings and source replies |
| Current judge source hashes | Both differ from their saved round 3 hashes |
| Bare terminal stop fragments | 17 in the original 3,600; 51 across the 10,800 |

---

**1. Lock the scope and evidence — 5 minutes**

Create `EXP/results/broken/finish_261016/` for new outputs.
Select exactly the unique reply keys with `type == "coh"` from `/Users/brendon/persona-manifolds/e3_judge2_coh_261016/key.json`.
Record source hashes, saved prompts, models, settings, and the source-version differences in a new manifest.
**Done check:** Every key resolves once; each group has 1,800 replies in a complete 20 × 18 × 5 grid.

| Scope rule | Required treatment |
|---|---|
| Runs | Main desc, ex2, ex2_filler, ex2_other; replication desc and ex2 |
| Questions | The 18 shared questions only |
| Other-role examples | Keep the donor already selected for each cell |
| Donor scope | No claim about all three donors per cell; 3,600 other-donor replies remain outside this set |
| Reply text | Use `reply`, not the later interview turns retained in `raw` |
| Old `valid` field | No exclusion based on the replaced attrition rule |

---

**2. Freeze the classification method — 5 minutes**

Use GPT as the uniform provisional judge because its existing output gives a label and a reason.
Choose this method before the extension; record that no judge passed validation.
Keep owner labels authoritative wherever they exist.
**Done check:** The manifest states these rules before any new output arrives.

| Component | Fixed rule |
|---|---|
| Final label | Owner label on the 80 known rows; GPT label elsewhere |
| GPT input | Original reply, except the verified fragment-only derivative described in step 4 |
| Fine reason | `none`, meaning no defined failure detected |
| Broken reasons | `loop`, `nonsense`, `format`, `off_question` |
| Jev | Reuse all saved round 3 outputs; Broken if any of the five Nouls is strictly above 0.5 |
| Jev coherence Score | Diagnostic only; no new cutoff |
| Opus | Reuse its existing 3,600 calls as a separate check |
| Combination | Keep all judge calls; add agreement and disagreement fields |
| Existing three-judge majority | Preserve its fixed rule and failed gate; retain its main desc/ex2 result as a sensitivity check |

---

**Combination limits**

Do not invent a two-judge majority when GPT and Jev disagree.
Do not change the main method between conditions.
Report pure GPT rates beside the owner-plus-GPT rates, so the owner substitutions remain visible.

---

**3. Complete GPT coverage — 35 minutes**

Create a separate helper, `EXP/finish_broken_261016.py`; preserve the round 3 scripts and saved results.
Reuse the 3,600 valid raw GPT calls and judge the remaining 7,200 raw replies through Codex CLI.
Use GPT-6.1-Sol, high effort, batches of 25, and eight workers.
**Done check:** Exactly 10,800 raw GPT calls pass schema, identifier, model, and input-hash checks.

| Execution rule | Required treatment |
|---|---|
| Prompt and schema | Copy the saved round 3 wording exactly; preserve the owner exclusions |
| Judge view | Question and answer only; blind identifiers; no condition, role, flags, owner labels, or other judge calls |
| Isolation | Empty working folder and read-only sandbox; check logs for outside file reads |
| Retries | Retry technical failures only; preserve failed attempts |
| Opus | No new calls; do not treat the small remaining budget as an extension budget |
| Jev | No new HTTP calls; reuse saved results |
| Time limit | Stop extension work at 45 minutes elapsed if coverage remains incomplete |

If coverage remains incomplete, report the missing keys and retain the full Jev-only diagnostic table.
Do not fill missing GPT calls with Fine or report complete GPT coverage.

---

**4. Separate stop fragments from answer failures — 10 minutes**

Make a derived answer by removing only a verified final bare `Interview` or `Interviewer` line from a length-limited reply.
Keep the source answer and every raw judge call unchanged.
Judge all 51 derived answers through the same GPT prompt, without showing the earlier calls.
**Done check:** Every text change has a key, original suffix, derived-text hash, and raw-versus-derived classification.

| Group | Verified fragment candidates |
|---|---:|
| Main desc | 7 |
| Main ex2 | 10 |
| Main ex2_filler | 17 |
| Main ex2_other | 6 |
| Replication desc | 8 |
| Replication ex2 | 3 |
| Total | 51 |

---

**Fragment decisions**

Confirm each candidate is only a stop fragment before removing it.
A real new question, speaker turn, copied question, loop, or unrelated answer remains assessable as Broken.
Do not turn all 51 replies into Fine; one can contain a real failure before the suffix.
Use the derived GPT call in the final automatic label, while keeping all owner labels unchanged.

---

**Fragment reporting**

Report the rate change in each group and identify disagreements with owner labels on these rows.
Use raw GPT versus raw Jev for the full judge comparison; label that comparison as raw-text agreement.
Do not call a raw Jev result a judgment of the derived answer.
Do not use any apparent agreement gain after this change to rescue the failed gate.

---

**5. Report owner agreement and keep the gate failed — 10 minutes**

Reproduce the original gate from the raw calls, before owner substitutions or fragment changes.
The rule stays kappa ≥ 0.6 on both sets, using the unrounded values.
The definition followed inspection of these labels, so neither set is a clean test of this procedure.
**Done check:** The report says `gate_failed`, with both separate label-set tables and no validated exclusion mask.

| Judge | Owner50 kappa | Fresh30 kappa | Gate |
|---|---:|---:|---|
| Opus | 0.593 | 0.561 | Failed |
| GPT | 0.568 | 0.636 | Failed |
| Jev | 0.541 | 0.769 | Failed |
| Three-judge majority | 0.568 | 0.636 | Failed |

| Agreement output | Required detail |
|---|---|
| Each judge and majority | Confusion counts, accuracy, kappa, Broken sensitivity, Fine specificity |
| Owner50 | Selected flagged desc examples; no population interpretation |
| Fresh30 | Separate condition/old-flag strata; preserve the 8/7 design; add weighted confusion counts using step 6 weights |
| Agreement intervals | 2,000 paired bootstrap draws; fresh draws within each of the four strata |
| Small class counts | Show denominators and Wilson 95% intervals for sensitivity and specificity; descriptive only on these selected labels |
| Undefined bootstrap kappa | Mark undefined and report its frequency; do not substitute zero |

---

**Agreement limits**

GPT agrees on 26/30 fresh labels but finds only 5/9 owner-marked Broken replies.
It misses both owner-marked Broken replies among the seven unflagged ex2 examples.
Show these counts; accuracy alone hides the missed failures.
Fresh30 agreement intervals describe this sample design, not validation across the other conditions.

---

**6. Report three separate sources of uncertainty — 10 minutes**

Keep the six run/condition groups separate, with counts and denominators beside percentages.
The observed-set rate is exact conditional on the assigned labels.
Use the following checks to show what those labels cannot establish.
**Done check:** Each reported interval or range states what it measures and what it leaves unknown.

| Source | Required calculation and limit |
|---|---|
| Roles and questions | 2,000 crossed bootstrap draws; resample 20 roles and 18 questions independently, keeping all five wordings |
| Conditional rate interval | Percentiles 2.5 and 97.5 for the final owner-plus-GPT rate; label it conditional on these classifications |
| Condition or seed contrasts | Use the same role/question draws on each side; preserve paired cells and fixed donor choices |
| Judge disagreement | Raw GPT/Jev 2×2 counts, disagreement percentage, both-Broken rate, and either-Broken rate in every group |
| Extra judge check | Opus rate, all-three disagreement, and majority rate on the original 3,600 only |
| Fragment handling | Raw versus derived rate change, with changed keys and retained owner calls |

---

**Rate limits**

The crossed interval describes variation across these selected roles and questions; it does not include classification error.
The both/either rates show judge dependence; they are not bounds on the true rate or a confidence interval.
Agreement between judges does not prove correctness.
Two decoding seeds do not establish uncertainty across a population of seeds.

---

**Owner sample**

Use fresh30 to give a separate, direct owner-based rate estimate for main desc and ex2.
First recover its sampling frame, including whether it excluded the earlier 50 owner-labelled replies.
For each random stratum, use its population count divided by the group total as its weight.
If the earlier 50 were excluded, add their known Broken count separately and weight the remaining sampled population.

| Main group / old-rule stratum | Full judged-set count | Fresh owner Broken / sampled |
|---|---:|---:|
| desc / flagged | 416 | 5/8 |
| desc / unflagged | 1,384 | 0/7 |
| ex2 / flagged | 145 | 2/8 |
| ex2 / unflagged | 1,655 | 2/7 |

---

**Owner sample intervals**

Sum stratum weight × stratum Broken fraction, then add the contribution from any fully observed group.
Invert the finite-population hypergeometric distribution for a 97.5% interval in each of the two sampled strata.
Weight and sum their endpoints for a conservative 95% interval per condition; include known labels as fixed counts.
State the sampling-frame assumption if it cannot be recovered; do not present an unverified design interval as exact.

---

**Owner sample limits**

Zero Broken in seven examples does not prove zero failures in that stratum.
Do not transfer these tiny-sample error estimates to filler, other-role examples, or replication replies.
Those four groups have no direct owner validation.
Do not give their true owner-defined rates a narrow interval by treating judge calls as truth.

---

**7. Build the complete label and reason files — 5 minutes**

Write one row per key to `labels.jsonl`, including final label, reason, method, and the preserved individual judge calls.
For the 26 owner-marked Broken rows, recover a reason from the owner's note or inspect the question and answer.
Mark reasons inferred by Claude as Claude's interpretation; preserve the owner's binary label.
**Done check:** Every row has a label and reason source; Fine plus Broken equals 1,800 in every group.

| Required field or table | Rule |
|---|---|
| Identity | Reply key, run, condition, role, question index, wording, selected donor |
| Final call | Fine/Broken, primary reason, label source, reason source |
| Audit fields | Owner call if present; raw GPT reason; derived GPT call if present; Opus call if present; all Jev Nouls |
| Evidence examples | One excerpt per observed failure type; include owner disagreement and fragment examples |
| Missing owner mechanism | Record `owner_broken_reason_unstated` if it cannot be supported; count this as an unresolved reason |
| Primary reason totals | One primary reason per Broken reply; totals must equal the Broken count |
| Secondary reasons | Optional overlapping counts, clearly marked as overlapping |
| Per-condition reason table | Fine, loop, nonsense, format, off_question, and unresolved owner reason; counts and percentages of all replies |

---

**Reason limits**

Keep GPT's single primary reason where GPT supplies the final call.
Show Jev's separate criterion rates as diagnostic counts; they can overlap.
Keep fragment-related format changes separate from real new turns and question copying.
If any owner mechanism remains unresolved, state that the reason audit is incomplete.

---

**8. Package the result and stop — 10 minutes**

Save the manifest, label file, fragment audit, agreement tables, rate tables, and report in the new results folder.
Make the report distinguish provisional classification, sample-based owner evidence, and unsupported claims.
Leave naming decisions and downstream exclusion masks unchanged.
**Done check:** A local summary command rebuilds every reported count and interval without model calls or network access.

| Final report item | Required content |
|---|---|
| Main rate table | Six groups; n; owner-plus-GPT count/rate and conditional interval; GPT rate before owner substitutions |
| Raw judge table | Original-text GPT and Jev rates; clearly separate from fragment-adjusted final rates |
| Method checks | Owner substitution counts, raw/derived changes, disagreement counts, and limited Opus comparison |
| Owner evidence | Both agreement tables; weighted fresh-sample rates with intervals and frame assumptions |
| Reasons | Primary reason table, overlapping Jev criteria, and unresolved reasons |
| Status | Gate failed; automatic results exploratory; exact coverage and any missing work |
| Limits | Selected roles/questions; one selected donor per cell; two seeds; no owner labels for four groups |

---

**Stop rule**

Stop at 90 minutes.
If any done check fails, name that gap and report a partial finish.
Do not start another prompt round today.

---

**What must not be done: validation**

- Do not lower the gate, round 0.593 up to 0.6, or drop the owner50 gate after seeing failure.
- Do not tune prompts, thresholds, weights, or judge choice on these 80 labels and claim clean validation.
- Do not select Jev because it wins fresh30 or GPT because it gives a preferred rate.
- Do not use majority voting as evidence that classification errors disappear.

---

**What must not be done: sampling**

- Do not pool the 50 selected examples with the 30 sampled examples as 80 random labels.
- Do not report unweighted fresh-sample prevalence; flagged replies were oversampled.
- Do not infer low failure rates from zero observed errors in seven examples.
- Do not count 10,800 replies as independent observations for a binomial confidence interval.

---

**What must not be done: labels**

- Do not apply a token-limit or repeated-4-gram rule as the new definition.
- Do not call a wrong, odd, rude, brief, or unusual answer Broken for that reason alone.
- Do not edit reply text, silently strip real turns, or change owner labels to fit a judge.
- Do not mix old-definition labels into the new majority or change scoring rules between conditions.

---

**What must not be done: claims and spend**

- Do not call a bootstrap interval, judge range, Noul value, or coherence Score a calibrated truth probability.
- Do not claim that every generated pilot reply or every donor was covered.
- Do not spend more on Opus, rerun Jev unnecessarily, or change the naming decision to fit this analysis.

---

**What I will check at review**

| Review check | Evidence required |
|---|---|
| Definition | Exact owner-approved scope; explicit exclusions for cutoff and harmless repetition |
| Coverage | Canonical key equality; six complete groups; no hidden exclusions or duplicate cells |
| Provenance | Matching input/output hashes; saved settings; explained source-version differences |
| Judge extension | Same GPT model, effort, prompt, and schema; blind batches; no owner-label access |
| Combination | One fixed rule; owner calls preserved; raw judge comparisons use the same text |
| Fragment handling | All 51 audited; only verified suffixes removed; real failures retained |
| Agreement | Original gate still failed; confusion counts and sensitivity shown before owner substitutions |
| Sampling design | Correct fresh strata and frame; no selected-label prevalence claim |
| Uncertainty | Crossed resampling; paired contrasts; classification uncertainty kept separate |
| Reasons | Traceable evidence; totals reconcile; unresolved mechanisms disclosed |
| Claim scope | Provisional automatic rates; limited owner evidence; no unsupported transfer to other groups |
| Completion | Rebuilt local outputs; no validated mask; all remaining gaps named |
