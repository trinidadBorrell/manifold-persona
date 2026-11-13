# Challenge of the pod C full-run result (30/10/2026)

Reviewer: Claude session 48adb487 (agent), started by session 681e31cc. **This is an agent review, not an independent human review.** Human review: none.

Method: `/challenge-result`, then the `presenting-results` checklist. I read `NOTE.md`, `THRESHOLDS.md` v1.7, `analysis.py`, `maps.py`, the pod C section of `../261029_stage_persona_plan/DESIGN.md`, `output/independent_review_2026-10-02/review.txt` and the pod C row of the vault `Reference/Experiment index.md`. I did not edit pod C code, rules or result files. No pod run, no judge run, no commit.

## Start here

| | |
|---|---|
| Short answer | D1 passes as reported, and every number I checked matches the HF results file. But D1 is a weak test here: a text-only reader (character n-grams) also clears the D1 floor. The "flat across stages" reading has no calibrated test behind it. |
| What holds | Base and Instruct give almost the same 20-role map when they read the same Instruct replies (r = 0.896 [0.767, 0.932]). The F3 role-free control is clean on pod A text. |
| What does not hold yet | "Training stages do not change the map" (no test for no change, no power check). "Model-general" (E3 `desc` fails the reliability floor). "More than the text": the gap is small (0.050), and about a third of it is noise in the text maps (anchor check). |
| Status | Exploratory. Candidate for human review only in the narrow wording below. |

---

## Evidence source

| Item | Value |
|---|---|
| Results file | HF `brend0nc/manifold-persona-experiment-artifacts` @ `c6755d666e0bcc207eecd8af37da0e18c305c577`, `experiments/q1_emergence/261029_fixed_text_read_through/run_full/analysis_full/results_full.json` |
| Hash check | HF file sha256 `16391870…43abb` = `records_full/analysis_full/results_full.json` = `SHA256SUMS.txt` entry (I downloaded it on 30/10 and compared). |
| Code in the file | `analysis.py` `a4275cfc…`, `maps.py` `718f4113…`, `frames.py` `ce906373…` (same as NOTE.md). |
| States | 10 checkpoint folders under `run_full/`, revisions in `NOTE.md` "Full run". The results file stores `hf_revision: null` for each checkpoint; only the receipts in `records_full/` bind states to HF revisions. |

---

## Claim ledger

| # | Claim text (as in NOTE.md / vault) | Source key in `results_full.json` | Population and method | Control and baseline | Contrary evidence | Open alternatives | Missing checks | Status | Allowed wording | Human review |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | D1 passes: base vs Instruct r = 0.896 [0.767, 0.932], p = 0.0001, both reliability lower bounds 0.77. | `pod_a.D1.r`, `.p`, `.reliability`, `.passed` | Pod A text, 21,000 rows, 20 roles (default excluded), F1 frame, hidden state 17; Spearman cross-half map r over 50 fixed halves; 2,000 paired draws; 9,999-permutation one-map null. | F3 role-free control clean (max relative difference 0, decoding 0.050 at all 10 checkpoints). Reliabilities base 0.900 [0.772, 0.937], Instruct 0.899 [0.774, 0.933]. | Text-only maps also clear the D1 floor: char n-grams vs Instruct 0.846 [0.696, 0.902], word n-grams 0.821 [0.660, 0.882], state 0 0.826 [0.595, 0.897]. | The role map is a property of the text, and any reader of it recovers the map. Base and Instruct share their pretraining weights. | No reader from another model family on the same text. | candidate for human review (narrow) | "When OLMo-2-7B base and Instruct read the same Instruct-written replies (F1, state 17), their 20-role maps agree (r = 0.896 [0.767, 0.932]). Text-only maps of the same replies agree with the Instruct map almost as well (char n-grams 0.846)." | none |
| 2 | "Base exceeds the strongest tested text-anchor agreement": 0.050 [0.014, 0.103]. | `pod_a.D1.minus_best_anchor`, `.text_anchors` | Same panel; per draw, base r minus the best of 4 anchors (word, char, embedding, Instruct state 0). The per-draw maximum is a fair choice. | Four text anchors. | Role decoding from states does not beat word n-grams (claim 4). E3 `desc`: not established (−0.010 lower bound). | The anchors are noisier maps, so their cross-half r is capped lower. A gap in raw r can come from anchor noise, not from extra role content. | Anchor self-reliability is computed in the panel but not saved. My Mac check gives 0.85–0.86 (see "Anchor check"); no paired interval for the normalized gap. | narrowed | "Base agrees with the Instruct map slightly more than the four tested text readers do (difference 0.050 [0.014, 0.103]). About a third of that gap is noise in the text maps (anchor check). This does not show information beyond the text." | none |
| 3 | "The map is flat across stages"; "the read-through map is fixed before SFT, so in this design training stages do not change it measurably". | `pod_a.D2`, `.stage_change`, `.D2_question_only_halves`, `.rl_steps`, `g5.equal_map_wording_case_recorded` | r_norm to Instruct: base 0.996 [0.991, 0.999], SFT 0.999, DPO 1.000; stage change 1 − r_norm: base→SFT 0.003 [0.001, 0.008], SFT→DPO 0.001 [0.000, 0.002], DPO→Instruct 0.000. | G5 equal-map case: a pure wording bias gives a D2 difference of 0.027 [0.012, 0.039], larger than every observed difference. | The state check finds base and SFT states differ by a median 31% at state 17, but the map does not move. The map measure (rank order of 190 role distances) can miss a change that keeps the order. | The measure is near its ceiling and insensitive, so it cannot show a change at this size. | D2 is report only by rule. No equivalence test, no power check with a planted map change. | exploratory | "I found no measurable change in the fixed-text role map between base, SFT, DPO, Instruct and the RL steps under this measure. The design has no calibrated test for no change." | none |
| 4 | Role decoding from states does not beat word n-grams. | `pod_a.decoding` | 20-class decoding, 50 splits, held-out cells. Base 0.761 [0.678, 0.838], Instruct 0.754. | Word n-grams 0.797, char 0.774, state 0 0.757, embedding 0.539. Gain base −0.036 [−0.070, 0.000], Instruct −0.043 [−0.078, −0.006]. | None against it. | None. | None. | candidate for human review | "Role decoding from base and Instruct states (0.76, chance 0.05) does not beat word n-grams of the same replies (0.80)." | none |
| 5 | Same comparison in other frames and layers: F2 0.889, state 9 0.888, state 23 0.889, F4 0.871, F1 on the F4 rows 0.899. | `pod_a.F2`, `.state_9`, `.state_23`, `.F4` | Same rows and draws. | Same as claim 1. | None. | These are sensitivities on the same data, not independent replications. | None. | candidate for human review | Report as same-data sensitivities, with their intervals (F2 [0.756, 0.931], state 9 [0.744, 0.930], state 23 [0.748, 0.931], F4 [0.714, 0.914]). | none |
| 6 | E3 text: `ex2` D1 r = 0.942 [0.753, 0.940]; `desc` not interpreted. | `e3_ex2.D1_report`, `e3_desc.D1_report` | E3 base replies, 18 questions, 1,800 rows each. | `desc` reliabilities 0.65 [0.19, 0.69] and 0.66 [0.19, 0.70]: below the floor. | `desc` fails the floor, so the two sources do not agree. | — | Interval coverage: `ex2` point lies above its interval; `desc` draw shift is −0.20. | narrowed | "On E3 `ex2` replies the result repeats (0.942). On E3 `desc` replies the map is too unreliable to read. A model-general claim needs both sources (THRESHOLDS) and is not supported." | none |
| 7 | Related pairs: counselor vs consultant above word n-grams at every checkpoint (+0.05). | `pod_a.related_pairs` | 2-way decoding per checkpoint. Base +0.052 [0.023, 0.081]. | Word n-grams on the same pair. | Chef vs bartender: states lose to n-grams (base −0.034 [−0.052, −0.017]). Detective vs criminal: not established. | The pair can differ in subject matter. | 3 pairs × 10 checkpoints = 30 report-only intervals, no correction. | narrowed | "Within one of three related pairs (counselor vs consultant), role decoding exceeds the n-gram baseline; in one pair n-grams win." | none |
| 8 | 4 components for 70% of the role-mean variance at every checkpoint. | `pod_a.components_70` | 20 role means, state 17. | None. | None. | None. | None. | candidate for human review | As stated. | none |

---

## Anchor check (30/10, Mac, $0)

Question: is the text anchors' lower agreement only noise in the anchor maps? I computed the anchors' own cross-half reliability with the pod C code (`maps.panel`, same 50 halves, 200 draws), from the frames and the HF embedding cache.

| Map | Own reliability (cross-half r) | r with the Instruct map (results file) | Normalized, r / sqrt(rel × rel_Instruct) |
|---|---|---|---|
| Base states | 0.900 [0.772, 0.937] | 0.896 | 0.996 |
| Char n-grams | 0.864 [0.695, 0.921] | 0.846 | 0.960 |
| Word n-grams | 0.851 [0.683, 0.903] | 0.821 | 0.939 |
| Embedding | 0.852 [0.696, 0.887] | 0.683 | 0.780 |
| State 0 | not computed (needs Instruct state 0 from HF) | 0.826 | — |

- The text maps are less reliable than the model maps (0.85–0.86 against 0.90). This explains part of the 0.050 gap: after the same correction, the gap to char n-grams is about 0.036.
- A gap remains in the point values, so the reading "base agrees a little more than the tested text readers" survives as a point. I have no paired interval for the normalized gap: my 200 draws are not the pod's 2,000 paired draws.
- Word and char n-gram maps agree with each other at 0.838; char and embedding at 0.654.
- Intervals in this table are plain percentile intervals over 200 draws; normalized values are points only.

Script and output: `/private/tmp/claude-501/-Users-brendon-persona-manifolds-manifold-persona/48adb487-2032-4b45-85ad-2b5d46f5116f/scratchpad/anchor_rel.py` and `anchor_rel.out` (scratch, not kept). The pod C panel already computes these values (`extra` in `analyse`); the fix is to save them in the results file.

---

## Method checks

| Check | Finding | Effect on the claim |
|---|---|---|
| Numbers against the file | Every number in claims 1–8 matches `results_full.json` at HF `c6755d6`. The canary D1 point (0.896) is the same, as expected: same models, same text. | The canary is not an independent replication. |
| F3 limit change (1e-3 → 1e-2, v1.6) | Made after the canary, before the full run, with the owner's go on the numbers. Pod A F3 max relative difference is 0.0 at all 10 checkpoints, so pod A passes the old limit too: D1 does not depend on the change. E3 F3 is 0.0057–0.0071 at all 10 checkpoints, so every E3 result exists only under the new limit. F3 decoding stays at chance (0.050–0.051). | D1 unaffected. E3 results are exploratory under a revised limit. |
| E3 F3 counted twice | E3 `desc` and `ex2` share one F3 frame (same digest `ddf28676…`), so the two F3 passes are one check. | Minor; say "one E3 F3 check". |
| Interval | Plain percentile interval; draws sit below the point (pod A D1 shift −0.020, E3 `desc` −0.20, E3 `ex2` point above its upper bound). No coverage check. | For D1 the shift makes the lower bound stricter, so the pass stands. The E3 `desc` and `ex2` intervals are not calibrated 95% intervals. |
| Multiple tests | D1 is the only pass rule, so no correction is needed. The file holds many report-only intervals; a few exclude zero (base→SFT change, counselor pair, E3 `ex2` anchor gap). | None of these is a finding. Do not select them. |
| G5 | Pass rule checks identical, rotated, scaled and one independent case. THRESHOLDS also lists "uniform p under the one-map null" and "the cross-half design removes planted same-row noise"; the code checks one null p (0.387) and has no planted-noise case. No case plants a map change to test detection. | Minor for D1. Major for any "no change" reading (claim 3). |
| Scope | Pod C measures how a checkpoint represents roles while it reads fixed Instruct text. It does not measure role play, the default a stage writes, or a persona in the model (THRESHOLDS "Question and scope"). | Any claim about persona formation or emergence across stages is out of scope. |
| Independent review 02/10 | F3 of that review asked for identical replies at every checkpoint; pod C does that, so the old confound (each stage writes its own reply) is gone for this measure. That review also asks to validate role expression on the same prompts; pod C has no judged subset (no approved judge panel). F2 asks for matched text baselines; pod C has them, and they nearly match the states. | Pod C answers the read-through question only. Persona formation stays unidentified. |
| Raw read | The frame read (`frames_full/frame_read.md`, 10 rows per frame) covers the full run, since frames were built once. Pod A did its own raw read of the replies. Translator rows are in Russian and French. | No gap for D1. |
| Records | NOTE.md status line names both v1.7 and v1.5. NOTE.md says `provenance.hf_revision` is "local"; the file has `null`. | Text fixes only. |

---

## Missing checks

| Check | Why it matters | Cost |
|---|---|---|
| Save the anchor self-reliabilities (and anchor r_norm) in the results file | Decides whether claim 2 means more than anchor noise. | $0, CPU, a code change by session 76e7aa2b. |
| A reader from another model family on the same text (for example a 7B base model of another family) | Tells whether 0.896 is special to OLMo or holds for any language model reading these replies. | One forward-pass checkpoint on a pod; needs the owner's go. |
| A planted-change G5 case for the stage measures | Shows what size of map change D2 and the stage change can detect. Without it, "flat" cannot be read as "no change". | $0, CPU. |
| Identifiable subset | Needs the approved judge panel (pod A D1). | Judge cost; owner go. |
| Axis | Needs pod A D3. | — |
| Human review | Required before any external claim. | Owner. |

---

## What the result supports

- Supported (narrow): when base and Instruct read the same Instruct-written replies, they lay out the 20 roles almost the same way, and a text-only reader recovers most of that layout.
- Supported: role decoding from the states does not beat word n-grams.
- Not supported: that base "holds a persona" or "plays roles"; that training stages leave the map unchanged; that the result is model-general (E3 `desc` fails); that the states hold role information beyond the text.

---

## Fixes sent to session 76e7aa2b

1. Save the anchor self-reliabilities `(a, a)` and the anchor r_norm in `results_full.json` (they are computed but dropped).
2. NOTE.md "What this does and does not show": replace "The read-through map is fixed before SFT, so in this design training stages do not change it measurably" with the claim 3 wording; replace "a bit more than the text alone predicts" with the claim 2 wording; add that a char n-gram map also clears the D1 floor.
3. NOTE.md status line: one rules version (v1.7). Say `hf_revision` is `null`, not "local".
4. Say that E3 F3 is one check (shared frame) and that E3 passes only under the 1e-2 limit.
5. Add a planted-change G5 case before any reading of D2 or the stage change.

---

## Addendum: re-run with the fixes (30/10/2026)

Owner go: "ok go ahead and do that". Session 76e7aa2b ran the new analysis (`analysis.py` `f6586f5d…`) on the A/B pod from HF copies. Results: HF @ `b343c96522365de638d7971d0a6969c294b4d41f`, `run_full/analysis_full_v2/results_full_v2.json`. I downloaded it: sha256 `1f91b354…a1c7` matches `records_full/analysis_full_v2/`. All 413 point values present in the first file are unchanged. `provenance.hf_revision` is now filled for all 10 checkpoints.

| Measure | Value | Effect on my ledger |
|---|---|---|
| Anchor reliability | word 0.851, char 0.864, embedding 0.852, state 0 0.879 | Confirms my Mac check (same points for word, char, embedding). |
| Anchor r_norm to Instruct | char 0.960 [0.912, 0.982], word 0.939, state 0 0.929, embedding 0.780; base 0.996 | — |
| Base minus best anchor, normalized | 0.037 [0.013, 0.081] (raw 0.050 [0.014, 0.103]) | Claim 2: the gap survives the noise correction. It stays small. |
| G5 null | mean p 0.585, share below 0.1 = 0.067 (30 pairs) | The null case now passes. |
| G5 shared same-row noise | cross-half r 0.036 | The planted-noise case now passes. |
| Planted change, all roles shifted (true map r 0.93) | change 0.071 [0.022, 0.145] | Detected. |
| Planted change, 2 of 20 roles moved (true map r 0.83) | change 0.156 [−0.006, 0.340] | Not detected. |
| Planted change, 5 of 20 roles moved (true map r 0.60) | change 0.362 [0.105, 0.697] | Detected. |

Updated wording:

- Claim 2 (still narrowed): "Base agrees with the Instruct map a little more than the four tested text readers, also after correction for map noise (0.037 [0.013, 0.081]). This does not show what all the information in the text can support."
- Claim 3 (still exploratory): "The observed stage changes (at most 0.003) are far below the change from a planted shift of all roles (0.071). A change in a few roles can go undetected (2 of 20 roles: interval includes 0)." The planted cases use synthetic noise, not the real data's noise, so their detection sizes only give a rough scale.

---

## Prior work: Lu et al. 2026, "The Assistant Axis" (arXiv 2601.10387v1)

I read the whole paper (`docs/papers/2601.10387.pdf`; text copy `output/lit_q1_2026-10-05/2601.10387.txt`) on 30/10.

| Point | Lu et al. | Pod C |
|---|---|---|
| Base reads Instruct text | Section 2.1.3 and App. B.3: Gemma 2 27B base reads the Instruct model's role rollouts. Top 3 role PCs match Instruct at cosine 0.93, 0.87, 0.83. Each role vector matches its Instruct vector at cosine 0.99. | OLMo-2-7B base reads pod A's Instruct replies. Centred 20-role map r = 0.896 [0.767, 0.932]. |
| Text used | Judge-filtered rollouts (only "fully" or "somewhat" role-playing replies), 275 roles, 5 system prompts × 240 questions. | All valid replies, 20 roles; no judge filter (no approved panel yet). |
| Interpretation | "These axes of persona differentiation … are likely already present in base models and inherited from the pre-training corpus." | Text-only maps recover most of the same map (char n-grams r_norm 0.960 against base 0.996). |
| Text baselines | None. | Word and char n-grams, sentence embedding, state 0. |
| Stages between base and Instruct | None. | SFT, DPO, 6 RL steps. |
| Noise, null, controls | None reported for B.3. | Split-half reliability, one-map null, F3 role-free control, bootstrap intervals. |
| Base behaviour | Section 3.2.2: steering base with the Instruct axis shifts "My job is to" completions toward helpful human roles. | No behavioural test (pods B and E3 cover base writing). |

What this changes:

- D1 replicates Lu et al. B.3 on a second model family and size. It is not a new finding.
- Their per-role cosine of 0.99 is on raw vectors, so the shared mean direction dominates it. Pod C centres the role vectors, which is a stricter comparison.
- Their B.3 cannot tell "base learned the map in pretraining" from "the text carries the map". Pod C's text anchors show that the text carries most of it. This qualifies their interpretation, and it is pod C's new contribution. Without our own human review, it stays exploratory.
- Their Section 8.2 proposes tracking how training data shifts persona dimensions. The pod C stage sweep is one such test, for fixed text only.
