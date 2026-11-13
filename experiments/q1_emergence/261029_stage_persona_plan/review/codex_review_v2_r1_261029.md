**Verdict:** Fix the nine text blockers below before code; the current canary cannot satisfy the full map rules.

---

**Blockers**

Paths below start at `experiments/q1_emergence/`, unless they start with `src/` or `extraction/`.
Aliases: **D** = `261029_stage_persona_plan/DESIGN.md`; **A**, **B**, **C** = the three new experiments' `THRESHOLDS.md` files.
**N** = `261029_instruct_persona_map/e3_noise_check.py`; **NOTE** = that folder's `NOTE.md`.
**JSON** = that folder's `results/e3_noise_check.json`; **E3** = `261008_base_persona_map/`.
Line references to these aliases identify the exact source file.
All nine blockers are [TEXT]. Their proposed fixes keep the owner's numerical thresholds and run counts.

---

**R1. [TEXT] The four-role canary cannot pass the full map test.**

Evidence: D:164 specifies four roles and one replicate. A:101 specifies four roles and five replicates.
A:103 records the owner's five-replicate choice. A:48 requires 15 passing roles; A:57 requires map p < 0.01.
The existing map bootstrap samples roles with replacement and rejects non-finite correlations (E3 `part2_analysis.py:368–390,327–338`).

Failing case: four roles permit only 24 role-label permutations. Even a perfect map has exact p ≥ 1/24 = 0.0417.
Drawing 9,999 permutations repeats those arrangements; it does not create more independent arrangements.
The four-role bootstrap has fewer than three distinct roles in 34.375% of draws.
My small call to the existing bootstrap raised `AnalysisError: non-finite values in map correlations`.

Exact fix: make D:164 match the accepted five replicates and 40 questions.
List the four role IDs, and state whether `default` is additional data for the axis check.
Call this an engineering canary. Require token, seed, reconstruction, storage and synthetic-null checks, not full D1–D3 significance.
Use an exact four-role permutation distribution as a diagnostic; report its minimum attainable p.

Keep full-panel thresholds unchanged. Do not claim this canary confirms precision for the 20-role map.
Record code, prompt and tokenizer hashes before the canary; discard affected rows if templates change afterward.
This also resolves the template-freeze timing conflict between D:84 and B:39.

---

**R2. [TEXT] The noise check calculates sums of squares, not independent noise variance components.**

Evidence: N:73–90 divides each sum of squares by the total observed sum of squares.
The seed term measures deviations from a two-seed prompt mean (N:76–77).
NOTE:29,34–37 uses these results to describe sampling noise and forecast precision.
A:103 uses the noise check as the basis for the accepted run count.

Failing case: I supplied independent Gaussian noise, with no role, question or wording signal.
The function reported seed 0.5008, role 0.0053, question 0.0048, role × question 0.0887 and wording 0.0225.
It assigned the remaining 0.3779 to other terms.
These values are valid descriptive sums of squares; they are not estimates of five independent signal/noise sources.

With two independent replies, the expected within-prompt residual captures half the single-reply sampling variance.
Noise in the two-reply mean remains in the other terms defined at N:78–89.
Thus the 26% and 15% residual shares do not estimate the full sampling variance of a single reply.
The saved values themselves reproduce exactly (JSON:40–46,82–88).

Exact fix: label this table “observed sums-of-squares shares; two-seed residual, not variance components.”
Remove the numerical precision forecast from NOTE:37, or give a separate, explicit variance model and its assumptions.
Keep the observed map correlations as descriptive comparisons, with their recorded split spread.
Keep three replicates and 40 questions; call their adequacy unconfirmed until the full-panel precision check.

The forecast cannot establish the proposed confidence-interval width: split spread is explicitly not a confidence interval (NOTE:11; A:98).
The check does not measure the default axis or Instruct naming precision (N:95–118; A:97–99).
Do not use it as evidence for those targets.

---

**R3. [TEXT] The uncertainty scope does not support the canary's precision claim.**

Evidence: A:23–25 averages replicates first, then resamples roles and questions while retaining all wordings and replicates.
B:25–26 and C:29–31 use the same structure.
A:90–93 says question, wording and sampling noise each receive a measurement.
A:101 proposes estimating all three sources from four canary roles.

Failing case: keep all replies fixed and replace one unusual wording for every role.
The reported role/question bootstrap conditions on that wording panel; it never samples a replacement wording panel.
Likewise, averaging three stochastic replies before this bootstrap does not separately resample their generation variation.
Role/question resampling can absorb some reply noise; it does not separately identify that noise or justify replicate-count forecasts.

Exact fix: state that the current intervals retain the five wordings and use the observed replicate means.
Report wording-held-out and replicate-held-out sensitivity separately; do not call their combined spread a crossed-bootstrap interval.
If unconditional intervals are required, define a hierarchical resampling rule before implementation.
Keep paired checkpoint/format draws, and keep copies of the same original question in one half.

Four selected roles cannot establish uncertainty across the selected 20-role panel.
Keep the accepted run count. If a full-panel precision target fails, report that failure and stop interpretation.
Remove the automatic “add replicates or questions” instruction at A:101, which conflicts with the later owner decision at A:103.
Any later increase then requires an explicit new owner decision.

---

**R4. [TEXT] Pod C D2 tests movement toward the final map, not whether a stage changes the map.**

Evidence: C:62 names the decision “A training stage changes the map.”
C:64–65 tests only a positive change in reliability-normalized agreement with final Instruct.
C:31 requires cross-half comparisons, and C:47 defines the within-checkpoint reliability anchor.

Failing case: two reliable stage maps can differ while having equal agreement with the final map.
My inverse-permutation example gave base–final r = SFT–final r = 0.002198, but base–SFT r = −0.067863.
Both stages can have perfect within-checkpoint reliability.
The stated D2 rule cannot detect this stage change, regardless of sample size.

Also, with identical splits and aggregation, final Instruct has r_norm = rel_Instruct / rel_Instruct = 1.
That endpoint is an algebraic anchor. It is not independent evidence of a developmental trend.
In the noiseless case, any earlier map with final-map r < 1 has a positive final transition by construction.
The size and uncertainty of the difference still matter; this is not a guaranteed significant result on finite data.

Exact fix: rename D2 “A transition increases agreement with the final Instruct map.”
State that negative and zero differences do not establish absence of map change.
Mark the final normalized value as fixed by the definition; report raw agreement and reliability beside every normalized value.
Apply the reliability floor to both checkpoints, in the same frame, source, layer and population, within the resampling calculation.

Keep the accepted positive-difference rule and thresholds. Do not turn it into an arbitrary-change detector silently.
An arbitrary-change decision would need a new owner-approved test.

---

**R5. [TEXT] The text-anchor and topic reading rules make claims the controls cannot support.**

Evidence: D:109 correctly says beating text baselines does not prove information beyond the text.
C:59 nevertheless permits “base holds more than the text.”
C:75 permits “role beyond topic” after beating only an n-gram baseline within the three chosen pairs.
D:110 says topic words alone cannot separate those pairs.

Failing case: chef replies discuss recipes; bartender replies discuss cocktails.
A semantic reader can separate them while an n-gram decoder transfers poorly across held-out questions.
Both within-pair decoding and map agreement can improve without removing output-topic information.
The roles can share a broad subject area and still differ in their actual subject matter.

Exact fix: replace the C:59 success wording with “base exceeds the strongest tested text-anchor agreement.”
Replace its failure wording with “no improvement over the tested text anchors was established.”
Replace “role beyond topic” with “within-pair role decoding exceeds the n-gram baseline.”
Call the selected pairs “related-topic role pairs”; retain all three pairs and all numerical rules.

Neither a pass nor a failure establishes what all information in the text can support.
These changes make C consistent with D:109 without reopening the owner's pairs or thresholds.

---

**R6. [TEXT] The promised axis projection definition is still missing.**

Evidence: D:66 says the projection origin and normalization are fixed in THRESHOLDS.md.
A:63–67 defines an axis and its repeatability, but no projection origin or normalization.
C:77 requests default-text projection onto the final axis without identifying its training and evaluation rows.
A:26 also applies floors to lower bounds, while A:64 states only axis cosine ≥ 0.80.

Failing case: use the evaluated default and role means to define the axis and projection origin.
Then `(default − role_mean) · unit(default − role_mean)` equals the positive axis length.
An implementation can report a positive default projection without any held-out confirmation.
The v1 concern survives despite the new scope label.

Exact fix: define a unit axis from Instruct training-half default minus role mean.
Use that training-half role mean as the origin, in the same plain frame and layer.
Project only the other half's default and role vectors; reverse halves and average.
Keep training origins and axes fixed when comparing another checkpoint, and state the shared-coordinate assumption.

Label the result as a fixed-text coordinate comparison, not stage occupancy.
Clarify that D3 uses the accepted E3 point-cosine gate; report its interval separately.
The stable fixed-default-offset case can still pass D3, as A:65 acknowledges.
That pass is acceptable only for the stated default-versus-role contrast, not for an Assistant Axis claim.

---

**R7. [TEXT] Pod B's variance decision lacks a rule for imbalance and missing cells.**

Evidence: D:93 and B:46 list different terms: B adds format × question.
B:46 specifies sums of squares without a projection formula, fit order or missing-cell rule.
B:34 permits up to 5% invalid replies per role × format cell.
B:47 then compares role and format shares.

Failing case: a role loses all replies for two of 40 questions in one format.
The cell still meets the 95% gate, but the prompt grid is incomplete.
Unweighted fits or sequential sums of squares can allocate correlated role, question and format effects differently.
A naive mean can also leave NaNs in an otherwise accepted run.

Exact fix: make B:46 the single term list and copy it into D:93.
Define grand-mean centering, equal prompt weights, each term's projection, the total denominator and the residual.
Average valid replicates within prompts; record replicate counts.
Use a declared balanced complete-grid sensitivity for the primary variance comparison, with its coverage shown.

If that grid is too small or any factor is absent, return “not estimable”; never impute a role mean.
Specify the minimum support before code, and report the all-valid-row fit separately if it uses different weights.
Add synthetic missingness and unequal-replicate cases to B G6.
Keep the accepted lower-bound difference > 0 rule.

---

**R8. [TEXT] Several Holm families have no defined decision p-value.**

Evidence: A:48 combines two judges, a macro bound and a 15-of-20 role rule; A:69 applies Holm.
B:47 defines only a bootstrap interval for D1; B:67 includes D1 in Holm.
C:65 defines interval rules for three transitions; C:69 includes them in Holm.
B:61–63 also leaves the joint permutation for the mean of six map comparisons unspecified.

Failing case: code computes the stated percentile intervals and pass booleans, then has no p-value for Holm.
Using zero for a positive lower bound, or counting positive bootstrap estimates without specifying the null, changes the test.
For A D1, one macro p-value alone does not specify how the complete multi-part decision enters Holm.
For B D3, independently permuting pair endpoints can destroy the shared-format dependence.

Exact fix: write the statistic, null, tail, decision p-value and aggregation formula for every Holm entry.
For the bootstrap contrasts, specify a null-calibrated paired bootstrap test or a stated test-inversion procedure.
For A, state how the two judges and the coverage condition form the decision; keep the 15-role requirement.
If coverage is a descriptive gate, say so and define Holm on the two-judge macro test explicitly.

For B D3, fix one format's labels and independently permute the other three formats once per null draw.
Use each format's same permutation in all its pairs; recompute the full six-pair mean on the fixed splits.
State centering within format and half, matching A:55.
Check false-positive rates with small synthetic data before fitting the real decoding nulls.

---

**R9. [TEXT] F4 is not yet a matched length and position control.**

Evidence: D:106 promises length- and position-matched versions.
C:22 defines only first-32-token pooling, and C:83 intends to obtain it from the F1 forward pass.
The reused E3 first-token rule uses `min(32, reply_length)` (E3 `context_extract.py:100–103`).

Failing case: a five-token robot reply and a 128-token chef reply yield five-token and 32-token means.
Their pooled lengths and position distributions still differ.
The main all-valid-row rule permits both (C:16,25).

Exact fix: define F4's primary sensitivity on replies with at least 32 kept tokens.
Pool exactly 32 tokens, retain the identical row mask at every checkpoint, and show per-role/question coverage.
Match question IDs and verify the first pooled position across roles within each question.
Compare F1 on that same subset, so row selection is not mistaken for a pooling effect.

If support is inadequate, report the control as unavailable.
Keep all valid replies as the main population; do not pad short replies with neutral text.

---

**Other issues**

**O1. [TEXT] Treat new-grid support as an explicit code acceptance test.**

Evidence: A:73 and B:53 reuse the E3 classifier and splits.
E3 `part2_analysis.py:137–144,147–158,292–299` has old grid defaults and no unique-cell assertion.
D:55 lists “changed grids,” but does not define the expected duplicate, missing-cell and replicate tests.

Failing case: one cell appears twice and another cell is absent, while role/question/wording level counts remain correct.
The old grid guard can accept that data; cell sums then give unequal weight to the duplicate.
Exact fix: derive every expected key from one frozen manifest, including format and replicate IDs.
Reject duplicate keys, distinguish invalid rows from absent rows, and test four-, ten-, twenty-role and default grids offline.

---

**O2. [TEXT] Harden the noise script even though these particular inputs are sound.**

Evidence: N:25–30 checks row count but does not verify the saved hash.
N:35–40 overwrites duplicate cells and checks only for NaNs.
N:22,56–62 uses a global RNG; N:108–118 consumes different half draws for desc and ex2.
N:112,114 uses only one cross-seed direction and one same-half side.

Failing case: my complete five-wording toy grid accepted an extra duplicate and silently replaced its first activation.
Changing condition order also changes which halves each condition receives.
Exact fix: verify hashes and finite values, count each expected cell once, and fail on duplicates.
Build one saved half list, reuse it across conditions, and average both directions and sides for seed comparisons.

These are safeguards, not evidence that the stored JSON is corrupt.
All four actual condition/run grids had exactly 1,800 unique valid cells; both activation hashes matched their metadata.
The relevant recorded hashes are E3 `run_main/reply_mean_acts_slim.json:14` and `run_rep/reply_mean_acts_slim.json:14`.

---

**O3. [TEXT] Pod A's stated reconstruction gate loses E3's scale check.**

Evidence: A:35 lists only centered cosine limits, while calling them the E3 C0 gate.
E3 `context_extract.py:33,136–145` also requires median relative norm error ≤ 0.02.
B:36 and C:40 refer to E3 C0 without restating which parts apply.

Failing case: replace every stored centered activation with ten times that vector, preserving the mean.
The centered cosine stays 1, but activation magnitudes are wrong.
My toy check had median relative norm error 8.918; E3 rejected it.
Exact fix: retain the existing E3 norm limit in all three reconstruction gates and specify checked layers and rows.

---

**O4. [TEXT] A failed positive test is not evidence for the opposite claim.**

Evidence: B:49 says D1 failure means document type contributes more than speaker.
B:57 says D2 failure means vectors do not carry over.
Those rules use lower confidence bounds and significance, not tests of equivalence (B:47,55).

Failing case: role share − format share has estimate +0.02 and interval [−0.01, +0.05].
The rule fails, but the data do not show format dominance.
Exact fix: say “role dominance was not established” and “transfer did not meet the stated criterion.”
Keep the rules and numerical values unchanged.

---

**O5. [TEXT] Finish exact release inputs before allocating the canary.**

Evidence: D:54,112–117,148–150 correctly requires pinned snapshots, storage recovery, installation and smoke checks.
C:23 names stages but leaves their immutable identities to the manifest.
The current loader still loads an unpinned tokenizer (`src/manifold_persona/extract.py:57–70`).
The override path is also unpinned (`extraction/generate_and_extract_roles.py:275–285`).

Failing case: code pins model weights but a later tokenizer revision changes the prompt IDs or template.
Exact fix: require a completed immutable manifest, dependency lock, bounded smoke command and restore command before allocation.
Require the new row adapter to accept pod A's output and reject the old schema clearly.
The old context adapter expects E3 items, replies and slim-state metadata (E3 `context_extract.py:74–97,149–155`).

These are implementation deliverables, not reasons to demand model loading during this read-only review.
I did not verify checkpoint availability, parent lineage, final/RL duplication or GPU behavior.

---

**O6. [TEXT] Scope the F3 gate and freeze its failure rule.**

Evidence: C:21 makes F3 identical across roles within each question; C:50 requires decoding at exactly 5%.
C:47 says unreliable checkpoint maps are not interpreted without explicitly excluding the role-free control.

Failing case: balanced F3 role means are identical. Centering gives zero vectors and undefined cosine-distance maps.
This is the intended negative control, not a broken checkpoint.
Exact fix: apply the map reliability floor to role-bearing frames only.
Keep F3 as a balanced 20-class decoding check, exclude `default`, and specify numerical tolerance or a calibrated leakage test.

---

**O7. [TEXT] The layer convention is consistent; some frame and denominator rules still need one source.**

Evidence: A:17, B:20 and C:24 all define state 17 as a `hidden_states` index, with state 0 as embeddings.
A:18 explicitly excludes `default` from maps and decoding; C:16 includes it in the source corpus.
A:55 centers role vectors; B:61 and C:56 do not repeat that operation explicitly.

Failing case: C code includes `default` in decoding and still reports 5% chance.
Exact fix: state “20 roles, default excluded” for every C map, decoder and F3 check.
Reuse A's within-half centering for every role map, including all text anchors.
Freeze the text feature definitions, vocabulary fit rows and embedding revision with the analysis manifest.

---

**What you verified**

**Disposition of all 13 v1 blockers**

“Fixed in text” means the proposed requirement is now correct. It does not claim that code already implements it.
The design itself says no code changed and no run occurred (D:3).
The source blocker list is `261029_stage_persona_plan/review/codex_review_design_261029.md:7–269`.

| v1 blocker | Assessment of v2 | Evidence and remaining work |
|---|---|---|
| B1: fixed-text specificity | Partly fixed | D:18,100–111 narrows scope and adds controls. R5 and R9 remain. |
| B2: map calibration and nulls | Partly fixed | D:121–128 fixes one-map nulls and cross-halves. R4 and R8 remain. |
| B3: axis and occupancy | Partly fixed | A:63–67 admits the fixed-offset case; D:111 removes occupancy. R6 remains. |
| B4: naming versus fidelity | Fixed in text for current claims | D:26–34 separates labels and postpones validated fidelity. A:9–10,77 keeps current claims narrower. |
| B5: format, topic and variance | Partly fixed | D:78–94 and B:71–80 define formats and expose limits. R5 and R7 remain. |
| B6: question and sampling seeds | Fixed in text | D:51–53 freezes questions and item keys. A:34 permits kernel-dependent batch differences without changing keys. |
| B7: uncertainty and repeat count | Partly fixed | D:130–137 narrows population and bootstrap scope. R1–R3 remain. |
| B8: wrong pooled tokens | Fixed in text | D:49–55 requires kept IDs, NaNs and edge cases. O3 strengthens reconstruction. |
| B9: omitted sampling/ID plumbing | Fixed in text | D:49,53,63 includes full settings, IDs and both generation paths. |
| B10: adapter and integrity | Correct frame/gate design; adapter still required | D:101,107,117 and C:38–40 fix the cross-frame gate. O5 names the required adapter acceptance check. |
| B11: old analysis grids | Not fully fixed | A:73 and B:53 retain E3 reuse. O1 makes the new manifest/grid requirements explicit. |
| B12: immutable checkpoint/tokenizer identity | Fixed as a pre-run requirement | D:54,112–115 and C:39 require identities and compatibility. O5 tracks the remaining implementation work. |
| B13: installation and recovery | Fixed as a pre-run requirement | D:117,148–150 adds a lock, atomic shards and verified recovery. O5 requires concrete commands before allocation. |

---

**Offline evidence checks**

I used `.venv/bin/python -B`, local JSON, local arrays and small synthetic arrays.
I imported the noise functions without calling `main()`, because `main()` writes the results file (N:119–126).
No model weights, GPU, model API or network service were used for the review checks.

| Check | Result |
|---|---|
| Saved activation hashes | Both slim arrays matched their recorded SHA-256 values. |
| Actual grids | desc and ex2, each run: 1,800 rows, 1,800 unique keys, all valid. |
| Saved noise output | Every reported desc and ex2 field reproduced exactly after rounding. |
| Halves | N:58–62 uses 9/9 questions and complementary 2/3 wordings. These match NOTE:11. |
| Map statistic | N:44–53 centers role means, normalizes them and compares upper-triangle cosine distances with Spearman r. |
| Variance arithmetic | N:73–90 forms an additive descriptive sums-of-squares partition on these complete grids. R2 explains its inferential limit. |
| Pure independent noise | Seed residual share 0.5008; apparent nonzero role/question/wording shares. |
| Duplicate toy grid | Accepted and overwritten; O2 gives the fix. |
| Four-role exact null | Minimum p = 0.0416667 for a perfectly matching generic map. |
| Four-role E3 bootstrap | Raised a non-finite map-correlation error. |
| Stage-change counterexample | Equal final-map agreement despite different stage maps. |
| Reconstruction scale case | Cosine 1.0; existing E3 norm check rejected the scale error. |

The NOTE's rounded correlation and share tables agree with JSON:8–46,50–88.
The descriptive second-seed improvement follows: desc 0.752/0.803 becomes 0.836; ex2 0.966/0.956 becomes 0.971.
The strong claim that another seed adds “no noise,” and the precision forecast, need the limits in R2.
Those are conclusions about two saved base runs, not proof about Instruct or additional seeds (NOTE:34–38).

---

**Next step**

Apply R1–R9 to the design text, then implement the shared row, token, seed and analysis rules.
Run the bounded offline checks before allocating RunPod; use the four-role canary for engineering evidence only.
Keep all accepted owner choices and threshold values.
