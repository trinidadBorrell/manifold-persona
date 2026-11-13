# Pod C `fixed_text_read_through`: measures, pass rules and gates

Status: **v1.7, 30/10/2026. Owner accepted the proposed values; v1.6 and v1.7 change the F3 equality limit and the bootstrap centring after the engineering canary, before the full run ("Changes").** No code, no run. Design: `../261029_stage_persona_plan/DESIGN.md` (pod C). Fixed before the run; later changes get a dated line in "Changes".

Three analyses on one job: `base_reads_instruct_text` (D1), `checkpoint_sweep` (D2), `checkpoint_calibration` (gates and anchors).

## Question and scope

When every checkpoint reads the same Instruct-written replies, does it hold the final model's role map? Where along base → SFT → DPO → RL does the map change?

- Claim scope: **fixed-text role representation**: how a checkpoint represents roles while reading supplied text. Every claim names the text source, frame, layer and population.
- Not claimed: role play by a checkpoint, where a stage's own default sits, or a persona in the model. The default text is "fixed Instruct default text".

## Data

- Text source 1 (main): pod A replies, 20 roles plus `default`, 5 wordings, 40 questions, 5 replicates (21,000 rows). All valid rows; the identifiable subset (pod A D1 judges) with one fixed row mask is reported beside it. That subset is defined only on pod A's approved judge panel: rows outside it are marked unjudged, never counted as misses, and the subset is compared with an unfiltered baseline on the same judged panel [round 2, B5]. It must pass the same half-support checks as F4, on the fixed halves and on every bootstrap draw.
- Every map, decoder and the F3 check use the 20 roles, `default` excluded (chance 5%). `default` is used only for the axis [O7].
- Text source 2: E3 base replies (`run_main`, `desc` and `ex2`, 20 roles, 18 shared questions), reply ids only.
- Frames, built once, ids and pooling indices stored and fed unchanged to every checkpoint:
  - F1 (main): `<|endoftext|>`, the E3 interview header, the question, `Guest:`, the reply. No description.
  - F2: `<|endoftext|>` and the reply only.
  - F3 (role-free control): F1 with one fixed neutral reply in place of every reply. F3 is built on the complete frozen role × question × wording grid, including rows whose generated reply is invalid or unjudged, so every role has the same questions and wordings [round 3, B2].
  - F4 (length control) [R9]: only replies with at least 32 kept tokens, pooled over exactly their first 32, with the same row mask at every checkpoint. F1 is reported on the same subset, so a row-selection effect is not read as a pooling effect. Coverage per role and question is reported; if any role keeps fewer than half of its rows (owner, 29/10), F4 is reported as unavailable. Short replies are never padded.
  - Half support [round 2, B4]: the F4 row mask is built and checked against every fixed half before any checkpoint is extracted. In each half, the map uses only the questions where every role has at least one F4 row (the half's common question set). If any half of any fixed split has no common question, F4 is unavailable. Roles are not dropped, halves are not replaced, replies are not padded. F1 is reported on the exact same rows.
  - The same support check runs on every paired bootstrap draw before any map statistic. If any draw has a half with no common question, that F4 interval, its normalized comparison and its outcome are unavailable. Draws are not deleted [round 3, O1].
- Checkpoints (owner, 29/10): base `main` (`7df9a82`), SFT, DPO, Instruct, Instruct RL `step_60` to `step_360`. 10 entries in a manifest with immutable revisions, parent, stage and step.
- Primary layer: hidden state 17 (`hidden_states` index; 0 is the embedding output). Stored: states 9, 17 and 23 for all checkpoints; all 33 for base and Instruct.
- Pooling: mean over kept reply tokens.

## Inference units

- As pod A: role × question cells, wordings nested in role, replicates nested in prompt and averaged first.
- Checkpoints read the same rows, so every checkpoint contrast is paired: the bootstrap resamples roles and questions once per draw and uses the same draw for every checkpoint. 2,000 draws.
- Halves split questions and wordings together. A cross-checkpoint comparison always puts checkpoint A on one half and checkpoint B on the other half, then the reverse, and averages (review B2).
- Every role map, including the text-anchor maps, centres role vectors within half, as pod A D2 [O7].
- Scope of the intervals [R3]: they keep the five wordings fixed and use the observed replicate means. Wording-held-out and replicate-held-out results are separate sensitivities.
- Bootstrap p-values [R8]: for a one-sided rule "lower bound > 0", the decision p is the share of paired bootstrap draws with the statistic ≤ 0.
- Floors apply to the 95% lower bound.

## Integrity gates (all must pass before any metric)

| Gate | Rule |
|---|---|
| G1 same input | Per row and frame, the hash of ids, mask and positions is identical at every checkpoint. |
| G2 manifest | 10 entries resolve to immutable revisions; tokenizer files present; vocabularies equal; record whether `step_360` and Instruct have identical weights. |
| G3 reconstruction | At Instruct, the stored pod A chat-frame ids reproduce pod A's stored states at hidden states 9, 17 and 23, on every valid row, with centring over valid rows only: centred cosine min ≥ 0.95, median ≥ 0.99, median relative norm difference ≤ 0.02 (the full E3 C0 gate) [O3]. Invalid rows are checked separately: invalid flag set, empty pooling span, NaN vector [round 2, O-A]. Run once; it is not a cross-checkpoint test. |
| G4 storage | Atomic shards, one namespace per checkpoint, refusal of a mismatched resume; each checkpoint is copied off the pod and hash-checked before the next one loads. |
| G5 synthetic | Before real data: identical, rotated and scaled maps give map r ≈ 1; independent maps give r ≈ 0 and uniform p under the one-map null; the cross-half design removes planted same-row noise. Also recorded, not a pass condition: equal complete role maps with checkpoint-specific wording sensitivity (role vectors S ± 0.2·k_w·E, k = (−3, −3, 2, 2, 2)); record r_norm, the D2 difference and the stage-to-stage value, which show the known bias [round 2, B3]. |
| Frame read | 10 rendered rows per frame printed and read before any metric. |

## Calibration (gates on interpretation)

- **Same anchor:** each checkpoint's within-checkpoint cross-half map r, in role-bearing frames only (F1, F2, F4) [O6]. A checkpoint whose lower bound is below 0.5 is not interpreted. Any comparison of two checkpoints applies this floor to both, in the same frame, text source, layer and row population [R4]. Each reliability lower bound comes from all paired draws. If either is below 0.5, the normalized comparison is unavailable and raw r is reported. Otherwise every paired draw is kept for the interval and the p-value; no draw is removed because of its reliability value. If any draw has a reliability ≤ 0, the normalized interval is unavailable [round 2, B2].
- **Text anchor:** maps built from the same rows by word and character n-grams, a sentence-embedding model and hidden state 0. Their agreement with the final map is reported next to every checkpoint's agreement. The text features are frozen in the analysis manifest before any real data: n-gram ranges, vocabulary fitted on training rows only, and the embedding model with its revision [O7].
- **Null:** permute the role identities of one map only, keeping wording blocks (9,999).
- **F3 check** [O6]: F3 gives every role the same input within a question, so F3 role means are identical and F3 has no map. The check: within each question, F3 pooled states are equal across roles (max relative difference ≤ 1e-2, for batch numerics; 1e-3 up to v1.5), and K-class role decoding on F3 is at chance, 1/K ± 0.5 points, where K is the number of non-default roles in the frozen manifest (full run K = 20, 5%; engineering canary K = 4, 25%) [round 2, B1]. Every role has the same question and wording weights in each train and test fold; balance is asserted before fitting. An unbalanced F3 grid is reported as an invalid control, not as role leakage [round 3, B2]. If either fails, the frame or the row order leaks the role and the run stops.

## Primary decisions

**D1. Base holds the Instruct map when it reads Instruct text (`base_reads_instruct_text`).**

- Measure: cross-half map r between base and Instruct, frame F1, state 17.
- Pass: lower bound ≥ 0.5 and one-map null p < 0.01.
- Reason: the same floor as pods A and B. Lu et al. (App. B.3) report top-3 PC cosines of 0.93, 0.87 and 0.83 for Gemma 2 27B base reading Instruct text.
- Decision p [R8]: the one-map permutation p, p = (1 + #null ≥ observed) / (1 + 9,999), on the cross-half r averaged over the 50 fixed halves.
- Reading rule [R5]: if the base–Instruct r exceeds the best text-anchor r (paired lower bound of the difference > 0), say "base exceeds the strongest tested text-anchor agreement". Otherwise say "no improvement over the tested text anchors was established". Neither result shows what all the information in the text can support.
- Also reported: F1 role decoding at base (chance 5%) and its gain over the text baselines.

**D2. Agreement with the final Instruct map across stages (`checkpoint_sweep`): report only, no pass rule** (owner, 29/10) [R4, round 3 B1].

- Measure: agreement with the final model, normalized by reliability, r_norm(c) = r(c, Instruct) / sqrt(rel_c × rel_Instruct), at base, SFT, DPO and Instruct, under the same-anchor rule (computed on all paired draws, or unavailable). Raw r and both reliabilities are reported beside every r_norm.
- Reported for each transition (base → SFT, SFT → DPO, DPO → Instruct): the paired difference in r_norm and in raw r, each with its 95% interval. No transition receives a pass from this statistic; checkpoint-specific wording sensitivity can cause positive differences (Codex round 3, B1: with identical complete maps, base → SFT gave an interval of 0.032 to 0.077 and bootstrap p 0/2,000). A claim that a stage moves the map needs a separately specified and calibrated test, approved by the owner.
- Reason: the reliability correction puts checkpoints with different split-half noise on one scale; pairing removes shared text noise.
- Limit of the correction [round 2, B3]: halves split a fixed panel of five wordings. A checkpoint whose map moves with the wordings can get a higher r_norm even when every complete role map is the same (G5 case; offline check: r_norm 1.00 at a checkpoint with no wording sensitivity, 1.05 at one with it). The correction does not remove checkpoint-specific wording sensitivity.
- Reported beside D2: r_norm with question-only halves (all five wordings in both halves, so wording effects count as part of the map). A difference that is absent there is flagged as possibly wording-driven.
- G5 keeps the equal-map, unequal-wording-sensitivity case as a required regression case: the analysis code must never label a transition as passed.
- Limits [R4]: r_norm(Instruct) = 1 by definition, so the last point is an anchor, not evidence of a trend. This rule tests movement toward the final map only. A zero or negative difference does not show that a stage left the map unchanged: two stages can differ from each other and still agree equally with the final map. The report-only stage-to-stage change measure ("Reported, no pass rule") covers any change between stages.
- The RL steps are a trajectory: reported, no pass rule. They are calibration points, not independent evidence.

**Multiple tests:** D1 is pod C's only primary decision, so no Holm correction is needed. D2 and every other result are report only.

## Reported, no pass rule

- **Stage-to-stage change** (owner, 29/10; report only). For each transition (base → SFT, SFT → DPO, DPO → Instruct), in frame F1 at state 17:
  - the cross-half map r between the two stages, r(c1, c2), with the same halves and both directions as D2;
  - its reliability-normalized form r_norm(c1, c2) = r(c1, c2) / sqrt(rel_c1 × rel_c2), under the same-anchor rule (both reliability lower bounds ≥ 0.5, then all paired draws); raw r and both reliabilities beside it;
  - the change 1 − r_norm(c1, c2) with its 95% interval from the same paired draws as D2. Values are not clipped; r_norm can exceed 1 by noise.
  - Reading [round 2, B3]: the value is a normalized cross-half comparison, conditional on the fixed five-wording panel. It does not separate map change from noise: checkpoint-specific wording sensitivity can move it either way when the complete role maps are identical (G5 case; Codex offline check: change −0.26, interval −0.41 to −0.17). An interval wholly below 0 flags a normalization problem, not a change. No claim of "change beyond noise" is made from it; that needs a separate, calibrated test.
  - Limit: Spearman map r uses the rank order of role distances, so a change that keeps that order is not detected. CKA between the two stages is reported beside it.
  - Adjacent RL steps are reported the same way, as a trajectory.
- F2 and F4 versions of D1 and D2 (frame and length sensitivity).
- Text source 2 (E3 base replies): D1 and the r_norm trajectory. A "model-general" claim needs both sources to agree.
- Related-topic role pairs (chef vs bartender, detective vs criminal, counselor vs consultant): 2-way decoding per checkpoint against the n-gram baseline on the same pair. Where the gain has a lower bound > 0, read it only as "within-pair role decoding exceeds the n-gram baseline" [R5]. The roles in a pair share a broad subject area but can still differ in subject matter.
- Number of components for 70% of the variance of the 20 role means (rank ≤ 19).
- Axis, only if pod A D3 passed: default minus role mean at each checkpoint (F1), cross-half cosine to the Instruct axis. The F1 axis is estimated and checked in F1 itself, at the same layer and row population: its training-half norm must be finite and nonzero in both directions, or the axis projection is unavailable; its F1 split-half cosine is reported beside the projection. Pod A's D3 pass is in the chat frame and does not carry over to F1 [round 2, O-C]. Also reported: the projection of the fixed Instruct default text onto the final axis. Projection uses pod A's rule [R6]: Instruct training-half origin and unit axis, held fixed for every checkpoint; only the other half's vectors are projected; halves swapped and averaged. Each result states the shared-coordinate assumption and is a fixed-text coordinate comparison, not stage occupancy.
- CKA as a second map-comparison measure.
- Layers 9 and 23.

## Cost and approval

- GPU: forward passes only. About 21,000 rows × 3 passes (F1 with F4, F2, F3) × 10 checkpoints, plus the E3 rows. Benchmark one checkpoint first.
- Storage: about 2 GiB per checkpoint for 3 states (4 pooled frames); about 21 GiB more for all 33 states in F1 and F2 at base and Instruct.
- No judge is needed for D1 and D2.

## Owner decisions (29/10)

Accepted ("Ye"): D1 floor 0.5, as pods A and B; D2 reliability-normalized difference with lower bound > 0, Holm over 3 transitions; text source 2 = E3 `desc` and `ex2` replies; the text-anchor reading rule for "more than the text".

Accepted ("Yes, start", 29/10): D2 becomes report only after Codex round 3 B1; pod C's only pass rule is D1.

Accepted ("Accept all three"): F4 is unavailable if any role keeps fewer than half its rows; a report-only stage-to-stage change measure is added; pod A's 5 replicates make 21,000 rows.

## Changes

- 29/10/2026, v1.1, before any code or data: text fixes from Codex round 1 (`../261029_stage_persona_plan/review/codex_review_v2_r1_261029.md`): D2 renamed to what it tests, with its limits (R4); reading rules for text anchors and related-topic pairs (R5); the axis projection rule (R6); decision p for D1 and D2 (R8); the F4 length control (R9); interval scope (R3); the full E3 C0 gate (O3); F3 check and floor scope (O6); `default` excluded, centring and frozen text features (O7). No threshold value changed.
- 29/10/2026, v1.2, before any code or data: owner decisions after round 1: the F4 support value; the report-only stage-to-stage change measure; 21,000 rows from pod A's 5 replicates, with GPU and storage estimates scaled. No threshold value changed.
- 29/10/2026, v1.3, before any code or data: text fixes from Codex round 2 (`../261029_stage_persona_plan/review/codex_review_v2_r2_261029.md`): F3 chance from the manifest (B1); reliability floor on lower bounds from all draws, no draw removed (B2); the stage-to-stage reading, the limit of the reliability correction, a G5 bias case and a question-only-halves sensitivity for D2 (B3); F4 and identifiable-subset half support (B4); identifiable subset on the judged panel (B5); reconstruction on valid rows (O-A); F1 axis checked in F1 (O-C). No threshold value changed.
- 29/10/2026, v1.4, before any code or data: text fixes from Codex round 3 (`../261029_stage_persona_plan/review/codex_review_v2_r3_261029.md`): F3 on the complete balanced grid (B2); half support checked on every bootstrap draw (O1). Round 3 B1 (D2 can pass with identical complete maps) needs an owner decision; D2 is unchanged until then.
- 29/10/2026, v1.5, before any code or data: owner decision on Codex round 3 B1: D2 is report only, with raw and normalized differences beside each other; Holm is dropped because D1 is the only primary decision; the G5 equal-map case is a required regression case.
- 30/10/2026, v1.6, after the engineering canary (base and Instruct), before the full run: the F3 equality limit goes from 1e-3 to 1e-2. On the canary, the two E3 sources failed equality on batch numerics alone (max relative difference 0.0057 at base and 0.0068 at Instruct: each question's 100 identical rows ran as batches of 64 and 36), while F3 role decoding stayed at chance (0.050–0.051) and the pod A source passed both parts. The same rounding moved pod A's role map by Spearman r 0.99999 (generation-time states against pod C's re-extraction; centred role means changed by 0.19%). Owner decision, 30/10, after these numbers: "ok sounds negligible, lets proceed with next steps". Decoding at chance remains the role-leak test. No other value changed. (`NOTE.md`, engineering canary.)
- 30/10/2026, v1.7, after the engineering canary, before the full run: the bootstrap centring changes and the interval does not (code review of 30/10; owner: "ok fix all"). (1) Each draw's role vectors are centred over the distinct roles that draw sampled, not over all roles, and pairs of draw positions with different roles map onto them, so a duplicated role adds its pairs once per copy (the E3 rule, `../261008_base_persona_map/part2_analysis.py`, `map_boot`; Codex code review of 30/10). (2) The interval stays the plain 95% percentile interval. Draws repeat questions, keep about 63% distinct questions per side and sit below the full-data value (canary: D2 question-only 0.984 against [0.946, 0.982]; E3 `ex2` D1 0.942 against [0.780, 0.944]), which makes the D1 lower-bound rule stricter than its 95% label; every result now reports that shift (median of draws minus the point). A shifted interval was tried and dropped: Codex showed it can flip a D1 floor decision without a coverage check (synthetic lower bound 0.4605 to 0.5405). Owner, 30/10: "Original interval". Point values, the one-map null and all thresholds (D1 lower bound ≥ 0.5, p < 0.01, reliability floor 0.5) are unchanged; D1 now also applies the same-anchor floors that "Calibration" already required. The canary intervals in `NOTE.md` used the v1.6 centring.
