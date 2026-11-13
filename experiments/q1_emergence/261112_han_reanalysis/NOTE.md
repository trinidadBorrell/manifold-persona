# Re-analysis of Han Xiao's released data (Olmo 3 7B)

This is the note for `261112_han_reanalysis`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

Every number here is from Olmo 3 7B (Han Xiao's release), not OLMo-2.

## Start here

| | |
|---|---|
| Question | Do Han Xiao's numbers hold when I re-analyse his released data? And at which stage is the default set apart from the roles? The default is the model with no role prompt. |
| Result | His main number reproduces. From SFT to DPO, the gap between desirable and undesirable roles is +0.180 in raw units (p 0.066). In role SD units the gap is near 0 (-0.026, p 0.86). The roles also move further apart at DPO. The default is not at an extreme of the role range at any stage. Its replies vary more than a typical role's. |
| How much I trust it | Part A is a gate, and it passed. Parts B to D are exploratory: I knew his results before I fixed the rules, and there is no family correction. All 2,484 files match the manifest. This is agent analysis with no human review. I make no external claim. |
| What I do not know | Whether OLMo-2 does the same. Why the roles move further apart at DPO. His release has no base model, so I cannot see what happens before SFT. |
| Status (13/11/2026) | Parts A to D ran on 12 and 13 Nov. The check of role SD with no judge filter ran on 13 Nov. |

---

## What I did

1. I approved the work on 12 Nov: "yes, we can do both", then "We can do modal ye for it".
2. The rules (THRESHOLDS v1.0) were written before any number in Parts B to D.
3. A Modal app downloaded his release and checked all 2,484 files against a manifest.
4. Part A reproduced his number. Then Parts B, C and D ran.
5. On 13 Nov I asked for one more check: the role SD with no judge filter.

---

## Results

### His SFT to DPO gap reproduces in raw units and is near 0 in role SD units

The gap is the shift of the desirable roles minus the shift of the undesirable roles on his axis.

| Version | Raw gap | p | Role-SD gap | p |
|---|---|---|---|---|
| Part A, his released vectors | +0.178 | 0.068 | | |
| Part B, rebuilt from the activations | +0.180 | 0.066 | -0.026 | 0.86 |
| Part B, without borderline roles | +0.090 | 0.51 | -0.218 | 0.34 |
| Part D, no judge filter | +0.408 | 0.0001 | +0.034 | 0.72 |

- Part A passes its gate: desirable +0.186, undesirable +0.008.
- The roles move further apart at DPO. The role SD on his axis goes from 0.613 to 0.715. With no judge filter it goes from 0.293 to 0.573.
- In role SD units the gap is near 0 in every version (p 0.34 to 0.86).
- An agent's reading, not reviewed: with no filter, the growth goes with the doubling of the role-play pass rate (0.26 to 0.52). So part of the spread may be more replies that play their role.
- From DPO to RLVR the gap is small: raw -0.033 (p 0.001), and +0.020 in role SD units (p 0.17).

### The default is not at an extreme at any stage

- On the roles-only main axis, the default's distance to the nearer end is 0.12 to 0.14. With no judge filter it is 0.075 to 0.10.
- The default is never outside the role range. Its signed z is about +1.0 to +1.2.
- The main axis holds 21 to 27% of the role variance.

### The default's replies vary more than a typical role's, and more from DPO on

The number is the log of the default's variance over the median role's variance. Above 0 means the default varies more.

| Stage | Log ratio | In percent |
|---|---|---|
| SFT | +0.031 [+0.017, +0.045] | about 3% |
| DPO | +0.125 | about 13% |
| RLVR | +0.126 | about 13% |

- The default cloud widens relative to the roles from SFT to DPO (+0.094 [+0.076, +0.108]).
- No other change between stages is present.

---

## How much I trust this

- The rules were fixed before any number in Parts B to D. But Han Xiao's results were known before, so those parts are exploratory.
- Many statistics are reported with no family correction.
- All 2,484 files of the release match the manifest by SHA-256.
- The method choices follow his own code where the rules allow it. Section 2 of the record lists each deviation.
- The PCA fit stretches the role range, which pushes the default toward the middle. So the test is harder on an "extreme" reading, not easier.
- At SFT, 27 clouds miss 1 or 2 of their 1,200 replies. The code handles them as his code does.
- The reading of the unfiltered role SD is an agent's, with no interval and no review.

---

## Open points

- A human review and `/challenge-result` must come before any external claim.
- The Mac copy of the release (`data/han_release`, 7.7 GB) is unchanged. I decide whether to delete it.
- His "matched" version of Part B is not in my rules and was not run.

---

## Words used here

| Word | Meaning |
|---|---|
| Han Xiao's release | The public dataset `han-xiao/assistant-axis-shift`: role vectors, axes, judge scores and layer-16 activations for Olmo 3 7B Instruct. |
| Olmo 3 | A different model family from OLMo-2. Its stages here are SFT, DPO and RLVR. |
| Default | Replies with no role prompt. |
| Desirable, undesirable, borderline | His labels for the roles (49 desirable, 25 undesirable, 201 neutral; 52 marked borderline). |
| His axis | The default vector minus the mean role vector. Each role has a projection on it. |
| Gap | The mean shift of the desirable roles minus the mean shift of the undesirable roles between two stages. |
| Raw units, role SD units | Raw units are the projection as it is. Role SD units divide by the standard deviation of the role projections at each stage. |
| `z_common`, `z_stage` | Two ways to compute role SD units: over the roles present in both stages (my rules), or over all roles of the stage (his notebook). They differ by one role. |
| Judge filter | His pipeline keeps only replies with judge score 3. "No judge filter" uses all replies. |
| Part A | The gate: reproduce his figure value from his released vectors. |
| Part B | The unit check: the same shifts in raw units and in role SD units, rebuilt from the activations. |
| Part C | Where the default sits. C1 is its position on the main role axis. C2 is its spread. C3 is the change between stages. |
| Part D | Parts A and C1 again with no judge filter. |
| Main axis, PC1 | The first principal component of the role vectors. |
| Modal | The cloud service where the analysis ran. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

Every number here is from Olmo 3 7B (Han Xiao's release), not OLMo-2. Rules: `THRESHOLDS.md` v1.0. Part A is a gate; Parts B to D are exploratory. Agent analysis; no human review; no external claim.

My words, 12 Nov: "yes, we can do both" (this re-analysis and the coin-flip test), then "We can do modal ye for it" (run it on Modal).

## 1. Files

| File | What |
|---|---|
| `han_reanalysis.py` | Parts A, B, C+D and the report (CPU). Imports Han Xiao's `persona_axes`, `split_half` and `permutation` from `data/han_repo` (commit 6f28e91) |
| `test_han_reanalysis.py` | 10 CPU tests on a synthetic release in his file layout |
| `modal_han.py` | Modal app `persona-han-reanalysis-261112`: anonymous download into volume `persona-han-reanalysis-261112`, SHA-256 check of all 2,484 files against `data/han_release_manifest.json`, then Part A, then B and C+D in parallel |
| `data/han_repo/notebooks/q1_robustness.py`, `assistant_axis.py`, `docs/writeup/lesswrong.md` | Fetched from GitHub at commit 6f28e91 (read-only reference) |
| `results/` | Small result files copied back from the volume |

The Mac copy of the release (`data/han_release`, 7.7 GB) is unchanged; I decide whether to delete it.

## 2. Deviations and choices

1. Loading: Han Xiao's `sum_activations_by_question` and `load_vectors` call `torch.load` without `weights_only=True`. The code uses copies that differ only in `weights_only=True`; a test shows the sums are identical.
2. Part A uses his released role vectors (`vectors-glm-5.3-flash`, layer 16 row) and his `axis_projections`; roles in both SFT and DPO; labels from his `role_desirability.json` (a missing label counts as neutral); his `permutation_test` with seed 0 (his notebook's seed).
3. Part B follows his `notebooks/q1_robustness.py` "full" version: roles with at least 50 score-3 replies, role vectors rebuilt from the layer-16 activations, 100 question halves with a fresh seed-0 generator per stage (same halves in both stages), SE = SD of half differences / 2, 10,000 shuffles with seed 0. His "matched" version is not in the rules and was not run.
4. Role-SD units: the rules z-score over the roles present in both stages (`z_common`). His notebook z-scores over all of the stage's roles, then keeps the common roles (`z_stage`). Both are reported; they differ by one role (translator, absent at SFT). Standard deviations use ddof 0, as his notebook.
5. C1: PC1 from the eigenvectors of the role Gram matrix (the same direction as the top right singular vector). Position t = (default - role min) / (role max - role min) on PC1; distance to the nearer end = min(t, 1 - t), and 0 with an "outside" flag when t < 0 or t > 1. The default is not in the PCA fit and the roles are, so the role range is stretched by the fit; this pushes the default toward the middle, which works against an "extreme" reading.
6. C1 sign: each resample is aligned to its stage's observed PC1; each stage's observed PC1 is aligned to RLVR's observed PC1; then all signs flip so RLVR's observed default z is positive.
7. A role with no kept reply in a resample is dropped from that resample; the count is reported.
8. C2: a cloud's variance is the mean over its 5 system prompts of the mean squared distance to the prompt's mean (all 4,096 dimensions, ddof 0). Resample weights act as question counts. The ratio uses the median over all 275 roles (all replies, no filter).
9. Resamples: 1,000 question resamples (seed 20261112); draw 0 is the observed data. The same draws serve every stage and Parts C and D, so stage differences are paired.
10. Part D: unfiltered vectors for all 275 roles at every stage (translator included at SFT). Shifts in raw and `z_common` units with the permutation p; DPO to RLVR is added beside the rules' SFT to DPO.
11. Missing replies: at SFT, 27 clouds miss 1 or 2 of their 1,200 replies (28 in all; no duplicates). DPO and RLVR are complete. C2 gives missing cells zero weight; role vectors (A, B, C1, D) count only existing replies, as Han Xiao's code does. Found when the first C+D run stopped on its completeness check; a test covers the fix.
12. Download: the first anonymous Modal download hit HTTP 429 (rate limit on the xet token endpoint). The rerun disabled xet (`HF_HUB_DISABLE_XET=1`) and downloaded one folder at a time with 2 workers, waiting 5 minutes on any 429. No token was used, and the Mac copy was not uploaded.

## 3. Runs (12-13 Nov 2026)

| App | What | Outcome |
|---|---|---|
| ap-WMvwNQSwObXdzlsgWoDQdA | first try | image build stopped: the run log was inside the uploaded folder; logs are now excluded |
| (second try) | download | HTTP 429 from Hugging Face; see choice 12 |
| ap-9pJftoMMr39LAHidetuNEw | download, A, B, C+D | 2,484 of 2,484 files match the manifest; A passed in 9 s; B done in 533 s; C+D stopped on missing SFT replies (choice 11) |
| ap-84HaGfo06yvLOFqfr7BBYj | C+D rerun | done: SFT 228 s, DPO 170 s, RLVR 147 s |

Results: `results/results.md`, `results/part_a.json`, `part_b.json`, `part_cd.json` (copies of `/vol/results` on the volume).

## 4. Results in brief (Olmo 3 7B, Han Xiao's release; B to D exploratory, no family correction)

- A (gate, released vectors): desirable +0.186, undesirable +0.008, gap +0.178, p 0.068 (48 desirable, 25 undesirable). Pass.
- B, SFT to DPO: raw gap +0.180 (SE 0.041), p 0.066, which reproduces his notebook. Without borderline roles: +0.090, p 0.51. In role-SD units: -0.026 (SE 0.041), p 0.86; without borderline roles: -0.218, p 0.34. Role SD on his axis: 0.613 to 0.715.
- B, DPO to RLVR: raw gap -0.033 (SE 0.011), p 0.001; role-SD gap +0.020, p 0.17. Small.
- C1: the default is not at an extreme of the roles-only PC1 at any stage. Distance to the nearer end 0.12 to 0.14 (judge-filtered; unfiltered 0.075 to 0.10), never outside the role range; signed z about +1.0 to +1.2. PC1 holds 21 to 27% of the role variance.
- C2: the default cloud is not tighter. It is wider than the median role: log ratio +0.031 [+0.017, +0.045] at SFT (about 3%), +0.125 at DPO and +0.126 at RLVR (about 13%).
- C3: the default cloud widens relative to the roles from SFT to DPO (+0.094 [+0.076, +0.108]); no other change is present.
- D (no judge filter), SFT to DPO: raw gap +0.408, p 0.0001 (49 desirable, 25 undesirable); role-SD gap +0.034, p 0.72.

## 5. Role SD with no judge filter (13/11/2026, exploratory check)

- Why: Part B's role SD (0.613 to 0.715) uses judge-filtered role vectors, and the pass rate doubles at DPO, so the filter could change which replies form each role. My words, 13 Nov: "check those three then rebuild the claims".
- Code: `role_sd_unfiltered.py` (Part B's definition: std of raw projections on Han's axis over the roles in both stages; Part D's unfiltered vectors: mean over all replies). CPU on this Mac, local release copy. Output: `results/role_sd_unfiltered.json`.
- Result (Olmo 3 7B, layer 16, 275 roles, all replies): role SD 0.293 at SFT, 0.573 at DPO, 0.561 at RLVR. The roles still move further apart at DPO, more than with the filter (0.613 to 0.715).
- Reading (agent, not reviewed): the unfiltered growth goes with the doubling of the role-play pass rate (0.26 to 0.52), so part of the spread may be more replies playing their role. No interval computed.
