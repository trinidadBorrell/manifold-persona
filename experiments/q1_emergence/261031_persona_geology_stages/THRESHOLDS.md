# Persona Geology pilot: fixed decision rules

Version 0.1, 31/10/2026. Fixed before training, activation extraction or factor analysis.
These are agent design choices under the brief, not owner approval of GPU or API spend.
The missing-data gate applies before implementation. No empirical result exists yet.

Recipe clarification before any training: the separate penalty is the source's squared
observed-token log-ratio. The explicit local schedule uses thirteen full 32-pair batches,
416 rows from a seed-0 permutation, with all 29 omitted row indices retained in the record.
This is a declared stepping adaptation; see NOTE.md. No pass threshold changed.

## Scope

The first zoo uses only pinned OLMo-2-1124-7B-Instruct, twenty Goldberg traits,
two words per pole per factor, chosen by the rule in NOTE.md.
It asks whether coherent pairs produce structure absent from matched scrambled pairs.
A pass justifies asking the owner about a sweep; it does not authorize that sweep.
There is no requirement to recover five factors from twenty traits.

## Gates in execution order

| Gate | Required evidence | Failure action |
|---|---|---|
| D0, data | Original 445-prompt chosen/rejected corpora, source revision, complete shared prompt lists and marker metadata. | Stop and report missing data. Do not generate or substitute data. |
| E0, CPU apparatus | Real code paths pass tiny-model tests; known factor and no-factor cases behave correctly; exact small dense products agree with the low-rank calculation to rtol 1e-6 and atol 1e-8 in float64. | Fix locally before requesting GPU use. |
| E1, rendered inputs | Saved chosen/rejected strings, IDs and masks for five traits per stage, read by the implementing agent; one initial base end-of-text, no base chat markup, no trained-stage examples. | Stop before GPU use. |
| Owner GPU gate | A new pod address, approved cap, complete source/config pins and resolved recipe details. | Stop and ask. The pod 213.173.102.5 is excluded. |
| E2, bounded timing | Two traits and their scrambled partners train with finite loss, correct parameter masks, shared initialization hashes and usable outputs inside the projected cap. | Report failure or a new budget; do not scale. |
| E3, run integrity | Every planned adapter and control has the correct resolved config, data/init hashes and finite nonzero update; raw losses and errors retained. Exact step count and effective examples are reported. | Withhold the pilot verdict. No silent retry with a changed recipe. |
| P1, pilot structure | All pilot conditions below hold. | Stop after the pilot either way. Report raw examples and the failure denominator. |
| Owner sweep gate | Owner says go after seeing the one-stage report. Stage-comparison rules and format sensitivity are then fixed before collection. | No sweep. |
| Owner API gate | Exact generator/judge, prompt, calibration, estimated tokens, maximum spend and owner approval. | No paid call. This gate can occur at any point. |

## Primary pilot rule

Use the cosine-normalized Frobenius Gram matrix over all twenty real adapters.
Run parallel analysis with 2,000 simulated Gaussian correlation matrices of twenty
variables and 150 observations, RNG seed 3102026. Keep consecutive leading eigenvalues
above their ordered null 95th percentiles. Use identical draws for every arm.
The assumed 150 observations follows the post's conservative setting; it is a design
convention, not an estimate of independent model parameters. Also report 75 and 300
as sensitivity checks. They cannot replace the primary decision.

| Condition | Fixed pass rule |
|---|---|
| Primary real zoo | At least one retained factor at the primary setting. |
| Matched scrambled zoo | Zero retained factors at the same setting. |
| Factor fit | Principal axis factoring converges with positive uniquenesses; no undefined rotation or improper solution. An inadmissible fit is inconclusive. |
| Seed check | Second real zoo retains the same count; every matched factor has absolute Tucker congruence at least 0.85 after sign/permutation matching. |
| Permutation control | After restoring data labels, the Gram agrees with the real zoo to rtol 1e-5 and atol 1e-6 under the deterministic recipe. Wrong-name alignment is not significant in the label test below. |
| Data baseline | Saved held-out predictions, R2, residuals and their spectrum exist. High text prediction cannot be called model-specific structure. |

All conditions must hold for a pilot pass. A real-versus-scrambled pass alone is
reported as preliminary if seed or permutation evidence is missing.
Different random initialization can preserve the map while changing individual weights.
Raw cross-seed weight cosine is never a pass criterion.

For the label test, use each factor's signed Goldberg key vector, cosine similarity
to the recovered loading columns, and the maximum absolute association over factor
columns per Goldberg group. Sum the five group maxima into one fixed omnibus score.
Compare the wrong-name score with 9,999 uniform trait-label permutations, seed 3102027;
p = (1 + null scores at least as large as observed) / 10,000.
Require wrong-name p greater than 0.05. Report its weak power on twenty traits.
This label check is not an independent test of factor existence.

## Measures that do not decide the pilot

| Measure | Required report and limit |
|---|---|
| Frobenius map | Full raw Gram, cosine Gram, eigenvalues, target modules, scaling, norms and any zero modules. |
| Factoring | Factor count, unrotated and oblimin loadings, factor correlations, uniquenesses, convergence, software version and matched congruences. |
| Text prediction | Balanced trait-held-out ridge prediction with tuning restricted to training traits; all predictions, held-out R2, map correlation and residual structure. No training-fit R2 as evidence. |
| Activation shifts | Per-input adapter-minus-reference answer states, no-adapter zero check, maps at resolved layer label 16, and layer labels 8/24 as secondary checks. |
| Behavioral examples | Fixed prompts, unedited replies, end reasons, lengths, loops and other failures, including a reference reply. No uncalibrated aggregate trait score. |
| Stage agreement | Spearman similarity-matrix correlation and Tucker matching, plus activation-map agreement. No main raw weight-vector comparison across checkpoints. |

No numerical rule for a stage-specific effect beyond the text baseline is set here.
The one-stage pilot cannot establish that claim. Freeze that rule and its detection
check before a separately approved sweep; report residuals now to expose the confound.
No judge prompt is published with the post. Any later judge is a declared adaptation,
with labeled calibration cases and errors reported separately from model behavior.

## Integrity and stopping

Save exact inputs and raw outputs before aggregate analysis. Report all attempted,
failed, retried and valid adapters; no complete-case replacement without a new note.
Do not truncate an overlength training row, drop a trait, change a seed or relax a
threshold to make a run pass. Stop and report the conflict.
The proposed ceiling is eighty adapters and 1,040 steps within four GPU hours,
including timing and extraction, subject to owner approval at the GPU gate.
Every rule amendment must have a date and reason and state whether outputs were seen.
