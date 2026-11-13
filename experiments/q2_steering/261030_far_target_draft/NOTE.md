# Far-target steering from several starting roles (draft, 30/10/2026)

This is the note for `261030_far_target_draft`. It is a draft design with no run. The first part is the short version. The second part ("Run record") holds the full draft, for checking.

## Start here

| | |
|---|---|
| Question | Does moving activations toward one distant reference point produce the target role, from several starting roles, while the reply stays a useful answer? |
| Result | None. No model was loaded, no code was run, no judge was called, and no threshold was changed. |
| How much I trust it | An agent wrote this draft without me, after my overnight brief of 30 Oct 2026. I have approved no choice in it. Every choice is a proposal until I approve a protocol and a budget. |
| What I do not know | What "far point" means: an observed distant centroid or an extrapolated point. That is my choice. I also do not know whether the old `steering/run_steering.py` fits this design. |
| Status (30/10/2026) | Draft only (task T-078). There are no outputs, commands, model revisions, run costs or verdict. |

---

## What the draft proposes

1. Use one observed target-role centroid as the far target. It has an unsteered behavior to compare with.
2. Choose the target and several source roles from training data only, before anyone sees a steering output.
3. Shift each source role toward the target with a static, source-specific delta: `delta_s(f) = f * (t - mu_s)`.
4. Run six arms on the same prompts: two unsteered baselines, the proposed shift, and three controls.
5. Measure the movement in activation space and the behavior separately. Score every attempt, with the failures.
6. First do an offline synthetic hook check. Then ask me to approve one bounded model pilot.

The two possible outcomes are these. The replies show the target role from every starting role. Or the coordinates move, but the replies are incoherent, generic or dependent on the source. The second outcome would reject a simple claim that coordinate proximity guarantees behavioral control.

---

## Open points

- I must choose the meaning of "far point". Neither target choice is approved.
- Pod A's map analysis was not written when this draft was made. A receipt is not a reliable-map verdict.
- The old replay mismatch (F71/F80) needs a new capture and replay check.
- Judge validation is a dependency, and paid judge calls need a separate approval.
- No numerical pass rule is frozen. The rule for several tests and the success rules must be chosen before a run.
- This draft authorizes no compute cost.

---

## Words used here

| Word | Meaning |
|---|---|
| Steering | Adding a vector to the model's activations during generation, to change its behavior. |
| Source role, target role | The role that a reply starts from, and the one role that every shift aims at. |
| Centroid | The mean activation of a role's replies. `mu_s` is the training centroid of source role `s`. `t` is the fixed target centroid. |
| Delta, fraction `f` | The vector that is added. `f` is the share of the way from the source centroid to the target. |
| Far | A large distance between source and target on the same centered activation representation. The exact distance must be declared. |
| Arm | One condition of the experiment. |
| Zero-dose replay | A check that an intervention of size zero gives the same reply as the baseline. |
| Hook | The code that changes the activations at a layer during the forward pass. |
| A | Pod A, my run `261029_instruct_persona_map`: the OLMo-2 Instruct collection. |
| T-078 | The id of this task in my task tracker. |
| F71/F80 | Two items in `research/reviews/2026-10-08_design_items.md` about the old replay mismatch. |
| F6, S2 | Two items of the independent audit of 2026-10-02. |
| HF | My private Hugging Face storage for run files. |

---

# Run record

No run exists. Everything below is the full draft, lightly cleaned. I only need it to check a detail.

## 1. What this draft is

Draft only: test whether a common target produces coherent behavior from different starting roles.

- Task T-078. A draft written without me, which I requested in the overnight brief, 30 Oct 2026.
- Mode proposed for a future pilot: de-risk; one checkpoint can reveal a bad intervention or measurement cheaply.
- No model was loaded, no code was run, no judge was called, and no threshold was changed.
- Every choice below is a proposal until I approve a protocol and budget.

## 2. Question and competing outcomes

Does moving activations toward one distant reference point produce the target role while preserving a useful answer?

- One possible outcome is target-like behavior across starting roles.
- Another is movement in the measured coordinates with incoherent, generic or source-dependent replies.
- The second outcome would reject a simple claim that coordinate proximity guarantees behavioral control.

## 3. Dependencies

| Needed input | Proposed source | Current limit |
|---|---|---|
| Valid model and tokenizer snapshot | The completed OLMo-2 Instruct collection, `../../q1_emergence/261029_instruct_persona_map/NOTE.md`. | Exact immutable revisions must be copied from the approved manifest before a run. |
| Reliable source and target centroids | A's future map analysis, using training questions and wordings only. | A analysis is not yet written. A receipt is not a reliable-map verdict. |
| Matched token basis | New capture and replay check at the same hidden-state index and token span. | The old replay mismatch is F71/F80 in `research/reviews/2026-10-08_design_items.md`. |
| Behavioral labels | Validated role-fidelity rubric and separate coherence/answer-quality labels. | Judge validation remains a dependency; paid calls need a separate approval. |
| Execution | Existing or new resources only under a later explicit run approval. | This draft authorizes no compute cost. |

- The old `steering/run_steering.py` already has near-to-far ablation arms and fraction-based doses.
- Read its geometry checks, hook implementation and manifests before deciding to reuse any part.
- Its existence does not establish that its model, token basis or outputs fit this design.

## 4. Proposed target and intervention

- Use one empirical target-role centroid rather than an arbitrary point outside the observed cloud.
- Select that target and several source roles using training data only, before viewing steering outputs.
- Define "far" with a declared distance on the same centered activation representation.
- Keep target choice separate from the test questions and the behavioral labels.

The meaning of "far point" remains my choice: an observed distant centroid or an extrapolated, unobserved point.

- The observed centroid is the proposed pilot because it has an unsteered behavioral reference.
- An extrapolated target would need a separate definition of successful behavior and a support check.
- Neither target choice is approved by this draft.

Let `mu_s` be the training centroid of source role `s`, and `t` the one fixed target centroid.

- The proposed static intervention is `delta_s(f) = f * (t - mu_s)` at the approved layer.
- The fraction `f = 1` moves the training centroid to `t` algebraically; it does not guarantee token-level or behavioral arrival.
- Report the actual token-level displacement and residual norms after every intervention.

Prefer a static source-specific delta for the first pilot.

- A feedback rule based on the live activation is a different intervention and needs its own design.
- Compare sources both at equal target fractions and at matched intervention norms.
- This separates a distant source's larger perturbation from a source-role effect.

## 5. Conditions and controls

| Arm | Purpose |
|---|---|
| Unsteered source role | Establish source behavior and baseline coherence on the same prompts. |
| Target prompt, unsteered | Show that the target is expressible without an activation intervention. |
| Source-specific delta toward the common target | Test the proposed transition. |
| Same delta with its sign reversed | Test direction specificity. |
| Norm-matched random directions | Measure generic damage or movement from perturbation size. |
| One common static vector from all sources | Distinguish source-specific targeting from a generic direction effect. |

- Use paired question ids, wording splits, generation seeds and decoding settings across arms.
- Do not select only prompts that answer under a treatment arm.
- Score every attempt; show empty, repeated, off-topic and truncated replies separately.
- Any answered-only analysis is secondary and uses an explicitly shared item set.

## 6. Measurement and checks

| Measure | What it can establish | What it cannot establish |
|---|---|---|
| Target-relative projection and distance | Measured movement in the declared activation space. | A role, identity, coherence or causal mechanism by itself. |
| Blind source/target role identification | Whether the reply expresses the intended target rather than its source. | Whether a classifier follows a lexical cue without control checks. |
| Coherence and question relevance | Whether the reply remains understandable and answers the question. | Safety or overall model capability. |
| Per-prompt paired effects | Which starting roles and questions retain or lose useful behavior. | Generalization to new models or training seeds. |
| Failure rates over all attempts | How often the intervention produces no usable answer. | A clean success rate after failures are removed. |

- Fit preprocessing on training data only. Compare text and activation measures on identical examples.
- Check that a zero intervention reproduces baseline generation at the same batch size and seed.
- Verify the hook-to-hidden-state mapping, unpadded token positions, exact delta and finite activations.
- Save rendered inputs, token ids, raw replies, stop reasons, per-item deltas, settings and code hashes.
- Inspect raw successes and failures before any aggregate interpretation.
- Use question-level paired uncertainty and show source-role effects separately.
- Choose multiplicity handling and quantitative success rules before a run.
- No numerical pass rule is frozen by this draft.

## 7. Budget and stop conditions

- First perform an offline synthetic hook check; then request approval for one bounded model pilot.
- The request must state the complete grid, maximum generations, storage estimate, GPU time and judge cost.
- Stop on an input/hash mismatch, invalid hook mapping, nonfinite values, failed zero-dose replay or exhausted approved budget.
- Stop before scaling when the target prompt fails or steering only produces degradation.
- Run heavy work on the pod. Upload results directly to private HF storage and verify the receipt and restore.
- Use the experiment's dated HF path; retain the exact remote revision in this note.
- No command is supplied because the execution entrypoint and run settings are not approved.

## 8. Evidence and assumptions

- Lu et al. show that steering effects depend on dose and can degrade at extremes. Source: [The Assistant Axis, v1, 15 Jan 2026, Section 3 and Appendix G](https://arxiv.org/html/2601.10387v1).
- Moskvoretskii et al. discuss norm calibration and the trade-off between trait expression and coherence. Source: [Tracing Persona Vectors, v1, 13 May 2026, Section 2 and Appendix C](https://arxiv.org/html/2605.13329v1).
- Assumptions: one checkpoint has a reliable map; source and target coordinates use compatible pooling and prompts.
- Failure points: format effects, prompt leakage, norm differences, unreliable centroids, token-level distribution shifts and judge error.
- The independent audit `output/independent_review_2026-10-02/review.txt`, F6 and S2, motivates paired behavior and all-attempt reporting.
- Current outputs, commands, model revisions, run costs and scientific verdict: [MISSING: design only].
