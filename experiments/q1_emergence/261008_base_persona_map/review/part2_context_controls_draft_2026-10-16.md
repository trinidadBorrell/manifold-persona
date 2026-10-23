# Draft change 16/10/2026: context controls for Part 2

Status: **draft, not approved.** Written before any Part 2 number on real data. When approved, it goes into THRESHOLDS.md under "Changes".

## Why

Reply-token states attend to the whole prompt. Every description wording names the role. In `ex2` and `ex2_other` the two example turns are the same text in the train and test splits. So the frozen decoder can read the role from the prompt context, not from what the model wrote.

The frozen text baselines read only the reply. So frozen rule 2 (gain over text) cannot detect this. A pass can come from the context alone.

Literature:
- The system-prompt type is linearly decodable at every layer, even when the behaviour does not change (arXiv 2609.38205, section 5.8).
- Probes at 100% fell to chance after format features were regressed out (arXiv 2606.02907). It recommends reporting the confound-only accuracy next to the main accuracy.
- Persona is decodable at the last prompt token (arXiv 2609.22522; Persona Vectors, arXiv 2507.21509, monitoring section).

## New features

One GPU pass, no generation. Same model, revision and dtype as `gen_extract.py` (OLMo-2-1124-7B base, bfloat16). Hidden states 9, 17 and 23. All items of `run_main` and `run_rep`.

| Name | Input | Vector |
|---|---|---|
| C0 check | Prompt ids + kept reply ids, exactly as `gen_extract.py` | Mean over reply positions |
| C1 prompt state | The same pass | State at the last prompt token (`Guest:`) |
| C2 reply without role context | Header, the test question, `Guest:`, then the same kept reply ids. The "About the guest" line and the example turns are removed. | Mean over reply positions |

- **C0 gate:** C0 must match the stored `reply_mean_acts.npy` at state 17. Median cosine ≥ 0.999 and 1st percentile ≥ 0.99. If the gate fails, stop. Do not use C1 or C2.
- C1: the token text is the same for every role. Role information there comes only from the context.
- C2: the reply tokens are the same ids. Role information there comes only from the reply text, as the model reads it.

## Analysis

Same classifier, same 50 splits, same crossed bootstrap as the primary. No new permutation null.

1. Target-role decoding from C1 and from C2, per condition, next to the primary.
2. Context share = primary − C2, as a paired difference over the same bootstrap draws.
3. `ex2_other`: decode the target role and the donor role from the reply state, C1 and C2. Report donor − target for each, paired.
4. `run_rep`: primary, C1 and C2 decoding for `desc` and `ex2`. Report the agreement with `run_main`.

## Wording rule (fixed now)

The four primary decisions do not change.

- A condition passes the frozen Part 2 rules and the C2 lower bound is ≥ 10%: "the reply states carry role information that the reply itself also carries; the context adds [context share]".
- A condition passes and the C2 lower bound is < 10%: "the role information in the reply states comes from the prompt context; the reply alone does not carry it".
- Every pass is reported with the C1 value next to it.

## Limits

- C2 moves the reply to earlier positions and removes the context the reply was written in. It measures what the model reads from the reply text, not what it computed while it wrote.
- C1 will probably be close to 100%. So plain accuracy has no room above it. The informative numbers are C2, the context share and the `ex2_other` contrast.

## Tests before the run

- C0 gate on the pod before any control is used.
- Self-test: a planted signal that lives only in the context features must fire the second wording rule. A planted signal in the reply features must fire the first.
- `/code-review` of all new code. Codex adversarial review of this draft and the code.
