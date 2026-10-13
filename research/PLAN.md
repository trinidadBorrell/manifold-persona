# Research plan — draft for owner review

As of 2026-10-05, all decisions below are open. This plan records no new experiment or result.

## Status of earlier Q1 results

On 2026-10-05 the owner decided: treat every earlier Q1 result as wrong. Do not use them as evidence or as assumptions.

- The base models got the wrong prompt. They received the chat template as plain text.
- The data was wrong. Trained-stage runs used the `--fewshot` example turns, which suppress role-play.
- Evidence: `output/results_2026-09/audit_2026-10-01/`, `output/results_2026-09/audit_2026-10-02/`, `output/independent_review_2026-10-02/review.txt`.
- Each Q1 draft in `output/` now starts with an "INVALID FOR Q1" banner. The `.docx` copies have no banner.

Facts we do not know now:

- Can base models or pretraining checkpoints play a role with a correct base prompt? Our only test used the wrong prompt.
- Does an assistant axis or role map exist at base, SFT, DPO or RL?

## Research questions

1. **Emergence:** at which training stage (pretraining, SFT, DPO, RL) does the Assistant persona's geometry take shape?
2. **Robustness:** how stable is that geometry under steering, adversarial prompting and further fine-tuning?
3. **Reward:** how does reward optimization, including reward hacking, reshape it?

Q2 and Q3 need the Q1 measurement first. They use it as the reference frame.

## Q1: two separate measures

| Measure | Question | Method |
|---|---|---|
| Geometry | Does the assistant axis and role map exist in the model at this stage? | Pass the same judged role replies through every checkpoint. |
| Occupancy | Where does the model's own default sit on that axis at this stage? | Project each stage's own default replies onto the fixed axis. |

"Takes shape" needs a number: for example, cosine to the final axis and role-map correlation to the final map. Both need a split-half reliability correction with the Spearman-Brown step-up.

## Q1 candidate experiments — proposals only

| # | Question | Cheapest test | Why this order | Status |
|---|---|---|---|---|
| E0 | Can a base model play a role with a base-native prompt? | One base model, a small role and question set, 2–3 prompt formats (for example story or interview frames). A blind judge scores each reply. Controls: an instruct model (positive) and wrong-character replies (negative). | The design of E1 and E2 at base and pretraining stages depends on this answer. | Proposed; not run |
| E1 | Does the axis and role map exist when each checkpoint reads fixed text? | Generate replies once without `--fewshot`. Keep the replies that the judge marks as played. Pass the same replies through every stage. | Holds the text constant across stages. | Proposed; not run |
| E2 | When does the model's own default move to the assistant end? | Each stage's own default replies, projected onto the E1 axis. | Separates "axis exists" from "model sits on it". | Proposed; not run |
| E3 | Does OLMo-2-7B base already contain the persona map, and can it express the roles? | Base writes its own role replies in two plain-text conditions (description; description + 2 same-role examples). A judge checks role-play; the persona map comes from base activations on those replies. 20-role pilot first. See "Next milestone". | Owner choice 08/10 (Step A dropped the same day). It runs before E1 and E2. | Pilot examples chosen; no base run |

Rules for every new run:

- Read raw transcripts and use a judge with a positive control before any number.
- Never pass `--fewshot` to trained stages.
- Fit all preprocessing on training data only.
- Compare text and activation decoders on the same questions and splits.

## Review concerns for Q2 and Q3 — proposals only

| Review concern | Cheapest proposed check | Status |
|---|---|---|
| Default contraction and role specificity (F5) | Compare role shifts with the default-contraction baseline on independent question halves; include benign-advice controls. | Proposed; not run |
| Steering interpretation and answered-only masks (F6, S2) | Compare directions at matched doses; inspect per-prompt role behavior, failures, and a common item set. | Proposed; not run |
| Option-score interpretation (F7) | Swap answer order and user opinion within each question; inspect full answer mass and free replies. | Proposed; not run |
| Changed Q3 tests (F4) | Show original and amended tests together; define the test family before any new run. | Proposed; not run |

## Next milestone

E0 ran on 05/10 (OLMo-2-1B). Base plays characters far above chance, but does not pass the fixed rule (fails example set 2). The Assistant Axis role rubric failed as a judge for this task.

On 08/10 the owner chose the next experiment, **E3: does OLMo-2-7B base already contain the persona map?** It replaces the E0 follow-up and the order E1 then E2.

- **Revised 08/10 by the owner:** base model only. Step A (Instruct text read by Base, a replication of Lu et al. B.3) is dropped. The base model writes its own role replies; we test (1) whether it plays the role and (2) the persona map from its activations on those replies.
- **Two prompt conditions:** Description (Moskvoretskii et al. 2026 main format) and Description + 2 examples of the same role. Plain text only.
- **Pilot:** 20 roles, examples chosen by a fixed rule from existing Instruct replies. See `experiments/q1_emergence/261008_base_persona_map/NOTE.md`.
- **Judge:** calibrate on about 200 human-labelled replies, including difficult negatives. Use a held-out validation set and report the uncertainty.
- **Stop rule:** if Step B still fails on 14/10/2026, report the negative result and move to the trained stages. Step A stands without Step B.
- Instruct default replies read by Base do not show Base's own default behaviour.
- Thresholds and null tests are fixed before the run. Status: design; not run.

## Owner decisions — open

- E3: success thresholds, the shared text format, the layer set, and GPU access for 7B.
- Which judge, and what cost limit?
- Do Exp 2 and Exp 3 results also need the banner? They use the same `--fewshot` clouds.
- Who will review shared code or give feedback if the work becomes extended?

## Feedback

No owner or collaborator feedback is recorded in this plan yet.
