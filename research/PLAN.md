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

E0 ran on 05/10 (OLMo-2-1B). Base plays characters far above chance, but does not pass the fixed rule (fails example set 2). The Assistant Axis role rubric failed as a judge for this task. Human labels are pending. Owner decides the next step: 7B base, a prompt change, or proceed with sets 1 and 3.

## Owner decisions — open

- E0 model: OLMo-2-7B base, or OLMo-2-1B base for a faster first check?
- Which judge, and what cost limit?
- Do Exp 2 and Exp 3 results also need the banner? They use the same `--fewshot` clouds.
- Who will review shared code or give feedback if the work becomes extended?

## Feedback

No owner or collaborator feedback is recorded in this plan yet.
