# Experiment index

Use one dated row for each new experiment. Link the note or code and actual raw output paths. Mark missing records as missing. As of 2026-10-02, no new run is recorded here.

| Date | Question | Mode | Code or note | Raw output | Decision | Status |
|---|---|---|---|---|---|---|
| 2026-10-02 | Which review concern should the first test address? | Pending owner choice | `research/PLAN.md` | None; no run | Owner decision open | Proposal only |
| 2026-10-05 | Q1 E0: does OLMo-2-1B base play characters from a plain-text in-context interview prompt? | De-risk | `experiments/q1_emergence/261005_base_incontext_elicitation/NOTE.md` | same folder: `replies_*.jsonl`, `answers/`, `ranswers/`, `results.json`, `results_role.json` | Base does not pass the fixed rule: blind top-5 30% / 16% / 24% (sets 1–3) vs about 2% chance; final 36–49%. Fails set 2. Role judge invalid (wrong-role 9.8%). Human labels pending | Run; partial pass |

## Historical pointers

These paths point to existing work. This index does not re-run or validate it.

- `output/results_2026-09/writeup/REPORT_persona_stages.md` — invalid for Q1 since 2026-10-05; do not cite.
- `output/independent_review_2026-10-02/review.txt`
- `research/ARTIFACT_STORAGE.md` maps historical files to private storage without treating them as new runs.
