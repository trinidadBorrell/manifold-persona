# Experiment index

Use one dated row for each new experiment. Link the note or code and actual raw output paths. Mark missing records as missing. As of 2026-10-02, no new run is recorded here.

| Date | Question | Mode | Code or note | Raw output | Decision | Status |
|---|---|---|---|---|---|---|
| 2026-10-02 | Which review concern should the first test address? | Pending owner choice | `research/PLAN.md` | None; no run | Owner decision open | Proposal only |
| 2026-10-05 | Q1 E0: does OLMo-2-1B base play characters from a plain-text in-context interview prompt? | De-risk | `experiments/q1_emergence/261005_base_incontext_elicitation/NOTE.md` | same folder: `replies_*.jsonl`, `answers/`, `ranswers/`, `results.json`, `results_role.json` | Base does not pass the fixed rule: blind top-5 30% / 16% / 24% (sets 1–3) vs about 2% chance; final 36–49%. Fails set 2. Role judge invalid (wrong-role 9.8%). Human labels pending | Run; partial pass |
| 2026-10-08 | Q1 E3 pilot: which existing Instruct replies can serve as in-context examples for 20 pilot roles? | De-risk | `experiments/q1_emergence/261008_base_persona_map/NOTE.md` | same folder: `v1/`, `v2/`, `v3/` (pools, judge answers), `pilot_examples_final.json` | 20 roles × 2 examples, all judge rank 1; tutor, analyst, librarian replaced by translator, counselor, consultant. Owner kept two weak examples | Examples chosen; no base-model run |
| 2026-10-09 | Q1 E3 pilot run: does OLMo-2-7B base play 20 roles from plain-text prompts (desc, ex2, ex2_other, ex2_filler), and do its reply states carry role information? | De-risk | `experiments/q1_emergence/261008_base_persona_map/NOTE.md`, `THRESHOLDS.md` | same folder `run_main/`, `run_rep/`, `judge/`; HF `brend0nc/manifold-persona-experiment-artifacts` @ `fb84c5028fee341aa8832620985b7a7d29c7c302` | Thresholds fixed before the run. Blind-naming judges done (secondary). Literature judge `judge2.py` reviewed and prepared (`/Users/brendon/persona-manifolds/e3_judge2_pilot`), not run. Owner scoring sheet `human/judge2_human_sheet.md` pending. Part 2 code reviewed, not run (pod down) | Run done; no result yet |

## Historical pointers

These paths point to existing work. This index does not re-run or validate it.

- `output/results_2026-09/writeup/REPORT_persona_stages.md` — invalid for Q1 since 2026-10-05; do not cite.
- `output/independent_review_2026-10-02/review.txt`
- `research/ARTIFACT_STORAGE.md` maps historical files to private storage without treating them as new runs.
