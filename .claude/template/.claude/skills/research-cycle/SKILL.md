---
name: research-cycle
description: Use when planning, choosing, running, or recording an empirical AI experiment.
---

# Research cycle

1. Read `research/PLAN.md` and `research/EXPERIMENT_INDEX.md`. State the question, next milestone, and open decisions.
2. Before a run, state what it will teach, the expected result, and whether a cheaper test can answer the question. Start with one model and one dataset when useful. Change one important factor at a time when possible.
3. Choose **de-risk** or **extended** mode and record one reason. Use de-risk mode for a small test that reduces uncertainty. Use extended mode for costly runs, reused core code, or shared work. Exploration can continue in extended mode.
4. In de-risk mode, use a short notebook or script and a linked note. Do not require a full preregistration, held-out run, code review, or general pipeline for each exploratory test.
5. In extended mode, keep run scripts and settings in `experiments/<topic>/<YYMMDD_name>/`. Move reused logic into stable modules. Test and review shared code. Use pre-commit and CI when useful. Save code IDs, config, intermediate outputs, checkpoints, and costly responses when needed for reruns. For LLM runs, use per-item JSONL with inputs, outputs, and metadata when it fits.
6. Agree on a cost limit and required approval before a costly run. If approval is needed but absent, stop before that run. If a needed tool is absent, use a plain-file method or state what is missing.
7. After an authorized run, inspect raw examples and errors before summary scores. Record contrary cases, what the test taught, and the next test or stop decision. If no run happened, say so.
8. Record the exact model, data, prompt, code path or version, parameters, config, seed, and command. Link actual raw output paths. For remote files, record the store ID, exact revision or version, and file path. Mark missing details as missing. Use `experiments/EXPERIMENT_TEMPLATE.md` or a shorter de-risk note.
9. Add a dated row to `research/EXPERIMENT_INDEX.md`. Update the plan's next milestone. Ask collaborators for feedback when they exist, and record the response or that it is pending.

A run record supports review; it does not certify a result. Use `challenge-result` before an external claim.
