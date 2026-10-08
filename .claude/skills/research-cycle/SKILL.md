---
name: research-cycle
description: Use when planning, choosing, running, or recording an empirical AI experiment in this project.
---

# Research cycle

Follow these steps for each new experiment. Treat `research/PLAN.md` as a draft until the owner approves it. Use `.claude/CLAUDE.md` for current project context.

1. Read `research/PLAN.md` and `research/EXPERIMENT_INDEX.md`. Update the question, candidate test, next milestone, and open owner decisions in the plan.
2. State what the test will teach, the expected result, and the cheapest useful test. Prefer one model and one dataset when that can answer the question. Change one important factor at a time when possible.
3. Choose **de-risk** or **extended** mode. Give one reason in the run note. Use de-risk mode when a small test can reduce uncertainty. Use extended mode when runs cost more, code is reused, or collaborators share the work. Exploration can continue in extended mode.
4. Match the work to the mode. In de-risk mode, a short notebook or script and a brief linked note are enough. In extended mode, keep run scripts and config in `experiments/<topic>/<YYMMDD_name>/`. Move reused logic to stable modules. Test and review shared core code. Use pre-commit checks and CI when useful. Save checkpoints or intermediate outputs when needed. Cache costly LLM responses and use per-item JSONL with inputs, outputs, and metadata where relevant.
5. Get owner approval before a costly run or an external action. If approval or a needed tool is absent, state what is missing and stop before that action.
6. After an authorized run, inspect raw examples and errors before relying on a summary score. Record contrary cases and what the run taught. If no run happened, say so; do not invent an output or result.
7. Record the exact model, data, prompt, code version or path, parameters, config, seed, and command. Add raw output paths and the next decision. If a file is on Hugging Face, record its dataset ID, exact revision, and path. Use `research/ARTIFACT_STORAGE.md` for the storage layout. Use `experiments/EXPERIMENT_TEMPLATE.md` or a shorter de-risk note. Mark missing details as missing.
8. Add a dated row to `research/EXPERIMENT_INDEX.md` with links to the code or note and actual raw outputs. Update the next milestone in the plan. Ask collaborators for feedback when they exist; record their feedback or that it is pending.

Neither mode has an automatic preregistration or held-out gate. A de-risk test needs no code review or general pipeline. Do not demand the inactive `save_result()` command or old sanity and provenance gates. Use scientific checks when the question needs them. A run record does not certify a scientific claim. Follow the separate claim review in `.claude/CLAUDE.md` before an external claim.
