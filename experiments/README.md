# New experiment folders

Put new work in `experiments/<topic>/<YYMMDD_name>/`. Use a short name that states the test. Link each new folder from `research/EXPERIMENT_INDEX.md`. Keep existing work at its current paths.

In **de-risk** mode, a short notebook or script and a brief note linked to the work are enough. State the question, expected result, cheapest useful test, raw examples, and next decision. A full preregistration, held-out run, code review, or general pipeline is not required for an exploratory test.

In **extended** mode, keep run scripts, config, and run notes in the dated folder. Move logic used by several runs to stable modules. Test and review shared code. Keep commands, code ID, model, data, prompt, parameters, seeds, and raw output paths needed for a rerun. Use caches, per-item JSONL, checkpoints, pre-commit checks, or CI when they fit the work.

Start from `experiments/EXPERIMENT_TEMPLATE.md` when a full note helps. Mark absent runs and missing records clearly. The note and index record evidence; they do not establish an external claim.
