# Review of the concurrent-work plan (12/11/2026)

A teammate wrote the plan below. Find what is wrong, missing or premature. Be blunt and truthful. Agree only where you checked a file yourself.

Rules: read only. Do not edit, create or delete any file. No GPU work, no pods, no paid API calls, no git commits. Small local reads only (`cat`, `sed`, `rg`, `python3 -I` or `.venv/bin/python -I`). Read each text source you rely on in full; say in a read-depth table what you read fully and what in part. Public web reads are allowed (for example Han Xiao's HF dataset card and file list at https://huggingface.co/datasets/han-xiao/assistant-axis-shift and his code at https://github.com/xiaohan2012/assistant-axis-shift); do not download large files.

## Inputs

- The plan: `experiments/q1_emergence/261029_stage_persona_plan/plan_concurrent_261112.md`
- Context: `tasks_261112.md`, `review/proof_ledger_261112.md`, `review/fable_rerun_check_261112.md`, `owner_review/NOTE.md`, `draft_post_thomas_v4_261112.md` (all in the same folder); `experiments/q1_emergence/261112_free_checks/NOTE.md`
- Han Xiao's post: `output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md` (two long lines are figure data)
- Code to reuse: `experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py`, `gates.py`, `analysis.py`; `experiments/q1_emergence/261029_stage_persona_plan/hf_archive.py`
- Shared banks: `refs/assistant-axis/` (roles, questions)
- Vault: `/Users/brendon/research-log/manifold-persona-log/Reference/Task tracker.md`, `Experiment index.md`, `Weeks/2026-W45.md`, `Weeks/2026-W46.md`

## Tasks

1. Does the plan's design actually test Han Xiao's three findings on OLMo-2? Check his method in the post (roles, labels, questions, judge filter, axis definition, layers, statistics) against the planned run. Name every difference that would stop a like-for-like comparison, and what it would take to close it.
2. How many roles are needed? Is the proposed free check on his released vectors sound, and what can it not tell us? Is 20, 40 or more roles right for his desirable-versus-undesirable test?
3. Are the "X and Y" additions true to the files, and are they worth a post? Name any overclaim.
4. Dangling experiments: search the repo (`experiments/`, `output/`, `research/PLAN.md`) and the vault task tracker for experiments that were started, designed or promised and not finished. For each, say whether it should be finished before this post, can wait, or should be closed. Find any the plan's list misses.
5. Cost, order and risk: is the order right (free check, then own-reply run)? What would make the run fail or the result unusable (code, gates, storage, upload paths, judge budget, base-model inputs)?
6. Is anything in the plan premature given the owner's rules (rules fixed before results, owner approval before paid runs, no history in shared docs, the 5 Oct rule on old Q1 results)?

## Output

Short sentences, plain words. A verdict line first. One small table per task: item, finding, evidence (`file:line` or URL), severity (blocker / fix before run / minor / ok). End with the five most important changes and a read-depth table.
