# Research workflow for manifold-persona

## What is active

`requirements.txt` defines the core Python environment. The project-specific `presenting-results` guide exists in `.claude/skills/presenting-results/`. The independent review is at `output/independent_review_2026-10-02/review.txt`.

Use `/research-cycle` at `.claude/skills/research-cycle/SKILL.md` and `/challenge-result` at `.claude/skills/challenge-result/SKILL.md`. Use `research/PLAN.md` and `research/EXPERIMENT_INDEX.md` for the project plan and dated record.

The old `.claude/scripts/`, `.claude/src/`, and `.claude/tests/` are retained but inactive. No hook or result gate from that scaffold runs as part of this workflow.

## Two research modes

Hughes and Perez describe de-risk mode and extended project mode. Use de-risk mode first when a small notebook, short script, or small run can test an important idea. Record the question, source paths, what happened, and the next decision. A de-risk result is information, not an automatic external claim.

Use extended mode when runs cost more, code is reused, or people share the work. Keep dated experiment folders, settings, code IDs, and outputs needed for reruns. Test and review shared code when useful. Both modes use the project plan and dated experiment index.

## Artifact storage

Use `research/ARTIFACT_STORAGE.md` for experiment-first paths and the private Hugging Face archive. Keep lightweight run code in `experiments/<topic>/<YYMMDD_name>/`. When a run uses remote data, record its exact dataset revision and file path. Historical files without reliable run details remain under `legacy/output/` with their original names. A matching file hash is not a scientific review.

## Before a claim

Use `/challenge-result` before an external claim, then use `presenting-results` for the report. This claim check is a project safeguard; it is not part of Hughes and Perez's published method.

Link every reported number to a real source path. Check examples, controls, contrary evidence, and method limits. Read `output/independent_review_2026-10-02/review.txt` before claims about persona formation, causal steering, or broad behavior. Show any missing check. An agent review does not replace an independent human review.

## Optional tools

| Tool | Current status | Use when |
|---|---|---|
| `uv` | Available on this machine | Setting up a future environment. |
| `tmux` | Available on this machine | Running a long or remote job. |
| `pandas` | Present in the current `.venv` | Reading per-item LLM records when useful. |
| Jupyter, Ruff, Black, pre-commit, nbstripout | Absent from the current `.venv` | A matching notebook or code-quality task needs them. |

Per-item JSONL can keep LLM inputs, outputs, and metadata together. Inspect or safety-tooling can help when a real LLM evaluation needs them. LiteLLM can help when a task uses many providers. Weights & Biases can help track shared runs. Slack and Notion are optional team tools; the local experiment index remains the project record.

## Copy to another project

The self-contained `.claude/template/` is available. Copy its contents to another project. Do not copy this project's data, results, local settings, or persona claims into another project. Keep Nanda's research-decision guide separate from this workflow.

## Limits

`.claude/settings.local.json` and the protected research folders remain local. This change installs no service, hook, or dependency. A file, check, or agent review cannot prove a scientific claim true.
