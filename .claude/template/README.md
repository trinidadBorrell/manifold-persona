# Empirical AI research starter

Copy the **contents** of this folder into a new, empty Git repository. For an existing repository, review and merge each file. Do not overwrite project instructions or configuration without checking them.

Edit `.claude/CLAUDE.md` and `research/PLAN.md` before the first test. Replace the examples in `research/EXPERIMENT_INDEX.md` and `experiments/EXPERIMENT_TEMPLATE.md` with records from your own work. The examples contain no measured results.

## Workflow

1. Write the question, expected result, and cheapest useful test in `research/PLAN.md`.
2. Use the `research-cycle` skill in `.claude/skills/research-cycle/SKILL.md` before and after a run.
3. Use **de-risk** mode for cheap tests. A small notebook or script and a short note are enough.
4. Move to **extended** mode when runs are costly, core code is reused, or several people share the work. Save settings and outputs; test and review shared code.
5. Add each run to `research/EXPERIMENT_INDEX.md`. Inspect raw cases and errors before summary scores.
6. Before an external claim, use `.claude/skills/challenge-result/SKILL.md`. Do not report an unreviewed result as verified.

Keep data, raw results, local settings, and secrets out of Git. The `.gitignore` covers common paths. Check it against your project's actual paths before you add files.

If a project uses remote storage, keep the same `experiments/<topic>/<YYMMDD_name>/` path for large artifacts. Record the store name, exact revision or version, and file path in the run note. Keep older files with unknown run details under a clearly marked legacy path; do not invent dates or findings. A backup is not a scientific review.

## Tools

| State | Tools and use |
|---|---|
| Included now | Plain Markdown records, a scoped pre-commit config, and `Makefile` commands. Copying files runs no check or hook. |
| Add when needed | `uv` for environment setup; Jupyter for quick notebooks; `tmux` for long jobs; Ruff, Black, pre-commit, and nbstripout for code and notebook checks. |
| Add for a matching task | Per-item JSONL and `pandas` for LLM records; Inspect or safety-tooling for structured evaluations; LiteLLM for several providers; Weights & Biases for shared run tracking. |
| Not installed by this starter | All optional packages, hooks, editors, services, and shared research tools. |

Use local files as the record. Teams can add Slack or Notion for updates. Add tools only when the work needs them. This starter does not add a submodule or a core `requirements.txt`.

VS Code or Cursor can help with notebooks and JSONL files. No editor is required or installed by this starter.

## Optional checks

Create `.venv` using your project's Python setup. If `uv` is available, `make install-dev` installs only the optional packages in `requirements-dev.txt` into that environment. `make hooks` opts in to Git hooks. `make check` runs scoped pre-commit checks. Read `.pre-commit-config.yaml` before enabling hooks, especially when merging this starter into an existing repository. These commands do not run when you copy the files.

The claim challenge is an added review step. It does not prove that a finding is true. A human must decide what an important result means.
