# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Manifold Persona Research

This project studies persona representations and behavior in language models. Treat `research/PLAN.md` as a draft until the owner approves it.

Before each experiment, use `/research-cycle`. Choose de-risk mode first when a small test can answer the question. Use extended mode when cost or scale justifies it. Keep a dated record in `research/EXPERIMENT_INDEX.md`.

Use `research/ARTIFACT_STORAGE.md` for private artifact paths. Keep new large files under the same dated experiment path as their Git run. Record exact remote revisions. Do not infer a historical run from an old file name or remove a local file before its remote copy passes verification and the owner approves removal.

Exploratory notes are not external claims. Before an external claim, use `/challenge-result`, then use the project-specific `presenting-results` guide. Link each reported number to its actual result path. State missing checks and contrary evidence. An agent review is not an independent human review.

Read `output/independent_review_2026-10-02/review.txt` before claims about persona formation, causal steering, or broad behavior. Do not change old research files to fit the new workflow.

Use `requirements.txt` for the core environment. See `.claude/README.md` for optional tools and the inactive old scaffold.

## Write in ASD-STE100 Simplified Technical English

ASD-STE100 Simplified Technical English is a controlled writing standard. Aerospace and defense groups made it. It helps people write clear technical text. All agent communication must follow it.

Key rules:

- Use approved words only. The standard gives a word list. Each word has one meaning.
- Use one word for one idea. Do not use two words for the same thing.
- Write short sentences. Use 20 words or less for instructions.
- Use active voice. Write "Turn the switch", not "The switch must be turned".
- Write short paragraphs. Keep one topic in each paragraph.

The goal is easy reading. Many readers are not native English speakers. Clear text helps them do the work in a safe and correct way.

## Response format (ADHD)

I have severe ADHD. Dense text loses me. Format every response like this:

- Start with the answer in one line. Give it a short bold label, for example **The one-line answer**.
- Break the rest into small chunks. Give each chunk a short bold header. Put a horizontal rule between chunks.
- Keep one topic per chunk. Use 4 sentences or less per chunk.
- Use a small table to compare items or to report statuses.
- End with a **Next step** chunk when I must act or decide. Offer numbered options when there is a choice.
- Cut everything that does not change what I do next.

This format applies to the main agent and to every subagent that writes to me.

## Commands

Use `.venv/bin/python` (Python 3.12). `requirements.txt` is the source of truth for dependencies; `pyproject.toml` lists none on purpose.

```bash
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest extraction/test_generate_and_extract_roles.py          # one test file
.venv/bin/python -m pytest extraction/test_generate_and_extract_roles.py -k NAME  # one test
.venv/bin/python diagnostics/01_activation_scales.py   # apparatus check, CPU, 0.5B model
make install-dev && make check   # optional: ruff + black --check, scoped to experiments/*.py
```

Robustness pins: `robustness/legacy/check_artifacts.py` (frozen record, reruns nothing), `robustness/current/check_invariants.py`, `robustness/check_olmo_invariants.py`. The repro tests from the 2026-10-08 code review are local only, in `output/code_review_2026-10-08/repro/`.

## Architecture

The code has two layers.

- **Package and legacy pipeline (Jul–Sep):** `src/manifold_persona/`, `extraction/`, `manifold/`, `exploratory/`, `diagnostics/`, `robustness/`. It builds role point clouds from the 276 Assistant Axis roles in `refs/assistant-axis/` and studies their geometry. The entry point is `python -m manifold_persona.cli extract|explore|manifold`. Settings and env vars (`MP_MODEL_NAME`, `MP_ROLE_DIR`, `MP_AGGREGATE`, `ASSISTANT_AXIS_DIR`, `PERSONA_VECTORS_DIR`) are in `src/manifold_persona/config.py`. See `README.md`.
- **Experiments (from 2026-10-05):** `experiments/<topic>/<YYMMDD_name>/`. Each folder is self-contained: plain scripts, `NOTE.md` (history and current state), `THRESHOLDS.md` (pass rules fixed before the run) and `review/` (Codex prompts and reviews). The scripts import almost nothing from the package.

Each experiment runs in three steps:

1. Generate on a rented GPU pod. The pod has no persistent volume. Copy the outputs back, check them against `SHA256SUMS_*.txt`, upload them to the private HF dataset `brend0nc/manifold-persona-experiment-artifacts`, and record the revision.
2. Judge with LLM CLIs (`codex exec`, `claude -p`) in an isolated folder outside the repo, so the answer keys (`KEY.json`) are out of reach. Judge runs cost money. Get owner approval first.
3. Score locally against the keys and the frozen rules in `THRESHOLDS.md`.

## Traps

- `hidden_states[0]` is the embedding output. Use `io.hidden_state_index` to map a layer label to an index. Several scripts refuse a label that differs from the loaded layer.
- Old `prompt_avg` clouds carry an attention-sink artifact. A prompt cloud is clean only if its `manifest.json` has a non-null `sink_factor`. Response-token clouds are clean.
- Base models get plain text with one `<|endoftext|>` at the start. Do not use a chat template or user/assistant/system markup.
- Do not pass `--fewshot` (`extraction/generate_and_extract_roles.py`) for trained stages. Its example turns suppress role-play.
- Q1 results from before 2026-10-05 are invalid. Do not use them as evidence.
- These paths are gitignored on purpose: `data/`, `output/`, `*.npy`, `*.jsonl`, `*.parquet`, judge `answers/` and `batches/`, `.labbook/`, `wiki/`, and `steering/` (Trinidad's code, tracked on the causal-study branch).
