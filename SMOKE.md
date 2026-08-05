# Setup and smoke tests (CPU laptop)

Verified 2026-07-30: Ubuntu, Python 3.12.3, 8 cores, 15 GB RAM, no GPU,
torch 2.13.0+cpu, transformers 5.14.1.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

`MP_MODEL_NAME` repoints the whole stack. Use Qwen2.5-0.5B-Instruct for smoke
tests — same architecture and chat template as the 3B default, ~1 GB, ~2.4
prompt-records/s on 8 CPU cores.

```bash
export MP_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
```

Measured at 0.5B: 610 prompt records = 253 s, `manifold.run` on 61 roles = 240 s,
`manifold.sweep --smoke` = 180 s, `local_id` on 61 roles ≈ 4 min. The full
6,900-record cloud is ~48 min at 0.5B and roughly 6× that at 3B.

## The smoke sequence

```bash
.venv/bin/python diagnostics/01_activation_scales.py

.venv/bin/python -m extraction.build_and_extract_roles --n_questions 2 \
  --roles $(ls refs/assistant-axis/roles/instructions/ | sed 's/.json//' | head -60 | tr '\n' ' ') default

.venv/bin/python exploratory/assistant_axis/run_all.py

.venv/bin/python -m manifold.run
.venv/bin/python -m manifold.local_id
.venv/bin/python -m manifold.sweep --smoke

.venv/bin/python -m extraction.generate_and_extract_roles \
  --n_questions 1 --limit 8 --max_new_tokens 24 --chunk 4 --restart
```

## Things that will bite you

- `--roles` must include `default`, or `02_clustering.py` has no Assistant Axis
  baseline and exits.
- `local_id` needs ≥ ~40 roles: `load_points` aggregates to one centroid per role,
  and with `k` near `n` local ID is constant by construction.
- `manifold.sweep` without `--smoke` refuses to run on anything but the real
  276-role 3B cloud. That is its regression guard, not a failure.
- `new_run_dir` uses minute-resolution timestamps with `exist_ok=False`, so an
  immediate retry raises `FileExistsError`. Delete the stale directory.
- A cloud whose `manifest.json` has no `sink_factor` key predates the attention-sink
  fix; its PC1 is sequence length. See [diagnostics/README.md](diagnostics/README.md).
