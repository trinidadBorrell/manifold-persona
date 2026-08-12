# Setup and smoke tests (CPU laptop)

Verified end-to-end on 2026-07-30: Ubuntu, Python 3.12.3, 8 cores, 15 GB RAM,
**no GPU**, torch 2.13.0+cpu, transformers 5.14.1.

## Setup

The README's `--system-site-packages` line assumes a system torch; there isn't one
here, and a CUDA torch would waste ~2 GB on a machine with no GPU. Use the CPU wheel:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Use a small model for every smoke test

`MP_MODEL_NAME` repoints the whole stack. Qwen2.5-**0.5B**-Instruct is the same
architecture and chat template as the 3B default, ~1 GB, and runs at ~2.4
prompt-records/s on 8 CPU cores.

```bash
export MP_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
```

Rough CPU costs at 0.5B, measured: 610 prompt records = **253 s**;
`manifold.run` on 61 roles = **240 s**; `manifold.sweep --smoke` = **180 s**;
`local_id` on 61 roles ≈ **4 min**. The full 6,900-record cloud would be ~48 min at
0.5B, and roughly 6× that at 3B — start it and walk away, don't iterate on it.

## The smoke sequence

Run in this order; each step feeds the next.

```bash
# 0. Apparatus checks first -- these gate everything else. ~1 min.
.venv/bin/python diagnostics/01_activation_scales.py

# 1. Point cloud. 61 roles is the smallest that keeps local_id non-degenerate.
#    'default' MUST be included -- it is the Assistant Axis baseline.
.venv/bin/python -m extraction.build_and_extract_roles --n_questions 2 \
  --roles $(ls refs/assistant-axis/roles/instructions/ | sed 's/.json//' | head -60 | tr '\n' ' ') default

# 2. Exploratory stage (01-06 + REPORT.md).
.venv/bin/python exploratory/assistant_axis/run_all.py

# 3. Manifold study.
.venv/bin/python -m manifold.run
.venv/bin/python -m manifold.local_id
.venv/bin/python -m manifold.sweep --smoke

# 4. Response-token path (paper-matched extraction). Generation is slow on CPU --
#    keep max_new_tokens tiny for a smoke run.
.venv/bin/python -m extraction.generate_and_extract_roles \
  --n_questions 1 --limit 8 --max_new_tokens 24 --chunk 4 --restart
```

## Things that will bite you

- **`--roles` must include `default`.** Without it `02_clustering.py` dies with
  `No 'default' role points found to define the axis` — the Assistant Axis is
  defined as a contrast against that baseline.
- **Small clouds need ≥ ~40 roles for `local_id`.** `load_points` aggregates to one
  centroid **per role**, so an 8-role cloud is 8 points. With `k` near `n` every
  neighbourhood is the whole cloud, local ID is constant by construction, and CV
  collapses to 0. `usable_k_list` now caps `k` at `n // 2`.
- **`manifold.sweep` without `--smoke` will refuse to run** on anything but the real
  276-role 3B cloud. Its `[0a]` regression check compares against stored plan-#1
  numbers and stops on mismatch. That is the guard working, not a failure.
- **Run directories collide within the same minute.** `new_run_dir` uses
  minute-resolution timestamps with `exist_ok=False`, so a failed run followed by an
  immediate retry raises `FileExistsError`. Delete the stale directory. Left as-is
  deliberately — the alternative risks overwriting a real run.
- **Old clouds are contaminated.** If `data/embeddings_roles/manifest.json` has no
  `sink_factor` key it predates the attention-sink fix and its PC1 is sequence
  length. See [diagnostics/README.md](diagnostics/README.md).

## Fixed while getting this to run

| where | problem |
|---|---|
| `src/manifold_persona/extract.py` | `prompt_avg` averaged over the attention-sink position, making PC1 = `1/seq_len` (r = 0.9998, 78% of variance). Now excludes sink positions; `--keep_sinks` restores. |
| `src/manifold_persona/extract.py` | `torch_dtype=` deprecated in transformers 5 → `dtype=`. |
| `src/manifold_persona/generate.py` | `apply_chat_template(return_tensors="pt")` returns a `BatchEncoding` in transformers 5, not a tensor → `AttributeError`. Now accepts both. |
| `exploratory/.../02_clustering.py` | `k_range = range(3, 31)` unclamped → silhouette crash when roles < 31. Also crashed composing fixed-k methods that were skipped. |
| `exploratory/.../04_role_families.py` | `n_families = 15` unclamped → `IndexError` indexing the Ward linkage of a small cloud. |
| `manifold/local_id.py` | `K_LIST = (15, 25, 40)` and `K_PRIMARY = 25` unclamped → `n_neighbors > n_samples`; then a degenerate zero-variance histogram. |
| `manifold/sweep.py` | log line and **saved manifest** hardcoded `layer 26` / `2048` / `Qwen2.5-3B-Instruct` regardless of the actual cloud. Now read from the cloud's manifest. |
| `manifold/sweep.py` | `n_list` entries above the available role count silently collapsed to the same degenerate cell under a wrong label. Now clamped and de-duplicated. |
| `extraction/*_roles.py` | logged `len(list_roles())` (276) instead of the roles actually in the records. |
| `requirements.txt` | missing `tabulate`, needed by `DataFrame.to_markdown` in `manifold/sweep_report.py`. |
