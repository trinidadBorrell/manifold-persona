# role_manifold — H1: do the 276 role centroids lie on a low-dimensional manifold?

**Status:** finished  ·  **When:** 2026-07-21 → 09-22  ·  **Plans:** `2026-07-21-role-manifold-reconstruction`, `2026-07-22-role-count-sweep` (cited in the module docstrings; the plan files are no longer in `plans/`)

## Question
Is there a low-dimensional manifold through the role centroids of the point cloud from `src/extraction`, in the spirit of the manifold-steering paper (arXiv:2605.05115)? The tests are a reconstruction metric against a permutation null, then a sweep over how many role centroids the manifold is built from, then local intrinsic dimension.

## Files
| File | What it does |
|---|---|
| `pipeline.py` | Plan 1: manifold constructions, the reconstruction metric, the permutation null |
| `run.py` | Runs plan 1 (positive control first) → `output/manifold_h1-2/<stamp>/` |
| `tps.py` | Thin-plate-spline manifold, a numpy port of causalab's spline |
| `report.py`, `plots.py` | Plan 1's REPORT.md and figures |
| `sweep.py`, `subsets.py`, `sweep_plots.py`, `sweep_report.py` | Plan 2: the same pipeline at 8 role-set sizes |
| `idim.py` | Intrinsic dimension against n, with a matched small-N reference |
| `local_id.py` | Local intrinsic dimension: does it vary from place to place? |
| `analysis_extra.py` | Post-hoc structure: ID, number of manifolds, MST skeleton |
| `PROJECT_README_2026-07.md` | The original top-level README of this phase. It covers extraction, the Assistant Axis and per-persona studies and the attention-sink withdrawal |
| `SMOKE.md` | CPU smoke-test sequence for the July pipeline |
| `refactors/` | The 2026-07-23 refactor note and its baselines |

## How to run
```bash
.venv/bin/python -m manifold_persona.cli manifold --run        # = python -m role_manifold.run
.venv/bin/python -m manifold_persona.cli manifold --sweep      # = python -m role_manifold.sweep [--smoke]
.venv/bin/python -m manifold_persona.cli manifold --local-id   # = python -m role_manifold.local_id
```
- **Input:** the role cloud under `data/embeddings_roles*/`. Pick it with `MP_ROLE_DIR`.
- **Output:** timestamped folders under `output/manifold_h1-2/`, each with a REPORT.md.

## Result
- Each run's `REPORT.md` holds its own numbers.
- The summary in `PROJECT_README_2026-07.md` (Qwen2.5-0.5B recompute, 2026-07-30):
  - Distance-based ID ≈ 10–13 against an ambient 896 survives the attention-sink fix.
  - The PCA count for 90% of the variance is 71.

## Used by / uses
- Uses `manifold_persona` (config, io, runlog).
- Not imported by later experiments.
