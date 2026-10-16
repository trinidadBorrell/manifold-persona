# chunk_ablation — does the knot-selection rule change the manifold path?

**Status:** finished  ·  **When:** 2026-09-17  ·  **Plan:** `plans/2026-09-17-chunk-selection-ablation.md` (local)

## Question
The September manifold arm (`steering_core.manifold_paths`) picks its knots as follows:
1. Cut the ε-cylinder around the A → B chord into `k` equal spans of chord projection.
2. In each span, take the persona nearest the chord.

This experiment (exp01) asks whether an equal-count ("density") cut picks different knots.
- The comparison is the median knot-set Jaccard between the two rules, over the ε × k grid.
- **Prediction:** the two differ at small ε and large k.
- **Falsification criterion:** a Jaccard of 0.95 or higher.

## Files
| File | What it does |
|---|---|
| `chunk_ablation.py` | exp01: both rules over the grid, with controls → `data/`, `logs/exp01.log` |
| `chunk_figures.py` | fig01–fig04 (PNG + CSV) from the run's data |
| `test_chunking.py` | Unit tests of the two rules on a synthetic cloud |

## How to run
```bash
.venv/bin/python -m chunk_ablation.test_chunking                     # CPU, seconds
# condor (jobs_condor/run_chunk_ablation.sh <run_dir> <geom_cache>):
python -m chunk_ablation.chunk_ablation --out RUN --geom-cache CACHE --layer 19
python -m chunk_ablation.chunk_figures --data RUN/data --figures RUN/figures --layer 19
```

## Result
Not written up in the repo. The plan defers the verdict on which rule is better. The figures are in the run folder that was passed to `run_chunk_ablation.sh`.

## Used by / uses
Uses `steering_core` (geometry, manifold_paths, path_cases, runmeta).
