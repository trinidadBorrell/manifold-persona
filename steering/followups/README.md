# steering/followups — calibrated alpha, 4 routes, re-anchored start

Contract: `plans/2026-09-24-steering-followups.md` (local-only). Context: `steering/README.md`.

This package only **imports** from `steering/` (`manifold_paths`, `interventions`,
`activation_steering`, `route_judge_v2/v3`, `run_steering`). It never edits them. The GPU driver and
Condor files are cluster infra and live in the gitignored `jobs_condor/`.

| File | What |
|---|---|
| `common.py` | routes, questions, seed rule, output root, h0 loaders, `degeneracy` |
| `selection.py` | `pick_nearest_segment`, `NearestPath`, `build_manifold` (the one path constructor) |
| `replace.py` | `ReplaceLastPosition` (paper-faithful h := π(t) at the last position), `CaptureLayer` |
| `grid.py` | the exp4/exp5 grid: cells, seeds, stages, and each cell's vector or target point |
| `calibrate.py` | exp1: alpha_neg / alpha_pos per route × direction |
| `knots.py` | exp3: distance vs density vs nearest, k ∈ {4, 8, 16} |
| `judge_followups.py` | route judge over Studies A and B (OpenRouter by default, interactive only) |
| `figures.py` | figA / figB / figC CSVs |
| `test_followups.py` | CPU tests (fake layer stack + a tiny random Qwen3 through the real `generate()`) |
| `jobs_condor/followups_steer.py` | GPU driver, every stage (untracked) |
| `jobs_condor/followups.submit`, `followups_exp2.submit`, `run_followups.sh` | Condor (untracked) |

Everything is written under **`output/steering-fix-24_09/`** (override with `STEERING_FIX_OUT`):
`exp1_calibration/ exp2_direction_check/ exp3_knots/ studyA/ studyB/ judge/ figures/ logs/`.
The GPU stages write to the same path from the execute nodes (it is on `/home`, like `MP_ROOT`).

## Run order

All commands from the repo root. `CONDOR_GUARD_OK=1` marks the short CPU steps as intended for the
login node (the guard hook blocks heavy local work otherwise). Never run `jobs_condor/run_*.sh`
locally.

```bash
# 0. tests (seconds)
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.test_followups

# 1. exp1 + exp3, CPU
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.calibrate
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.knots

# 2. init: output layout, cells.csv, manifest.json, Geometry control (CPU)
CONDOR_GUARD_OK=1 .venv/bin/python jobs_condor/followups_steer.py --stage init
CONDOR_GUARD_OK=1 .venv/bin/python jobs_condor/followups_steer.py --stage init --dry-run   # cell list + counts

# 3. GPU smoke (4 generations, 8 tokens) and exp2
condor_submit STAGE=smoke NSHARDS=1 jobs_condor/followups.submit
condor_submit jobs_condor/followups_exp2.submit
#    gate: exp2_direction_check/exp2_summary.csv pass_criterion for float16 (plan, Controls)

# 4. Study A without the target-vector doses (140 cells)
condor_submit STAGE=A NSHARDS=4 jobs_condor/followups.submit

# 5. h0 from Study A's unsteered (alpha = 0 additive) cells, CPU
CONDOR_GUARD_OK=1 .venv/bin/python jobs_condor/followups_steer.py --stage h0
#    optional: redo exp1 for the assistant routes with the measured h0
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.calibrate \
    --h0 'assistant>vampire=output/steering-fix-24_09/studyA/h0_assistant-vampire.npy' \
    --h0 'assistant>bard=output/steering-fix-24_09/studyA/h0_assistant-bard.npy' --tag measured

# 6. Study A target-vector doses (32 cells) + Study B (56 cells); independent, submit both
condor_submit STAGE=Atv NSHARDS=2 jobs_condor/followups.submit
condor_submit STAGE=B   NSHARDS=4 jobs_condor/followups.submit

# 7. collect all cells into studyA|studyB/generations.{parquet,csv} + steer/text_acts.npy
CONDOR_GUARD_OK=1 .venv/bin/python jobs_condor/followups_steer.py --stage collect

# 8. judge: dry run first (one full prompt + judge/token_estimates.json), then spend
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.judge_followups --dry-run
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.judge_followups --yes   # resumable

# 9. figure tables
CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.figures
```

Any stage accepts `--dry-run` (no model loaded): GPU stages resolve every intervention for their
shard and run the Geometry control, so a missing h0 fails in seconds, not in a queue slot. GPU
stages are resumable: a cell whose table exists under `studyA|B/cells/` is skipped.

## Design decisions the plan left open

Recorded here so they are not silent; each is also in the module docstrings.

- **Seed unit** = (route, arm, alpha, question), the plan's nesting read literally; Study B
  continues the count after Study A (all 4,104 seeds unique).
- **Stages.** The target-vector arm needs h0 and Study B needs h0(q), both measured by Study A's
  alpha = 0 additive cells (no hook, so unsteered). Hence A → h0 → {Atv, B}.
- **h0_mean** = mean text footprint over the identity questions of those cells (75 per route);
  **h0(q)** = that question's 15.
- **nearest_B past alpha = 1** is the linear continuation, i.e. identical to linear_B there
  (flagged `beyond_path`); the plan's count (56) requires those cells.
- **Study B arms are additive**, like the target-vector control.
- **Judge options** per route = source + nearest-k(8) knots of c_A→c_B + target + other;
  `collapsed` (rep4 > 10) is assigned mechanically and never sent; only identity questions are
  judged (canary metric is "Paris").
- **Replacement sanity** (‖h − c_B‖ < 1e-2) is below fp16's own representation error of c_B
  (1.4e-2 for vampire, 1.2e-2 for bard), so the driver records both `write_err` (vs π(t)) and
  `write_err_fp16` (vs fp16(π(t)), which the hook meets exactly).
