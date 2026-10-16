# pm_addition_round — PersonaManifold paper reproduction, first round with additive steering

**Status:** superseded by `src/pca_pga_steering` (clamp round)  ·  **When:** 2026-10-08 (was `steering/personamanifold/` until 2026-10-16)  ·  **Plan:** `plans/2026-10-08-personamanifold-steering-and-geodesic-components.md` (local)

## Question
Part 1: does the PersonaManifold paper's steering (eq. 6, h += delta at the extraction layer, every token, no system
prompt) move Qwen3-8B along a geodesic between two personas, and does the geodesic beat a straight line between the
same endpoints? Part 2.5 (exploratory): does adding along the top PGA / PCA components move behaviour the expected way?

## Files
| File | What it does |
|---|---|
| `build.py` | `--consolidate[-only]`: E1 chunks → layer-major `last_L*.npy`, `mean_L*.npy`, `meta.csv`. Then per layer: persona vectors, 99% PCA, d*, metric graph, geodesic + straight-line eq. 6 deltas for the 6 routes (assistant>bard, assistant>vampire, assistant>leviathan, grader>tree, examiner>criminal, scheduler>void) at t = 0, .25, .5, .75, 1 → `G1/geom_L<L>.npz`, `G1/summary.json`. `load()` is also used by `calibration.cells`. |
| `component_paths.py` | Part 2.5 cells at one layer: PGA 1-3 and PCA 1-3, 3 starts (graph Frechet mean, assistant, a median-distance persona), both signs (target = nearest persona to start ± 2 sigma), steps 0-1, 5 held-out questions x 2 paraphrases x 2 samples → `PC/cells.json`. |
| `judge_batches.py` | Writes subagent-judge batches of 100 answers (`--kind pm` for S1, `--kind pc` for PC); judge prompt = `ace_steering.judge_prompt.system_for_route`; options = start, path personas, target. |
| `collect_labels.py` | Reads `labels_strict/` then `labels/` per batch → `judgements.jsonl`; prints missing batches. |
| `sweep_add.py` | Adds the two new arms (`pm_geodesic`, `pm_linear`) to the existing layer-sweep page: distance/cosine to target, judge shares per t, examples, strength table. |

## How to run
PM = `/data/project/eeg_foundation/data/manifold_persona/personamanifold`. E1 extraction first (see `src/pm_geometry/README.md`).
1. Geometry (CPU): `python -m pm_addition_round.build --e1 $PM/E1 --out $PM/G1 --consolidate-only`, then one job per layer
   (`condor_submit jobs_condor/pm_g1_layers.submit`, `--layers L-L`), then merge `--layers 12-32` (`pm_cpu.submit`, `MOD=pm_addition_round.build`).
2. Steering S1 (GPU): `condor_submit STAGE=gen NSHARDS=30 jobs_condor/pm_steer.submit`, then `--stage collect`
   (`jobs_condor/pm_steer.py`, engine `ace_steering.steer_gpu`). Layers 15, 19, 23, 25, 27, 31, 32; 25 answers per cell;
   alpha 0.5 / 2 at t = 1; follow-up alpha 4, 8 via `pm_steer_extra.submit`. → `$PM/S1/generations.csv`, `S1/steer_acts.npy`.
3. Component steering: `python -m pm_addition_round.component_paths --e1 $PM/E1 --p2 $PM/P2 --out $PM/PC [--layer L]`,
   then `pc_steer.submit` (`jobs_condor/pc_steer.py --stage gen|collect --pc $PM/PC`) → `$PM/PC/generations.csv`.
4. Judge: `python -m pm_addition_round.judge_batches --kind pm --gen $PM/S1 --g1 $PM/G1 --out $PM/J_PM` (or `--kind pc --gen $PM/PC --out $PM/J_PC`);
   one Claude subagent per batch following `$PM/JUDGE_AGENT_STRICT.md`; then `python -m pm_addition_round.collect_labels --judge $PM/J_PM`.
5. Page: `python -m pm_addition_round.sweep_add --live LIVE.html --s1 $PM/S1 --g1 $PM/G1 --judge $PM/J_PM --out page.html`
   (the two other pages were built by what is now `src/pca_pga_components/pages.py`) → `$PM/pages/`.
Steps 1-3 ran as one DAG: `condor_submit_dag $PM/dag/pm.dag`.

## Result
From `$PM/RESULTS.md` (exploratory): the target persona is rarely reached (8% of answers at t = 1, alpha 1; 25% collapse);
alpha 2 collapses 93%. Mid-path, geodesic vs straight line: target 5.8% vs 3.7%, path personas 2.0% vs 0.2%.
Component steering (layer 28): little moves at alpha 1; at alpha 4 (37-51% collapse) moving series go the expected way.
Full write-up: `$PM/RESULTS.md`, `$PM/README.md`, pages linked there.

## Used by / uses
- Uses: `pm_geometry` (geometry, components), `ace_steering` (common, grid, judge_prompt, steer_gpu via the condor drivers),
  `steering_followups.common.degeneracy` (in the drivers).
- Used by: `calibration.cells` (`build.load`), `pca_pga_steering` (clamp_paths, paper_measures import `component_paths`),
  `pca_pga_components.pages` (`judge_batches.pm_key`).
