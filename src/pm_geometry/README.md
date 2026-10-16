# pm_geometry — PersonaManifold-paper geometry, persona components, and the E1/E1b state extraction

**Status:** shared library  ·  **When:** 2026-10-08 onwards (was `steering/personamanifold/{geometry,components}.py` and `jobs_condor/pm_extract.py` until 2026-10-16)  ·  **Plan:** `plans/2026-10-08-personamanifold-steering-and-geodesic-components.md` (local; E1b design from `plans/2026-10-09-calibration-stability-direction.md`)

## Question
Shared code, not a study. It ports the geometry of the PersonaManifold paper (Xu, Xu, Wu, arXiv 2609.34571) and
answers Part 2 of the 2026-10-08 plan: how many components / principal personas the 276 persona vectors need,
which personas they are, and whether that holds at one layer or across layers.

## Files
| File | What it does |
|---|---|
| `geometry.py` | Paper equations by number: eq. 1 persona vectors (answer minus the no-prompt answer to the same question), 99% PCA, d* (Marchenko-Pastur + Levina-Bickel), eq. 2-4 metric kNN graph (k = 5 d*, eps 1e-4), Dijkstra path + cubic-spline smoothing, eq. 6 steering deltas, straight-line baseline. Choices taken from the released repo are marked `[repo]`, our own `[ours]`. |
| `components.py` | Part 2 per view (`last`, `mean`) and layer: flatness (graph path / straight line, Isomap vs PCA residuals), intrinsic dimension (Levina-Bickel, TwoNN), global PCA, Fletcher 2004 PGA on the data graph, Pennec 2018 barycentric subspaces (greedy principal personas, max k from `PM_BSA_MAX_K`, default 40), split-half stability; multi-layer M0/M1/M2 (M2 = stacked PCA instead of the plan's PARAFAC2). |
| `extract_states.py` | E1 / E1b extraction on Qwen3-8B (`PM_MODEL` to change): greedy answers to EOS (cap 1024), then one teacher-forced pass keeping the last answer token and the answer-token mean at hidden_states 12-32 (`PM_LAYERS`). Sharded, resumes from chunk files. Designs: `rot40` (40 questions per role, question j under system prompt j mod 5; = E1), `q20p3` (20 questions x prompts 0-2; = E1b), `full` (5 x 40). |

## How to run
PM = `/data/project/eeg_foundation/data/manifold_persona/personamanifold`.
1. Extraction (GPU): `condor_submit jobs_condor/pm_extract.submit` (SMOKE=48 by default: throughput check only).
   Real run: `condor_submit SMOKE=0 NSHARDS=32 jobs_condor/pm_extract.submit` (E1 used 32 shards) → `$PM/E1/chunks/`.
   E1b: same with `pm_extract_e1b.submit` (`PM_DESIGN=q20p3`) → `$PM/E1b/`; Qwen3-14B: `pm_extract_14b.submit` → `$PM/qwen3_14b/E1b/`.
   Wrapper `jobs_condor/run_pm_extract.sh` runs
   `python -m pm_geometry.extract_states --out $PM_OUT --n-shards N --shard i --design $PM_DESIGN`.
   `pm_wait*.submit` (`run_pm_wait.sh`) polls until all chunks exist, as the first node of a DAG.
2. Consolidate chunks into `last_L*.npy` / `mean_L*.npy` / `meta.csv`:
   `python -m pm_addition_round.build --e1 $PM/E1 --out $PM/G1 --consolidate-only` (lives in `pm_addition_round`).
3. Components (CPU): `python -m pm_geometry.components --e1 $PM/E1 --out $PM/P2 [--layers 12-32] [--views last,mean] [--per-layer-only]`.
   Condor: `pm_p2_layers.submit` (one job per view x layer) then `pm_cpu.submit` with `MOD=pm_geometry.components`;
   rerun with up to 150 principal personas: `pm_p2full_layers.submit` + `pm_cpu_full.submit` → `$PM/P2full/` (DAG `$PM/dag/p2full.dag`).
   Outputs: `P2/layers/<view>_L<L>.json`, `P2/components_{last,mean}.json`.

## Result
From `$PM/RESULTS.md` (2026-10-08, exploratory): geometry reproduces the paper's shape (d* 14-15, 99% PCA keeps
~170 dims, 57% of persona pairs have graph path > 1.2 x straight line; paper ~60%). Fletcher PGA needs 9-11
directions for 90% of variance at layers 12-23, PCA 17-24; Pennec needs ~57-66 real personas, and the top-10
picks are unstable between question halves; first PGA direction stable (cos 0.98). Full write-up: `$PM/RESULTS.md`, `$PM/README.md`.

## Used by / uses
- Uses: `manifold_persona.prompts_roles` (role records), `jobs_condor/mp_compat` + `mp_qwen3` (cluster transformers shims).
- Used by: `pm_addition_round` (build, component_paths), `calibration` (routes, cells), `pca_pga_steering`
  (clamp_paths, paper_measures), `route_judge_sweep` (clamp_build2, qwen3_route).
