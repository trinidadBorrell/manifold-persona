# ace_steering — ACE vs linear vs manifold persona steering on Qwen3-8B

**Status:** finished; parts are a shared library (see below)  ·  **When:** 2026-09-29 to 2026-10-02  ·  **Plan:** `~/.claude/plans/piped-shimmying-simon.md` (local, outside the repo)  ·  Was `steering/ace_study` until 2026-10-16

## Question
Steering from persona A to persona B: does ACE (sets the coordinate along A→B, arXiv 2411.09003 eq. 5) or the
manifold path (h + S(α) − S(0)) get more answers judged as B, and fewer broken answers, than linear
h + α(μ_B − μ_A)? Then: across layers 15–32, from other sources, on reverse routes, and with activation capping.

## Files
| File | What it does |
|---|---|
| `common.py` | Constants every script shares: routes, layers, α grids, prompts, seeds, sampling, clouds, `OUT_ROOT` (`$ACE_OUT`, default `output/ace-steering-29_09`) |
| `grid.py` | The cells (step, route, layer, arm, α) and the hook each arm applies (none / linear / ace / manifold / ace_manifold, floors for step 5) |
| `steer_gpu.py` | GPU driver, stages `plan`, `smoke`, `gen`, `collect`; also the generation engine other packages reuse |
| `select_routes.py` | Proposes step 2's routes → `routes_step2_candidates.json` (user picks, copies to `routes_step2.json`) |
| `judge.py` | OpenRouter judge (follow-ups' judge, this study's option list); steps 0–2 |
| `judge_prompt.py` | Judge prompt with the fix for routes that include `assistant` (used from step 3 on) |
| `subagent_batches.py`, `subagent_labels.py`, `subagent_redo.py` | Batches for the Claude Code subagent judge, turning workflow journals into `judgements_<step>_subagent.jsonl`, re-batching missed answers |
| `relabel_ai.py` | Re-judges `other` answers as `ai_assistant` / `other` (Assistant Axis paper wording) and overlays the result |
| `readout.py` | `shares.csv` (share per judge option, bootstrap CI) and `page_data.json` (3-D PCA) per step |
| `page.py` + `page_template.html` | Step page `step<N>/ace_step<N>.html` |
| `sweep_page.py`, `sweep_q_page.py` (+ templates) | Layer-sweep pages (distance/cosine to μ_B, shares, examples), the second with a question filter |
| `followup_pages.py` | Pages for follow-ups A+B (`--page AB`) and D (`--page D`) |
| `followup_summary.py` | Tables for `output/behavior-latent/RESULTS.md` → `output/behavior-latent/followups/*.csv` |
| `fig_behavior_latent.py` | % target vs distance to μ_B plots → `output/behavior-latent/layer_<L>/` |
| `explainer.py` + `explainer_template.html` | Explanation page of the four arms → `<root>/steering_arms.html` |

## How to run
Outputs go to `output/ace-steering-29_09/` (gitignored); its `README.md` lists every job, cluster id and judge workflow.
1. Geometry caches and route clouds (CPU): `condor_submit WHAT=geom jobs_condor/ace_geom.submit`, then `WHAT=clouds`
   (`jobs_condor/ace_geom.py` → `geom/geom_cache_L{15,23,27,31}.npz`, `clouds/<route>_L<n>.npz`).
2. Plan the cells: `.venv/bin/python -m ace_steering.steer_gpu --stage plan --step 1` → `cells_step<N>.csv`, `routes.json`.
3. Smoke, then generate (GPU, 2×P100 per job):
   `condor_submit STAGE=smoke jobs_condor/ace.submit`;
   `condor_submit STAGE=gen ACESTEP=1 NSHARDS=6 jobs_condor/ace.submit` (runs `run_ace.sh` → `ace_steering.steer_gpu`);
   `STAGE=collect` → `step<N>/generations.csv`, `steer_acts.npy`. Steps: 1, 2 (sweep), 3 (follow-up A), 5 (follow-up D).
   Extra flags (`--arms ace_manifold`, `--include-csv`, `--exclude-csv`) via `EXTRA=...`.
4. Step-2 routes: `condor_submit MOD=ace_steering.select_routes ARGS="" jobs_condor/ace_cpu.submit` (any CPU module runs this way).
5. Judge: `.venv/bin/python -m ace_steering.judge --step 1` (dry run; `--yes` spends), or the subagent path:
   `python -m ace_steering.subagent_batches --tag <T> --gen step<N> ...` → workflow → `python -m ace_steering.subagent_labels --step <N> --journal <journal.jsonl>` → `subagent_redo --tag <T>` for misses.
6. Read out and pages: `python -m ace_steering.readout --step 1`, `python -m ace_steering.page --step 1`,
   `sweep_page`, `sweep_q_page`, `followup_pages --page AB|D`, `followup_summary`, `fig_behavior_latent`, `explainer`.

## Result
Unsteered, the model answers as the assistant (79% on the sweep). At α 1.5 linear gives 30% target with 34% collapsed,
ACE 26% with 12% collapsed; layers 19–27 work best; the target decides difficulty (bard up to 96%, vampire ≤ 24%);
steering back toward assistant-like roles mostly fails. ace_manifold kept the position exact but collapsed 33–67% of answers.
Full write-up: `output/behavior-latent/RESULTS.md`; run log: `output/ace-steering-29_09/README.md`.

## Used by / uses
- **Used by:** `steer_gpu.py` (load_model, run_batch, cell tables) is the generation engine for `pca_pga_steering.steer_gpu`,
  `route_judge_sweep.qwen3_route`, `giovanni_repro.qwen3_pin`, and `jobs_condor/{cal_gen,pc_steer,pm_steer}.py`.
  `common.py` is imported by `pca_pga_steering.steer_gpu`, `pca_pga_components.pages`, `pm_addition_round.{sweep_add,judge_batches}`,
  `jobs_condor/{ace_geom,pc_steer,pm_steer}.py`; `grid.geom` by `pm_addition_round.sweep_add`;
  `judge_prompt.py` by `calibration.judge`, `pm_addition_round.judge_batches`, `route_judge_sweep.prompts`.
- **Uses:** `steering_core` (`interventions` — ACE, subspace-pin and floor hooks — `manifold_paths`, `activation_steering`,
  `run_steering`, `route_judge_v2`; the prompt itself is route_judge_v3's, via `steering_followups.judge_followups`), `steering_followups` (`judge_followups`, `common`, `replace`, `selection`).
