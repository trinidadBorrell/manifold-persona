# calibration — what α means, steering stability, steering along a principal direction

**Status:** finished (`routes.py` / `cells.py` output still read by `src/route_judge_sweep`)  ·  **When:** 2026-10-09 to 2026-10-13  ·  **Plan:** `plans/2026-10-09-calibration-stability-direction.md` (local)  ·  was `steering/calib/` until 2026-10-16

## Question
Three experiments on Qwen3-8B (C also on Qwen3-14B), all built from one persona representation (E1b, PersonaManifold eq. 1) and one injection rule (eq. 6, no system prompt):
- **C** — what α and s do to the latent state and to the judged persona, per layer and route.
- **E1** — between the same start A and target B, which route (LIN, PMG, MAN3/4/5, FL, FG) passes through its personas in order and reaches B without collapsing.
- **E2** — steering along a Fletcher principal direction (FG, FL) vs the k-th PCA axis: does behaviour move consistently?

## Files
| File | What it does |
|---|---|
| `routes.py` | `LayerGeom`: 99% PCA space of the 276 E1b persona vectors at one layer, paper graph, Fletcher graph + PGA; builds LIN, PMG, MANk, FL, FG, PCA routes |
| `cells.py` | builds the cells of C / E1 / E2: `OUT/cells.json`, `OUT/deltas.npy`, `OUT/personas_L<L>.npz` |
| `ends.py` | prints the personas at both ends of the top 3 Fletcher directions (for writing E2 hypotheses first) |
| `judge.py` | writes judge batches: `persona` (options per pair) and `trait` (E2, reads `hypotheses_E2.json`) |
| `analyse.py` | per-row latent measures (fraction of the way to B, distance/cosine to B, nearest persona) + judge labels → `rows_measured.csv`, `page_data.json` |
| `answers.py` | answer-browser data (every answer, setting, label, judge prompts) |
| `pages.py` + `cal_template.html`, `st_template.html`, `dir_template.html` | the three HTML pages |

## How to run
GPU generation is `jobs_condor/cal_gen.py` (run by `run_cal_gen.sh` / `cal_gen.submit`; `cal_gen_14b.submit` sets `PM_MODEL=Qwen/Qwen3-14B`; `cal_gen_rev.submit` walks shards from the end). CPU modules run through `jobs_condor/pm_cpu.submit MOD=... ARGS=...`. The DAGs are in `$PM/dag/` (`cal.dag`, `e1e2.dag`, `st19.dag`) and `$PM/qwen3_14b/dag/` (`cal14b.dag`, `st21.dag`, `e1e2_14b*.dag`). `PM=/data/project/eeg_foundation/data/manifold_persona/personamanifold`.

1. E1b extraction (see `src/pm_geometry`): `condor_submit jobs_condor/pm_extract_e1b.submit` → `$PM/E1b`; then `pm_addition_round.build --e1 $PM/E1b --out $PM/E1b/G --consolidate-only`.
2. Cells, e.g. E1 at layer 15:
   ```
   .venv/bin/python -m calibration.cells --exp E1 --e1b $PM/E1b --out $PM/ST15 --layers 15 --alphas 1,2,4 \
       --starts assistant,navigator,ambassador,producer,blogger,validator,rogue,adolescent
   ```
   C: `--exp C --out $PM/CAL --layers 15,19,28 --alphas 0.5,1,2,4`; E2: `--exp E2 --out $PM/DIR --layers 15` (same starts).
3. Generate: `condor_submit DIR=$PM/ST15 STAGE=gen NSHARDS=20 MAXCH=0 jobs_condor/cal_gen.submit`, then `--stage collect` (via `pm_cpu.submit MOD=cal_gen`) → `generations.csv`, `states.npz`.
4. Judge: `.venv/bin/python -m calibration.judge persona --dir $PM/ST15 --out $PM/J_ST15` (E2 also `trait --hyp $PM/hypotheses_E2.json --out $PM/J_DIRT`); Claude subagents per `$PM/JUDGE_AGENT_STRICT.md` write `labels_strict/`; `.venv/bin/python -m pm_addition_round.collect_labels --judge $PM/J_ST15` → `judgements.jsonl`.
5. Measures and pages:
   ```
   .venv/bin/python -m calibration.analyse --dir $PM/ST15 --judge $PM/J_ST15 --out $PM/ST15/page_data.json
   .venv/bin/python -m calibration.pages E1 --dir $PM/ST15 --notes $PM/pages/notes_E1.json --cal $PM/CAL --out steering_stability.html
   .venv/bin/python -m calibration.answers --exp ST15,ST19 --judge J_ST15,J_ST19 --out <dir>
   ```

Data on disk: `$PM/CAL`, `$PM/ST15`, `$PM/ST19`, `$PM/DIR` + `$PM/J_CAL`, `J_ST15`, `J_ST19`, `J_DIR`, `J_DIRT`; 14B under `$PM/qwen3_14b/` (`CAL`, `ST21`, `ST28`, `DIR28`, ...); pages in `$PM/pages/`. Run notes: `$PM/README_2026-10-09.md`.

## Result
From `$PM/RESULTS.md` (exploratory): α=1 moves the state ~40–56% of the way to B, α=2 ~80–110%, α=4 past B; target answers ≤6% for α≤2, no α* except assistant (0.5). In E1 (L15, L19) the seven routes are within each other's intervals for order and arrival; layer and α set target share and collapse. In E2, FG/FL move judgements towards the steered end and PCA mostly does not. Summary: the direction steered along matters; with endpoints fixed, the route between them mostly does not. Pages: Calibration https://claude.ai/artifact/LokLuziN1Gofz1EZK4WsUM, Stability https://claude.ai/artifact/K2PwDWUZWJWrEsJdj5V1ju, Principal direction https://claude.ai/artifact/GzauMgcxr3zD1koq8AK2iw.

## Used by / uses
- Uses: `pm_geometry.geometry`, `pm_geometry.components`, `pm_addition_round.build.load`, `ace_steering.judge_prompt`, `steering_core.route_judge_v2/v3`; `cal_gen.py` uses `ace_steering.steer_gpu` and `steering_followups.common.degeneracy`.
- Used by: `src/route_judge_sweep` reads `$PM/ST15/cells.json` (and `$PM/qwen3_14b/ST21`) for its pairs and routes; its clamp runs (`STC15`, `STD15`) are generated with `cal_gen.py` (`pins.npz` instead of deltas).
