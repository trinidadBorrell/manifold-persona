# pca_pga_steering — do PCA or PGA directions steer more coherent personas?

**Status:** current (16-10 meeting, Part 1, slides 5–7)  ·  **When:** 2026-10-14 → 10-16  ·  **Plan:** `plans/2026-10-08-personamanifold-steering-and-geodesic-components.md` (local), Part 2.5

## Question
Steer Qwen3-8B along PCA components 1–3 and Fletcher PGA directions 1–3, from `../pca_pga_components`. Do the answers along the geodesic directions behave more coherently? Coherence is scored with the PersonaManifold paper's three path measures (arXiv 2609.34571 §4.1).

## Setup
- **Model and layer:** Qwen3-8B, hidden state 28 (the most stable layer).
- **Hook:** Giovanni's clamp on one axis, `h ← h + (goal − h·u) u` at every position.
- **Goal:** `a + S · sign · step · 2σ_k`.
  - `a` is the start persona's coordinate on the axis; `σ_k` is the spread of the 276 personas along it.
  - `step` ∈ {0, .25, .5, .75, 1}; `S` ∈ {1, 4}; the sign covers both ends.
- **Questions:** 5 held-out questions × 2 paraphrases × 2 samples, so 20 answers per cell.

## Measures
| Measure | Definition | Code |
|---|---|---|
| **IC**, Interpolation Coherence | Mean pairwise cosine of MiniLM embeddings across the answers at one step. The meeting shows the *within-question* version: paraphrases and samples of the same question. | `paper_measures ic` |
| **TS**, Transition Smoothness | Share of steps where a judge's 1–5 rating of each Big Five trait moves in the direction of the path's overall change. The rating instructions are based on BFI-44. | `paper_measures bigfive-*`, `BIGFIVE_AGENT.md` |
| **MA**, Manifold Adherence | 1 − mean geodesic distance from each step to the nearest real persona / the 95th-percentile distance between personas. | `paper_measures ma-clamp` |

## Files
| File | What it does |
|---|---|
| `clamp_paths.py` | Builds the cells and axes and writes `$PM/PCC/{axes.npy, cells.json}`. |
| `steer_gpu.py` | GPU generation. Uses the `ace_steering.steer_gpu` engine and writes `$PM/PCC/S{1,4}/generations.csv`. |
| `paper_measures.py` | IC (MiniLM), MA, and the Big Five judge batches and collection. |
| `BIGFIVE_AGENT.md` | Instructions for the Claude subagents that rated Big Five (used for the meeting numbers). |
| `bigfive_api.py` | The same Big Five judge through the Claude API. Written later; not used for the meeting numbers. |
| `pm_measures_page.py` + `pm_measures_template.html` | The IC/TS/MA page. |
| `pm_measures_figs.py` | PNGs of that page: findings 1–3, `table3`, and the meeting summary `table3_IC_TS_MA`. |

## How to run
1. `python -m pca_pga_steering.clamp_paths --out $PM/PCC`. Inputs: `$PM/PC/cells.json` and `$PM/E1/mean_L28.npy`.
2. `condor_submit S=1 NSHARDS=17 jobs_condor/pcc_steer.submit`, then the same with `S=4`. The job runs `python -m pca_pga_steering.steer_gpu --stage gen …` and ends with `--stage collect`.
3. Measures, CPU, on condor: `condor_submit MOD=pca_pga_steering.paper_measures ARGS="ic PM_MEAS_OUT=$PM/PM_MEAS_CLAMP PM_MEAS_RUNS=1=$PM/PCC/S1,4=$PM/PCC/S4" jobs_condor/pm_meas_cpu.submit`. Repeat with `ma-clamp`.
4. Big Five: `python -m pca_pga_steering.paper_measures bigfive-build`, then the subagent judges following `BIGFIVE_AGENT.md`, then `bigfive-collect`. Output: `$PM/PM_MEAS_CLAMP/bigfive/bigfive.csv`.
5. Page: `python -m pca_pga_steering.pm_measures_page --live <saved component-steering page> --notes notes.json --out page.html`. The page used at the meeting and its notes are saved in `$PM/PM_MEAS_CLAMP/page/`.
6. Figures: `python -m pca_pga_steering.pm_measures_figs --page $PM/PM_MEAS_CLAMP/page/page.html --out output/11-oct --label S`.

## Result (as presented, S = 1; `output/11-oct/table3_IC_TS_MA.png`)
| Direction | IC | TS | MA |
|---|---|---|---|
| PCA 1 / 2 / 3 | 0.76 / 0.76 / 0.74 | 0.40 / 0.54 / 0.49 | 0.75 / 0.75 / 0.81 |
| Fletcher 1 / 2 / 3 | 0.75 / 0.75 / 0.75 | 0.46 / 0.43 / 0.53 | 0.76 / 0.84 / 0.74 |

There is no meaningful difference between PCA and Fletcher directions, so these metrics show no gain in coherence from geodesic axes. At S = 4, MA drops to 0.15–0.49 (`table3.csv`).

## Used by / uses
Uses `pm_geometry`, `ace_steering.steer_gpu` / `ace_steering.common`, `steering_followups.common.degeneracy`, and `pm_addition_round.component_paths` (questions and steps of the earlier addition round).
