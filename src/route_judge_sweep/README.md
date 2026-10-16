# route_judge_sweep — steering routes between personas, scored by many LLM judges

**Status:** current (16-10 meeting, Part 2, slides 9–12)  ·  **When:** 2026-10-14 → 10-16  ·  **Plan:** `plans/2026-10-14-judge-prompt-sweep.md` (local)

## Question
The PersonaManifold paper scored coherence loosely and never asked whether the target persona is *reached*, either at the end point or along the way. Here we steer along four routes from a start persona to a target and ask:
- How often does each route reach the target?
- How often does each route collapse into broken text?
- How much do these answers depend on the judge prompt?

## Setup of the run shown at the meeting (run 2, `qwen3_route.py`)
- **Model, questions, sampling:** Qwen3-8B; Giovanni's 10 open questions; 140 tokens; T 0.7, top-p 0.9; empty system block; thinking off; 1 answer per question.
- **Pairs:** assistant → vampire, assistant → leviathan, grader → tree, scheduler → void. The start persona is replaced by the empty-prompt model.
- **Routes** through the personas' mean answer-token states (E1b), built in `calibration/routes.py`:
  - **LIN:** the straight line.
  - **PMG:** the PersonaManifold geodesic. A shortest path on the paper's metric kNN graph, then a natural cubic at equal arc length.
  - **MAN** (`MAN4` in code and data): the nearest real persona to the straight line in each of 4 spans, then the cubic.
  - **FL:** a straight line in Fletcher PGA coordinates, mapped to the chain of nearest real personas, then the cubic.
- **Hook:** Giovanni's clamp in the route's span, `h ← h + U Uᵀ (G − h)`, every position.
  - Goal: `G = M₀ + S (m(s) − M₀)`.
  - s ∈ {0.2, …, 1}; layers 15 and 23; S ∈ {1, 4}.
  - That gives 800 answers per (layer, S) set.

## Judges (all Claude Sonnet: run 2 through the API, run 1 through subagents; full description and one worked case: `output/11-oct/all-judges.txt`)
| Judge | What it is | Kept? |
|---|---|---|
| J1 | Original route-judge prompt (control). The pair's 10 personas + other; "other is a last resort". | kept |
| **J2** | J1 cleaned up: other not discouraged, a `collapsed` option, quotes before deciding. | **shown** |
| J3 / J4 | Assistant Axis paper's judge prompt, original / revised. | kept |
| J5 | Variant A of the prompt doc: closed list + a definition of persona. | dropped |
| J6 | Open vocabulary: it names the persona freely (stage-2 mapping not run). | kept |
| J7 | Minimal prompt, 275 role names. | kept |
| **J8** | J2 choosing from all 275 roles. | **shown** |
| J9 | J8 with ranked probabilities. | dropped |
| **J10** | Giovanni's Ranking Judge. Reads 15 answers at once (first 500 characters, no question) and ranks 10 of 276 names, plus a nonsense flag. | **shown** |

- **Prompt versions:** J2–J9 prompts are built in `prompts.py`. A version `(a, c, d)` varies the wording, the option order and the worked examples. In run 2 each answer gets one judgement, and the version varies by batch.
- **J10:** `j10.py` is an exact port of Giovanni's judge.
- **Tables:** `judges_table.py` draws the judge tables, `output/11-oct/table*-judges*.png`.

## Files
| File | What it does |
|---|---|
| `qwen3_route.py` | Run 2 generation, one (layer, pair, S) per call. |
| `build.py` | Run 1 answers and batches (`$PM/JS`). Also holds the constants `PM`, `PAIRS` and `ROUTES` used everywhere. |
| `clamp_build.py`, `clamp_build2.py` | Clamp versions 1 and 2 of the run 1 cells. Run 2 replaced them. |
| `clamp_sweep.py` | Judge inputs (`build`) and label merge (`labels`) for the clamp runs. Its environment variables pick the set. |
| `prompts.py`, `j10.py` | Judge prompts J1–J9, and J10. |
| `api_judge.py` | Runs any judge through the Claude API: interactive, budget-capped, keys from `token/`. |
| `collect.py` | Validates and collects subagent or API labels. |
| `JUDGE_SWEEP_AGENT.md`, `J10_AGENT.md` | Instructions for the subagent judges used in run 1. |
| `export.py` | Artifact data: label tables, per-pair answer files and the metrics. |
| `run2_judge_figs.py` | One figure per judge, the four routes as panels. |
| `all_judges_txt.py`, `judges_table.py`, `doc.py` | Judge documentation: the txt, the PNG tables and `docs/judge_prompts_2026-10-14.md`. |
| `pages/` | The three artifact pages: all 8 judges; J2/J8/J10; J2/J8/J10 without FL. Each page reads `data_L*_S*/` written by `export.py`. |

## How to run (run 2)
1. Routes: `python -m calibration.cells --exp E1 --e1b $PM/E1b --out $PM/ST15 --layers 15 --alphas 1,2,4 --starts assistant,navigator,ambassador,producer,blogger,validator,rogue,adolescent`.
2. Generation: `condor_submit_dag jobs_condor/gio_q3r.dag`. It has 16 nodes, each running `python -m route_judge_sweep.qwen3_route --layer L --pair P --strength S --out $PM/GIO8B_R`.
3. Judge inputs: `CLAMP_V=2 GIO_L=15 CLAMP_S=4 python -m route_judge_sweep.clamp_sweep build`. Output: `$PM/JSG_L15_S4/`.
4. Judging, on the login node because it needs the API:
   - `API_BUDGET=… python -m route_judge_sweep.api_judge --dir $PM/JSG_L15_S4 --jobs manifest_api.json --workers 16` covers J2, J4 and J10.
   - Then run the same with `manifest_api2.json` for J1, J3, J6, J7 and J8.
   - Labels are written only when a run ends.
5. Labels: `CLAMP_V=2 GIO_L=15 CLAMP_S=4 python -m route_judge_sweep.clamp_sweep labels`. Output: `labels.csv` and `J10/labels.csv`.
6. Page data: `JSWEEP_DIR=$PM/JSG_L15_S4 EXPORT_JUDGES=J2,J8,J10 python -m route_judge_sweep.export --out <page dir>`. Copy `data/` to `data_L15_S4/` next to a page from `pages/`. The published pages are saved in `$PM/JSG_pages/`.
7. Figures: `python -m route_judge_sweep.run2_judge_figs --layer 15 --strength 4`. Output: `output/11-oct/run2_L15_S4_{J}.png`.

**Earlier runs**, same judges:

| Run | Data | Answers |
|---|---|---|
| Run 1 | `$PM/JS` | Additive E1 steering, α = 2, 8B L15 + 14B L21. All 10 judges, through subagents. |
| Clamp v1 | `$PM/STC15` → `$PM/JSC` | Clamp to h0 + δ(s). |
| Clamp v2 | `$PM/STD15` → `$PM/JSD_S{1,4}` | Mean-state units. |

## Result (as presented: L15, S = 4)
- **Reaching the target** (J2, s = 1): FL 57%, PMG 52%, MAN 48%, LIN 42%. LIN reaches 55% at s = 0.8.
- **Collapse** (J2, s = 1): PMG 18%, FL 22%, LIN 28%, MAN 30%. No route collapses clearly less.
- **By pair** (J2 target / J10 top 5): leviathan 72% / 88%, tree 52% / 92%, vampire 32% / 28%. Void reaches 90% at s = 0.8, then 52% collapsed at s = 1.
- **Judges disagree a lot on the same answers.** At s = 1 the target share is:
  - J1: 57–78%
  - J7: 22–45%
  - J8: 28–38%
  - J6: 8–18%
  - J10 top 5: 70–80%, but 31% of its top-5 hits are flagged nonsense.
- **Other sets:** L15 S1 barely moves; L23 S4 breaks.
- **Conclusion on the slide:** no technique shows clearly fewer collapses or reaches the persona more strongly.
