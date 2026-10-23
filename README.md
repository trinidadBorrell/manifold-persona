# manifold-persona

The project asks whether persona representations in an LLM (Qwen) lie on a low-dimensional manifold, and whether steering *along* that manifold moves the model between personas more coherently than straight-line steering does.

Each experiment lives in its own folder under `src/`, with a README that gives its question, files, exact commands, data paths and result. The three experiments presented at the **16-10 meeting** (`output/11-oct/Meeting 16_10.pdf`) are listed first.

## Current experiments (16-10 meeting)

| Slides | Folder | What | Main outputs |
|---|---|---|---|
| Part 1, slides 3–4 | [`src/pca_pga_components`](src/pca_pga_components/README.md) | How many PCA vs Fletcher PGA components the 276 persona vectors need, by layer, and which personas sit at each direction's ends | `output/11-oct/fig1_components_by_layer.png`, `table_direction_ends.png` |
| Part 1, slides 5–7 | [`src/pca_pga_steering`](src/pca_pga_steering/README.md) | Clamp steering of Qwen3-8B (layer 28) along PCA / PGA directions 1–3, scored with the PersonaManifold paper's IC, TS and MA | `output/11-oct/table3_IC_TS_MA.png`, `finding*.png` |
| Part 2, slides 9–12 | [`src/route_judge_sweep`](src/route_judge_sweep/README.md) | LIN / PMG / MAN / FL routes between persona pairs (Giovanni's clamp, L15, S = 4), each answer judged by up to 10 LLM judges (J2, J8, J10 shown) | `output/11-oct/run2_L15_S4_*.png`, `table-judges-J2-J8-J10.png`, `all-judges.txt`; pages in `src/route_judge_sweep/pages/` |

**After the meeting (20-10):** [`src/axis_interpretability`](src/axis_interpretability/README.md) tests whether PCA / PGA directions (1–3 and 1–50) are interpretable, with a blind LLM judge. Outputs are in `output/20-oct/`. [`src/curvature_null`](src/curvature_null/README.md) then checks whether persona space is curved at all, against flat nulls. Outputs are in `output/20-oct-AI-pilled/`.

**Started 20-10:** [`src/persona_axes`](src/persona_axes/README.md) asks how many *interpretable* axes (named behavioural dispositions, e.g. hedging ↔ confident) cover the personas, which ones, and whether they beat random interpretable axes; Stage A (coverage, identifiability, per-axis purity, full-vector ceiling check), Phase 3 (reachability inside the axes' span), the single-axis and the alternative-direction runs are done. Results in `output/21-oct/RESULTS.md`.

**Conclusions (slide 13):**
- **No gain in coherence from geodesic axes.** IC, TS and MA are about the same for PCA and Fletcher directions.
- **No route shows clearly fewer collapses, or reaches the target persona more strongly.**
- **The judges disagree a lot on the same answers.**

**Open questions:**
- Which metric should define "reaching" a persona?
- Should out-of-distribution behaviour be defined separately from collapse?
- Should the persona set go beyond the Assistant Axis roles?
- Next on the psychology side: Big Five (OCEAN) axes as the basis.

## All experiments

| Folder | When | Question | Status |
|---|---|---|---|
| `src/extraction` | Jul–Aug | Build the 276-role point cloud (prompt and response tokens) | finished |
| `src/diagnostics` | Jul–Aug | Are the activations measured correctly (scales, attention sink, length)? | finished |
| `src/role_manifold` | Jul–Sep | H1: do roles lie on a low-dimensional manifold? Reconstruction, role-count sweep, intrinsic dimension | finished |
| `src/assistant_axis` | Jul–Sep | Assistant Axis view of the cloud: ID, clusters, UMAP, role families | finished |
| `src/per_persona_geometry` | Aug–Sep | Geometry *within* each role (ID, curvature, density) against the axis | finished |
| `src/robustness` | Aug | Invariant checks on the H1 results | finished |
| `src/steering_rebuild` | Sep 08–24 | First steering study: axis vs pair arms, dose, route judge | finished |
| `src/chunk_ablation` | Sep 17 | Which chunk of the manifold path to steer along | finished |
| `src/steering_followups` | Sep 24–28 | Calibrated α, four routes, re-anchored start, pin vs add | finished |
| `src/ace_steering` | Sep 29–Oct 02 | ACE vs linear vs manifold steering | finished |
| `src/pm_addition_round` | Oct 08–12 | PersonaManifold paper reproduction, additive steering on components | superseded by `pca_pga_steering` |
| `src/calibration` | Oct 09–12 | Steering strength calibration, route stability, principal direction | finished; `routes.py` is used by `route_judge_sweep` |
| `src/giovanni_repro` | Oct 10–13 | Reproduce Giovanni's clamp steering (Qwen2.5-3B, then Qwen3-8B) | finished; its clamp is used by the current runs |
| **`src/pca_pga_components`** | Oct 08–16 | see above | **current** |
| **`src/pca_pga_steering`** | Oct 14–16 | see above | **current** |
| **`src/route_judge_sweep`** | Oct 14–16 | see above | **current** |
| **`src/axis_interpretability`** | Oct 20 | Are PGA directions easier to interpret than PCA ones? LLM rating, label then sort held-out roles, order roles along the whole axis, Assistant Axis Table 1 check; directions 1–3 and 1–50 | **new**; PGA no better than PCA; `output/20-oct/RESULTS.md` |
| **`src/curvature_null`** | Oct 20 | Is persona space curved, or does it only look curved? Bend test and path ratio against flat nulls and synthetic controls, layers 12–32 | **new**; curved at layers 19–32 (bend test), PGA mostly does not capture it; `output/20-oct-AI-pilled/RESULTS.md` |
| **`src/persona_axes`** | Oct 20– | How many interpretable axes cover persona space, which, and are they better than random interpretable axes? 31 pre-registered axes + 240-trait pool, held-out coverage and identifiability vs k, nulls, purity, full-vector ceiling | **Stage A + Phase 3 done** (Oct 21): 8 named axes ≈ 80% of PCA-8 for coverage, but steering inside their span reaches 0.1 vs 0.44 for the full vector, no better than random traits; a 2-direction clamp (minus-assistant + prompt-token) is the best steering setup found (43 / 65 personas); `output/21-oct/RESULTS.md` |

**Shared libraries** (no experiment of their own):

| Folder | What it holds |
|---|---|
| `src/manifold_persona` | config, I/O, extraction, generation |
| `src/steering_core` | steering hooks, interventions, paths, route judges v2/v3 |
| `src/pm_geometry` | PersonaManifold geometry, PCA / PGA / BSA components, E1/E1b state extraction |

Dates come from the plans and git history; see each folder's README.

## Data and outputs (not in git)

| Where | What |
|---|---|
| `$PM` = `/data/project/eeg_foundation/data/manifold_persona/personamanifold` | Data for the October experiments; see `$PM/README.md` |
| `$PM/E1`, `$PM/E1b` | Persona states per role and layer: last-token persona vectors and mean answer-token states (`pm_geometry.extract_states`) |
| `$PM/P2full` | Component counts and direction ends (Part 1) |
| `$PM/PCC`, `$PM/PM_MEAS_CLAMP` | Part 1 steering answers and their IC / TS / MA |
| `$PM/ST15` | Route personas per pair (Part 2) |
| `$PM/GIO8B_R`, `$PM/JSG_L{15,23}_S{1,4}` | Part 2 answers and judge labels |
| `$PM/AXI_L28` | Axis interpretability: role scores, judge items and every judge reply |
| `$PM/CURV` | Curvature against flat nulls: `means_L<L>.json`, `answers_L28.json`, `control.json` |
| `$PM/JSG_pages` | The published judge pages |
| `$PM/JS`, `JSC`, `JSD_*` | Earlier judge-sweep runs |
| `$PM/PC`, `PC4`, `PM_MEAS`, `CAL`, `DIR`, `GIO`, `GIO8B` | Earlier October runs |
| `data/` (symlink) | Point clouds for the July–September studies |
| `output/` | Figures and reports. `output/11-oct/` has everything shown on 16-10 |

## Setup and running

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # every src/<folder> with an __init__.py becomes importable
.venv/bin/python -m route_judge_sweep.run2_judge_figs --layer 15 --strength 4   # example
```

- **Cluster jobs:** GPU and heavy CPU stages run through HTCondor. The wrappers (`run_*.sh`, `*.submit`, `*.dag`) are in the gitignored `jobs_condor/`, and each experiment README names the one it uses. `jobs_condor/_common.sh` puts `src/` on `PYTHONPATH`.
- **API keys:** LLM-judge stages call the Claude API with keys from the gitignored `token/`.
- **Script folders:** `assistant_axis`, `per_persona_geometry`, `diagnostics` and `robustness` are run by path (`python src/<folder>/<script>.py`), not with `-m`.
- **Older docs:** `src/role_manifold/PROJECT_README_2026-07.md` is the original README of the July–September phase. Plans (`plans/`) and reading notes (`docs/`) are local and gitignored.
- **Before the reorganisation:** tag `pre-reorg-2026-10-16` is the tree before everything moved into `src/`.
