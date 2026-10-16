# pca_pga_components — how many directions does the persona set need?

**Status:** current (16-10 meeting, Part 1, slides 3–4)  ·  **When:** 2026-10-08 → 10-16  ·  **Plan:** `plans/2026-10-08-personamanifold-steering-and-geodesic-components.md` (local), Part 2.0–2.4

## Question
How many components do the 276 Assistant Axis persona vectors of Qwen3-8B need for 90% of their variance? Straight-line PCA is compared with Fletcher's principal geodesic analysis (PGA). Which personas sit at the two ends of the first directions?

Fletcher PGA works like PCA but uses geodesic components:
1. Find the graph Fréchet mean, the persona that minimises the summed squared geodesic distance.
2. Map every persona into the flat tangent space at that mean (log map).
3. Run PCA there.

## Files
| File | What it does |
|---|---|
| `../pm_geometry/components.py` | The computation (shared). For each layer 12–32 and each representation (last answer token, answer-token mean) it computes PCA, Fletcher PGA, Pennec barycentric subspaces, the count for 90% of the variance, the personas at each direction's ends, and split-half stability. |
| `pages.py count` + `count_template.html` | The "How Many Persona Components" page. |
| `meeting_figs.py` | The static figures and tables shown at the meeting. |
| `pages.py steer` + `steer_template.html` | The page for the earlier addition round (`../pm_addition_round`). Its saved page is an input to `../pca_pga_steering`. |

## How to run
1. Persona states E1, done once and shared: `python -m pm_geometry.extract_states` on condor (`jobs_condor/pm_extract.submit`). Output: `$PM/E1/{meta.csv, last_L*.npy, mean_L*.npy}`.
2. Components, one condor job per layer and view: `condor_submit jobs_condor/pm_p2full_layers.submit` runs `python -m pm_geometry.components --e1 $PM/E1 --out $PM/P2full --layers L-L --views V --per-layer-only`. Then `jobs_condor/pm_cpu_full.submit` (`MOD=pm_geometry.components`, no `--per-layer-only`) merges them into `$PM/P2full/components_{last,mean}.json`. The DAG is `$PM/dag/p2full.dag`.
3. Figures: `python -m pca_pga_components.meeting_figs` (defaults: `--p2 $PM/P2full --out output/11-oct`). It writes:
   - `fig1_components_by_layer`, which is slide 4 left;
   - `table_direction_ends`, which is slide 4 right, at the best layer (28);
   - `fig2`, `fig3`, and `table_overlap_pca_pga` / `overlap_pca_pga_ends`.
4. Page (optional): `python -m pca_pga_components.pages count --p2 $PM/P2full --notes notes.json --out count.html`.

`$PM` = `/data/project/eeg_foundation/data/manifold_persona/personamanifold`.

## Result (as presented)
- With the last answer token, PGA needs about 10 components for 90% of the variance at layers 12–24. PCA needs about 21–23. Both rise at later layers: PCA reaches 40 at layer 31, PGA 20.
- The first directions separate recognisable traits. At layer 28:
  - PCA 1 (29%) runs from prophet / wraith / vampire to consultant / teacher.
  - PGA 1 (29%) runs from theorist / polymath / robot to ascetic / rebel / demon.
- Conclusion on the slide: there is a manifold, and its axes seem to capture different traits.

## Used by / uses
- Uses `pm_geometry` (components, geometry) and `route_judge_sweep.build.PM`, the data root.
- The axes feed `../pca_pga_steering`.
