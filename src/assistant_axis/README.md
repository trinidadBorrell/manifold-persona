# assistant_axis — the role cloud seen through the Assistant Axis (Stage 1)

**Status:** finished  ·  **When:** 2026-07-23 → 09-22  ·  script folder, run by path

## Question
Six views of the 276 role-mean points:
1. intrinsic dimension
2. clustering
3. UMAP, and how well the Assistant Axis is recovered
4. role families
5. the role map
6. cluster maps

Is there a dominant identity direction, as the Assistant Axis paper (arXiv:2601.10387) claims?

## Files
| File | What it does |
|---|---|
| `01_intrinsic_dimension.py` | ID estimators (TwoNN, MLE, lPCA, MOM, TLE, CorrInt), plus the PCA curve |
| `02_clustering.py` | KMeans, GMM, HDBSCAN and DBSCAN in PCA-95% space, with internal scores |
| `03_umap_axis.py` | UMAP 2D/3D coloured by axis projection; cos(PC1, axis) |
| `04_role_families.py` | Ward dendrogram cut into families |
| `05_role_map.py` | UMAP / PCA role map coloured by family |
| `06_cluster_maps.py` | Every clustering drawn in four embeddings |
| `run_all.py`, `make_report.py` | Runs 01–06 into one dated folder and writes its REPORT.md |
| `layer_compare.py` | The same ID estimators at hidden states 19, 25 and 32 (added 09-22) |

## How to run
```bash
.venv/bin/python src/assistant_axis/run_all.py     # or: python -m manifold_persona.cli explore
```
- **Output:** `src/assistant_axis/figures/<DD-Mon-YYYY-HHMM>/` (gitignored), with REPORT.md.
- **Condor:** `jobs_condor/run_manifold_layers.sh` runs `01_intrinsic_dimension.py` per layer.

## Result
- **Prompt-token findings:** withdrawn on 2026-07-30. In the clean recompute, |cos(PC1, axis)| fell from 0.945 to 0.276, so the apparent alignment came from sequence length.
- **Write-up:** `src/role_manifold/PROJECT_README_2026-07.md`.

## Used by / uses
- **Uses:** `manifold_persona.common` (figure folder, loading).
- **Reused by:** `per_persona_geometry` follows the same recipe.
