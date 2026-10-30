# cloud_predicts_position — does the shape of a persona's answer cloud predict where it lies on the assistant axis? (Oct 29–)

**Status:** Experiments 1 and 2 done 2026-10-30 (Part B of `plans/2026-10-29-geometry-predicts-position-and-misalignment.md`, local)  ·  **Data:** the full-design clouds (276 roles × 5 prompts × 240 questions, Qwen3-8B, response-token mean, layers 19, 25, 32; `data/embeddings_roles_qwen3_8b*`)  ·  **Outputs:** `$PM/CPP/`, `output/29-oct/partB/`

## Question
For each persona, the 1,200 answer activations form a cloud. Metrics of that cloud (intrinsic dimension, spectral shape, density, topology, curvature, anisotropy relative to the assistant axis, design decomposition, alignment with the default cloud, consistency) are the predictors. The target is the projection of the persona's centroid on the assistant axis (Lu et al.: mean default − mean of all roles, in the full 4096-d residual stream).
- **Experiment 1 (hold out personas):** fit position ~ metric on i personas, predict the rest. Baselines: training mean, permutation null, scale-only model (log cloud variance, centroid norm, response length).
- **Experiment 2 (hold out questions):** metric from the first i questions, fit the line, then place the metric from a matched number of held-out questions on the same line; split-half reliability of each metric vs i.

## Setup
- Clouds: question-centred (`q`: answer minus the `default` role's answer to the same question under the same prompt index; primary) and raw. Layer 25 primary, 19 and 32 robustness; view `last` at layer 25 as a further column.
- Metrics (46 candidates, 9 pre-registered primaries: `MLE`, `effective_rank`, `knn_dist_mean`, `H0_total_persistence`, `orc_mean`, `var_frac_along_axis`, `interaction_frac`, `cka_linear`, `between_prompt_disp`): definitions in `panel.py` (`FAMILIES`) and plan §2. Low-dimensional metrics use the cloud's own PCA-50; topology on 400-point subsamples × 3 (ripser, maxdim 2); fixed seeds.
- Decision rule (Exp 1, i = 200): median test R² ≥ 0.25, above the 95th percentile of the permutation null, and ΔR² over the scale-only model ≥ 0.05.

## Files
| File | What |
|---|---|
| `data.py` | Paths and loaders for the full-design clouds |
| `axis.py` | Stage 0: role centroids, assistant axis, target `axis_proj`, default-cloud summaries → `$PM/CPP/axis_<view>_L<L>.npz` |
| `panel.py` | Stage 1: the metric panel per chunk of roles → `$PM/CPP/panel/<view>_L<L>/chunk_*.csv`; `--silhouette`; `--collect` → `panel_<view>_L<L>.csv` |
| `exp1.py` | Experiment 1 → `$PM/CPP/exp1/exp1_<tag>.csv`, `_curves.json` |
| `exp2.py` | Experiment 2: one job per (seed, i) → `$PM/CPP/exp2/metrics_*.csv`; `--analyse` → `exp2_<tag>_summary.csv` |
| `figs.py` | F1–F6 and the tables into `output/29-oct/partB/` |

## Commands
```bash
condor_submit jobs_condor/cpp_axis.submit                                            # 4 jobs
condor_submit LAYER=25 VIEW=avg NCH=69 jobs_condor/cpp_panel.submit                  # and 19/32 avg, 25 last
condor_submit MOD=cloud_predicts_position.panel ARGS="--layer 25 --view avg --silhouette" jobs_condor/cpp_misc.submit
condor_submit MOD=cloud_predicts_position.panel ARGS="--layer 25 --view avg --collect" jobs_condor/cpp_misc.submit
condor_submit MOD=cloud_predicts_position.exp1 ARGS="--layer 25 --view avg" jobs_condor/cpp_misc.submit
condor_submit LAYER=25 VIEW=avg jobs_condor/cpp_exp2.submit                           # 70 jobs (10 seeds × 7 i)
condor_submit MOD=cloud_predicts_position.exp2 ARGS="--layer 25 --view avg --analyse" jobs_condor/cpp_misc.submit
python -m cloud_predicts_position.figs --layer 25 --view avg
```

## Result (layer 25, response-token mean, question-centred; `output/29-oct/partB/avg_L25_q/tables.md`)
Experiment 1, hold-out over personas (i = 200 training personas, 100 orderings; sign: + = toward the default assistant).
- **Single cloud metrics predict the position well.** Test R² 0.85 for the graph path / straight-line ratio (`path_ratio_p50`), 0.81 Forman curvature mean, 0.78 correlation dimension, 0.76 PCA participation ratio, 0.75 `interaction_frac`, 0.71 `MLE`, 0.61 `knn_dist_mean`, 0.60 `H0_total_persistence`, 0.49 `effective_rank` and `cka_linear`. The permutation null's 95th percentile is 0.02. A ridge over the whole panel reaches 0.93.
- **But cloud size and answer length already predict it.** The scale-only model (log cloud variance, raw centroid norm, mean response length) has test R² 0.80. Personas closer to the assistant have smaller clouds (Spearman of log variance with position −0.82) and longer answers (+0.78). Under the pre-registered rule (ΔR² over scale ≥ 0.05), 20 of 53 metrics add information beyond scale, by ΔR² 0.05–0.10: the curvature family (`path_ratio_p50`, `frc_mean`, `frc_sd`), `corr_dim`, `PCA_participation_ratio`, `knn_dist_cv`, persistence entropies, `interaction_frac`, `knn_dist_mean`, `effective_rank`. `MLE` and `H0_total_persistence` are scale proxies; `orc_mean`, `var_frac_along_axis` and `between_prompt_disp` do not predict at all.
- **Direction of the relations:** assistant-like personas have higher local intrinsic dimension (`MLE` +0.87, `local_id_mean` +0.80), a lower participation ratio and higher top-1 share (variance concentrated in fewer directions), shorter MST and lower total persistence (tighter), a larger interaction share in the prompt × question decomposition (+0.91), and question geometry more aligned with the default role's (`cka_linear` +0.79). This is the opposite sign for intrinsic dimension from the August Qwen2.5-3B result, which was on raw prompt-token clouds.
- **Raw clouds** (no question-centring): single metrics reach 0.58 (`orc_mean`) to 0.83 (`subspace_overlap_k10`), but the scale-only model is at 0.87 and nothing passes the ΔR² rule.
- **Caveat to settle with the `last` view and the other layers:** with a 128-token cap and mean pooling, longer answers mechanically give tighter token-mean clouds, so part of the "scale" effect may be answer length; the single-token `last` view (running) has no such averaging.
- **Layer 19 is stronger** (`output/29-oct/partB/avg_L19_q/`): the scale-only model drops to R² 0.69, while `top10_share` 0.86, `effective_rank` 0.86, `frc_mean` 0.84, `persistence_entropy_H0` 0.84, `knn_dist_mean` 0.83, `path_ratio_p50` 0.82 and `interaction_frac` 0.82 each add ΔR² 0.19–0.23 over scale; 38 of 53 metrics pass the rule, including 7 of the 9 primaries (`orc_mean` and `between_prompt_disp` fail at both layers).

Experiment 2, hold-out over questions (layer 25, question-centred, 10 orderings; `output/29-oct/partB/avg_L25_q/tables.md`, F4–F6).
- **The position itself is pinned down by 5 questions** (reliability 0.99 between the first 5 and 5 held-out questions, all 5 prompts).
- **The cheap metrics are question-stable almost as fast:** `H0_total_persistence`, `interaction_frac`, `between_prompt_disp` reach reliability ≥ 0.8 and a held-out / training RMSE ratio ≤ 1.2 at i = 5, `MLE` at 10, `frc_mean` and `var_frac_along_axis` at 20. `effective_rank`, `cka_linear`, `orc_mean` need 40 questions, `knn_dist_mean` 60, `top1_share` 120.
- The metric → position lines fitted on the first i questions hold for the held-out questions: the RMSE ratio is 1.0–1.1 at i ≥ 40 for every primary metric. So the relation is a property of the persona, not of the question sample.

**Last-token view, layer 25** (`output/29-oct/partB/last_L25_q/`). Single metrics again predict well (`orc_mean` 0.80, `persistence_entropy_H1` 0.76, `frc_mean` 0.72, `corr_dim` 0.69), but the scale-only model is at 0.88 and **no metric adds ≥ 0.05 over it**. Two things differ from the mean-pooled view: (i) the sign of cloud size flips, log variance is +0.76 with position on the last token against −0.82 on the token mean, so "assistant-like personas have tighter clouds" is a property of mean pooling (longer answers, more tokens averaged, tighter mean), not of the last-token state; (ii) `orc_mean`, useless on the token mean, is the best single metric on the last token. What is common to both views is response length, +0.78 / +0.80 with position: personas closer to the assistant write longer answers under the 128-token cap. Any claim that cloud shape predicts position has to be read against that.

**Layer 32** (`output/29-oct/partB/avg_L32_q/`): the same metrics predict (path ratio 0.79, Forman curvature 0.73, correlation dimension 0.70), but the scale-only model is at 0.94 and nothing adds to it.

**Across layers (question-centred, token mean):** the scale-only R² rises with depth, 0.69 at layer 19, 0.80 at 25, 0.94 at 32, and the number of metrics that add ≥ 0.05 over it falls, 38 → 20 → 0 of 53. The shape of the cloud carries information of its own at the middle layers; by layer 32 the position is a function of cloud size and answer length alone.

## Used by / uses
Uses `per_persona_geometry.{density,curvature,topology}` (flat modules), `role_manifold.idim`, `curvature_null.run` (bend test, path-ratio trend, nulls), `manifold_persona.common.assistant_axis` (definition copied into `axis.py`).
