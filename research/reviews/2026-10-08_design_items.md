# Design items for the owner, 2026-10-08

These need a decision before the code changes. I did not change them. Suggested option first.

## Findings marked as design items

| Item | Question | Suggested option |
|---|---|---|
| F2 | Base models always get chat-template markup (no plain-text path). | Add a plain-text prompt builder for base checkpoints (role text + question + an answer cue), used when the model has no chat template; keep the chat path for aligned stages. Re-extract base clouds after. |
| F3 | The `default` role's prompt contains the stage's repo name, so the Assistant anchor differs per stage. | Use one fixed name (or drop instruction 3) for every stage of a lineage. Changes the default rows, so re-extract. |
| F5 | Model `generation_config` silently adds repetition_penalty / top_k / top_p to "greedy" runs. | Pass explicit decoding settings (repetition_penalty=1.0, no top_k/top_p when greedy) and record them in the manifest. Changes Qwen-Instruct clouds on re-extraction. |
| F18 | `cos_axis` equals `cos_centroid`, and `axis_proj` = `mean_norm` × `cos_centroid`; after controls axis_proj and cos_centroid still correlate 0.785. | Keep one centroid-level predictor (cos_centroid) plus the two cloud-level ones, or report axis_proj only with mean_norm as a control and say they overlap. |
| F21 | The clustering design null has no residual term, so any noise looks like persona structure. | Add a residual drawn from the pooled within-cell residual covariance. |
| F22 | The planted-manifold calibration in compute_budget is almost noise-free and its scale setting does nothing. | Match noise to the data's within-role residual, and drop the no-op scale. |
| F25 | The residual-noise null compares disjoint question halves within roles but the same halves between roles. | Use disjoint halves in both (role i-1 even vs role i odd). |
| F36 | Topology lifetimes change with cloud scale; density and curvature do not. | Divide H*_total_persistence and H*_max_lifetime by the cloud diameter (the _frac columns already do). |
| F41 | HDBSCAN `min_cluster_size` is 2, not a group size as the comment says; results are not harmed. | Either set it to n/(4·max(n_i,n_q)) as the comment says, or fix the comment. |
| F52 / F59 | The H1 positive control uses the role-shuffle null but H1 is decided by covgauss_null, which cannot detect a planted curved manifold; the TPS kernel is wrong (r^(k-2) log r). | Replace the TPS with scipy RBFInterpolator(thin_plate_spline, degree=1); gate the positive control on the deciding null and add a flat-Gaussian negative control. All H1 numbers change. |
| F53 | local_id neighbour_corr has no null; a homogeneous cloud gives 0.3–0.5. | Compute it on the negative-control draws and report the real value against that band (or Moran's I). |
| F71 / F80 | The steering replay and steering prompts differ from the cloud's token basis and prompt format (few-shot, EOS). | Decide one comparison basis; then make capture_activations pool the same span, with the same prompt format, as the cloud. F71 part (c), the silent truncation, is fixed (F72 prints the count). |
| F92 | adoption_report outlier flags are per-role 5% tests with no multiplicity correction (~14 of 275 flagged under the null). | Apply BH to the per-role score-test p-values, or label the count "unadjusted". |
| F98 / F99 | OLMo pins rest on base clouds made with the Instruct template; the "current" robustness tier is a byte copy of the legacy tier. | Re-pin after the base-prompt decision (F2) and after a fresh baseline run. |
| F26 / F97 pin | The fold-change pin (30.16x) was measured with the prompt cloud at layer 26 and the response cloud at 19. At matched depth (19/19) it is 53.7x. The exact pin now fails; the "> 10" rule still holds. | Re-pin with `check_invariants.py --full --pin` after review. |
| F17 (rest) | CI overlap between nested tiers is a weak stability test; a 0.20 shift in r still overlaps. | Bootstrap the paired difference r(tier) − r(40) over the shared roles; flag tiers whose CI excludes 0. |
| F24 (rest) | calib_estimators.py says it is a hard gate, but no pipeline runs it. MLE fails it today (worst error 0.38 > 0.2). | Run it first in run_geometry and stop on failure, OR drop MLE from the headline figures and say so. |
| F101 (rest) | gate_check G5 passes trivially when a source is a large CSV (almost every 2-decimal value appears). | Cite the exact column or ledger key and match only there. |
| F45 (rest) | When there is no system prompt, some chat templates insert their own default system text; that text is not counted. | Count system tokens from the rendered prompt (with vs without the system turn). |

## Custom-method items (Step B) that are design or refactor choices

| Item | Question | Suggested option |
|---|---|---|
| KDE column | kde_logdens_mean is a deterministic function of the kNN distance (R² ≥ 0.99998), so the density family counts one measurement twice. | Remove it from PANEL_COLS, or use a bandwidth that does not depend on kNN (likelihood CV). |
| Bending energy | `bending_energy` depends on scale (×4 when scale doubles) and is nonzero for a straight path. | Use ∫κ² ds, or normalise by chord length, before comparing across cases. |
| Holm floor in the sweep | With N_PERM=100 the sweep's Holm step over 8 n-values cannot go below 0.079. | Raise permutations to 999 or drop the median-p Holm step. |
| B4 bootstrap | B4 uses B_BOOT=20, so the "range/sd" ratio varies ~2x between seeds. | Raise B_BOOT to ≥200 or report the Monte Carlo error. |
| Eigengap / persistence N | The eigengap N changes with knn and a Gaussian null gives N=2; persistence_sweep fails a 4-blob control. | Treat both as weak; use single-linkage merge heights (the H0 barcode) instead. |
| Deduplicate participation ratio (8 copies), assistant axis (≈6 copies), ID wrappers (5 copies, MLE k=10 vs 20), bootstrap loops (9 copies), Gaussian samplers | Copies agree numerically today but differ in edge handling and hidden choices (PCA-50 vs uncentred, MLE k). | One shared helper per method; name the variants in output keys (e.g. PR_pca50, PR_uncentred). Changes output column names. |
| ID calibration spread | 'max_recovered_spread_across_radii' is seed noise (radii span only 1.7%). | Sweep radii over a real range at a fixed seed, or rename it to a seed spread. |
