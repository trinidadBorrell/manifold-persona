# Changelog

## 2026-08-03 — Review repairs (steps 1–8)

Source: full-codebase review with adversarial verification
(`output/code_review_2026-08-03.html`). 41 critical/major findings
checked by independent agents: 38 confirmed, 3 partial, 0 refuted.
This entry lands every repair that needs no new extraction. Step 9
(clean re-extraction of the prompt cloud + reruns) is still open.

### Safety

- `extraction/generate_and_extract_roles.py`: the completed-run guard
  now compares every `run_config` key plus `n_records`. Before, it
  compared 3 of 8 keys and a `--limit 8` smoke run silently overwrote
  the 6,900-record response cloud. Manifests now record all config
  keys (canonical name `model_name`; `seed`, `temperature`, `limit`,
  `chunk` added).
- `extraction/build_and_extract_roles.py`: new refusal guard. The
  script stops when the output dir holds a completed run, unless
  `--force` is given. The docstring's smoke-test command now includes
  a safe `--out_dir`.
- `src/manifold_persona/io.py`: `load_layer` refuses a prompt cloud
  whose manifest lacks a non-null `sink_factor` (a pre-fix or
  `--keep_sinks` cloud). `MP_ALLOW_UNCLEAN=1` bypasses with a stderr
  banner. Response clouds (`token_basis: response`) load unchanged.
  Layer indices are now bounds-checked with a clear error.

### Statistics

- `manifold/pipeline.py`: new `coordinate_null` — permutes each
  coordinate of the real role means across roles. It keeps the
  marginals and destroys joint structure, so it tests manifold
  structure. The old `permutation_null` shrank its fake means ~4x
  toward the global mean and only measured between-role spread.
- `manifold/run.py` + `manifold/report.py`: the preregistered decider
  now judges against the coordinate null and reports both nulls.
  **Result change:** on the current cloud the verdict flips from
  SUPPORTED (rel_reduction 0.451 vs the old null) to WEAK
  (rel_reduction 0.0065 vs the 0.30 floor). `manifold/sweep.py` is
  unchanged; its frozen PRIOR gate predates the new null (open item).
- `manifold/idim.py`: MLE now sets `n_neighbors` at fit time.
  `MLE(K=...)` was dead in skdim 0.3.6 — K is ignored and every MLE
  ran with the library default of 20 neighbors. **Result change:**
  resp40 median MLE moves 6.60 → 4.87; prompt-cloud 4.09 → 3.78
  (still outside the design null).
- `exploratory/per_persona/04_id_vs_axis.py` + `common.py`: new
  role-level text-length control. The script now reports partials
  with and without mean response length. **Result change:** the
  resp40 headline weakens from partial r −0.78 (scale only) to
  −0.61 (scale + length); direction intact.
- `output/e8_crossfit_40q/role_length_control.py` (new) +
  `LENGTH_CONTROL.md` (new): role-level length audit of the E8
  endpoints. E8's primary endpoint (`relative_gplvm_gain_avg_d5`) is
  unrelated to length (r = 0.003). The local-ID vs axis link does not
  survive the length control (+0.21 → +0.01). 88.6% of responses hit
  the 128-token cap.

### Reports

- `manifold/sweep_report.py`: the verdict text now branches on the
  computed decision; all directional claims are computed from the
  metrics frame or deleted; the section for the never-generated
  `fig08_mst_skeleton` is removed; mesh references now match the
  plots.
- `exploratory/assistant_axis/make_report.py`: counts, cosines, and
  the interpretation ladder derive from the manifest and computed
  tables; a new provenance section prints `sink_factor` and the cloud
  path, so clean and pre-fix runs produce different documents.
- `manifold/analysis_extra.py`: the manifold-count conclusion now
  branches on the computed eigengap / persistence values.
- `exploratory/per_persona/run_all.py`: the generated report's
  explanation of `default`'s position now states the measured length
  confound first (matches METHODS.md).

### Docs

- `README.md`: clean rule corrected to "`sink_factor` present and not
  null"; states that the on-disk prompt cloud fails it; both the 0.5B
  probe numbers and the measured 3B numbers given; Stage-1 rewritten
  to the six real scripts; per-persona study documented; layout and
  paths fixed (`exploratory/assistant_axis/figures/`,
  `data/embeddings_roles_resp_40q/`).
- `data/embeddings_roles/README.md`: contamination warning added;
  the "Reproduce" command marked not reproducible as-is.
- `exploratory/per_persona/README.md` + `METHODS.md`: contamination
  banners; the design-null claim corrected (the null reproduces the
  clustering, not the ID — real ID outside the null IQR on 5/5
  estimators); resp40 commands fixed to
  `MP_ROLE_DIR=data/embeddings_roles_resp_40q` with `--layer 19`;
  `default`'s axis position explained by the measured length confound.
- `output/e5_algo_gaps/REPORT.md`: aligned with RESULTS.md (9/10
  splits; the d=2 response minimum is k=6; prompt arm relabeled a
  confounded sensitivity check).
- `output/e8_crossfit_40q/RESULTS.md` + `output/e1b_gplvm_repro/RESULTS.md`:
  "100% converged" scoped to held-out convergence; training-selected
  convergence at d=12 is 25%.

### Build

- `.gitignore`: `output/` ignore rules restructured so experiment
  scripts can be tracked (`!output/**/*.py`). The script that made
  E8's primary result was unrecoverable; this closes that hole.

### Known open items

- Step 9: re-extract the prompt cloud with the sink fix; rerun
  assistant-axis, per-persona prompt arm, E5/E7 prompt arms; decide
  the `sweep.py` PRIOR gate.
- Same dead-`K` MLE pattern still present (out of scope here):
  `diagnostics/02_sink_impact.py`, `exploratory/assistant_axis/01_intrinsic_dimension.py`,
  `output/e7_id_reconciliation/run_e7.py`.
- ~30 minor review findings not yet individually verified or repaired.
- Rerun outputs under `output/manifold_h1-2/2026-08-03T13-41` and the
  per-persona figures tree were computed on the pre-fix cloud with
  `MP_ALLOW_UNCLEAN=1` — comparable, not publishable.
