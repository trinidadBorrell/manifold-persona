# diagnostics/

Checks on the measurement apparatus, not on personas. CPU, 0.5B model.

```bash
.venv/bin/python diagnostics/01_activation_scales.py
.venv/bin/python diagnostics/01_activation_scales.py --model Qwen/Qwen2.5-3B-Instruct
```

## 01_activation_scales.py

Qwen2.5-0.5B-Instruct, 90 records, `hidden_states[12]`.

1. Residual stream is not normalized: mean ‖h‖ runs 0.44 → 91 over layers 0–23.
2. `hidden_states[L]` is post-norm, `0..L-1` are not — a 3.95× scale jump. Do not
   include index `L` in a layer sweep.
3. Position 0 (`<|im_start|>`) has ‖h‖ = 1728 vs a median of 15.4, concentrated in
   channel 62 (1705 vs a median channel max of 1.75).
4. It is constant across records (max deviation `0.000e+00`), so mean-pooling gives
   `prompt_avg = (1/T)·h_sink + (1/T)·Σ(content)` and PC1 becomes `1/T`.

Raw `prompt_avg`:

| quantity | value |
|---|---|
| cosine(`prompt_avg`, same mean without sinks) | 0.24 |
| variance held by channel 62 alone | 73.4% |
| PC1 explained variance | 78.1% |
| \|r(PC1, 1/T)\| | 0.9998 |
| r( d(`prompt_avg`), d(sink-excluded) ) | 0.42 |

After `sink_factor=5.0`: PC1 78.1% → 17.9%, \|r(PC1, 1/T)\| 0.9998 → 0.6127.

A `prompt_avg` cloud whose `manifest.json` has no `sink_factor` key predates the
fix; its PC1 is length.

## 02_sink_impact.py

Two clouds, identical except sink handling: 276 roles × 5 instructions × 3
questions = 4,140 records, Qwen2.5-0.5B-Instruct, layer 17, seed 0.

```bash
.venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 --out_dir data/embeddings_roles_full_fixed
.venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 --keep_sinks --out_dir data/embeddings_roles_full_raw
.venv/bin/python diagnostics/02_sink_impact.py
```

| quantity | RAW | FIXED |
|---|---|---|
| per-record PC1 share | 0.440 | 0.104 |
| per-record \|r(PC1, 1/T)\| | 0.998 | 0.598 |
| role-mean PC1 share | 0.493 | 0.166 |
| role-mean \|r(PC1, 1/mean_T)\| | 0.997 | 0.614 |
| role-mean \|cos(PC1, Assistant Axis)\| | 0.945 | 0.276 |
| between/within role ratio | 1.443 | 1.232 |
| PCs for 90% of variance | 42 | 71 |

Intrinsic dimension of the 276 role centroids (ambient 896):

| estimator | RAW | FIXED |
|---|---|---|
| TwoNN | 11.82 | 11.31 |
| MLE | 12.96 | 12.59 |
| TLE | 12.21 | 11.88 |
| CorrInt | 9.85 | 10.95 |
| MOM | 8.31 | 10.61 |
| lPCA | 4.00 | 25.00 |

- Low-dimensional manifold: survives. Distance-based estimators unmoved, ID ≈ 10–13.
- Linear picture: does not survive. lPCA 4 → 25, PCs for 90% variance 42 → 71.
- Assistant Axis: does not survive. \|cos(PC1, axis)\| 0.945 → 0.276.

The axis is also confounded with prompt length by design, which the sink fix does
not touch. `default` is the shortest condition in the design (mean 32.6 tokens vs
41.7 over the 275 roles, no role below it), and in the fixed cloud
`r(axis projection, 1/mean_T) = 0.64`.

Both papers average over response tokens, so the prompt's position-0 sink is
excluded from their means. `prompt_last` is likewise unaffected.
