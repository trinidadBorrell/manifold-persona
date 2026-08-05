# diagnostics/

Checks on the *measurement apparatus*, not on personas. Run these before trusting
any geometry result. They are laptop-sized: CPU only, a 0.5B model, seconds to a
couple of minutes.

```bash
.venv/bin/python diagnostics/01_activation_scales.py                       # default: 0.5B, 6 roles
.venv/bin/python diagnostics/01_activation_scales.py --model Qwen/Qwen2.5-3B-Instruct
```

---

## 01_activation_scales.py — findings on Qwen2.5-0.5B-Instruct, 2026-07-30

Measured with 90 records of the real role pipeline at `hidden_states[12]`.

### Finding 1 — the residual stream is not normalized

Mean ‖h‖ ranges **0.44 → 91** across layers 0–23. These are pre-norm
architectures: RMSNorm is applied to a *copy* entering each sublayer, so the
residual stream itself is an unnormalized running sum whose scale grows with
depth. Any analysis that assumes unit-norm vectors, or that compares layers
without rescaling, is wrong.

### Finding 2 — `hidden_states[L]` is normalized and the rest are not

`output_hidden_states` returns `L+1` tensors. HF applies the model's final norm to
the last one, so `hidden_states[24]` is post-norm while `0..23` are raw residual —
a measured **3.95×** scale jump. Verified by `torch.allclose` against the inner
model's `last_hidden_state`. **Do not include index `L` in a layer sweep.**

### Finding 3 — massive activations at the attention sink

At layer 12: median ‖h‖ = 15.4, but position 0 (`<|im_start|>`) has
‖h‖ = **1728** — 112× the median — concentrated almost entirely in **channel 62**
(1705 vs a median channel max of 1.75).

### Finding 4 — this made `prompt_avg` measure sequence length, not persona

This is the one that mattered. Under causal attention, position 0 attends only to
itself, so its hidden state is **bit-identical across every record** (measured max
deviation `0.000e+00`). It is a constant vector — but the mean divides by `T`:

```
prompt_avg  =  (1/T)·h_sink  +  (1/T)·Σ(content positions)
```

so its contribution varies as `1/T` and nothing else. Consequences measured on the
raw (pre-fix) `prompt_avg`:

| quantity | value |
|---|---|
| cosine(`prompt_avg`, same mean without sink positions) | **0.24** (mean over 90 records) |
| share of variance held by channel 62 alone | **73.4%** |
| PC1 explained variance | **78.1%** |
| \|r(PC1, 1/T)\| | **0.9998** |
| pearson r( d(`prompt_avg`), d(sink-excluded) ) | **0.42** |

PC1 of the main point cloud was prompt length to four decimal places. And because
system-prompt length differs by role (measured 38.0–47.6 tokens across 6 roles),
this produced a **fake between-role signal**: `prompt_avg` showed the *highest*
between/within-role ratio (1.95) of the three views, purely from length.

Why intrinsic-dimension estimates were affected: TwoNN/MLE/correlation-dimension
consume pairwise distances and are translation-invariant, so a *constant* offset
would have been harmless. But the offset is scaled by `1/T`, so it **varies** —
r = 0.42 between the two distance matrices confirms the sink was inside the
measured geometry, not outside it.

### The fix

`extract_prompt_activations(..., sink_factor=5.0)` drops positions whose norm
exceeds 5× the record's median norm, per layer, before averaging. Recorded in
`manifest.json` as `sink_factor`. `--keep_sinks` restores the old behaviour.

After the fix, on the same 90 records: PC1 drops **78.1% → 17.9%** and
\|r(PC1, 1/T)\| drops **0.9998 → 0.6127**.

**Remaining confound, not fixed.** r = 0.61 is still substantial. Prompt length is
genuinely correlated with role (different archetypes get different-length system
prompts), and averaging over more tokens changes the estimate's variance. Length is
a real confound in this design, not only an artifact — it needs a design answer
(match prompt lengths, or regress length out, or condition on it), not a pooling
tweak.

**Any `prompt_avg` cloud whose `manifest.json` has no `sink_factor` key predates
this fix.** Its PC1 is length. Recompute before interpreting.

---

## 02_sink_impact.py — what the fix changed, at full scale

Two clouds, identical in every respect except sink handling:
**276 roles × 5 instructions × 3 questions = 4,140 records**,
Qwen2.5-0.5B-Instruct, layer 17, seed 0. ~30 min each on 3 CPU cores.

```bash
.venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 --out_dir data/embeddings_roles_full_fixed
.venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 --keep_sinks --out_dir data/embeddings_roles_full_raw
.venv/bin/python diagnostics/02_sink_impact.py
```

| quantity | RAW | FIXED |
|---|---|---|
| per-record PC1 share of variance | 0.440 | 0.104 |
| per-record \|r(PC1, 1/T)\| | **0.998** | 0.598 |
| role-mean PC1 share | 0.493 | 0.166 |
| role-mean \|r(PC1, 1/mean_T)\| | **0.997** | 0.614 |
| role-mean **\|cos(PC1, Assistant Axis)\|** | **0.945** | **0.276** |
| between/within role ratio | 1.443 | 1.232 |
| PCs for 90% of variance | 42 | **71** |

Intrinsic dimension of the 276 role centroids:

| estimator | RAW | FIXED |
|---|---|---|
| TwoNN | 11.82 | 11.31 |
| MLE | 12.96 | 12.59 |
| TLE | 12.21 | 11.88 |
| CorrInt | 9.85 | 10.95 |
| MOM | 8.31 | 10.61 |
| lPCA | **4.00** | **25.00** |

### Reading this

**The low-dimensional-manifold claim survives.** Every *distance-based* estimator
is essentially unmoved (TwoNN 11.8 → 11.3, MLE 13.0 → 12.6, TLE 12.2 → 11.9). That
is what theory predicts: they consume pairwise distances, and the sink adds
essentially one extra varying direction to an already ~11-dimensional cloud.
Intrinsic dimension ≈ **10–13** against ambient 896 is a real result and is not an
artifact.

**The linear/PCA picture does not survive.** lPCA jumps 4 → 25 and the components
needed for 90% of variance go 42 → 71, because in the raw cloud one eigenvalue (the
sink) dwarfed everything. The cloud is substantially *higher*-dimensional in the
linear sense than the raw view implied.

**The Assistant-Axis result does not survive.** `|cos(PC1, axis)| = 0.945` in the
raw cloud looked like a strong reproduction of the paper's finding. But raw PC1 is
sequence length at r = 0.997 — so that alignment was length, not identity. After
the fix it falls to **0.276**.

### Why: the Assistant Axis is structurally confounded with prompt length

This is a design issue, not an implementation bug, and the fix above does not
remove it.

The axis is defined as `mean(default points) − mean(all points)`. But the `default`
baseline is *by construction* the shortest condition in the whole design — its five
instructions are `""` (empty), "You are an AI assistant.", "You are a large language
model.", "You are {model_name}.", "Respond as yourself.", while every role gets a
descriptive multi-clause system prompt. Measured:

| | n | mean tokens | range |
|---|---|---|---|
| `default` | 15 | **32.6** | 26–40 |
| the 275 roles | 4,125 | **41.7** | 34–62 |

**No role's mean prompt length falls below the default's mean** (0.0 percentile).
So "distance from the Assistant" and "prompt length" are collinear by design. Even
in the *fixed* cloud:

```
r(assistant-axis projection, 1/mean_T) = +0.64   over 276 roles
r(assistant-axis projection,   mean_T) = −0.62
```

Any claim that PC1 measures Assistant-likeness has to rule this out first —
by matching prompt lengths, regressing length out, or conditioning on it.

### Does this affect the papers? Probably not — and that's the point

Both Persona Vectors and the Assistant Axis average over **response** tokens. The
attention sink sits at position 0 of the *prompt*, so it is **excluded from their
average automatically**. They are immune to this specific artifact.

This repo introduced it by deviating from them — averaging over *prompt* tokens
instead. Persona Vectors
already warned that response tokens work better (their footnote 2); this is a
concrete, quantified reason why. `prompt_last` is also immune, since the final
position is never the sink.

The length confound is a separate matter and could in principle touch the papers
too, via response length rather than prompt length. Untested here.
