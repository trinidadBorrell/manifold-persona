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
