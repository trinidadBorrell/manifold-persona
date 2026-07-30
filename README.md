# manifold-persona

Geometry of persona/trait activations in an LLM. We reproduce the **embedding
step** of *Persona Vectors* and then study the resulting point cloud through the
lens of two follow-up ideas: that persona representations lie on a low-dimensional
**manifold**, and that persona space has a dominant identity direction (the
**Assistant Axis**).

**References**
- Persona Vectors: Monitoring and Controlling Character Traits in Language Models —
  Chen, Arditi, Sleight, Evans, Lindsey (arXiv:2507.21509). Source code +
  trait artifacts reused from the sibling `../persona_vectors` repo.
- Manifold Steering Reveals the Shared Geometry of Neural Network Representation
  and Behavior (arXiv:2605.05115) — motivates intrinsic-dimension estimation.
- The Assistant Axis (arXiv:2601.10387) — motivates the PCA / leading-component view.
- Anthropic, *Persona Selection Model* (alignment.anthropic.com/2026/psm).

## What it does

**Stage 0 — embedding generation** (`extraction/`). The main cloud is built from
the **276 character-role archetypes** vendored from the Assistant Axis repo (see
`refs/assistant-axis/`, `PROVENANCE.md`). For every role we form persona-conditioned
chats — each role's 5 instruction phrasings × 5 shared sampled questions
(`system(role) + user question`) — render them through the model's chat template,
and extract **prompt activations** (mean-over-tokens `prompt_avg` and final-token
`prompt_last`) from **all 37 layers** of `Qwen/Qwen2.5-3B-Instruct`. This yields a
labelled point cloud of **6,900 points** (276 roles × 5 instructions × 5 questions),
saved under `data/embeddings_roles/`. Run it with:

```bash
.venv/bin/python -m manifold_persona.cli extract              # default: prompt tokens
.venv/bin/python -m manifold_persona.cli extract --response   # answer tokens, ~0.5 depth
```

The default reads **prompt** tokens at layer **26** — the depth-scaled equivalent of
Persona Vectors' layer 20/28 for Qwen2.5-7B (~0.71 depth). `--response` instead
generates an answer and reads the **response** tokens at ~half depth
(`generate_and_extract_roles.py`), matching the Assistant Axis paper's extraction.

**Stage 1 — exploratory** (`exploratory/`):
1. `01_intrinsic_dimension.py` — non-linear ID estimators (TwoNN, MLE, lPCA,
   MOM, TLE, correlation dimension) + PCA baselines, globally and per-trait.
2. `02_clustering.py` — centroid (KMeans auto-k, fixed k=7/k=14, GMM) and
   density (HDBSCAN, DBSCAN) clustering, scored by silhouette / Davies-Bouldin
   and — since labels exist — **ARI / NMI vs trait and polarity**.
3. `03_umap.py` — UMAP 2D (static) + 3D (interactive HTML), plus a PCA(1,2)
   "Assistant Axis" view.
4. `04_traits_per_cluster.py` — cluster×trait contingency heatmap, composition
   bars, dominant-trait/purity table.

## ⚠️ Findings below are WITHDRAWN pending recompute (2026-07-30)

The `prompt_avg` view they were computed from was contaminated by a first-token
**attention sink**, which made its **PC1 essentially `1/sequence_length`**
(|r| = 0.9998, 78% of variance) rather than anything about personas. Because
system-prompt length differs by role, it also produced a *fake* between-role
signal. Fixed in `src/manifold_persona/extract.py`; full measurement and
mechanism in **[diagnostics/README.md](diagnostics/README.md)**.

Distance-based ID estimators are translation-invariant, so a constant offset
would have been harmless — but the sink term is scaled by `1/T`, so it varies.
Measured `r(d(prompt_avg), d(sink-excluded)) = 0.42`: the artifact was inside the
geometry being measured. **Both numbers below need recomputing on a clean cloud.**

A cloud is clean iff its `data/embeddings_roles/manifest.json` contains a
`sink_factor` key.

Recomputed on 2026-07-30 (Qwen2.5-**0.5B**, layer 17, 276 roles × 5 × 3 = 4,140
records, both with and without the sink). Verdict is split:

- **Low-dimensional manifold — SURVIVES.** Every distance-based estimator is
  essentially unchanged by the fix (TwoNN 11.8 → 11.3, MLE 13.0 → 12.6, TLE
  12.2 → 11.9). ID ≈ **10–13** vs ambient **896** is real. *But* the linear
  picture was badly understated: lPCA 4 → 25, and PCs for 90% variance 42 → 71.
- **Assistant Axis — DOES NOT SURVIVE.** `|cos(PC1, axis)|` falls **0.945 → 0.276**.
  The apparent alignment was sequence length (raw PC1 correlates with `1/T` at
  0.997). Worse, the axis is *structurally* confounded with length: the `default`
  baseline is the shortest condition in the design by construction (mean 32.6
  tokens vs 41.7 for roles; no role's mean falls below it), and the projection
  still correlates with `1/T` at **0.64** even after the fix.

Caveat: this is 0.5B and *prompt* tokens, so it does not test the papers' claims —
they use response tokens (which excludes the sink automatically) at 27B–70B.

## Setup & run

On a CPU-only machine see **[SMOKE.md](SMOKE.md)** instead — it has a verified
CPU-wheel setup, a small-model smoke sequence, and measured runtimes.

```bash
python -m venv --system-site-packages .venv     # inherits an existing torch, if present
.venv/bin/pip install -r requirements.txt

# Sanity-check the apparatus before trusting any geometry (~1 min, CPU, 0.5B model)
.venv/bin/python diagnostics/01_activation_scales.py

# Stage 0: build the role point cloud (data/embeddings_roles/)
.venv/bin/python -m extraction.build_and_extract_roles     # prompt-token activations
# Push the point cloud to Hugging Face (<username>/manifold-persona)
.venv/bin/python -m extraction.push_to_hf                  # token read from token/huggingface.txt

# Stage 1: exploratory — runs 01-06 + writes REPORT.md into one dated folder
.venv/bin/python exploratory/assistant_axis/run_all.py
```

Each run writes to a **timestamped folder**
`exploratory/persona_vectors/figures/<DD-Mon-YYYY-HHMM>/` containing every figure
(PNG + interactive 3D HTML), the raw metric dumps (`*.json`, `*.csv`), and an
auto-generated **`REPORT.md`** that pulls the run's real numbers, embeds each
figure, interprets it, and cites the code that produced it. The orchestrator
fixes one timestamp and shares it across the four scripts via the `MP_RUN_DIR`
env var; each script also accepts `--outdir`.


## Layout

```
docs/papers/ + docs/notes/     # reference PDFs + reading notes (see docs/README.md)
diagnostics/                   # apparatus checks: activation scales, attention sinks
src/manifold_persona/          # config, io, extract, common, runlog; prompts_roles.py (roles)
extraction/                    # build_and_extract_roles.py (prompt tokens),
                               # generate_and_extract_roles.py (response tokens), push_to_hf.py
manifold/                      # H1 manifold study: pipeline, tps, run, sweep, local_id
exploratory/persona_vectors/   # trait study: 01–04 + common + run_all + make_report + figures/<stamp>/
exploratory/assistant_axis/    # role study:  01–04 + common + run_all + make_report + figures/<stamp>/
data/embeddings/               # trait point cloud (gitignored; mirrored on HF)
data/embeddings_roles/         # role point cloud (gitignored)
token/huggingface.txt          # HF token (gitignored)
```

Config knobs live in `src/manifold_persona/config.py` (model, traits, paths, and the
layer helpers `primary_layer` / `half_depth_layer` / `half_depth_hidden_state`). Point
the extractor at a different model with `MP_MODEL_NAME`, or a relocated source repo with
`ASSISTANT_AXIS_DIR` / `PERSONA_VECTORS_DIR`. `MP_ROLE_DIR` repoints the whole stack at a
different cloud; `MP_AGGREGATE=none` disables role-mean collapsing.

Install once so the packages import from any working directory:

```bash
.venv/bin/pip install -e .
```
