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

**Stage 1 — exploratory** (`exploratory/assistant_axis/`). Six scripts, all on the
276 **role-mean** points (one mean vector per role):
1. `01_intrinsic_dimension.py` — ID estimators (TwoNN, MLE, lPCA, MOM, TLE,
   correlation dimension) + PCA baselines, plus the PCA cumulative-variance curve.
2. `02_clustering.py` — KMeans (auto-k and fixed k), GMM, HDBSCAN and DBSCAN in a
   PCA-95% space. **Internal** scores only (silhouette, Davies-Bouldin, cluster
   counts, noise fraction) — after the role-mean collapse there is no label left
   to score against. Each cluster gets a size, a mean axis projection, a
   holds-`default` flag and representative roles; assignments go to
   `clusters_rolemean_<view>_L<layer>.parquet`.
3. `03_umap_axis.py` — UMAP 2D + interactive 3D HTML, plus PCA(1,2) coloured by
   the axis projection, and `cos(PC1, assistant_axis)`.
4. `04_role_families.py` — Ward dendrogram of the 276 role centroids cut into K
   families, and per-cluster composition (size, top roles, %default, mean axis
   projection).
5. `05_role_map.py` — the role map: UMAP and PCA of the role centroids, coloured
   by Ward family, with role labels + 3D HTML.
6. `06_cluster_maps.py` — every clustering drawn in four embeddings (UMAP-2D/3D,
   PCA-2D/3D) with `default` starred, plus a per-method interpretability dump.

`run_all.py` runs all six into one dated folder; `make_report.py` writes that
folder's `REPORT.md`.

## ⚠️ Findings below are WITHDRAWN pending recompute (2026-07-30)

The `prompt_avg` view they were computed from was contaminated by a first-token
**attention sink**, which tied its **PC1 to `1/sequence_length`** rather than to
anything about personas. On the 0.5B layer-12 probe that link is near-total
(|r(PC1, 1/T)| = 0.9998, PC1 = 78% of variance — see
**[diagnostics/README.md](diagnostics/README.md)**); measured on *this* 3B
layer-26 cloud it is weaker but still large — PC1 holds ~26% of the variance,
|r(PC1, 1/T)| ≈ 0.60, and the axis projection itself correlates with `1/T` at
r ≈ 0.78–0.80. Because system-prompt length differs by role, it also produced a
*fake* between-role signal. Fixed in `src/manifold_persona/extract.py`; full
measurement and mechanism in **[diagnostics/README.md](diagnostics/README.md)**.

Distance-based ID estimators are translation-invariant, so a constant offset
would have been harmless — but the sink term is scaled by `1/T`, so it varies.
Measured `r(d(prompt_avg), d(sink-excluded)) = 0.42`: the artifact was inside the
geometry being measured. **Both numbers below need recomputing on a clean cloud.**

A prompt-token cloud is clean iff its `manifest.json` carries a `sink_factor`
that is **present and not null**: a pre-fix run writes no key at all, and a
`--keep_sinks` run writes `null`. Response-token clouds are clean by
construction, since position 0 is never pooled. The cloud now in
`data/embeddings_roles/` **fails** this test — it predates the fix — so
`load_layer` refuses to load it unless you set `MP_ALLOW_UNCLEAN=1`.

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
`exploratory/assistant_axis/figures/<DD-Mon-YYYY-HHMM>/` (set in
`src/manifold_persona/common.py:26`) containing every figure (PNG + interactive
3D HTML), the raw metric dumps (`*.json`, `*.csv`), and an auto-generated
**`REPORT.md`** that pulls the run's real numbers, embeds each figure,
interprets it, and cites the code that produced it. The orchestrator fixes one
timestamp and shares it across the six scripts via the `MP_RUN_DIR` env var;
each script also accepts `--outdir`.

**Stage 1b — per-persona** (`exploratory/per_persona/`). The same cloud read the
other way round: instead of one mean point per role, it keeps each role's own
points and measures the geometry *within* each role. Two stages, each with its
own orchestrator and its own dated folder:

```bash
# ID stage — 276 intrinsic dimensions and 276 clusterings, against both nulls
MP_ROLE_DIR=data/embeddings_roles_resp40 \
  .venv/bin/python exploratory/per_persona/run_id_stage.py --layer 0 --stamp resp40

# geometry stage — the per-role metric panel (curvature, density, topology,
# closeness) and how it tracks the Assistant Axis
MP_ROLE_DIR=data/embeddings_roles_resp40 \
  .venv/bin/python exploratory/per_persona/run_geometry.py --layer 0
```

Its prompt-cloud arm is contaminated by the same length artifact and measures the
extraction grid, not persona geometry. Its 40-question **response**-cloud arm is
clean — response-token pooling never sees the sink — and finds that more
Assistant-like roles have **lower** within-role ID, weakened but directionally
intact once cloud scale is controlled.

Read `exploratory/per_persona/README.md` before quoting any number, and
`exploratory/per_persona/METHODS.md` for how the two headline quantities are
actually computed.


## Layout

```
docs/papers/ + docs/notes/     # reference PDFs + reading notes (see docs/README.md)
diagnostics/                   # apparatus checks: activation scales, attention sinks
src/manifold_persona/          # config, io, extract, common, runlog; prompts_roles.py (roles)
extraction/                    # build_and_extract_roles.py (prompt tokens),
                               # generate_and_extract_roles.py (response tokens), push_to_hf.py
manifold/                      # H1 manifold study: pipeline, tps, run, sweep, local_id
exploratory/assistant_axis/    # role study: 01–06 + run_all + make_report + figures/<stamp>/
exploratory/per_persona/       # per-role study: two stages (run_id_stage, run_geometry)
                               # + common + METHODS.md + README.md + figures/<stamp>/
data/embeddings_roles/         # role cloud, prompt tokens 5x5, L26 (gitignored; UNCLEAN, see above)
data/embeddings_roles_resp/    # role cloud, response tokens 5x5, L19 (gitignored)
data/embeddings_roles_resp_40q/ # role cloud, response tokens 5x40, 37 layers, use --layer 19 (gitignored)
data/embeddings_roles_resp40/  # the same cloud thinned to L19 only, use --layer 0 (gitignored; from HF)
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
