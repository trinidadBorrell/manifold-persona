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

## Key findings (Qwen2.5-3B, layer 26, `prompt_avg`)

- **Low-dimensional manifold**: intrinsic dimension ≈ **3–13** (most estimators
  4–7) versus ambient **2048** → persona activations occupy a thin manifold.
- **Assistant Axis**: PCA **PC1** orders **neutral → negative → positive**
  personas — a single leading direction capturing distance from the model's
  default identity.
- **Geometry recovers the traits**: KMeans **k=14** recovers trait×polarity with
  **0.86 dominant-trait purity** (NMI≈0.73 vs trait); neutral prompts form their
  own cluster. Linear structure carries the persona signal, while local UMAP is
  dominated by question-level micro-structure.

## Setup & run

```bash
python -m venv --system-site-packages .venv     # inherits an existing torch, if present
.venv/bin/pip install -r requirements.txt

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

## Three studies

`exploratory/` holds **three parallel studies** that share the identical pipeline
(local Qwen2.5-3B, prompt activations, per-example point cloud) but differ in the
personas — kept separate so results never mix:

| | `persona_vectors/` | `assistant_axis/` | `assistant_axis_traits/` |
|---|---|---|---|
| Personas | 7 behavioral **traits** × pos/neg | 276 character-role **archetypes** | 240 behavioral **traits** × pos/neg |
| Source | `../persona_vectors` artifacts | `../assistant-axis/data/roles` | `../assistant-axis/data/traits` |
| Points | 1,540 | ~6,900 | 3,600 |
| Polarity | yes (pos/neg/neutral) | no (identities) | yes (pos/neg/neutral) |
| Data | `data/embeddings/` | `data/embeddings_roles/` | `data/embeddings_aa_traits/` |
| Signature analyses | polarity PCA axis; traits-per-cluster; 8 cluster-3D | Assistant-Axis recovery; role families | polarity axis; trait families; per-polarity ID |

The **assistant-axis role study** is the live one and the basis of the H1 manifold
work; it builds its own cloud rather than using the paper's uploaded aggregate vectors:

```bash
.venv/bin/python -m extraction.build_and_extract_roles
.venv/bin/python exploratory/assistant_axis/run_all.py
```

The `persona_vectors/` and `assistant_axis_traits/` clouds are **retained on disk but
are no longer regenerable from this repo** — their extraction scripts were removed as
unused by H1 (`RESEARCH.md`). Their exploratory scripts still run against the existing
`data/embeddings*/` directories.

## Layout

```
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
