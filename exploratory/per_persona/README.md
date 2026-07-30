# Per-persona manifold

The assistant-axis study collapses each role to one mean point and studies the
276-point cloud **between** roles. This study inverts that: it keeps the raw
6,900-point cloud and asks, for each role separately, what the geometry **within**
that role looks like — 276 intrinsic dimensions and 276 clusterings.

```bash
# prompt cloud, 5x5 (default)
.venv/bin/python exploratory/per_persona/run_all.py

# response cloud, 5x40 — MP_ROLE_DIR repoints the whole stack, --layer 0
# because only the primary layer was kept when it was fetched
MP_ROLE_DIR=data/embeddings_roles_resp40 \
  .venv/bin/python exploratory/per_persona/run_all.py --layer 0 --stamp resp40
```

Output lands in `figures/<DD-Mon-YYYY-HHMM>/` — the same dated-folder contract as
the other exploratory stages (`MP_RUN_DIR` / `--outdir`).

**[METHODS.md](METHODS.md)** derives the two calculations that carry the
interpretation — the within-role variance fractions, and the ID-vs-Assistant
correlations with their scale control. Read it before quoting either.

## Read this before using any number here

Each role's points are a complete two-factor grid — n_i instruction phrasings ×
n_q shared questions, one point per cell — so the cloud's rank is capped by the
design before any model geometry enters:

```
additive (no-interaction) rank = (n_i - 1) + (n_q - 1)
```

`grid_shape()` reads that layout from the data and asserts every role agrees;
nothing here hardcodes it. **Interaction** — the residual after fitting
`instruction_effect + question_effect` — is the only term a grid does not force,
so it is the number that says whether a per-persona manifold exists at all.

## Two clouds, opposite answers

| | `embeddings_roles` | `embeddings_roles_resp40` |
|---|---|---|
| tokens / layer | prompt, L26 | **response**, L19 |
| grid | 5 × 5 = 25 pts | 5 × 40 = 200 pts |
| additive rank | 8 | 43 |
| instruction / question / **interaction** | 68% / 31% / **0.6%** | 2% / 80% / **17.7%** |
| clusters recover | instruction (ARI 1.00) | question (ARI 0.92) |
| verdict | the grid, nothing else | real role-specific structure |

The prompt cloud is a **negative result**: a design null with no persona in it
reproduces both the ID and the clustering exactly, because 99.4% of the variance
is the 5×5 grid. The response cloud is not — interaction carries 17.7%, and the
real roles separate from the null on every panel.

`embeddings_roles_resp40` is fetched from
`triniborrell/manifold-persona-roles-response-40q` (8.4 GB on the hub; only the
primary layer is kept locally, 226 MB, so `manifest.json` reports `n_layers: 1`
and `source_layer: 19`).

## ID vs the Assistant Axis (`04`)

Run separately against an existing run folder:

```bash
MP_ROLE_DIR=data/embeddings_roles_resp40 .venv/bin/python \
  exploratory/per_persona/04_id_vs_axis.py --layer 0 --rundir figures/resp40
```

On the response cloud, **4 of 6 estimators agree strongly: more Assistant-like
roles have LOWER-dimensional within-role manifolds** (partial r −0.49 to −0.80
after controlling cloud scale, BH q < 1e-17). `lPCA` dissents (+0.44) and
participation ratio is near zero once scale is removed (+0.13).

Two warnings the script exists to enforce. **Cloud scale is a huge confound** —
ID correlates with log within-role variance at r up to −0.78, so the raw column
is not the answer; read the partial. And **`axis_proj` and `dist_default` are
mirror images here** (their r-vectors correlate at 0.999), so they are one
finding stated twice, not two independent confirmations.

The prompt cloud gives the **opposite sign** — but its ID is 99.4% grid, so that
correlation is about grid saturation, not geometry.

## Scaling further

Only the **question** factor scales: `refs/assistant-axis/extraction_questions.jsonl`
holds 240 and the response cloud samples 40. `03` plants manifolds of known
dimension and finds the N at which the estimators recover them, converting "more
data" into a GPU-hour figure per role budget. Note its default rate is a
prompt-only forward pass — for a response cloud it warns and reports a lower
bound, since generation is several times slower.

## Scripts

| script | what it does |
|---|---|
| `METHODS.md` | how the variance fractions and the ID-vs-axis correlations are computed |
| `common.py` | raw-cloud loader, ANOVA design split, and the two nulls |
| `01_per_persona_id.py` | 276 IDs vs both nulls; variance decomposition; ID vs assistant-axis position |
| `02_per_persona_clustering.py` | 276 clusterings, ARI/NMI vs instruction and vs question |
| `03_compute_budget.py` | measured analysis cost, ID-recovery-vs-N curve, extraction budget |
| `fetch_resp40.py` | pull the HF response cloud, keep the primary layer only |
| `04_id_vs_axis.py` | does per-role ID track closeness to the Assistant? all 6 estimators, scale-controlled |
| `run_all.py` | all three into one dated folder + `REPORT.md` |

## Reuse

Nothing here re-implements an estimator. `manifold.idim.id_estimates` supplies
TwoNN/MLE/lPCA (its adaptive `K = min(10, n-2)` is what makes MLE defined at
n=25 — the assistant-axis `01` hardcodes `MLE(K=20)`, which needs n > 21).
`manifold.subsets.kmeans_medoid_roles` supplies role-subset selection,
`manifold_persona.common` supplies `load_points`, `center`, `assistant_axis`.
Methods and scoring in `02` mirror `exploratory/assistant_axis/02_clustering.py`
and `exploratory/persona_vectors/02_clustering.py`.
