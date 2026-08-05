# Per-persona manifold

> **⚠️ Contamination notice.** Every number here computed from
> `data/embeddings_roles` — the **prompt** cloud — is downstream of the
> attention-sink length artifact: that cloud predates the sink fix, and on it the
> assistant-axis projection correlates with `1/prompt_length` at r ≈ 0.78. Read
> those columns as "needs recompute on a clean cloud", not as results. The
> loader now refuses that cloud unless `MP_ALLOW_UNCLEAN=1` is set. The
> response-cloud (`resp40`) numbers are on a clean cloud: response-token pooling
> never sees the sink.

The assistant-axis study collapses each role to one mean point and studies the
276-point cloud **between** roles. This study inverts that: it keeps the raw
6,900-point cloud and asks, for each role separately, what the geometry **within**
that role looks like — 276 intrinsic dimensions and 276 clusterings.

```bash
# prompt cloud, 5x5 (default) — needs MP_ALLOW_UNCLEAN=1, see the notice above
.venv/bin/python exploratory/per_persona/run_all.py

# response cloud, 5x40 — MP_ROLE_DIR repoints the whole stack. --layer 19 is the
# manifest's primary_layer; run_all uses it only to name the report's input files,
# while each script reads primary_layer from the manifest itself.
MP_ROLE_DIR=data/embeddings_roles_resp_40q \
  .venv/bin/python exploratory/per_persona/run_all.py --layer 19 --stamp resp40
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

| | `embeddings_roles` | `embeddings_roles_resp_40q` |
|---|---|---|
| tokens / layer | prompt, L26 | **response**, L19 |
| grid | 5 × 5 = 25 pts | 5 × 40 = 200 pts |
| additive rank | 8 | 43 |
| instruction / question / **interaction** | 68% / 31% / **0.6%** | 2% / 80% / **17.7%** |
| clusters recover | instruction (ARI 1.00) | question (ARI 0.92) |
| verdict | the grid reproduces the clustering | real role-specific structure |

The prompt cloud is a **negative result**, but a narrower one than first written
here: the design null — synthetic points with the same grid and no persona in
them — reproduces the **clustering** exactly (ARI 1.000 against the real 1.000).
It does **not** reproduce the ID: the real per-role median sits *outside* the
design-null IQR on all 5 estimators. That does not rescue the ID, because
interaction still carries only 0.6% of the within-role variance — the cloud is
its 5×5 grid, just not the same grid the null draws. The response cloud is
different in kind: interaction carries 17.7%, and the real roles separate from
the null on every panel.

`embeddings_roles_resp_40q` is a full local copy of
`triniborrell/manifold-persona-roles-response-40q` (37 layers, `primary_layer:
19`, `prompt_avg.npy` of shape `(55200, 37, 2048)`, 8.4 GB). `fetch_resp40.py`
instead writes a thinned `data/embeddings_roles_resp40` holding that one layer
(226 MB, `n_layers: 1`, `primary_layer: 0`, `source_layer: 19`) — for **that**
directory, and only that one, the layer to pass is `0`.

## ID vs the Assistant Axis (`04`)

Run separately against an existing run folder:

```bash
MP_ROLE_DIR=data/embeddings_roles_resp_40q .venv/bin/python \
  exploratory/per_persona/04_id_vs_axis.py --layer 19 \
  --rundir exploratory/per_persona/figures/resp40
```

(`--layer 19` is this cloud's `primary_layer`; omitting the flag reads the same
layer from the manifest.)

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
TwoNN/MLE/lPCA (its adaptive `fit(n_neighbors = min(10, n-2))` is what keeps MLE
local at n=25 — the assistant-axis `01` passes `MLE(K=20)`, which skdim 0.3.6
ignores under the default `neighborhood_based=True` and then fits with its own
default of 20 neighbours anyway).
`manifold.subsets.kmeans_medoid_roles` supplies role-subset selection,
`manifold_persona.common` supplies `load_points`, `center`, `assistant_axis`.
Methods and scoring in `02` mirror `exploratory/assistant_axis/02_clustering.py`
and `exploratory/persona_vectors/02_clustering.py`.
