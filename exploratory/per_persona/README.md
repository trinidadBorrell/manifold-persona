# Per-persona manifold

The assistant-axis study collapses each role to one mean point and studies the
276-point cloud **between** roles. This study inverts that: it keeps the raw
6,900-point cloud and asks, for each role separately, what the geometry **within**
that role looks like — 276 intrinsic dimensions and 276 clusterings.

```bash
.venv/bin/python exploratory/per_persona/run_all.py            # 01-03 + REPORT.md
.venv/bin/python exploratory/per_persona/01_per_persona_id.py  # or individually
```

Output lands in `figures/<DD-Mon-YYYY-HHMM>/` — the same dated-folder contract as
the other exploratory stages (`MP_RUN_DIR` / `--outdir`).

## Read this before using any number here

Every role owns exactly **25 points: 5 instruction phrasings × 5 shared
questions**. That is a complete 5×5 factorial grid, so the cloud's rank is capped
by the design before any model geometry enters:

```
additive (no-interaction) rank = (5-1) + (5-1) = 8
```

`01` measures how much of each role's variance is actually the grid. The answer
is **99.4%** — instruction ~68%, question ~31%, interaction ~0.6%. lPCA returns
exactly 8.00 for essentially every role, and every clustering method recovers the
5 instruction phrasings at ARI ≈ 1.0.

A **design null** — 25 synthetic points on the same grid built from the empirical
instruction/question effect covariances, containing no persona at all — reproduces
both results. So the per-persona ID and the per-persona clustering, as computed on
today's cloud, are measurements of the extraction grid. They are reported in full
anyway, because the null is what makes that readable.

## What would fix it

Only the **question** factor can be scaled: `refs/assistant-axis/extraction_questions.jsonl`
holds 240 questions and the current cloud samples 5. `03` plants manifolds of known
dimension in the data's ambient space and finds the N at which the estimators
recover them — that converts "more data" into a GPU-hour figure, per role budget.

## Scripts

| script | what it does |
|---|---|
| `common.py` | raw-cloud loader, ANOVA design split, and the two nulls |
| `01_per_persona_id.py` | 276 IDs vs both nulls; variance decomposition; ID vs assistant-axis position |
| `02_per_persona_clustering.py` | 276 clusterings, ARI/NMI vs instruction and vs question |
| `03_compute_budget.py` | measured analysis cost, ID-recovery-vs-N curve, extraction budget |
| `run_all.py` | all three into one dated folder + `REPORT.md` |

## Reuse

Nothing here re-implements an estimator. `manifold.idim.id_estimates` supplies
TwoNN/MLE/lPCA (its adaptive `K = min(10, n-2)` is what makes MLE defined at
n=25 — the assistant-axis `01` hardcodes `MLE(K=20)`, which needs n > 21).
`manifold.subsets.kmeans_medoid_roles` supplies role-subset selection,
`manifold_persona.common` supplies `load_points`, `center`, `assistant_axis`.
Methods and scoring in `02` mirror `exploratory/assistant_axis/02_clustering.py`
and `exploratory/persona_vectors/02_clustering.py`.
