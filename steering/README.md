# steering/ — manifold-aware steering of persona expression

Replicates the role-susceptibility evaluation of **The Assistant Axis**
(arXiv:2601.10387 §3.2.1) on Qwen2.5-3B-Instruct, and adds two targeted interventions the paper
does not have.

- **Plan (the contract):** `plans/2026-08-17-manifold-steering-role-susceptibility.md`
- **Context (stable facts):** `RESEARCH.steering.md`
- **Paper:** `docs/papers/2601.10387.pdf`; notes in `docs/notes/assistant-axis.md`

> **This run is exploratory.** It produces three dose-response figures and stops. There is no
> deciding metric, no threshold and no statistical test — by the user's explicit choice. **Nothing
> from this run may be presented as a confirmatory result**, including by a later plan quoting it.
> No prediction was recorded, so the report may not call any outcome expected or surprising.

## The three arms

All three perturb the residual stream at **layer 19** (`hidden_states[19]`, which is the output of
`model.layers[18]` — the off-by-one is asserted numerically in `smoke.py`), at **every token
position**, all rescaled to the same dose `‖Δh‖ = α · N̄`, where `N̄ = 48.169` is the mean
response-token residual norm on the resp240 cloud. **The rescaling is the point**: without it,
"arm X is better" just means "arm X pushed harder".

| Arm | Δh | What it knows |
|---|---|---|
| `arm1_axis` | `−α·N̄·v̂_axis` | nothing about the target — the paper's §3.1 contrast vector, and the replication |
| `arm2_linear` | `α·N̄·normalize(c_T − h)` | the target; straight line through ambient space. **The honest baseline** |
| `arm3_manifold` | follows the fitted curve toward the target | the target *and* where roles actually live |

Arm 1 runs on the vendored `addition` path unchanged — the authors' own code. Arms 2 and 3 use the
one added `intervention_type="dynamic"`, because their direction depends on the current activation.

## The manifold

A **natural cubic spline through the 276 role centroids, parameterized by Assistant-Axis
projection**. `intrinsic_dim = 1`, so **there is no intrinsic dimension to choose anywhere** — the
coordinate is externally given by the paper's contrast vector rather than manufactured by PCA.

The bending penalty `λ = 0.001` is chosen by **GCV** (`spline1d.py::fit_gcv`), which sees only the
276 centroids and never a steering outcome. `λ = 0` is not a neutral default here: it makes the
spline interpolate, which drives the target term `r_T = c_T − S(u_T)` to exactly zero and leaves
Arm 3 target-blind. See plan Amendment A1 and Observations O2.

## Files

| File | What it is |
|---|---|
| `activation_steering.py` | **Vendored verbatim** from `../assistant-axis/assistant_axis/steering.py`, plus one added `dynamic` mode. Existing modes untouched. |
| `spline1d.py` | numpy port of causalab's `CubicSpline1D` (Reinsch natural cubic) + GCV. Validated against scipy to 2e-14. |
| `geometry.py` | axis, role centroids, the fitted curve, `N̄`, the 50 near / 50 far role sets, the degeneracy gate |
| `interventions.py` | the three arms and the dose scaling — the only file that knows what an arm *is* |
| `run_steering.py` | the generation driver: builds the grid, generates, checkpoints per cell |
| `judge.py` | the paper's D.1.3 rubric **verbatim**, via the Batches API. **Off unless a key exists.** |
| `figures_steering.py` | the three figures |
| `validate_judge.py` | Cohen's κ between the judge and your blind human labels |
| `smoke.py` | proves the plumbing before any GPU time |
| `runmeta.py` | run dirs, the `.run-active` marker, the manifest |

## Running it

**1. Smoke test — local, no API calls. Do this first.**

```bash
.venv/bin/python -m steering.smoke
```

Checks the spline port against scipy, asserts the hook lands on `hidden_states[19]` and not its
neighbours, asserts every arm realises the requested dose, confirms the arms point in different
directions, and generates from all four conditions. It fails if any unsteered response is
degenerate — an earlier version passed 4/4 while every response was `!!!!!!!!` (Observations O3),
which is why the check now reads the text rather than counting rows.

**2. The full grid — on a GPU machine, not this laptop.**

```bash
.venv/bin/python -m steering.run_steering --out <run_dir>
```

31,750 generations. Resumable: each (condition, role) cell is one shard, so a kill loses at most one
cell and re-running the same command continues. A config mismatch on resume aborts rather than
mixing runs.

> **MPS is capped at batch size 1, and that is not a tuning choice.** A left-padded batch of ≥2 on
> MPS produces non-finite logits and the model emits `!!!!!!` — at every dtype and with both `sdpa`
> and `eager` attention. Batch 1 is fine and CPU batch 20 is fine, so it is an MPS bug, not ours.
> CUDA is unaffected and batches 20. See `run_steering.py::default_batch_size`.

**3. Judging — costs money, so it is off by default.**

Put the key in `token/anthropic.txt` (one line). `token/` is gitignored at the repo root under
"Secrets — never relax these". `ANTHROPIC_API_KEY` takes precedence if set.

```bash
.venv/bin/python -m steering.smoke --judge --max-calls 50   # ≤50 interactive calls, ~$0.20
```

The smoke path raises rather than truncating if asked to exceed `--max-calls`, so a mistake costs an
error message instead of money. The full grid goes through the **Batches API** at 50% of standard
rates (~$53 for 31,750 calls).

**4. Judge validation — your labels, blind.**

The run writes `judge_validation_blank_L19.csv` (100 stratified responses, empty `human_label`
column, **no judge label**) and a separate key file. Label it whenever; then:

```bash
.venv/bin/python -m steering.validate_judge --labels <filled> --key <key> --controls <controls.json>
```

The blind/key split is deliberate: showing the model's answer next to an empty box measures
agreement-with-a-suggestion and inflates κ. Until labels exist the judge reads
**PENDING HUMAN VALIDATION** on every figure caption and every quoted fraction; if they never
arrive it becomes **UNVALIDATED**.

## Controls, and what stops the run

| # | Control | Pass condition | Status |
|---|---|---|---|
| 0 | Persona-shift rank | a number exists and is quoted; gates nothing | — |
| 1 | Judge validation | κ reported; no gate | pending your labels |
| 2 | **Positive control** — Arm 1 reproduces §3.2.1 | `assistant` fraction falls ≥ 10 points from α=0 to α=3.0 | needs generation |
| 3 | **Degeneracy gate** — centroids are distinguishable targets | nearest-centroid top-1 ≥ 10% | **PASS — 14.9%** vs 0.36% chance |
| 4 | Negative control — random direction, matched norm | reported as a band on every figure | needs generation |

Controls 2 and 3 **stop the run** if they fail. A positive control that fails means the figures are
unreadable; a degeneracy gate that fails means Arms 2 and 3 have nothing to aim at. Neither is
repairable by adjusting a parameter, and the plan does not permit trying.

## Outputs

Everything lands under
`/Users/trinidad.borrell/Documents/Work/MARS-V/code/manifold-persona/output/steering-manifold/<YYYY-MM-DDTHH-MM>-<slug>/`.

**Results stay local.** `output/` is fully gitignored and that is left as it is. Only `plans/`,
`RESEARCH.steering.md` and this code are committed. The consequence is real and worth stating:
`manifest.json` inside the run dir is the *only* link between a figure and the code that produced
it — its git sha, dirty flag, seeds, input hashes and the full GCV grid. **A run whose manifest is
lost is unreproducible.**

Run dirs are timestamped to the minute and never overwritten. `.run-active` marks the live run and
is deleted at close to seal it.
