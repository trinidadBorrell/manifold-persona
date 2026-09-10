# steering/ — manifold-aware steering of persona expression

Replicates the role-susceptibility evaluation of **The Assistant Axis**
(arXiv:2601.10387 §3.2.1) on Qwen2.5-3B-Instruct, and adds two targeted interventions the paper
does not have.

- **Context (stable facts):** the next section of this file. It replaces `RESEARCH.steering.md`,
  which has been deleted — nothing outside this README is needed to read this code.
- **Paper:** arXiv:2601.10387.
- **Paper-fidelity plan (current):** `docs/notes/paper-fidelity-plan.md` — what still differs from
  the paper, and the order to fix it in. Read it before running anything **if you have it**.
- **Plan (the original contract):** `plans/2026-08-17-manifold-steering-role-susceptibility.md`.
- Those last two, and the paper PDF and reading notes, are **local-only**: `.gitignore` excludes
  `docs/` and `plans/` at any depth, so none of them reach a clone. Everything a clone needs is
  here.

> **This run is exploratory.** It produces three dose-response figures and stops. There is no
> deciding metric, no threshold and no statistical test — by the user's explicit choice. **Nothing
> from this run may be presented as a confirmatory result**, including by a later plan quoting it.
> No prediction was recorded, so the report may not call any outcome expected or surprising.

## Context — stable facts

*The repo root's `RESEARCH.md` governs the geometry track and states as stable facts that `data/`
is read-only, that runs do no re-extraction, and that all analysis is local CPU/numpy with no CUDA
and no vLLM. **Half of those are false here**: this track generates new text under forward hooks
on a GPU and calls an external API. Different data, different environment, different off-limits
rules — which is why the steering track carries its own context instead of an edit to that one.
`RESEARCH.md` is untouched and still governs `output/manifold_h1-2/`, `output/per_persona_*` and
`output/per-persona-entropy/`.*

### Data

- **Read-only input.** `data/embeddings_roles_resp240/` — response-token activations,
  Qwen2.5-3B-Instruct, layer 19, **published pre-thinned so index 0 *is* layer 19** (hence
  `--layer-index 0`, while `--layer 19` is a filename label). 331,200 records = 276 roles ×
  5 instructions × 240 questions, hidden 2048. Source of the role centroids, the Assistant Axis,
  the fitted curve and the persona knots. **Nothing in this track writes under `data/`.**
- The cloud is an argument everywhere (`--cloud`; `geometry.RESP240_DIR` is only the default), so
  the same code runs against a larger single-layer cloud staged outside the repo without an edit.
- **Role prompts are vendored, not read from a sibling checkout:**
  `refs/assistant-axis/roles/instructions/*.json` (276 roles × 5 system prompts) and
  `refs/assistant-axis/extraction_questions.jsonl` — see `refs/assistant-axis/PROVENANCE.md` for
  the sha256 list and why the upstream `data/` path segment was dropped. `config.py` reads these by
  default; `ASSISTANT_AXIS_DIR` overrides to a live checkout.
- **Everything this track generates** — text, labels, figures, CSVs, manifests — lands under the
  output root and nowhere else.
- Two gaps the paper leaves, and what fills them: the 50-role susceptibility system prompts were
  never released (Appendix D.1.1 says they were regenerated for that eval), so `gen_role_prompts.py`
  builds a held-out bank; LMSYS-Chat-1M *is* used for the dose unit, streamed rather than
  downloaded, and it is **gated** — accept its terms with the HF account whose token is in `token/`
  or `lmsys_norm.py` fails at load with a 401.

### Environment

- **Analysis, smoke, and all pure geometry:** `.venv/bin/python` at the repo root, **Python 3.11**,
  CPU numpy. No GPU is needed for anything in `geometry.py`, `manifold_paths.py`, `spline1d.py` or
  `geometry_check.py`.
- **Generation:** a separate GPU machine, HF `transformers` with forward hooks — **not vLLM**, which
  does not expose the residual stream for intervention. Its launcher scripts are site-specific
  (absolute paths, queue config) and are deliberately untracked, so a run is reproduced from the
  documented CLI, not from a submit file. The card is not assumed: every run measures its own
  generations/sec before committing to a grid.
- **Judges:** Anthropic API — `claude-sonnet-5` for the perspective judge (`judge.py`) and the
  all-personas judge (`persona_judge.py`), `claude-haiku-4-5` for the role-expression filter
  (`rolefilter.py`). **Off unless a key exists**, `--yes` required, and the smoke path is
  hard-capped at 50 calls. The interactive API is the default path, not Batches.
- **Output root:** `<repo>/output/steering-manifold/`, overridable with `STEERING_OUTPUT_ROOT`
  (`runmeta.py`). It was once an absolute path into one laptop's home directory, which meant every
  run on any other machine — the GPU box this track was always going to need — died in `mkdir`
  before generating a token.

### Reuse, do not reimplement

- `assistant_axis/steering.py::ActivationSteering` — layer resolution across architectures, hook
  registration and teardown, `positions="all"`. **Vendored verbatim** as `activation_steering.py`
  and extended with one new `intervention_type`; the existing modes are untouched, so the
  replication arm runs on the authors' own code.
- `causalab/methods/spline/cubic.py::CubicSpline1D` — Reinsch natural cubic. Ported to numpy in
  `spline1d.py` the way `manifold/tps.py` ported the TPS solve. causalab cannot be imported here:
  it needs Python 3.12.
- `manifold_persona.common::load_points`, `::assistant_axis` — the study's loader/aggregator and the
  paper's §3.1 contrast vector. `geometry.py` calls both rather than re-deriving either.
- `extraction/generate_and_extract_roles.py` — the chunked, resumable, shard-checkpointed
  generation loop. `run_steering.py` copies its checkpoint discipline.
- `diagnostics/03_experiment1_shift_rank.py` — the persona-shift rank diagnostic, re-run for its
  number (Control 0 below).

### Don't touch

`data/`; any existing run dir under `output/` (the `.run-active` marker plus the guard hook enforce
this, not discipline); the causalab-port semantics in `manifold/tps.py`; anything under
`exploratory/per_persona/`.

### Conventions

- Run dirs are `<output_root>/<YYYY-MM-DDTHH-MM>-<slug>/`, timestamped to the minute and **never**
  overwritten, each holding `figures/ data/ logs/ manifest.json REPORT.md`. `.run-active` marks the
  live run and is deleted at close to seal it.
- Figures are 300 dpi PNG named `figNN_<slug>_L<layer>.png`, and each ships the CSV or JSON it was
  drawn from — a figure whose numbers exist only inside a PNG is not a result.
- Plot functions are defensive: a plotting failure logs and returns, and never kills a run that has
  already produced numbers.
- Seeds are fixed and logged in `manifest.json`; the git sha and dirty flag are hard failures to
  collect, not best-effort.
- Generation is greedy (`do_sample=False`, `max_new_tokens=128`), matching the resp240 cloud, so
  steered and unsteered text stay comparable to the cloud the geometry came from.
- Reports are read by someone who knows the method but not this run: define terms, state what
  failed.

### Standing references

- **The Assistant Axis**, arXiv:2601.10387 — §2.1.1 role expression, §3.1 the contrast vector,
  §3.2.1 the role-susceptibility eval, Appendix D.1.1 (50-role selection), D.1.2 (the five
  introspective questions), D.1.3 (the judge prompt and its seven categories).
- **Manifold Steering**, arXiv:2605.05115 — the spline manifold; implementation
  `causalab.methods.spline`.
- `diagnostics/README.md` — the attention-sink and length confounds. Response-token averaging is
  immune to the sink, which is why this track uses the response cloud.

*Both PDFs and the reading notes sit under `docs/`, which `.gitignore` excludes at any depth. The
arXiv IDs are the pointers that survive a clone; the paths are not.*

## The three arms

Both perturb the residual stream at **layer 19** (`hidden_states[19]`, which is the output of
`model.layers[18]` — the off-by-one is asserted numerically in `smoke.py`), at **every token
position**, both rescaled to the same dose `‖Δh‖ = |α| · N̄`, where `N̄` is the mean per-token
response residual norm on LMSYS-Chat-1M (`lmsys_norm.py`; the resp240 value 48.169 is the
fallback, and is a different quantity on a different corpus). **The rescaling is the point**:
without it, "arm X is better" just means "arm X pushed harder".

| Arm | Δh | What it knows |
|---|---|---|
| `linear_axis` | `α·N̄·v̂_axis` | nothing about any role — the paper's §3.1 contrast vector, and the replication |
| `manifold_axis` | follows the fitted curve along the same axis | where roles actually live along that axis |

Neither arm has a target role, so there is no near/far split: the paper's intervention is
targetless and these are its straight and curved forms. α is **signed** and is the paper's own
x-axis (Fig. 4): negative is away from the Assistant, positive toward it.

`linear_axis` runs on the vendored `addition` path unchanged — the authors' own code.
`manifold_axis` uses the one added `intervention_type="dynamic"`, because its direction depends on
the current activation.

*(The earlier three-arm design — `arm1_axis` / `arm2_linear` / `arm3_manifold`, with target
centroids and a near/far split — is gone. `figures_steering.py` still draws those runs; nothing
else does. Its `make_arm2_delta_fn` / `make_arm3_delta_fn` remain in `interventions.py` for the
same reason.)*

## The manifold

A **natural cubic spline through the 276 role centroids, parameterized by Assistant-Axis
projection**. `intrinsic_dim = 1`, so **there is no intrinsic dimension to choose anywhere** — the
coordinate is externally given by the paper's contrast vector rather than manufactured by PCA.

The bending penalty `λ = 0.001` is chosen by **GCV** (`spline1d.py::fit_gcv`), which sees only the
276 centroids and never a steering outcome. `λ = 0` is not a neutral default here: it makes the
spline interpolate, which drives the target term `r_T = c_T − S(u_T)` to exactly zero and leaves
Arm 3 target-blind. See plan Amendment A1 and Observations O2.

## The path construction: chord vs personas

`manifold_paths.py` is the other geometry in this track, and it answers a different question from
the axis spline above. There the curve is a *fit* to all 276 centroids parameterised by axis
projection. Here we are given **two fixed points** `P0` and `P1` in activation space and asked for
two routes between them:

| Arm | Class | The route |
|---|---|---|
| linear | `LinearPath` | the straight chord `P0 → P1` |
| manifold | `PersonaPath` | a cubic through **k real persona centroids** lying near that chord, with the same two endpoints |

**`PersonaPath` is the class to use.** Every interior knot is a real, fully-role-playing persona
centroid — never a mean of several. A mean of `soldier` and `observer` is not a persona, and a
curve through it passes through nothing that exists; `pick_centroids` therefore selects actual rows
of `C` and the report records their role names, so you can read off which personas a route threads.

**How the knots are chosen.** `eps` is a tube radius around the chord, in **absolute activation
units**, not a fraction of chord length. `mode="relative"` exists and is wrong for this cloud: at
`eps=0.35` the same fraction gives a tight tube around a 38-unit near→far chord and a
proportionally huge one around a 5.8-unit near→midway chord, and the detour ratio then explodes
because its denominator is small. Absolute `eps` gives every route the same physical
neighbourhood. Candidates are the centroids inside the tube whose projection lands strictly
between the endpoints; the chord is then cut into **k equal spans** and the candidate **nearest
the chord** is taken from each. So k is an upper bound, not a promise — **empty spans yield fewer
than k knots**, and a few hundred centroids in a space of thousands of dimensions is sparse enough
that this is normal. Centroids within `END_MARGIN` (1e-3 of the chord) of either endpoint are
dropped: an endpoint centroid is itself a member of `C` and in floating point lands a hair either
side of `u=1`, so it can be admitted as an interior knot *while also being the pinned endpoint*.
Two knots ~1e-16 apart blow up the cubic's second derivative and the path loops instead of
travelling. Measured, of 16 routes: the 7 whose control set contained an endpoint had detour
5.7–9.1; the 9 that did not had 1.24–1.72. Perfect separation.

**`alpha ∈ [0,1]` is normalised arc position.** Not a dose, and not a fraction of the knot
coordinate. Both arms are sampled at the same alphas, so they have the same endpoints and the same
number of stops and differ only in route. The spline is keyed by a scalar abscissa, so arc length
can only be built *after* fitting: quadrature on `‖dS/du‖`, cumulative table, monotone inversion.

**`lam=0` is the default and it is not a compromise.** For a natural cubic, `lam=0` interpolates
every chosen persona exactly *and* is still C2-continuous — smooth, no kinks. Both properties at
once. **The axis spline's `lam=0.001` is a different curve solving a different problem** (a
GCV-chosen fit to 276 points, where `lam=0` would make the fit interpolate and leave a target term
identically zero) — do not carry that value across. The one thing `lam=0` cannot prevent is
overshoot *between* knots that sit far apart, which is what `excursion` reports and what `lam`
stays available for. Endpoints survive any `lam`: `_fit_fixed_ends` pins `gamma[0] = gamma[-1] = 0`,
which is exactly the condition `y_hat[0]=y[0]`, `y_hat[-1]=y[-1]`, so "same endpoints" holds at
every smoothing level rather than only at zero.

**`param` is the knot abscissa, and `centripetal` is the default.** Measured on the real cloud, 16
routes at `eps=9`, mean spline overshoot above the polyline floor:

| `param` | mean overshoot | worst | knot error |
|---|---|---|---|
| `projection` (position along the chord) | 1.173 | 18.97 | ~1e-14 |
| `length` (cumulative ambient distance) | 0.135 | 0.43 | ~1e-14 |
| `centripetal` (the same, distances to the 0.5 power) | **0.114** | 0.43 | ~1e-14 |

Exact interpolation is unaffected — the knot error stays ~1e-14 in all three — so this is purely
about what the curve does *between* knots. `projection` fails for a reason: the u-gap between two
knots is `‖dy‖·cos θ / L` while the curve must actually travel `‖dy‖`, so mean speed on a segment is
`L/cos θ`, unbounded as knots go lateral. As k grows the u-gaps shrink like 1/k but the ambient
gaps stay at the cloud's lateral scatter, so the speed ratio grows *with k* and a C2 cubic
overshoots — the suspected cause of detour rising 1.07 → 2.08 from k=2 to k=8. `length` is what
causalab implements and never enables; `centripetal` is the Catmull-Rom choice, provably free of
cusps and self-intersections.

**Selection and fitting both use the chord coordinate**, never `axis_proj = h·â`. Those are
different orderings whenever the chord is not parallel to the Assistant Axis, which is the normal
case — measured on this cloud, ~72% of a near→far displacement is orthogonal to the axis. Fitting
on one ordering while selecting on the other makes the curve zigzag.

### Diagnostics, and which of them are controls

`PersonaPath.report()` returns all of these, and they go into the manifest row:

| Quantity | What it says | Control? |
|---|---|---|
| `endpoint_drift` | distance from the path's own ends to `P0`/`P1` | **positive control** — must be ~0 for both arms at every eps, or "same endpoints" is a claim and not a fact |
| eps→0 collapse | at `eps=0` nothing is selected, so `PersonaPath` falls back to the chord: `n_centroids` 0 and `detour_ratio` exactly 1 | **negative control** — if the curve is still curved with no personas to visit, the curvature came from the code, not the cloud |
| `polyline_ratio` | length of the straight-line path *through the same knots*, over chord length — **the floor**; no interpolating curve can be shorter | — |
| `detour_ratio` | arc length over chord length | — |
| `overshoot` | `detour_ratio − polyline_ratio`: what the spline adds on top of an unavoidable cost | — |
| `knot_error` | max distance from a **chosen persona centroid** to the fitted curve — measured against the centroids, not against `y_hat`. Comparing to `y_hat` once reported 7e-15 while the curve actually missed the personas by 12.79 | — |
| `excursion` | how far the curve strays from the chord over how far its knots do; > 1 means it overshoots past the personas it visits | — |

Splitting `detour_ratio` into floor + overshoot is what makes the number readable: a long route
through far-apart personas is the knots' own doing, and only the overshoot is the
parameterisation's fault.

### Running the geometry check

`geometry_check.py` sweeps eps and k, draws figures 01–02 and writes the controls. Pure numpy — no
generation, no GPU, no judge calls — which is the point: if the manifold path is visually
indistinguishable from the chord, or wanders somewhere absurd, that is knowable now for the price of
some CPU rather than after a GPU run and a judge bill.

```bash
.venv/bin/python -m steering.geometry_check \
  --cloud data/embeddings_roles_resp240 \
  --labels <run_dir>/data/role_labels.parquet \
  --outdir <run_dir>/figures \
  --layer 19 --layer-index 0 \
  --eps-fig2 9.0 --k-fig 5 --persona-lam 0.0 --param centripetal \
  --export-json
```

**`--labels` is required in practice.** Without it `load_geometry` silently falls back from `fully`
to `somewhat` to unfiltered centroids, and a curve anchored on vectors the paper's ≥10 rule
discards is exactly the class of error this rebuild exists to remove. `geometry_check` now
hard-fails in that case instead of proceeding: no fully-role-playing counts in `axis_report`, or
more centroids falling back to unfiltered than the one expected for `default`, and it exits rather
than claim a filter it did not apply.

It builds 16 routes — one along the Assistant Axis itself (a segment of the axis line through the
cloud centre, spanning the axis extent the centroids reach, deliberately *not* the chord between
the two extreme centroids, which is just another A→B pair) plus 3 near-Assistant sources × (3
rank-midway + 2 far) targets — and writes, with `--layer 19`:

| File | What is in it |
|---|---|
| `fig01_epsilon_ablation_L19.png` | PCA small multiples per eps, plus eps vs bending energy and eps vs interior centroids |
| `fig02_pca_paths_L19.png` | chord and curve over the role cloud, markers at the 8 alpha stops, the threaded personas named |
| `fig01_epsilon_ablation_L19.csv` | the eps sweep behind fig01, including `endpoint_drift` per row |
| `fig01_epsilon_ablation_L19.json` | the controls block: eps→0 collapse, max endpoint drift, PCA explained variance, roles picked, the full args, the git sha and dirty flag |
| `persona_k_sweep_L19.csv` | k ∈ {3,4,5,6,8,10,12} × `param` ∈ {projection, length, centripetal} at `--eps-fig2` — the table the `param` default was chosen from |
| `ablation3d_L19.json`, `paths3d_L19.json` | `--export-json` only: PCA-3 cloud, both paths, the eps×k grid and the per-route persona supply, for the interactive viewer |

Paths are **projected into** the centroid PCA for drawing, never fitted in it.

> **eps and k are fixed from these figures and never from a steering outcome.** Tuning the geometry
> on how the generations turn out makes the route a free parameter fitted to the result, and any
> difference between the two arms then measures the tuning rather than the manifold.

## Files

| File | What it is |
|---|---|
| `activation_steering.py` | **Vendored verbatim** from `../assistant-axis/assistant_axis/steering.py`, plus one added `dynamic` mode. Existing modes untouched. |
| `spline1d.py` | numpy port of causalab's `CubicSpline1D` (Reinsch natural cubic) + GCV. Validated against scipy to 2e-14. |
| `geometry.py` | axis, role centroids, the fitted curve, `N̄`, the 50 near / 50 far role sets, the degeneracy gate |
| `manifold_paths.py` | the two routes between two fixed points: `LinearPath` (the chord) and `PersonaPath` (a cubic through k real persona centroids, same endpoints). Cylinder selection, the endpoint-pinned fit, arc-length parameterisation, and the detour / overshoot / knot-error diagnostics |
| `geometry_check.py` | Experiment 1: sweeps eps and k over 16 routes, draws figs 01–02, writes the controls JSON. Pure numpy — no GPU, no API. eps and k are frozen here, before any generation |
| `interventions.py` | the arms and the dose scaling — the only file that knows what an arm *is* |
| `run_steering.py` | the generation driver: builds the grid, generates, checkpoints per cell |
| `judge.py` | the paper's D.1.3 **perspective** rubric, verbatim, plus the row-identity keys every judge joins on. Batch path included. **Off unless a key exists.** |
| `judge_concurrent.py` | the same judge over a thread pool — the default path, since a batch stalled once for 18.5 h |
| `rolefilter.py` | the paper's **other** judge — §2.1.1 role expression (`fully`/`somewhat`/`no`). Rebuilds the axis from `fully` rows (WP1) and gates the fresh prompts (WP3b) |
| `lmsys_norm.py` | the dose unit measured the paper's way: per-token residual norm on LMSYS-Chat-1M (WP2) |
| `gen_role_prompts.py` | fresh evaluation system prompts for the 50 roles, held out from extraction (Appendix D.1.1, WP3). Pass the same `--labels` `run_steering` gets, or it builds a bank for a different 50 |
| `capture_activations.py` | hookless replay of a finished run: re-reads the generations and captures the residual stream, so the PCA/divergence figures plot what the model actually did rather than the intended path |
| `figures_steering.py` | the Figure-4 analogues, one per arm |
| `persona_judge.py` | the all-personas judge: which role is this, out of all 276 |
| `adoption_report.py` | which roles and which questions refuse to be adopted |
| `validate_judge.py` | Cohen's κ between the judge and your blind human labels |
| `smoke.py` | proves the plumbing before any GPU time |
| `runmeta.py` | run dirs, the `.run-active` marker, the manifest |

### Figure scripts (present on disk, gitignored)

Plot-only stages: each reads a finished run dir and emits figures, and none
computes a number the write-up depends on, so they are deliberately kept out of
the tree. If one ever produces a *result*, track it — see the E8 note in
`.gitignore`.

| file | what it draws |
| --- | --- |
| `figures_pca.py` | trajectories in the role-vector PCA plane; `html_pca.py` is the interactive 3-D version and imports from it |
| `figure_transition.py` | per-rollout identity vs dose, no averaging |
| `figures_divergence.py` → `figure_why_linear.py` | a producer/consumer pair over `fig08_arm_divergence.json`; not wired into any job yet |
| `figures_poster.py` | standalone poster panel |

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

19,000 generations: (1 unsteered + 2 arms × 9 signed alphas) × 50 near-Assistant roles × 20 rows
per cell. The figure is derived in `run_steering.py::FULL_GRID_ROWS`, so it cannot go stale behind
a change to `ALPHAS` or `ARMS` — the old 31,750 was left over from the three-arm near/far design
and overstated every wall-clock projection by 67%.

Resumable: each (condition, role) cell is one shard, so a kill loses at most one cell and
re-running the same command continues. A config mismatch on resume aborts rather than mixing runs.

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
error message instead of money. The full grid goes through **`judge_concurrent.py`**, the
interactive API with a thread pool (~$45 for 19,000 calls): the Batches API's 50% discount buys
nothing when a batch sits 18.5 h at 0/13, which one of ours did. `judge.py`'s `--submit`/`--collect`
batch path is kept and correct, but it is not the default. Both require `--yes`.

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
`<repo>/output/steering-manifold/<YYYY-MM-DDTHH-MM>-<slug>/`.

**Results stay local.** `output/` is fully gitignored and that is left as it is. The code and this
README are the only committed part of the track — `plans/` and `docs/` are gitignored too. The
consequence is real and worth stating:
`manifest.json` inside the run dir is the *only* link between a figure and the code that produced
it — its git sha, dirty flag, seeds, input hashes and the full GCV grid. **A run whose manifest is
lost is unreproducible.**

Run dirs are timestamped to the minute and never overwritten. `.run-active` marks the live run and
is deleted at close to seal it.
