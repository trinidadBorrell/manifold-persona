---
run_id: 2026-07-23-selfcontained-one-command
status: approved       # draft | approved | executed
interface: in-scope    # existing module entry points frozen; packaging + a CLI layer are added
approach: incremental
branch: refactor/selfcontained-one-command
baseline_sha: a86d375d0ca9c1ef8c5ead728a18659a194635b8
revision: 3            # rev1: 7 blocking. rev2: those closed, 3 new. rev3: 0 blocking; 3 should-fix applied.
---

# Make manifold-persona self-contained and runnable in one command per stage

## Honesty note — this is not a pure refactor

Steps 1–4 are refactors: behavior identical, numbers identical, verified against a golden run.
Steps 5–7 are **additive** — vendored data, a CLI layer, a rewritten README. They add surface that
did not exist, so "behavior is the invariant" cannot cover them; they get a smoke check, not a
numeric one. Every step is marked with which kind it is.

**capability-scout was skipped** (the user declined the agent spawn); the inventory was done by hand.
That cost two failed pre-audits: rev 1 raised 7 blocking findings, rev 2 closed them but introduced
3 more. Every line number in this revision has been verified against the tree at `a86d375` by an
independent reader, twice.

## Motivation

- **Can't do today:** hand this repo to anyone, or to yourself on another machine. `config.py:13`
  and `:27` resolve `../persona_vectors` and `../assistant-axis` as sibling directories; without both
  checked out alongside, extraction cannot run. Nothing is installed — `.venv/lib/python3.9/
  site-packages` contains no `manifold*` and no `.pth` — so every consumer reaches the code through
  one of **8** `sys.path.insert` hacks, one of which (`common.py:21`) is a load-bearing *import side
  effect* that `make_report.py:13-15` silently depends on. And there is no single command per stage.
- **Will enable:** clone → install → run, from any working directory.
- **Trigger / why now:** H1 is executed and audited, so the code is momentarily stable.
- **Cost of not doing it:** every new stage copies the harness a fifth time, and the writeup ships
  pointing at two repos that aren't in it.

**Scope note on the second sibling.** `../persona_vectors` (`config.py:13`, `TRAIT_ARTIFACTS_DIR`)
is **not** vendored. Its only consumers are `build_and_extract.py` and `build_and_extract_aa_traits.py`,
which step 4 deletes. Self-containment w.r.t. `persona_vectors` is achieved by *removing the
consumers*, not by copying the data. Stated plainly here because the motivation would otherwise
imply both siblings get vendored.

## Scope

**In:**
- `pyproject.toml` (new — required, see step 1)
- `manifold/` — `run.py`, `sweep.py`, `local_id.py`, `analysis_extra.py`, `plots.py`,
  `sweep_plots.py`, `pipeline.py` (lines 20-24), `meeting_report.py` (deleted)
- `src/manifold_persona/` — `config.py`, plus new `common.py`, `runlog.py`, `cli.py`
- `exploratory/assistant_axis/` — `common.py` (moved), and the import blocks of all **9** files that
  import it (`01`–`06`, `make_report.py`, `run_all.py`)
- `extraction/` — `build_and_extract.py`, `build_and_extract_aa_traits.py` (deleted); the other
  three keep their behavior; see the Out list for exactly what changes in each
- `refs/assistant-axis/` (new), `README.md`, `.gitignore`

**Out:**
- `manifold/tps.py` — **frozen.** `RESEARCH.md:37-39`. Not one line.
- `data/`, `output/` — no *existing* run dir is read, written, or compared in place.
- `manifold/report.py`, `sweep_report.py` — 692 lines, duplicate `_f()`. Deliberately left.
- `exploratory/persona_vectors/`, `exploratory/assistant_axis_traits/` — untouched, untracked.
- `extraction/generate_and_extract_roles.py` and `push_to_hf.py` — kept; only their `sys.path`
  block changes.
- `extraction/build_and_extract_roles.py` — kept; **its `sys.path` block and the stale docstring
  reference at `:3`** ("Same pipeline as extraction/build_and_extract.py") change, nothing else.
  Named here because step 4 edits it and a bare "sys.path only" scope line would contradict that.
- The statistics. No method, threshold, seed, or estimator changes.

**One refactor or several?** Several, bundled on one branch, **separate commits per step**.

## Behavior contract — must NOT change

- **Public API:** every name in `pipeline.py` (`load_cloud`, `fit_manifold`, `construction_C_role`,
  `construction_C_tau`, `construction_C_raw`, `permutation_null`, `permutation_null_tau`,
  `separation_stats`, `positive_control`, `pca_plane_r2`, `cosine_*`), the `Cloud` and
  `ConstructionResult` fields, and all of `tps.py`.
- **CLI:** `.venv/bin/python -m manifold.run` and `-m manifold.sweep [--smoke]` keep working
  identically. New flags may be *added* with defaults reproducing today's values.
- **Exploratory output root:** `exploratory/assistant_axis/figures/<stamp>/` — `RESEARCH.md:27`.
  `common.py:26-27` sets it `__file__`-relative and mkdirs at import; moving the file relocates it.
  Step 1 must re-anchor it explicitly.
- **Output formats:** run dir layout `figures/ data/ logs/ manifest.json REPORT.md`, stamped
  `%Y-%m-%dT%H-%M`, `mkdir(exist_ok=False)` at `run.py:61`, `sweep.py:127`, `local_id.py:229`.
  `metrics.csv` columns and row order. `manifolds_C_role.npz` array names. All **four** manifest key
  sets.
- **Figure DPI, per helper, as it is today** — there are **five** save helpers, not four:
  | helper | dpi |
  |---|---|
  | `plots.py:18` `_save` | 300 |
  | `sweep_plots.py:22` `_save` | 300 |
  | `analysis_extra.py:30` `_save` | 300 |
  | `local_id.py:161` `_save` | **160** |
  | `common.py:163` `savefig` (used by all 6 of `01`–`06`) | **150** |
  The 160 and the 150 are deviations from `RESEARCH.md:43`. Both are **preserved and logged in
  `## Found`**. `common.savefig` is **excluded from the unification entirely** — it keeps its own
  identity in the moved `common.py`.

  **How DPI is asserted.** matplotlib writes resolution into the PNG `pHYs` chunk as integer
  pixels-per-metre, so the round-trip is lossy — measured in this venv, 150 → 150.0124,
  160 → 159.9946, 300 → 299.9994. Every DPI gate below is therefore
  `round(PIL.Image.open(p).info["dpi"][0]) == expected`, **not** equality. Written as equality the
  gates fire on a correct build, and the executor either "fixes" a DPI that was right or learns to
  ignore the gate.
- **Determinism:** seed 0 where it is today; `D_AMBIENT=50`, `K_INTRINSIC=3`, `MIN_COMPONENT=3`,
  `FLOOR=0.30`, `ALPHA`, `N_PERM_DECIDER` unchanged in value.
- **Stop-on-fail:** `run.py:102-107` (posctrl) and `sweep.py:159-168` (regression check).
- **Defensive plotting:** `RESEARCH.md:44`. Preserved where it exists (6 try/except in `plots.py`,
  10 in `sweep_plots.py`) and **extended** to the `--extra` path, which today has zero.
- **Performance budget:** `manifold.run` ≤ **210 s** with `--extra` off. **Tightened during
  step 0** (deviation, recorded): the 470 s figure came from the 2026-07-21 manifest's 411 s,
  but both step-0 passes ran in 183.9 s / 178.2 s with bit-identical output. Against a 184 s
  measured baseline, a 470 s ceiling would pass a 2.5x regression. 210 s is +14% on measured.
  With `--extra` on the budget is **900 s**, because step 7 pulls `skdim`, `umap` and an Isomap fit
  onto the run path — an addition the 470 s figure was never meant to cover.

### Preserved capabilities

| Capability | Where | How it'll be verified |
|---|---|---|
| `MP_ROLE_DIR` repoints the stack at another cloud | `common.py:69` | Step 1: set/unset, compare shapes |
| `MP_AGGREGATE` switches role-mean vs raw (default `"role"`) | `common.py:71-72` | Step 1: explicit test. **Golden cannot see it** — `pipeline.py:53` passes `aggregate="none"` |
| `MP_RUN_DIR` passes the run dir to exploratory subprocesses | `run_all.py:33`, `common.py:152` | Step 1: `run_all.py` completes 7 scripts **into `exploratory/assistant_axis/figures/<stamp>/`, asserted by path** |
| `MP_MODEL_NAME`, `ASSISTANT_AXIS_DIR`, `PERSONA_VECTORS_DIR` overrides | `config.py:36,27,13` | Step 5: must still win over the vendored default |
| `primary_layer()` = round(20/28·L); `half_depth_layer()` = round(0.5·L) | `config.py:54,67` | Not touched; asserted in step 5 |
| `TRAITS` order "fixed for reproducibility" | `config.py:39-47` | Not touched |
| Positive control runs FIRST and can stop the run | `run.py:89-107` | Golden reproduces the `posctrl` row |
| `--smoke` = `n_list=[10,25]`, 1 seed, 20 perms; `-smoke` dir suffix | `sweep.py:119-122`, suffix `:126` | Step 2: completes, suffix present |
| A run refuses to overwrite | `run.py:61`, `sweep.py:127`, `local_id.py:229` | Step 2: run twice in one minute, second must raise |
| `common.REPO_ROOT = parents[2]` | `common.py:20` | Step 1: **re-derived explicitly**, not inherited — it resolves correctly from both locations only by coincidence |
| `common.savefig` dpi=150 | `common.py:163` | Step 2: PIL asserts exploratory PNGs are 150 dpi |

- **Unresolved `[?]` items:** none.

## Licensed to change

- **Packaging.** Add `pyproject.toml` and `pip install -e .` into the existing `.venv`.
  It must declare **all three** top-level packages, or the CLI and the "run from anywhere"
  motivation are false:
  ```toml
  [build-system]
  requires = ["setuptools>=64"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "manifold-persona"
  version = "0.1.0"
  requires-python = ">=3.9"
  dependencies = []            # deliberately empty; requirements.txt stays the source of truth

  [tool.setuptools]
  packages = ["manifold_persona", "manifold", "extraction"]

  [tool.setuptools.package-dir]
  manifold_persona = "src/manifold_persona"
  manifold = "manifold"
  extraction = "extraction"
  ```
  Explicit `packages` — not `find` — because the flat+src mixed tree (`manifold/`, `extraction/`,
  `exploratory/`, `data/`, `output/` at root) would otherwise be left to setuptools' heuristics.
  **`package-dir` is a section, not an inline table** — TOML inline tables may not span lines, and
  the multi-line form raises `TOMLDecodeError` in pip's parser. This exact config was verified by
  editable-installing a mirror of this tree into a throwaway 3.9.6 venv: all three packages import
  from cwd `/`, including `extraction`, which has no `__init__.py` (setuptools' editable finder maps
  it anyway). Nothing named `manifold*` exists in the venv today, so there is no collision.
  **Known constraint:** the venv has **setuptools 58.0.4**, which predates PEP 660, so
  `pip install -e .` succeeds only by fetching setuptools ≥64 into an isolated build env. This works
  today (index reachable); it fails offline or with `--no-build-isolation`. Recorded so an offline
  failure is a known cause, not a mystery. Reversible: `pip uninstall manifold-persona`.
  `requirements.txt` content unchanged; no new third-party dependency.
- Removal of **all 8** `sys.path.insert` sites — packaging makes every one redundant.
- Four `_save()` bodies → one with a **defaulted** `dpi=300`; `local_id` passes `dpi=160`.
  Two `holm()` → one. Four manifest-write sites → one parameterized. `common.savefig` is **not**
  part of this.
- **Deletion of exactly three files**, user-approved: `manifold/meeting_report.py` (664),
  `extraction/build_and_extract.py`, `extraction/build_and_extract_aa_traits.py` (152 together).
  **Consequences, accepted not fixed:** (a) `data/embeddings/` and `data/embeddings_aa_traits/`
  become non-regenerable from this repo — they stay on disk; (b)
  `output/manifold_h1-2/MEETING-2026-07-22/MEETING-REPORT.md` cites `manifold/meeting_report.py`
  (written by `meeting_report.py:445`) and will permanently name a script that no longer exists.
  `output/` is frozen, so this stands as a recorded decision.
- New files: `pyproject.toml`, `src/manifold_persona/{common,runlog,cli}.py`, `refs/assistant-axis/**`.

## Safety net

**Never waived.**

1. **Tests covering the changed paths: none exist.** Zero `test_*.py`/`*_test.py`/`conftest.py`.
2. **Green right now?** N/A. Baseline is `a86d375`.
3. **Therefore: characterization by golden output, as step 0, before any edit.**

**Step 0 (blocking):**
```
.venv/bin/python -m manifold.run     # pass A  (~411 s)
.venv/bin/python -m manifold.run     # pass B  (~411 s)
# pass A is the golden. Pass B exists only to prove determinism and is then discarded.
cp -R <A> refactors/baselines/2026-07-23-selfcontained-one-command/golden/
cp -R <A> refactors/baselines/2026-07-23-selfcontained-one-command/extra-reference/
.venv/bin/python -m manifold.analysis_extra <that copy>     # today's post-hoc output
```
If A ≠ B, `manifold.run` is not deterministic, the golden is worthless, and **this plan stops and is
re-planned.** Determinism was judged plausible by both audits: every RNG on this path is seeded
(`pipeline.py:55,77,169,192,206,229,288`), `tps.py` has no RNG, ordering is explicit (`sorted()`
`:58`, `np.unique` `:128`, reindex `:208,:231`), and no time-dependent value reaches `metrics.csv`.
Residual risk is BLAS float reassociation.

Golden → `refactors/baselines/2026-07-23-selfcontained-one-command/golden/`:
- `data/metrics.csv` — byte-identical
- `data/manifolds_C_role.npz` — exact (`atol=0, rtol=0`)
- `manifest.json` — identical after dropping `started`, `finished`, `elapsed_sec`, `run_id`
  (`run.py`'s manifest has no `platform` key; that one is `local_id.py:301`)
- `REPORT.md` — identical after normalizing the timestamp line
- **PNG/HTML figures: filename + existence only, not bytes.** Matplotlib is not guaranteed
  byte-reproducible; claiming otherwise would be a false green.

**Coverage gap.** `python -m manifold.run` imports only `pipeline.py`, `plots.py`, `tps.py`,
`report.py` (lazily, `:269`) and `common.py` (transitively, `pipeline.py:24`). It does not execute:

| Uncovered, and edited here | Holds | Compensating check |
|---|---|---|
| `manifold/sweep.py` | `holm` #2, manifest #2/#3 | `sweep --smoke` runs to a written `manifest.json` |
| `manifold/sweep_plots.py` | `_save` #2 | imported by the smoke run; figures exist |
| `manifold/local_id.py` | `_save` #4 (**160 dpi**), manifest #4 | runs to `local_id.json`; **PIL asserts 160 dpi** |
| `manifold/analysis_extra.py` | `_save` #3, all of step 7 | diffed against step 0's `extra-reference` |
| 8 exploratory scripts | `common.savefig` (**150 dpi**) | `run_all.py` completes 7 scripts into the correct root; **PIL asserts 150 dpi** |
| all of `extraction/` | — | **import + `--help` only.** Running it needs model forward passes at ~7–10 s/record |
| `subsets.py`, `idim.py` | — | reached by `sweep --smoke` |

**Would break silently and go unnoticed for a month — two of them:**

1. **Vendored refs drifting from the cloud on disk.** If `refs/assistant-axis/extraction_questions.jsonl`
   differs from the sibling's copy, every *future* extraction produces a cloud subtly inconsistent
   with `data/embeddings_roles/`. **Guard:** sha256 of every vendored file in
   `refs/assistant-axis/PROVENANCE.md`; assert the role-instruction count is exactly 276.
2. **`.gitignore:19` swallowing the vendored refs.** `data/` is unanchored, so it matches at any
   depth — verified: `git check-ignore refs/assistant-axis/data/roles/instructions/x.json` → ignored
   via `.gitignore:19`. Mirroring the upstream layout (`config.py:30-31` builds
   `.../data/roles/instructions`) is the natural implementation and would make every step-5 check
   pass on local disk while git holds none of the files — surfacing only on a fresh clone.
   **Two guards, both required:** (a) the vendored layout omits the `data` segment —
   `refs/assistant-axis/roles/instructions/` and `refs/assistant-axis/extraction_questions.jsonl`
   (verified not ignored); (b) step 3 anchors the rule to `/data/`. Step 5 additionally asserts
   `git check-ignore` reports **no** vendored file as ignored, and that `git ls-files refs/ | wc -l`
   is 277+.

## Blast radius

- **`sys.path.insert` — 8 sites**, independently re-verified: `pipeline.py:21,22`; `common.py:21`;
  `extraction/build_and_extract.py:18`, `build_and_extract_roles.py:23`,
  `build_and_extract_aa_traits.py:22`, `push_to_hf.py:20`, `generate_and_extract_roles.py:40`.
- **`common` importers — 9 files:** `pipeline.py:24` + all 8 exploratory scripts. **6** (`01`–`06`)
  import `load_points`; `make_report.py:13` and `run_all.py:10` import other names. All 9 edited.
- **Save helpers — 5, none interchangeable.** See the DPI table in the contract.
- **`holm()` — 2**, `run.py:37` / `sweep.py:55`, behaviorally identical. Genuinely mechanical.
- **Manifest writes — 4 sites, 4 distinct key sets:** `run.py:264`; `sweep.py:163`
  (`{"status": "stopped-regression-failed", "regression": reg}` — reachable **only** on regression
  failure, which no planned run exercises, so step 2 verifies it by a **direct call** with that
  status rather than by a run); `sweep.py:302`; `local_id.py:298`.
- **`config.py` constants** — read by all 5 `extraction/*.py`, `common.py:23`, `io.py`.
  `generate_and_extract_roles.py` needs `RESP_ROLE_EMBEDDINGS_DIR` (`config.py:21`) left alone.
- **Dynamic dispatch:** exactly one, confirmed twice — `run_all.py:19` `subprocess.run` with the 7
  filename strings at `:34-40`. **No `exploratory/assistant_axis/NN_*.py` is renamed.** It passes
  `env` from `os.environ` and does **not** set `PYTHONPATH`, so it offers no path rescue.
- **Outside this repo:** the two siblings are read from, never written. Nothing outside imports this
  repo. `manifold-temporal/framing/plots.py` is a *style* reference (`RESEARCH.md:35`), not an import.
- **In-flight branches:** none.

## Target state

```
manifold-persona/
  pyproject.toml                  NEW — declares all 3 packages; kills every sys.path hack
  refs/assistant-axis/            NEW — note: NO "data/" segment, or .gitignore:19 eats it
      roles/instructions/         276 role JSONs (2.1 MB)
      extraction_questions.jsonl  (24 KB)
      PROVENANCE.md               arXiv:2601.10387, source path, sha256 per file
  src/manifold_persona/
      config.py                   vendored refs as DEFAULT; env vars still override
      common.py                   MOVED; FIGURES_DIR + REPO_ROOT re-anchored explicitly;
                                  savefig() keeps its own dpi=150
      runlog.py                   NEW — run_dir(), write_manifest(), save_fig(dpi=300), holm()
      cli.py                      NEW — thin dispatcher, calls existing main()s unchanged
  manifold/                       pipeline, tps (frozen), run, sweep, local_id, analysis_extra
  exploratory/assistant_axis/     same 8 filenames, import blocks updated, figures/ stays here
```

Three commands — the yes/no test for "done":
```
python -m manifold_persona.cli extract  [--model ... --limit ...]
python -m manifold_persona.cli explore  [--view ...]
python -m manifold_persona.cli manifold [--run | --sweep | --local-id]
```
`python -m manifold.run` and `-m manifold.sweep` keep working identically — the CLI wraps, it does
not replace. That is what keeps the golden valid.

## Migration path

Reordered again: packaging and the `common.py` move are **one step** (splitting them leaves the tree
unimportable), argparse now precedes the `--extra` fold, and the fold is last.

| # | Step | Kind | Verified by |
|---|---|---|---|
| 0 | Two golden runs + `extra-reference` capture, untouched tree | — | A == B, else **stop and re-plan** |
| 1 | `pyproject.toml` + `pip install -e .`; move `common.py` → `src/manifold_persona/` **re-anchoring `FIGURES_DIR` and `REPO_ROOT`**; update 9 import blocks; delete all 8 `sys.path` blocks | additive + mechanical | **Full golden**, run from a **different cwd**; `run_all.py` writes into `exploratory/assistant_axis/figures/<stamp>/` **asserted by path**; `MP_ROLE_DIR`/`MP_AGGREGATE`/`MP_RUN_DIR` each tested |
| 2 | `runlog.py`: 2 `holm`→1; 4 `_save`→1 (`dpi=300` default, `local_id` passes 160); 4 manifest sites→1 parameterized | mechanical | **Full golden** + `sweep --smoke` + `local_id` + PIL asserts 160/150 dpi + all 4 manifest key sets compared, the regression-failure one by **direct call** |
| 3 | Rewrite `README.md`; anchor `.gitignore` `data/`→`/data/`; fix stale `exploratory/figures/` in `.gitignore:30-31` and `config.py:23` | docs | Every README command executed; `git check-ignore` spot-checks |
| 4 | Delete the 3 approved files; update the prose reference at `build_and_extract_roles.py:3` | deletion | Import-all check; **zero *executable* references** (the one remaining mention is a docstring, updated in this step) |
| 5 | Vendor `refs/assistant-axis/` (no `data/` segment); repoint `config.py` defaults, env still overrides | additive | sha256 manifest; 276-file assert; `git check-ignore` clean; **`../assistant-axis` temporarily renamed away and the default path still resolves** |
| 6 | `cli.py` dispatcher + argparse on `manifold/run.py` (defaults = today's constants, incl. the `--extra` flag, wired to nothing yet) | additive | `--help` on 3 subcommands; **full golden** via bare `-m manifold.run` and via the CLI |
| 7 | Fold `analysis_extra.py` behind `--extra` (**default off**), wrapping the whole extra block in **one outer try/except** per `RESEARCH.md:44` | structural | **Full golden** unchanged flag-off; flag-on `POSTHOC-manifold-structure.md` diffed against step 0's `extra-reference` (the `.md` is the only byte-comparable artifact — the ~8 PNGs fall under the existence-only policy); **≤900 s** |

Five full golden runs — steps 1, 2, 6, and both legs of 7. Four run at ~411 s and the flag-on leg
at ≤900 s, so **≈ 42 min**, plus step 0's ~21 min. Budgeted deliberately: it is the only verification that exists.

**On step 7's risk, corrected.** An earlier revision claimed folding it in swaps a saved run dir's
arrays for the live cloud. That is wrong: `analysis_extra.py:204` calls `P.load_cloud(seed=0)`
itself and never reads the run dir's arrays — the run dir is an output destination only, and its
defaults match `run.py:78` exactly. So no number can change on that account. The real cost is
runtime (`skdim`, `umap`, Isomap), which the 900 s flag-on budget covers.

- **Point of no return:** step 7, now the final step. Everything before it is a clean `git revert`.

## Policy

- **Bug found mid-refactor:** preserve + log to `## Found`. Fix separately.
- **Dead code:** delete only the three approved files. Nothing else, however dead it looks.
- **Formatting churn:** none. No reformatter, no import re-sorting beyond lines being edited.
- **Gitignore:** `exploratory/*/figures/` and `.DS_Store` added pre-baseline. Step 3 anchors
  `data/`→`/data/`, corrects the stale `# exploratory/figures/*.png` hint, and adds
  `refactors/baselines/`. `plans/`, `RESEARCH.md`, `DESIGN.md`, `output/` stay ignored per the
  user's instruction to honour the file as written.
- **Stale ignore paths:** `.gitignore:30-31` and `config.py:23` both name `exploratory/figures/`,
  which has never existed. Both corrected in step 3.

## Rollback

- **Branch:** `refactor/selfcontained-one-command`, off `a86d375`.
- **How to back out:** `git reset --hard a86d375` plus `pip uninstall manifold-persona`. Per-step:
  `git revert` the single commit; steps 1–6 are independently revertible, step 7 is not.
- **What this refactor writes outside git:** roughly **9 new timestamped dirs** under
  `output/manifold_h1-2/` and `output/local_id/` (step 0 ×2, five goldens, a smoke, a local_id), plus
  `refactors/baselines/`. No *existing* run dir is modified, but the claim "nothing under `output/`
  is written" would be false — backing out means deleting those new dirs by hand.

## Done

- **Stopping rule:** the three CLI commands run end to end from an arbitrary cwd; `-m manifold.run`
  reproduces the golden numerics exactly; `refs/` makes the role/question path resolve with
  `../assistant-axis` absent; every README command executes; the 8 `sys.path` hacks are 0; the 10
  duplicate helper definitions are 3 (`runlog.save_fig`, `runlog.holm`, `runlog.write_manifest`)
  plus `common.savefig`, which stays separate on purpose. Stop.
- **Non-goals — stays ugly on purpose:** `report.py` + `sweep_report.py` (692 lines, duplicate
  `_f()`) untouched. `meeting_report.py`'s HTML/base64 embedding is deleted, not reimplemented.
  `local_id.py` keeps 160 dpi and `common.savefig` keeps 150. `../persona_vectors` is not vendored.
  **No test suite is written** — that is its own job, and pretending otherwise would hide that this
  entire refactor rests on one golden run.
- **Reviewer / max reviewable diff size:** self-review + post-audit; any commit over ~400 lines splits.

## Approval

- [x] Pre-audit (rev 3) passed — **no blocking findings**. Three should-fix items applied after:
      invalid TOML in the pyproject snippet, DPI gates rewritten with `round()`, and the
      `build_and_extract_roles.py` scope/step disagreement.
- [x] Approved by user on 2026-07-23

---

## Found

Pre-existing, preserved, each its own change afterward:

- **`manifold/local_id.py:162` saves at `dpi=160`** and **`common.py:165` `savefig` at `dpi=150`**,
  while `RESEARCH.md:43` specifies 300 dpi and the other three helpers use it. Preserved —
  normalizing either would change output in files the golden does not cover.
- **`manifold/analysis_extra.py` has zero defensive try/except** (vs 6 in `plots.py`, 10 in
  `sweep_plots.py`), contrary to `RESEARCH.md:44`. Harmless today as a separate post-hoc
  invocation; step 7 adds a single outer guard rather than rewriting the module.
- **`src/manifold_persona/config.py:23`** `FIGURES_DIR = REPO_ROOT / "exploratory" / "figures"`
  points at a path that has never existed. Currently unused — no importer.
- **`manifold/run.py:252`** hardcodes `"git": "not-a-git-repo (reproducibility via manifest+seeds)"`
  into every manifest. Untrue as of `a86d375`. It is a string literal, so the golden is unaffected —
  but every future run will record a false provenance claim.
- **`manifold/local_id.py:99-102`** has a `try/except` on the changed path that the
  defensive-plotting inventory ("6 in `plots.py`, 10 in `sweep_plots.py`") does not account for.

## Deferred

- Unify `report.py` / `sweep_report.py` formatters.
- A real test suite. `pipeline.py`'s pure functions (`cosine_components`, `separation_stats`,
  `holm`) are trivially unit-testable and would replace the 35-minute golden loop.
- Finish or delete `data/embeddings_roles_resp/` (5/33 shards) and `generate_and_extract_roles.py`.
- Publish-oriented vs local-baseline `.gitignore` — one file serving two purposes, which is why
  `RESEARCH.md` and `plans/` currently have no revert.
- Normalize the 150/160 dpi deviations to 300; fix the `run.py:252` git literal.
- Vendor `../persona_vectors` trait artifacts, if the deleted trait-extraction scripts are ever
  wanted back.
