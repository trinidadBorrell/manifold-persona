# Guide for LLM assistants working in this repo

Start with `README.md`. It lists every experiment, which three are current, and where their data lives. Each `src/<folder>/README.md` gives that experiment's question, files, exact commands and result.

## Layout
- **One folder per experiment:** `src/<experiment>/`. After `pip install -e .`, every folder with an `__init__.py` is an importable package: `python -m route_judge_sweep.export …`.
- **Script folders**, run by path: `assistant_axis`, `per_persona_geometry`, `diagnostics`, `robustness`.
- **Shared code:** `manifold_persona` (core), `steering_core` (hooks, interventions, paths, route judges) and `pm_geometry` (PersonaManifold geometry, components, state extraction).
  - Changing a shared module changes every experiment that imports it. Grep the importers first.
- **Older experiments import some modules from each other.** These are on the path of the current code and must keep working:
  - `ace_steering.{common, grid, judge_prompt, steer_gpu}`
  - `steering_followups.{common, judge_followups, grid, replace, selection}`
  - `pm_addition_round.{build, component_paths}`
  - `calibration.routes`
- `refs/assistant-axis/` holds the vendored roles and questions. Do not edit it.

## Data and outputs
- **October data root:** `PM = /data/project/eeg_foundation/data/manifold_persona/personamanifold`, defined in `route_judge_sweep/build.py`. Never overwrite an existing run folder; write a new one.
- **Gitignored:** `output/`, `data`, `jobs_condor/`, `token/`, `plans/`, `docs/`, `extra/`. Code that produces a published number must live in `src/`, not in `output/`.

## Running things
- **Login node:** it is shared. GPU and heavy CPU stages run only through HTCondor: `condor_submit jobs_condor/<name>.submit` or `condor_submit_dag`. A hook (`.claude/hooks/condor_guard.sh`) blocks heavy commands. Prefix `CONDOR_GUARD_OK=1` only for short local checks, and say why.
- **Generation drivers:**
  - `ace_steering.steer_gpu` (shared engine), `pca_pga_steering.steer_gpu` and `pm_geometry.extract_states` are tracked.
  - Older drivers (`cal_gen.py`, `pc_steer.py`, `pm_steer.py`, `followups_steer.py`, …) are still in `jobs_condor/`.
  - Site-specific shims (`mp_compat.py`, `mp_qwen3.py`) stay in `jobs_condor/`.
- **LLM judges:** use the Claude API with interactive requests, not the Batches API.
  - Keys are files in `token/`. Never print them.
  - Ask the user before spending API money, and give an estimate.
  - Subagent judges get one batch each and must read every answer.

## Conventions
- **Steering adds** a vector in the full residual space, unless the experiment is explicitly a clamp or pin (Giovanni's clamp: `h ← h + U Uᵀ (G − h)`).
- **Before porting a method**, quote the paper's equation and parameters (papers are in `docs/papers/`). Do not re-parameterise silently.
- **Labels** on plots and tables say literally what the quantity is. Do not coin terms.
- **Reports** are short. A correction replaces the old number in place, not in an addendum.
- **Docs:** when you add an experiment, give it a `src/<name>/` folder with a README in the same format, and add it to the tables in `README.md`.
