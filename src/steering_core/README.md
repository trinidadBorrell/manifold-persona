# steering_core — shared steering library

**Status:** shared library  ·  **When:** 2026-09-08 → (used by every steering experiment)  ·  **Plan:** `plans/2026-09-08-steering-rebuild.md` (local)

These modules were the core of the top-level `steering/` folder until 2026-10-16. Changing one changes every experiment listed under "Used by". Grep for importers before editing.

## Files
| File | What it does | Used by |
|---|---|---|
| `activation_steering.py` | Steering hooks from the Assistant Axis repo (add, replace, capture) | ace_steering, steering_followups, steering_rebuild, jobs_condor |
| `interventions.py` | The steering arms as hook functions: linear / manifold (axis, journey, contrast, path), ACE, floor, subspace pin (clamp); dose scaling | ace_steering, steering_followups, steering_rebuild, jobs_condor |
| `geometry.py` | Layer geometry: the resp240 role axis, the fitted curve, dose units | chunk_ablation, steering_rebuild, jobs_condor |
| `manifold_paths.py` | Paths from A to B: chord, fixed-end spline through chosen personas (`PersonaPath`, `FixedEndSpline`) | ace_steering, chunk_ablation, steering_followups, steering_rebuild |
| `path_cases.py` | The A → B cases and path defaults | chunk_ablation, steering_rebuild |
| `spline1d.py` | Natural cubic spline | geometry, manifold_paths |
| `run_steering.py` | The September generation driver; also holds `INTROSPECTIVE_QUESTIONS` and `render_chat_prompt` | ace_steering, steering_followups, steering_rebuild |
| `route_judge_v2.py`, `route_judge_v3.py` | The route judge (pair options, definitions, system prompt). J1 in `route_judge_sweep` is this prompt | route_judge_sweep, calibration, ace_steering, steering_followups |
| `runmeta.py` | Run-folder manifest (git sha, command, inputs) | chunk_ablation, steering_rebuild |
| `test_paths.py` | Contract tests for the paths and hooks on a synthetic cloud | — |

## How to run
- Tests: `.venv/bin/python -m steering_core.test_paths`, CPU, seconds.
- `run_steering.py` is the September full-grid driver. Run it on condor through `jobs_condor/steering_full.submit`, which calls `run_steering_full.py` to apply the ppc64le shims first. See `src/steering_rebuild/README.md`.

## Uses
`manifold_persona` (config, prompts_roles) and the vendored roles in `refs/assistant-axis/`.
