# persona_axes — interpretable persona axes (Stage A: Phases 0–2)

**Status:** Stage A done, at the checkpoint (started 2026-10-20)  ·  **Plan:** `plans/2026-10-20-principal-axes-v2.md` (local)  ·  **Outputs:** `output/21-oct/`  ·  **Data:** `$PM/AX21/`

## Question
How many interpretable axes (named behavioural dispositions) cover the personas Qwen3-8B can express, which ones, and are they better than arbitrary interpretable axes? Stage A answers the representational part (coverage M1, identifiability M2), measures each axis's purity and interpretability (M4), and runs the full-vector ceiling check that Phase 3 (reachability, not yet approved) depends on.

## Setup
- **Model, layer, view:** Qwen3-8B; hidden_states 15; `mean` = answer-token mean minus the neutral answer (the states the run-2 clamp used). Robustness column: L28, `last`.
- **Role vectors:** the 276 Assistant Axis roles from `$PM/E1` (reused). Set (b) = the 226 human roles (`personas_b_human.json`); set (b-all) = all 276; set (a) = 41 deployment personas in 20 categories (`personas_a.json`, written here — see the caveat in that file).
- **Axes:** 31 pre-registered candidate axes + 4 nuisance axes (`axes.json`), extracted with the Persona Vectors contrastive protocol: 5 pos/neg system-prompt pairs (generated with the Assistant Axis App. C template, `gen_prompts.py`), 40 shared questions disjoint from E1's (`questions.json`), cap 256 tokens, vector = mean(pos) − mean(neg).
- **Trait pool:** the Assistant Axis paper's 240 traits with their own prompt pairs (`traits.json`, from the public repo @ a989619), same protocol, judge-filtered with each trait's eval prompt by Claude Haiku 5.5 (keep if ≥ 3 of 5 pairs have mean score difference ≥ 40).
- **Steering (Phase 0 ceiling, Phase 2 doses):** Giovanni's clamp along one direction at L15, `h ← h + (c − h·u) u` every position; Giovanni's 10 questions, 140 tokens, T 0.7, top-p 0.9, 2 seeds.
- **Judges:** J10 (ranking judge, 317 names, nonsense flag) on every ceiling answer; J8 on a 25% subsample; Phase 2 ratings 1–5 on all 35 scales, 12 per call; blind naming with a 5-way match. All Claude Sonnet 5.5, interactive API.

## Files
| File | What |
|---|---|
| `axes.json`, `personas_a.json`, `personas_b_human.json`, `questions.json`, `traits.json` | Frozen inputs (`spec.py` writes the last three) |
| `gen_prompts.py` | Sonnet: prompt pairs for the 35 axes, instructions for the 41 personas → `$PM/AX21/prompts/` |
| `extract.py` | GPU extraction (`--part traits|axes|seta`), E1 engine, both views, layers 12–32 |
| `collect.py` | Chunks → arrays; `--vectors` → `$PM/AX21/vectors/` (traits, axes, set (a), roles; raw means; split halves) |
| `filter.py` | Haiku trait filter → `filter/traits_kept.json` |
| `human_vote.py` | Haiku check of the human-role list (report only) |
| `ceiling.py` | `--plan` picks the 24 set (b) roles; GPU: S ∈ {1,2,4,6} + prompted cell for 65 personas. `--plan-hiS` / `--run ceiling_hiS`: the 35 failed personas with an absolute push S × 9.0, S ∈ {4,6,8,12} |
| `judge_build.py` | J10 / J8 inputs for `route_judge_sweep.api_judge`; `--collect` → `labels.csv` |
| `phase1.py` | M1, M2, nulls B1–B3, ablation, sparse decomposition, residual, stability → `phase1/results_*.json` |
| `dose.py`, `rate.py`, `phase2.py` | Phase 2 dose response, rating + naming judges, analysis + P2 figures |
| `altdir.py` | Pilot of other clamp directions (contrast vs assistant state, prompt-token states) on 13 personas; `--analyse` → P0_4 |
| `phase3.py` | Phase 3 reachability: `--plan` (37 reachable personas, persona-specific S_p, 9 conditions, basis) then GPU shards |
| `phase3_axes.py` | Phase 3b: one-axis clamp toward each persona for every set-6 axis → persona × axis reach |
| `figs.py` | P0 and P1 figures and tables |
| `api.py` | API helper: interactive calls, budget cap, cost print |

## How to run (in order)
```bash
python -m persona_axes.spec
condor_submit MOD=persona_axes.extract ARGS="--part traits" NSHARDS=32 jobs_condor/ax21_gpu.submit
CONDOR_GUARD_OK=1 python -m persona_axes.gen_prompts            # API, login node
condor_submit MOD=persona_axes.extract ARGS="--part axes" NSHARDS=4 jobs_condor/ax21_gpu.submit
condor_submit MOD=persona_axes.extract ARGS="--part seta" NSHARDS=8 jobs_condor/ax21_gpu.submit
for p in traits axes seta; do condor_submit MOD=persona_axes.collect ARGS="--part $p" jobs_condor/ax21_cpu.submit; done
condor_submit MOD=persona_axes.collect ARGS="--vectors" jobs_condor/ax21_cpu.submit
CONDOR_GUARD_OK=1 python -m persona_axes.filter --part traits   # Haiku, ~19k calls
python -m persona_axes.ceiling --plan
condor_submit MOD=persona_axes.ceiling NSHARDS=8 jobs_condor/ax21_gpu.submit
python -m persona_axes.judge_build --stage ceiling
CONDOR_GUARD_OK=1 API_BUDGET=25 python -m route_judge_sweep.api_judge --dir $PM/AX21/judge_ceiling --jobs manifest.json --workers 16
python -m persona_axes.judge_build --stage ceiling --collect && python -m persona_axes.figs --phase 0
python -m persona_axes.ceiling --plan-hiS                       # rerun of the 35 failed personas, absolute push
condor_submit MOD=persona_axes.ceiling ARGS="--run ceiling_hiS" NSHARDS=7 jobs_condor/ax21_gpu.submit
python -m persona_axes.judge_build --stage ceiling_hiS; CONDOR_GUARD_OK=1 API_BUDGET=12 python -m route_judge_sweep.api_judge --dir $PM/AX21/judge_ceiling_hiS --jobs manifest.json --workers 16
python -m persona_axes.judge_build --stage ceiling_hiS --collect && python -m persona_axes.figs --phase 0hi
condor_submit MOD=persona_axes.phase1 ARGS="--view mean --layer 15" jobs_condor/ax21_cpu.submit   # and --view last --layer 28
python -m persona_axes.figs --phase 1
condor_submit MOD=persona_axes.dose NSHARDS=5 jobs_condor/ax21_gpu.submit
CONDOR_GUARD_OK=1 API_BUDGET=60 python -m persona_axes.rate --stage rate; python -m persona_axes.rate --stage name
python -m persona_axes.phase2
```

## Result (Stage A, 2026-10-21; full write-up `output/21-oct/RESULTS.md`)
- **Coverage:** 8 pre-registered axes (utility + LLM-specific set) explain 0.59 of the held-out variance of the 226 human-role vectors at L15 = 79% of PCA-8, 94th percentile of random 8-trait sets (H1's bar was 95; the 31-axis union passes at 95.4, greedy-from-pool at 99). The identity family (human-like ↔ AI, grounded ↔ mystical) is the one whose removal costs most (−0.04). Deployment personas are not better covered than roles.
- **Identifiability:** 8 axes retrieve a role from a disjoint half of its answers with MRR 0.81 (PCA-8 0.88, full vector 0.94).
- **Purity:** steering any one axis at L15 mostly moves one "formal, detached ↔ playful, chatty" direction; 10 of 31 axes pass the drop rule but have effective rank 3.7. Grounded ↔ mystical is the one axis with its own signature.
- **Ceiling:** the full-vector clamp at S* = 4 reaches the target in 22–25% of answers (prompt: 86%); 30 of 65 personas pass. The push S·|v| is 12–16 units for the professional roles (|v| 3–5) against 65 for vivid ones; a rerun of the 35 failures with an absolute push (36–108) rescues a few (reporter 0.85, medical_cautious 0.95 at 72) but on average reach peaks at 0.20 with collapse 0.48, and 15 never reach. Phase 3 waits for the checkpoint.
- **Steering direction (65 personas):** a 2-dim clamp on span(persona − assistant state, last-prompt-token state − empty-prompt state) reaches 43 / 65 personas at ≥ 0.25 (Phase 0 direction: 33), with less collapse; 14 sharp deployment personas are reached by nothing (`P0_5_directions_all.png`).
- **Phase 3 (37 reachable personas):** steering inside the span of 4–16 named axes reaches 0.09–0.14 vs 0.44 for the full vector; named sets = random kept traits > random PCA-span directions; coverage does not predict reach (H3 ρ = 0.26 [−0.08, 0.56]). One-axis clamps reach 8 / 37 personas at the persona's own coordinate and 14 / 37 at a matched 54-unit push (`phase3_axes.py`, P3b): single named axes evoke the cheerful, chatty personas only, and the epistemic / agency axes collapse the model when pushed hard.
- Deviations from the plan are in its 2026-10-21 addendum (upstream trait prompts, Haiku `max_tokens` fix, 4σ dose unit, checkpoint decisions).

## Used by / uses
Uses `pm_geometry.extract_states` (engine), `ace_steering.steer_gpu.load_model`, `route_judge_sweep.{api_judge, j10, prompts}`, `manifold_persona.prompts_roles`, `$PM/E1`, `$PM/E1b`.
