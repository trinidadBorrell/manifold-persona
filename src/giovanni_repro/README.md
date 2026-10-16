# giovanni_repro — reproduction of Giovanni Marraffini's clamp steering, and our addition on his setup

**Status:** finished (results not written up)  ·  **When:** cluster runs dated 2026-10-16 (`__init__.py` says 10-10 to 10-12)  ·  **Plan:** none; was `steering/gio_repro` until 2026-10-16

## Question
Does Giovanni's layer-23 clamp ("pin") steering on Qwen2.5-3B reproduce when run with his own code?
On the same setup, how does our additive steering (E1 persona vector, LIN route) compare?
Does a single-direction clamp also work on Qwen3-8B?

The clamp: u = (M_t − M_default)/‖·‖, a = M_default·u, d = ‖M_t − M_default‖ (M = mean answer-token state);
at every position h ← h + (a + s·d − h·u)·u. Our addition: h ← h + α·s·x_B at every position.

## Files
| File | What it does |
|---|---|
| `run.py` | Runs his modules unchanged from his branch (`origin/persona-definition` @ b860780, exported by `git archive` to `$PM/GIO/repo`). Stages: export, targets, acts, vectors (his L23 u, a, d), steer (`steer_pin`, s ∈ {3,4,6,8} + zero-vector baseline), ceiling (`steer_pin2`, role prompt, no steering). `shim()` maps `dtype`→`torch_dtype` for transformers 4.53. |
| `shard.py` | His targets + activations for one persona, one question shard (`i/10`), in one process. Replaces the targets/acts stages in the DAG. |
| `ours.py` | Our steering on his setup: `extract` (E1-style last-answer-token states, 40 questions), `vectors` (x_B), `steer` (α ∈ {1,4} × s ∈ {0.25,0.5,0.75,1}, 8 cells). |
| `encode.py` | Re-reads every steered answer (his clamp, our addition, baseline, ceiling) through the unsteered model; mean over answer tokens at layers 18, 23, 28, 33, 36. |
| `pca_page.py` + `pca_template.html` | 3D PCA page of those encodings for spy, ghost, vampire, pirate, musician. |
| `judge.py` | His ranking judge J10 (port of his `make_rank_chunks.py`, 15 answers per chunk, recall@k) + our J8 on the clamp answers. `build` / `score`. |
| `ours_judge.py` | J10 on our-method answers. |
| `qwen3_pin.py` | His clamp on Qwen3-8B, M from our E1b states (`$PM/E1b/mean_L<L>.npy`, default = `__neutral__`), s ∈ {0,1,3,4,6,8}, his 10 questions. Uses `ace_steering.steer_gpu` to load the model. |
| `q3_judge.py` | J10 chunks and recall@k for the `qwen3_pin` answers → `scores.csv`. |

`qwen3_route.py` moved to `src/route_judge_sweep/`.

## How to run
All on condor; every DAG node is `jobs_condor/gio_stage.submit` → `run_gio_stage.sh <mode> ...`. `$PM` = `/data/project/eeg_foundation/data/manifold_persona/personamanifold`.

1. His reproduction (Qwen2.5-3B, 5 personas): `condor_submit_dag jobs_condor/gio_repro.dag`.
   Nodes: `sh` (`giovanni_repro.shard`, 10 shards per persona and for `default`) → `vec` (`run --stage vectors`) → `steer` (5 shards) + `ceil`.
   Output: `$PM/GIO/<persona>/{vectors.npz, gen_steer_<k>.jsonl, gen_ceiling.jsonl}`.
2. Ours on his setup: `condor_submit_dag jobs_condor/gio_ours.dag` (`oex` → `ovec` → `ost`, 8 cells). Output: `$PM/GIO/ours/<persona>/gen_<cell>.jsonl`.
3. Encodings: `condor_submit_dag jobs_condor/gio_enc.dag` (`enc`) → `$PM/GIO/pca/enc_<persona>.{npz,json}`; then
   `.venv/bin/python -m giovanni_repro.pca_page --root $PM/GIO --out page.html`.
4. Judges (Qwen2.5-3B): `.venv/bin/python -m giovanni_repro.judge build --out $PM/GIO/spy`; subagents write `rank/verdict_NN.json` (instructions: `src/route_judge_sweep/J10_AGENT.md`) and `j8/labels/J8_gio.json`; then `... judge score --out $PM/GIO/spy` → `results.json`. Same for `ours_judge build|score --dir $PM/GIO/ours/<persona>`.
5. Qwen3-8B clamp: `condor_submit_dag jobs_condor/gio_q3.dag` (`q3 <layer> <target>`, layers 15 and 23 × spy, pirate, vampire, tree, void, leviathan) → `$PM/GIO8B/gen_L<L>_<t>.jsonl`; then `.venv/bin/python -m giovanni_repro.q3_judge build --dir $PM/GIO8B`, judge the chunks, `... q3_judge score --dir $PM/GIO8B` → `$PM/GIO8B/scores.csv`.
6. `jobs_condor/gio_q3r.dag` (`q3r`) runs `route_judge_sweep.qwen3_route`, not this package → `$PM/GIO8B_R`.

## Result
Not written up. Per-cell recall@1/5/10, assistant-top-1 share and nonsense share are in
`$PM/GIO/<persona>/results.json` (clamp, with J8), `$PM/GIO/ours/<persona>/results.json` (ours) and
`$PM/GIO8B/scores.csv` (Qwen3-8B, per answer).

## Used by / uses
- Uses: his exported tree (`persona_lora.*`) at `$PM/GIO/repo`; `manifold_persona.prompts_roles`; `route_judge_sweep.j10` and `.prompts`; `ace_steering.steer_gpu`; `$PM/E1b`.
- Used by: `route_judge_sweep.qwen3_route` and `route_judge_sweep.run2_judge_figs` import `PM` (and `QUESTIONS`) from `qwen3_pin`.
