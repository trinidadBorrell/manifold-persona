# extraction — the role point cloud (Stage 0)

**Status:** finished  ·  **When:** 2026-07-23 → 08-12

## Question
Build the labelled point cloud the July–September studies run on.
- **Roles:** the 276 Assistant Axis character roles (`refs/assistant-axis/`).
- **Prompts:** 5 instruction phrasings × the shared questions.
- **Model:** Qwen2.5-3B-Instruct.
- **Activations:** taken from the prompt tokens, or from generated response tokens as in the Assistant Axis paper.

## Files
| File | What it does |
|---|---|
| `build_and_extract_roles.py` | Prompt-token activations (`prompt_avg`, `prompt_last`), all layers → `data/embeddings_roles/` |
| `generate_and_extract_roles.py` | Generates answers, then reads response-token activations at about half depth → `data/embeddings_roles_resp*/` |
| `push_to_hf.py` | Pushes the cloud to Hugging Face as a dataset (token from `token/huggingface.txt`) |

## How to run
```bash
.venv/bin/python -m manifold_persona.cli extract              # prompt tokens
.venv/bin/python -m manifold_persona.cli extract --response   # response tokens
.venv/bin/python -m extraction.push_to_hf
```
- **Full 240-question response cloud:** sharded on condor with `jobs_condor/shard_extract.py` (`run_extract_shard.sh`), then uploaded with `jobs_condor/push_cloud_to_hf.py`.
- **Later October runs:** these use their own extraction in `pm_geometry.extract_states`.

## Result
- **Prompt-token cloud:** contaminated by a first-token attention sink. See `src/diagnostics/README.md` and `src/role_manifold/PROJECT_README_2026-07.md`.
- **Response-token clouds:** clean.

## Used by / uses
- **Uses:** `manifold_persona` (extract, generate, prompts_roles, config).
- **Read by:** `role_manifold`, `assistant_axis`, `per_persona_geometry` and `steering_rebuild` read its clouds.
