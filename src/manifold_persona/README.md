# manifold_persona — core library

**Status:** shared library  ·  **When:** 2026-07-23 → 09-22

Configuration, prompt building, activation extraction and run bookkeeping. The July–September studies use all of it. The October code uses `config` and `prompts_roles`.

## Files
| File | What it does |
|---|---|
| `config.py` | Model, paths (`REPO_ROOT`, `DATA_DIR`, the vendored `refs/assistant-axis`), layer helpers. Override with `MP_MODEL_NAME`, `MP_ROLE_DIR`, `ASSISTANT_AXIS_DIR` |
| `prompts_roles.py` | The 276 character-role prompt set (5 instructions per role × questions) |
| `prompts.py` | The Persona Vectors trait prompt set (from the sibling `persona_vectors` repo) |
| `extract.py` | Prompt-token activation extraction, with the attention-sink fix |
| `generate.py` | Answer generation plus response-token activations, as in the Assistant Axis paper |
| `io.py` | Saving and loading the point cloud. Refuses an unclean prompt cloud unless `MP_ALLOW_UNCLEAN=1` |
| `common.py` | Helpers for the Assistant Axis study: figure folder, timestamps, loading |
| `runlog.py`, `provenance.py` | Run folders, manifests, figure saving, Holm correction; provenance stamp |
| `hf_utils.py` | Hugging Face login and dataset push |
| `cli.py` | One command per July stage: `python -m manifold_persona.cli extract | explore | manifold` |
