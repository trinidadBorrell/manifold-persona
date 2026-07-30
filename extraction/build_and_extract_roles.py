"""Stage 0 (assistant-axis study): build role prompts, extract activations, save.

Builds the 276 character-role personas from the assistant-axis repo into a
prompt-token point cloud: prompt activations (prompt_avg + prompt_last), all
layers, Qwen2.5-3B-Instruct. Saved distinctly under data/embeddings_roles/.

For the paper-matched RESPONSE-token cloud see generate_and_extract_roles.py.

Usage:
    .venv/bin/python -m extraction.build_and_extract_roles
    .venv/bin/python -m extraction.build_and_extract_roles --n_questions 5
    .venv/bin/python -m extraction.build_and_extract_roles --limit 16   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


from manifold_persona.config import MODEL_NAME, ROLE_EMBEDDINGS_DIR, primary_layer
from manifold_persona.prompts_roles import (build_role_records, render_prompts,
                                            records_to_metadata, list_roles)
from manifold_persona.extract import (load_model_and_tokenizer, extract_prompt_activations,
                                      DEFAULT_SINK_FACTOR)
from manifold_persona.io import save_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=str(ROLE_EMBEDDINGS_DIR))
    ap.add_argument("--n_questions", type=int, default=5, help="questions sampled per role")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #records (smoke test)")
    ap.add_argument("--roles", nargs="+", default=None,
                    help="restrict to these roles (smoke test). Include 'default' -- the "
                         "neutral Assistant baseline that the Assistant Axis is defined "
                         "against; without it 02_clustering.py cannot build the axis.")
    ap.add_argument("--keep_sinks", action="store_true",
                    help="do NOT exclude attention-sink positions from prompt_avg. "
                         "Reproduces the pre-fix behaviour, whose PC1 is ~1/seq_len "
                         "(r=0.9998). See diagnostics/01_activation_scales.py.")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading model {args.model} ...")
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    print(f"device={device}  n_hidden_states={n_layers}  hidden={hidden}")

    print("Building role prompt records from assistant-axis artifacts ...")
    model_display = args.model.split("/")[-1]
    records = build_role_records(n_questions=args.n_questions, seed=args.seed,
                                 roles=args.roles, model_display=model_display)
    if args.limit:
        records = records[: args.limit]
    records = render_prompts(records, tokenizer)
    # Count roles actually present -- with --limit this is far fewer than list_roles().
    print(f"{len(records)} prompt records over {len({r.role for r in records})} roles "
          f"(of {len(list_roles())} available)")

    texts = [r.text for r in records]
    sink_factor = None if args.keep_sinks else DEFAULT_SINK_FACTOR
    avg, last, n_dropped = extract_prompt_activations(
        model, tokenizer, texts, device, sink_factor=sink_factor)

    meta_df = pd.DataFrame(records_to_metadata(records))
    # Prompt length per record. The attention sink made prompt_avg's PC1 = 1/T
    # (see diagnostics/), and length stays a real confound with role even after
    # the fix -- so it has to be a first-class column, not something recovered later.
    meta_df["n_tokens"] = [len(tokenizer(r.text, add_special_tokens=False)["input_ids"])
                           for r in records]
    meta_df["n_sink_dropped"] = n_dropped
    p_layer = primary_layer(model.config.num_hidden_layers)
    manifest = {
        "study": "assistant_axis",
        "model_name": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "primary_layer": p_layer,
        "n_records": len(records),
        "n_roles": int(meta_df["role"].nunique()),
        "n_questions": args.n_questions,
        "seed": args.seed,
        "views": ["prompt_avg", "prompt_last"],
        "dtype": "float16",
        # Provenance for the attention-sink fix. Absent in clouds built before it.
        "sink_factor": sink_factor,
        "sink_positions_dropped_max": int(max(n_dropped)) if n_dropped else 0,
        "final_layer_is_normalized": n_layers - 1,
    }
    print("Saving to", args.out_dir)
    save_embeddings(avg, last, meta_df, manifest, out_dir=Path(args.out_dir))

    mb = (avg.nbytes + last.nbytes) / 1e6
    print(f"Done in {time.time()-t0:.1f}s. primary_layer={p_layer}. "
          f"Arrays: {avg.shape} x2 = {mb:.0f} MB fp16")


if __name__ == "__main__":
    main()
