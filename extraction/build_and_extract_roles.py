"""Stage 0 (assistant-axis study): build role prompts, extract activations, save.

Same pipeline as extraction/build_and_extract.py, but over the 276 character-role
personas from the assistant-axis repo. Prompt activations (prompt_avg +
prompt_last), all layers, Qwen2.5-3B-Instruct. Saved distinctly under
data/embeddings_roles/.

Usage:
    .venv/bin/python -m extraction.build_and_extract_roles
    .venv/bin/python -m extraction.build_and_extract_roles --n_questions 5
    .venv/bin/python -m extraction.build_and_extract_roles --limit 16   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from manifold_persona.config import MODEL_NAME, ROLE_EMBEDDINGS_DIR, primary_layer
from manifold_persona.prompts_roles import (build_role_records, render_prompts,
                                            records_to_metadata, list_roles)
from manifold_persona.extract import load_model_and_tokenizer, extract_prompt_activations
from manifold_persona.io import save_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=str(ROLE_EMBEDDINGS_DIR))
    ap.add_argument("--n_questions", type=int, default=5, help="questions sampled per role")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #records (smoke test)")
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
                                 model_display=model_display)
    if args.limit:
        records = records[: args.limit]
    records = render_prompts(records, tokenizer)
    print(f"{len(records)} prompt records over {len(list_roles())} roles")

    texts = [r.text for r in records]
    avg, last = extract_prompt_activations(model, tokenizer, texts, device)

    meta_df = pd.DataFrame(records_to_metadata(records))
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
    }
    print("Saving to", args.out_dir)
    save_embeddings(avg, last, meta_df, manifest, out_dir=Path(args.out_dir))

    mb = (avg.nbytes + last.nbytes) / 1e6
    print(f"Done in {time.time()-t0:.1f}s. primary_layer={p_layer}. "
          f"Arrays: {avg.shape} x2 = {mb:.0f} MB fp16")


if __name__ == "__main__":
    main()
