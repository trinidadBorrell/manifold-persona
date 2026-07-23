"""Stage 0 (assistant-axis TRAITS study): build prompts, extract, save distinctly.

240 behavioural traits × n_instructions × n_questions × {pos, neg} + neutral,
Qwen2.5-3B-Instruct prompt activations, all layers. Saved to
data/embeddings_aa_traits/.

Usage:
    .venv/bin/python -m extraction.build_and_extract_aa_traits
    .venv/bin/python -m extraction.build_and_extract_aa_traits --n_instructions 2 --n_questions 3
    .venv/bin/python -m extraction.build_and_extract_aa_traits --limit 16   # smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


from manifold_persona.config import MODEL_NAME, AA_TRAIT_EMBEDDINGS_DIR, primary_layer
from manifold_persona.prompts_aa_traits import (build_aa_trait_records, render_prompts,
                                                records_to_metadata, list_traits)
from manifold_persona.extract import load_model_and_tokenizer, extract_prompt_activations
from manifold_persona.io import save_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=str(AA_TRAIT_EMBEDDINGS_DIR))
    ap.add_argument("--n_instructions", type=int, default=2)
    ap.add_argument("--n_questions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading model {args.model} ...")
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    print(f"device={device}  n_hidden_states={n_layers}  hidden={hidden}")

    print("Building assistant-axis TRAIT prompt records ...")
    records = build_aa_trait_records(n_instructions=args.n_instructions,
                                     n_questions=args.n_questions, seed=args.seed)
    if args.limit:
        records = records[: args.limit]
    records = render_prompts(records, tokenizer)
    print(f"{len(records)} prompt records over {len(list_traits())} traits")

    texts = [r.text for r in records]
    avg, last = extract_prompt_activations(model, tokenizer, texts, device)

    meta_df = pd.DataFrame(records_to_metadata(records))
    p_layer = primary_layer(model.config.num_hidden_layers)
    manifest = {
        "study": "assistant_axis_traits",
        "model_name": args.model,
        "n_layers": n_layers, "hidden": hidden, "primary_layer": p_layer,
        "n_records": len(records), "n_traits": int(meta_df["trait"].nunique()),
        "n_instructions": args.n_instructions, "n_questions": args.n_questions,
        "seed": args.seed, "views": ["prompt_avg", "prompt_last"],
        "polarity_counts": meta_df["polarity"].value_counts().to_dict(),
        "dtype": "float16",
    }
    print("Saving to", args.out_dir)
    save_embeddings(avg, last, meta_df, manifest, out_dir=Path(args.out_dir))
    mb = (avg.nbytes + last.nbytes) / 1e6
    print(f"Done in {time.time()-t0:.1f}s. primary_layer={p_layer}. "
          f"Arrays: {avg.shape} x2 = {mb:.0f} MB fp16")


if __name__ == "__main__":
    main()
