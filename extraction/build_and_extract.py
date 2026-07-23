"""Stage 0: build persona prompts, extract activations, save locally.

Usage:
    .venv/bin/python -m extraction.build_and_extract
    .venv/bin/python -m extraction.build_and_extract --limit 40   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


from manifold_persona.config import MODEL_NAME, EMBEDDINGS_DIR, TRAITS, primary_layer
from manifold_persona.prompts import build_prompt_records, render_prompts, records_to_metadata
from manifold_persona.extract import load_model_and_tokenizer, extract_prompt_activations
from manifold_persona.io import save_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=str(EMBEDDINGS_DIR))
    ap.add_argument("--no_neutral", action="store_true", help="skip neutral (no-instruction) records")
    ap.add_argument("--limit", type=int, default=None, help="cap #records (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading model {args.model} ...")
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    print(f"device={device}  n_hidden_states={n_layers}  hidden={hidden}")

    print("Building prompt records from persona_vectors artifacts ...")
    records = build_prompt_records(traits=TRAITS, include_neutral=not args.no_neutral)
    if args.limit:
        records = records[: args.limit]
    records = render_prompts(records, tokenizer)
    print(f"{len(records)} prompt records")

    texts = [r.text for r in records]
    avg, last = extract_prompt_activations(model, tokenizer, texts, device)

    meta_df = pd.DataFrame(records_to_metadata(records))
    p_layer = primary_layer(model.config.num_hidden_layers)
    manifest = {
        "model_name": args.model,
        "n_layers": n_layers,
        "hidden": hidden,
        "primary_layer": p_layer,
        "n_records": len(records),
        "traits": TRAITS,
        "views": ["prompt_avg", "prompt_last"],
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
