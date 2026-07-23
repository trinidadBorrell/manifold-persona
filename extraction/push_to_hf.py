"""Push the local point cloud to Hugging Face as a dataset repo.

Builds a datasets.Dataset where each row carries its metadata plus the
per-layer activation arrays (prompt_avg, prompt_last) as Array2D
[n_layers, hidden] fp16, then push_to_hub(<username>/manifold-persona).

Usage:
    .venv/bin/python -m extraction.push_to_hf
    .venv/bin/python -m extraction.push_to_hf --private
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from manifold_persona.config import EMBEDDINGS_DIR, HF_REPO_NAME
from manifold_persona.io import load_metadata, load_manifest, AVG_FILE, LAST_FILE
from manifold_persona.hf_utils import read_token, whoami


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(EMBEDDINGS_DIR))
    ap.add_argument("--repo_name", default=HF_REPO_NAME)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from datasets import Dataset, Features, Value, Array2D

    in_dir = Path(args.in_dir)
    meta = load_metadata(in_dir)
    manifest = load_manifest(in_dir)
    n_layers, hidden = manifest["n_layers"], manifest["hidden"]

    avg = np.load(in_dir / AVG_FILE)     # [N, n_layers, hidden] fp16
    last = np.load(in_dir / LAST_FILE)

    token = read_token()
    user = whoami(token)
    repo_id = f"{user}/{args.repo_name}"

    features = Features({
        "trait": Value("string"),
        "polarity": Value("string"),
        "instruction_idx": Value("int32"),
        "question_idx": Value("int32"),
        "question": Value("string"),
        "system": Value("string"),
        "text": Value("string"),
        "prompt_avg": Array2D(shape=(n_layers, hidden), dtype="float16"),
        "prompt_last": Array2D(shape=(n_layers, hidden), dtype="float16"),
    })

    def gen():
        for i, row in meta.iterrows():
            yield {
                "trait": row["trait"],
                "polarity": row["polarity"],
                "instruction_idx": int(row["instruction_idx"]),
                "question_idx": int(row["question_idx"]),
                "question": row["question"],
                "system": row["system"] if row["system"] is not None else "",
                "text": row["text"],
                "prompt_avg": avg[i],
                "prompt_last": last[i],
            }

    print(f"Building dataset ({len(meta)} rows, layers={n_layers}, hidden={hidden}) ...")
    ds = Dataset.from_generator(gen, features=features)

    print(f"Pushing to {repo_id} (private={args.private}) ...")
    ds.push_to_hub(repo_id, token=token, private=args.private)
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
