"""Local persistence + loading of the embedding point cloud.

Layout under data/embeddings/:
- prompt_avg.npy   [N, n_layers, hidden] fp16
- prompt_last.npy  [N, n_layers, hidden] fp16
- metadata.parquet  (N rows: trait, polarity, instruction_idx, question_idx,
                     question, system, text)
- manifest.json     (model_name, n_layers, hidden, primary_layer, counts)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import EMBEDDINGS_DIR

AVG_FILE = "prompt_avg.npy"
LAST_FILE = "prompt_last.npy"
META_FILE = "metadata.parquet"
MANIFEST_FILE = "manifest.json"


def save_embeddings(prompt_avg, prompt_last, metadata_df: pd.DataFrame,
                    manifest: dict, out_dir: Path = EMBEDDINGS_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / AVG_FILE, prompt_avg)
    np.save(out_dir / LAST_FILE, prompt_last)
    metadata_df.to_parquet(out_dir / META_FILE, index=False)
    with open(out_dir / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def load_metadata(in_dir: Path = EMBEDDINGS_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(in_dir) / META_FILE)


def load_manifest(in_dir: Path = EMBEDDINGS_DIR) -> dict:
    with open(Path(in_dir) / MANIFEST_FILE) as f:
        return json.load(f)


def load_layer(view: str = "prompt_avg", layer: int = None,
               in_dir: Path = EMBEDDINGS_DIR) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """Load a single layer as [N, hidden] float32 + metadata + manifest.

    Uses mmap so we never read the whole [N, n_layers, hidden] array into RAM.
    ``view`` is "prompt_avg" or "prompt_last". ``layer`` defaults to the
    manifest's primary_layer.
    """
    in_dir = Path(in_dir)
    manifest = load_manifest(in_dir)
    if layer is None:
        layer = manifest["primary_layer"]
    fname = AVG_FILE if view == "prompt_avg" else LAST_FILE
    arr = np.load(in_dir / fname, mmap_mode="r")   # [N, n_layers, hidden]
    X = np.asarray(arr[:, layer, :], dtype=np.float32)
    meta = load_metadata(in_dir)
    return X, meta, manifest
