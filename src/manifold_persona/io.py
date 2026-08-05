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
import os
import sys
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


def check_attention_sink(manifest: dict, in_dir: Path) -> None:
    """Refuse point clouds extracted before the attention-sink fix (0bd7d0b).

    Rule:
    - ``token_basis == "response"`` -> clean by construction (position 0, the
      sink token, is never pooled), so nothing to check.
    - otherwise (prompt-token clouds) -> the manifest MUST carry a non-null
      ``sink_factor``, which only the post-fix extractor writes. Missing means
      an old cloud; null means it was extracted with ``--keep_sinks``. Both are
      unclean and raise.

    Set ``MP_ALLOW_UNCLEAN=1`` to load anyway (warns loudly on stderr).
    """
    if manifest.get("token_basis") == "response":
        return
    if manifest.get("sink_factor") is not None:
        return
    if os.environ.get("MP_ALLOW_UNCLEAN") == "1":
        print("\n" + "!" * 78 +
              f"\n!! UNCLEAN POINT CLOUD: {in_dir}"
              "\n!! Its manifest.json has no non-null 'sink_factor', so this prompt-token"
              "\n!! cloud predates (or opted out of) the attention-sink fix 0bd7d0b."
              "\n!! Position 0 (the attention sink) is pooled into every point and"
              "\n!! dominates the geometry. Loading anyway because MP_ALLOW_UNCLEAN=1."
              "\n!! Results from this cloud are NOT publishable.\n" + "!" * 78 + "\n",
              file=sys.stderr, flush=True)
        return
    raise RuntimeError(
        f"Refusing to load unclean point cloud: {in_dir}\n"
        "Rule: a cloud is clean if its manifest.json has token_basis == 'response' "
        "(sink token excluded during pooling) OR a non-null 'sink_factor' "
        "(written only by the post-0bd7d0b prompt extractor). This manifest has "
        f"token_basis={manifest.get('token_basis')!r} and "
        f"sink_factor={manifest.get('sink_factor')!r}, so it predates the "
        "attention-sink fix and its geometry is dominated by position 0.\n"
        "Re-extract the cloud, or set MP_ALLOW_UNCLEAN=1 to load it anyway.")


def load_layer(view: str = "prompt_avg", layer: int = None,
               in_dir: Path = EMBEDDINGS_DIR) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """Load a single layer as [N, hidden] float32 + metadata + manifest.

    Uses mmap so we never read the whole [N, n_layers, hidden] array into RAM.
    ``view`` is "prompt_avg" or "prompt_last". ``layer`` defaults to the
    manifest's primary_layer.
    """
    in_dir = Path(in_dir)
    manifest = load_manifest(in_dir)
    check_attention_sink(manifest, in_dir)
    if layer is None:
        layer = manifest["primary_layer"]
    fname = AVG_FILE if view == "prompt_avg" else LAST_FILE
    arr = np.load(in_dir / fname, mmap_mode="r")   # [N, n_layers, hidden]
    n_layers = arr.shape[1]
    if not -n_layers <= layer < n_layers:
        raise IndexError(
            f"layer={layer} is out of range for {in_dir / fname}: the array has "
            f"{n_layers} layers (manifest n_layers={manifest.get('n_layers')}), "
            f"so layer must be in [{-n_layers}, {n_layers - 1}].")
    X = np.asarray(arr[:, layer, :], dtype=np.float32)
    meta = load_metadata(in_dir)
    return X, meta, manifest
