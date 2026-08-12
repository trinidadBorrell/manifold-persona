"""Fetch the 40q response cloud, keep only the primary layer, drop the 8.4GB blob.

Writes `data/embeddings_roles_resp40`: the same cloud as the HF repo but thinned
to its primary layer, with `n_layers: 1` and `primary_layer: 0`, so scripts
reading it pass `--layer 0`. (The unthinned 37-layer copy lives in
`data/embeddings_roles_resp_40q` and takes `--layer 19`.)

Everything is behind `main()` and a `__main__` guard on purpose. The body
downloads 8.4 GB and then unlinks a file out of the shared HF cache, and this
module sits in a directory that gets swept by import — a bare import used to
start the download and the deletion as a side effect.

Usage:
    .venv/bin/python exploratory/per_persona/fetch_resp40.py
    .venv/bin/python exploratory/per_persona/fetch_resp40.py --keep-blob
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from manifold_persona.hf_utils import read_token

RID = "triniborrell/manifold-persona-roles-response-40q"
OUT = Path("data/embeddings_roles_resp40")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-blob", action="store_true",
                    help="leave the 8.4 GB source array in the HF cache instead "
                         "of deleting it after the thinned copy is written")
    ap.add_argument("--outdir", default=None, help=f"default {OUT}")
    args = ap.parse_args()
    out = Path(args.outdir) if args.outdir else OUT
    out.mkdir(parents=True, exist_ok=True)
    tok = read_token()

    man = json.load(open(hf_hub_download(RID, "manifest.json", repo_type="dataset", token=tok)))
    meta = pd.read_csv(hf_hub_download(RID, "metadata.csv", repo_type="dataset", token=tok))
    L = man["primary_layer"]
    print(f"manifest primary_layer={L}  n_records={man['n_records']}  n_layers={man['n_layers']}")

    print("downloading prompt_avg.npy (8.4 GB) ...", flush=True)
    p = hf_hub_download(RID, "prompt_avg.npy", repo_type="dataset", token=tok)
    arr = np.load(p, mmap_mode="r")
    print("downloaded shape", arr.shape, arr.dtype, flush=True)
    assert arr.shape == (man["n_records"], man["n_layers"], man["hidden"])

    # Keep the primary layer only, but preserve the [N, n_layers, hidden] loader
    # contract with n_layers=1 so io.load_layer works unchanged.
    sub = np.ascontiguousarray(arr[:, L, :])[:, None, :]
    np.save(out / "prompt_avg.npy", sub)
    print("wrote", out / "prompt_avg.npy", sub.shape, sub.nbytes / 1e6, "MB")

    meta.to_parquet(out / "metadata.parquet", index=False)
    man2 = dict(man)
    man2.update({"n_layers": 1, "primary_layer": 0, "source_layer": L,
                 "source_repo": RID, "views": ["prompt_avg"],
                 "note": f"single layer {L} extracted from the 37-layer HF cloud"})
    json.dump(man2, open(out / "manifest.json", "w"), indent=2)

    del arr
    if args.keep_blob:
        print(f"done. source blob kept at {p}")
    else:
        Path(p).unlink(missing_ok=True)     # drop the 8.4 GB blob
        print("done. cache blob removed.")


if __name__ == "__main__":
    main()
