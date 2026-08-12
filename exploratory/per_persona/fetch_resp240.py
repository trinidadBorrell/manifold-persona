"""Fetch the 240-question response cloud into `data/embeddings_roles_resp240`.

Unlike the 40q repo, this one is already published thinned — `n_layers: 1`,
`source_layers: [19]` — so there is nothing to strip out. This script only
mirrors it into the layout the loader expects (metadata as parquet, a manifest
with `primary_layer: 0`) and checks the grid is complete before writing.

The stored layer IS layer 19; it just sits at index 0 in a one-layer array.
Scripts therefore take `--layer 0` to read it and `--label-layer 19` to name
their outputs, exactly as for resp40.

Clean by construction: `token_basis: response` means position 0 — the attention
sink — is never pooled, so this cloud passes `check_attention_sink` without a
`sink_factor` key.

Usage:
    .venv/bin/python exploratory/per_persona/fetch_resp240.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from manifold_persona.hf_utils import read_token

RID = "triniborrell/manifold-persona-roles-response-240q"
OUT = Path("data/embeddings_roles_resp240")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help=f"default {OUT}")
    ap.add_argument("--view", default="prompt_avg",
                    help="which pooled view to mirror (prompt_avg or prompt_last)")
    args = ap.parse_args()
    out = Path(args.outdir) if args.outdir else OUT
    out.mkdir(parents=True, exist_ok=True)
    tok = read_token()

    man = json.load(open(hf_hub_download(RID, "manifest.json", repo_type="dataset", token=tok)))
    print(f"manifest: token_basis={man['token_basis']!r} n_records={man['n_records']} "
          f"n_roles={man['n_roles']} n_questions={man['n_questions']} "
          f"source_layers={man.get('source_layers')}")
    if man.get("token_basis") != "response":
        raise SystemExit(f"expected a response-token cloud, got {man.get('token_basis')!r}")

    meta = pd.read_csv(hf_hub_download(RID, "metadata.csv", repo_type="dataset", token=tok))

    # The whole subset sweep assumes a complete role x instruction x question
    # grid -- every rank and variance-fraction statement downstream depends on
    # it. Check here, once, rather than letting grid_shape fail per tier.
    per_role = meta.groupby("role").size()
    n_i, n_q = meta.instruction_idx.nunique(), meta.question_idx.nunique()
    if per_role.min() != per_role.max() or per_role.iloc[0] != n_i * n_q:
        raise SystemExit(f"grid is not complete: {per_role.min()}-{per_role.max()} points "
                         f"per role against a {n_i}x{n_q} grid")
    print(f"grid complete: {meta.role.nunique()} roles x {n_i} instructions x {n_q} questions")

    src = hf_hub_download(RID, f"{args.view}.npy", repo_type="dataset", token=tok)
    arr = np.load(src, mmap_mode="r")
    print(f"downloaded {args.view}.npy {arr.shape} {arr.dtype}")
    if arr.shape != (man["n_records"], man["n_layers"], man["hidden"]):
        raise SystemExit(f"array shape {arr.shape} disagrees with the manifest")

    shutil.copyfile(src, out / f"{args.view}.npy")
    meta.to_parquet(out / "metadata.parquet", index=False)
    man2 = dict(man)
    man2.update({"primary_layer": 0, "views": [args.view], "source_repo": RID,
                 "note": "published pre-thinned; index 0 IS layer 19"})
    json.dump(man2, open(out / "manifest.json", "w"), indent=2)
    print(f"wrote {out} ({(out / f'{args.view}.npy').stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
