"""Per-depth within-role variance decomposition of the response cloud.

`confound_variance.py` has no --layer flag: it always loads each cloud's
manifest default layer, so it cannot answer "what is the interaction
fraction at depth L?". This script does exactly that, for ONE cloud (the
response cloud) at an explicit layer, using the same `design_fractions`
as every published split.

Written 2026-08-07 for the three-layer-depth experiment
(wiki/experiments/three-layer-depth/prereg.md, decision rule 1).

Usage:
    MP_ROLE_DIR=data/embeddings_roles_resp_40q \\
      .venv/bin/python exploratory/per_persona/depth_variance.py \\
      --layer 10 --outdir <run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import load_role_clouds, design_fractions, grid_shape, small_matrix_ops

TERMS = ("instr_frac", "quest_frac", "interaction_frac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True,
                    help="hidden-state index to slice from the stored cloud")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    roles, clouds, factors, _ = load_role_clouds("prompt_avg", args.layer)
    n_i, n_q, add_rank = grid_shape(factors)
    with small_matrix_ops():
        f = pd.DataFrame([design_fractions(clouds[r], *factors[r])
                          for r in roles])
    out = {"exploratory": True, "layer": args.layer, "n_roles": len(roles),
           "points_per_role": int(len(next(iter(clouds.values())))),
           "grid": [n_i, n_q], "additive_rank": add_rank,
           **{k: {"median": float(f[k].median()),
                  "q25": float(f[k].quantile(.25)),
                  "q75": float(f[k].quantile(.75))} for k in TERMS}}
    p = Path(args.outdir) / "data" / f"depth_fractions_L{args.layer}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"L{args.layer}: " + "  ".join(
        f"{k}={out[k]['median']:.4f}" for k in TERMS))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
