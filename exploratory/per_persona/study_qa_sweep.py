"""Does the per-persona geometry result hold as you add questions?

The published result rests on one question budget: 40 questions per role, 5
instructions each, 200 points. This runs the SAME geometry study at 40, 80,
120, 180 and 240 questions and puts the tiers side by side. If the per-role
geometry is real it should sharpen, or at least survive, as points are added.
If it is an artifact of a thin cloud it should move.

THE SUBSETS ARE NESTED, AND SHARED ACROSS ROLES
-----------------------------------------------
Both properties are load-bearing and neither is automatic:

  nested       40 questions are a subset of the 80, which are a subset of the
               120, and so on. Tier-to-tier differences are then "what adding
               these questions did", not "what two unrelated question samples
               happen to score". Independent samples per tier would confound
               budget with which questions were drawn.

  shared       every role is asked the SAME question set at a given tier.
               `question_idx` indexes the global 240-question pool, so a fixed
               set of indices means a fixed set of questions for all 276 roles.
               Without this, between-role comparisons at a tier would be partly
               a comparison of different questions.

The 40-question tier is not an arbitrary draw: it is EXACTLY the question set
of `data/embeddings_roles_resp40`, verified to be a subset of the 240 pool. So
the smallest tier reproduces the published result rather than merely resembling
it, and any movement across tiers is attributable to budget alone. Tiers above
40 extend that anchor with a seeded permutation of the remaining questions.

The grid stays complete at every tier — all 5 instructions are kept, only
questions are subsetted — so `grid_shape`'s balance assertion holds throughout
and the variance decomposition stays interpretable.

WHAT IT RUNS
------------
`run_geometry.py` unchanged, once per tier, against a materialised per-tier
cloud directory. Nothing in the existing study is modified: the subsetting
happens by writing a filtered cloud to `data/` and pointing `MP_ROLE_DIR` at
it, which is the same mechanism the resp40 arm already uses.

Layer: the 240q cloud is published pre-thinned, one layer, whose `source_layers`
is `[19]`. Index 0 IS layer 19, so every tier runs `--layer 0 --label-layer 19`
and the outputs are named `*_L19` like the rest of the study.

Usage:
    .venv/bin/python exploratory/per_persona/study_qa_sweep.py \\
        --outdir output/per_persona_axis_centroid_11-Aug-2026

    # what it would do, without running or writing clouds
    .venv/bin/python exploratory/per_persona/study_qa_sweep.py --outdir X --dry-run

    # one tier only, e.g. to time it first
    .venv/bin/python exploratory/per_persona/study_qa_sweep.py --outdir X --tiers 40
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PY = sys.executable

BASE = Path("data/embeddings_roles_resp240")     # the full 240q cloud
ANCHOR = Path("data/embeddings_roles_resp40")    # supplies the 40-question tier
TIERS = (40, 80, 120, 180, 240)
SEED = 0


def question_tiers(meta: pd.DataFrame, tiers, anchor_qs, seed: int = SEED) -> dict:
    """tier -> sorted list of question_idx, nested and anchored.

    `anchor_qs` seeds the smallest tier; the rest of the pool is shuffled once
    with `seed` and consumed in that fixed order, so every tier is a prefix of
    the next and re-running gives the same sets.
    """
    pool = np.array(sorted(meta["question_idx"].unique()))
    anchor = np.array(sorted(anchor_qs))
    missing = set(anchor) - set(pool)
    if missing:
        raise SystemExit(f"anchor questions absent from the 240 pool: {sorted(missing)}")

    rest = np.array([q for q in pool if q not in set(anchor)])
    rng = np.random.default_rng(seed)
    rest = rng.permutation(rest)
    ordered = np.concatenate([anchor, rest])          # anchor first, then filler

    out = {}
    for k in sorted(tiers):
        if k > len(ordered):
            raise SystemExit(f"tier {k} exceeds the {len(ordered)}-question pool")
        if k < len(anchor):
            raise SystemExit(f"tier {k} is smaller than the {len(anchor)}-question anchor")
        out[k] = sorted(int(q) for q in ordered[:k])
    return out


def build_tier_cloud(k: int, qs, meta: pd.DataFrame, view: str, dest: Path) -> Path:
    """Write a cloud directory holding only the rows whose question is in `qs`."""
    dest.mkdir(parents=True, exist_ok=True)
    keep = meta["question_idx"].isin(set(qs)).to_numpy()
    n_exp = meta["role"].nunique() * meta["instruction_idx"].nunique() * k
    if keep.sum() != n_exp:
        raise SystemExit(f"tier {k}: kept {keep.sum()} rows, expected {n_exp}")

    arr = np.load(BASE / f"{view}.npy", mmap_mode="r")   # [N, 1, hidden]
    np.save(dest / f"{view}.npy", np.ascontiguousarray(arr[keep]))
    del arr
    meta.loc[keep].reset_index(drop=True).to_parquet(dest / "metadata.parquet", index=False)

    man = json.load(open(BASE / "manifest.json"))
    man.update({"n_records": int(keep.sum()), "n_questions": k,
                "views": [view], "question_subset": list(map(int, qs)),
                "subset_of": str(BASE),
                "note": f"{k}-question nested subset; index 0 IS layer 19"})
    json.dump(man, open(dest / "manifest.json", "w"), indent=2)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True,
                    help="parent run folder; one subfolder per tier")
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--tiers", type=int, nargs="+", default=list(TIERS))
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--clouddir", default="data/_qa_sweep",
                    help="where the per-tier clouds are materialised")
    ap.add_argument("--keep-clouds", action="store_true",
                    help="keep each tier's .npy after its run (default: delete "
                         "it, since the five together are several GB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tier plan and exit without writing anything")
    args = ap.parse_args()

    if not (BASE / "manifest.json").exists():
        raise SystemExit(f"{BASE} missing — run fetch_resp240.py first")
    meta = pd.read_parquet(BASE / "metadata.parquet")
    anchor_qs = pd.read_parquet(ANCHOR / "metadata.parquet")["question_idx"].unique() \
        if (ANCHOR / "metadata.parquet").exists() else []
    if len(anchor_qs) == 0:
        raise SystemExit(f"{ANCHOR} missing — it defines the 40-question tier")

    tiers = question_tiers(meta, args.tiers, anchor_qs, args.seed)
    n_i = meta["instruction_idx"].nunique()
    n_roles = meta["role"].nunique()

    print(f"anchor: {len(anchor_qs)} questions from {ANCHOR}")
    print(f"{'tier':>6}  {'points/role':>11}  {'total rows':>10}  nested-in-next")
    ks = sorted(tiers)
    for a, b in zip(ks, ks[1:] + [None]):
        ok = "yes" if b is None or set(tiers[a]) <= set(tiers[b]) else "NO"
        print(f"{a:>6}  {n_i * a:>11}  {n_roles * n_i * a:>10}  {ok if b else '(top)'}")
    if args.dry_run:
        print("\ndry run — nothing written")
        return

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump({"tiers": {str(k): v for k, v in tiers.items()},
               "anchor_source": str(ANCHOR), "seed": args.seed,
               "base_cloud": str(BASE), "view": args.view,
               "nested": True, "shared_across_roles": True},
              open(out_root / "question_tiers.json", "w"), indent=2)

    timings = {}
    for k in ks:
        cloud = Path(args.clouddir) / f"q{k}"
        tier_out = out_root / f"q{k}"
        print(f"\n{'=' * 70}\n=== tier {k} questions -> {tier_out}\n{'=' * 70}", flush=True)
        t0 = time.time()
        build_tier_cloud(k, tiers[k], meta, args.view, cloud)
        env = dict(os.environ, MP_ROLE_DIR=str(cloud))
        r = subprocess.run(
            [PY, str(HERE / "run_geometry.py"), "--view", args.view,
             "--layer", "0", "--label-layer", "19",
             "--n-null", str(args.n_null), "--outdir", str(tier_out)],
            env=env)
        timings[k] = round(time.time() - t0, 1)
        print(f"--- tier {k}: {'ok' if r.returncode == 0 else f'FAILED rc={r.returncode}'} "
              f"in {timings[k]}s", flush=True)
        if not args.keep_clouds:
            (cloud / f"{args.view}.npy").unlink(missing_ok=True)
        if r.returncode != 0:
            raise SystemExit(f"tier {k} failed; stopping rather than reporting a partial sweep")

    json.dump(timings, open(out_root / "tier_runtimes.json", "w"), indent=2)
    if not args.keep_clouds:
        shutil.rmtree(args.clouddir, ignore_errors=True)
    print(f"\nDone. {len(ks)} tiers in {out_root}")


if __name__ == "__main__":
    main()
