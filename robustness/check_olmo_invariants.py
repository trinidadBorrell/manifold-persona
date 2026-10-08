"""OLMo-2-7B staircase invariants: pin and re-check a few headline numbers.

A smoke alarm for the OLMo staircase, in the spirit of check_invariants.py. It
recomputes the headline numbers from the committed activation clouds, compares
them to the pinned values, prints one PASS/FAIL row each, and exits 0 (all pass)
or 1. A FAIL is a question, not a verdict: it can be a bug OR a discovery.

Every number is recomputed RAW from the depth clouds, the one time, here. No
pipeline math is duplicated.

Inputs (committed, read-only)
  output/olmo7b_hf_inputs_clean/compact/<stage>/prompt_avg_depths.npy
      (55200, 3, 4096) float16, row-aligned to the parquet below. The three
      depths are hidden_states 9 / 17 / 23 (early / mid / late) at index 0/1/2.
  output/meta_olmo_<stage>.parquet
      55200 rows, one per prompt: 276 roles x 40 questions x 5 instructions.
      The `default` role is the neutral-Assistant baseline.

What it pins, per training stage (base, dpo, rlvr) and depth
  assistant_axis = default_centroid - mean(all 276 role centroids), unit-norm.
      Roles are balanced (200 rows each), so this equals the canonical
      grand-mean axis in manifold_persona.common.assistant_axis.
  cos_base_rlvr  angle between the base and rlvr assistant axes (per depth).
  pr.<stage>     PCA participation ratio (Sum L)^2 / Sum L^2 of the 276
      role-centroid cloud (covariance PCA, no z-score), per depth.

The story the structural pins hold
  cos(base, rlvr) small  -> the post-trained assistant axis is NOT the base
                            axis; RLVR moves it.
  cos(dpo, rlvr) large   -> the two post-trained axes agree; the direction the
                            method finds is stable (a sanity control).
  PR(rlvr) > PR(base)    -> post-training widens the persona manifold.

Usage
  .venv/bin/python robustness/check_olmo_invariants.py          # check
  .venv/bin/python robustness/check_olmo_invariants.py --pin    # accept current
                                                                # numbers as pins

To update pins after an intentional change: run with --pin, read the
`git diff robustness/olmo_invariants.json`, and say why in the commit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COMPACT = REPO / "output" / "olmo7b_hf_inputs_clean" / "compact"
PINS_FILE = Path(__file__).resolve().parent / "olmo_invariants.json"

STAGES = ["base", "dpo", "rlvr"]
DEPTH_NAMES = ["early", "mid", "late"]     # compact index 0 / 1 / 2
HIDDEN_STATE_IDX = [9, 17, 23]             # provenance for the depth names
N_ROLES = 276

# Exact pins: the measured value must sit within tol of the pinned value.
# Everything here is deterministic (no bootstrap, no RNG), so the tols only
# need to cover float / BLAS noise while still firing on a real change.
EXACT = (
    [(f"cos_base_rlvr.{n}", 0.01) for n in DEPTH_NAMES]
    + [(f"pr.{s}.{n}", 0.25) for s in STAGES for n in DEPTH_NAMES]
)

# Structural pins: fixed comparisons that define the published story. They hold
# no pinned number, so --pin never touches them.
STRUCT = [
    ("cos(base,rlvr) < 0.20 at every depth  (post-trained axis != base axis)",
     lambda v: all(abs(v[f"cos_base_rlvr.{n}"]) < 0.20 for n in DEPTH_NAMES)),
    ("cos(dpo,rlvr) > 0.90 at every depth  (post-trained axes agree)",
     lambda v: all(v[f"cos_dpo_rlvr.{n}"] > 0.90 for n in DEPTH_NAMES)),
    ("PR(rlvr) > PR(base) at every depth  (post-training widens the manifold)",
     lambda v: all(v[f"pr.rlvr.{n}"] > v[f"pr.base.{n}"] for n in DEPTH_NAMES)),
]


def stage_axis_and_pr(stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis[depth, hid] unit-norm, pr[depth]) recomputed from the cloud."""
    npy = COMPACT / stage / "prompt_avg_depths.npy"
    # The labels written with the array, not a separate run's metadata: a
    # row-count match cannot show two runs share their row order.
    meta_path = COMPACT / stage / "metadata.parquet"
    A = np.load(npy, mmap_mode="r")
    meta = pd.read_parquet(meta_path)
    if len(meta) != A.shape[0]:
        raise SystemExit(f"{stage}: meta rows {len(meta)} != npy rows {A.shape[0]}")
    roles = sorted(meta["role"].unique())
    if len(roles) != N_ROLES or "default" not in roles:
        raise SystemExit(f"{stage}: expected {N_ROLES} roles incl. 'default', "
                         f"got {len(roles)}")
    rvals = meta["role"].to_numpy()
    n_depth = A.shape[1]

    # per-role centroid at each depth: (n_roles, n_depth, hid)
    C = np.empty((len(roles), n_depth, A.shape[2]), dtype=np.float64)
    counts = []
    for i, r in enumerate(roles):
        idx = np.where(rvals == r)[0]
        counts.append(len(idx))
        C[i] = np.asarray(A[idx], dtype=np.float64).mean(0)
    if len(set(counts)) != 1:
        print(f"  warning: {stage} roles are not balanced ({sorted(set(counts))} "
              f"rows); the role-centroid-mean axis then differs from the "
              f"grand-mean axis")

    di = roles.index("default")
    axis = np.empty((n_depth, A.shape[2]))
    pr = np.empty(n_depth)
    for d in range(n_depth):
        ax = C[di, d] - C[:, d].mean(0)              # default - mean(role centroids)
        axis[d] = ax / (np.linalg.norm(ax) + 1e-8)
        M = C[:, d] - C[:, d].mean(0, keepdims=True)  # covariance PCA, no z-score
        lam = np.linalg.svd(M, compute_uv=False) ** 2
        pr[d] = float(lam.sum() ** 2 / (lam ** 2).sum())
    return axis, pr


def measure() -> dict:
    """Recompute every pinned number from the depth clouds."""
    values: dict[str, float] = {}
    axes: dict[str, np.ndarray] = {}
    for st in STAGES:
        print(f"  measuring {st} ...")
        axis, pr = stage_axis_and_pr(st)
        axes[st] = axis
        for d, name in enumerate(DEPTH_NAMES):
            values[f"pr.{st}.{name}"] = float(pr[d])

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    for d, name in enumerate(DEPTH_NAMES):
        values[f"cos_base_rlvr.{name}"] = cos(axes["base"][d], axes["rlvr"][d])
        values[f"cos_base_dpo.{name}"] = cos(axes["base"][d], axes["dpo"][d])
        values[f"cos_dpo_rlvr.{name}"] = cos(axes["dpo"][d], axes["rlvr"][d])
    return values


def check(values: dict) -> int:
    if not PINS_FILE.exists():
        print("No olmo_invariants.json yet. Run with --pin first.")
        return 2
    pins = json.loads(PINS_FILE.read_text())
    exp = pins["values"]
    rows, n_fail = [], 0
    for key, tol in EXACT:
        got, want = values[key], exp[key]
        ok = abs(got - want) <= tol
        n_fail += not ok
        rows.append(("exact", key, f"{want:.4g}", f"{got:.4g}", f"±{tol:g}", ok))
    for desc, fn in STRUCT:
        ok = bool(fn(values))
        n_fail += not ok
        rows.append(("struct", desc, "", "", "", ok))

    w = max(len(r[1]) for r in rows)
    print(f"\n{'kind':7s} {'invariant':{w}s} {'pinned':>10s} {'now':>10s} "
          f"{'tol':>8s} verdict")
    for kind, name, want, got, tol, ok in rows:
        print(f"{kind:7s} {name:{w}s} {want:>10s} {got:>10s} "
              f"{tol:>8s} {'PASS' if ok else '** FAIL **'}")

    print(f"\npinned {pins['pinned_at']} at {pins['git_commit'][:9]}")
    if n_fail:
        print(f"\n{n_fail} FAIL. A fail is a question, not a verdict.")
        print("  Bug?       Compare your diff against this script and the clouds.")
        print("  Discovery? If the change is intentional, rerun with --pin and "
              "commit the olmo_invariants.json diff with a reason.")
    else:
        print("\nAll OLMo invariants hold.")
    return 1 if n_fail else 0


def pin(values: dict) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    out = {
        "pinned_at": date.today().isoformat(),
        "git_commit": commit,
        "provenance": {
            "stages": STAGES,
            "depth_names": DEPTH_NAMES,
            "hidden_state_indices": HIDDEN_STATE_IDX,
            "n_roles": N_ROLES,
            "axis": "default_centroid - mean(role centroids), unit-norm, per depth",
            "pr": "(sum L)^2 / sum L^2 of the 276 role-centroid cloud, covariance PCA",
        },
        "values": {k: round(v, 6) for k, v in values.items()},
    }
    PINS_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nPinned {len(out['values'])} values -> {PINS_FILE}")
    print("Review with: git diff robustness/olmo_invariants.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pin", action="store_true",
                    help="write the measured values as the new pins")
    args = ap.parse_args()

    if not args.pin and not PINS_FILE.exists():
        print("No olmo_invariants.json yet. Run with --pin first.")
        raise SystemExit(2)
    print("recomputing OLMo staircase headline numbers from the depth clouds")
    values = measure()
    if args.pin:
        pin(values)
    else:
        raise SystemExit(check(values))


if __name__ == "__main__":
    main()
