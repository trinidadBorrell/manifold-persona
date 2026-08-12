"""Residual-noise null for the early interaction rise (three-layer-depth, follow-up 2).

interaction_frac is the residual ANOVA share (see `design_fractions`), so
extra per-point noise inflates it by construction — the fraction alone
cannot separate noise from structure. This script tests the residual's
STRUCTURE instead, per role and per depth:

- within-role split-half replication: split the questions into even/odd
  halves, remove main effects inside each half, take each half's top-k
  residual subspace, measure overlap ||Va^T Vb||_F^2 / k. Noise that is
  independent across cells does not replicate across halves.
- between-role baseline: the same overlap between DIFFERENT roles
  (role i half-A vs role i+1 half-A). Generic structured noise (shared
  token covariance) lands here; role-specific structure exceeds it.

Decision rule (wiki/experiments/three-layer-depth, follow-up 2): the noise
explanation for the L10 interaction rise is refuted if within > between at
L10 with a 2000-draw role-bootstrap CI excluding 0, and the L10 margin is
not smaller than the L19 margin.

Usage:
    MP_ROLE_DIR=data/embeddings_roles_resp_40q \\
      .venv/bin/python exploratory/per_persona/residual_noise_null.py \\
      --outdir output/residual_noise_null_2026-08-12
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import load_role_clouds, small_matrix_ops

LAYERS = (10, 19, 26)
K = 10          # residual subspace dims per half
N_BOOT = 2000
SEED = 0


def half_residual(Xr: np.ndarray, instr: np.ndarray, quest: np.ndarray,
                  keep: np.ndarray) -> np.ndarray:
    """Main-effect residual computed inside one question-half."""
    X = Xr[keep]
    ih, qh = instr[keep], quest[keep]
    qh = np.unique(qh, return_inverse=True)[1]
    Xc = X - X.mean(0)
    A = np.stack([Xc[ih == u].mean(0) for u in range(ih.max() + 1)])
    B = np.stack([Xc[qh == u].mean(0) for u in range(qh.max() + 1)])
    return Xc - (A[ih] + B[qh])


def topk(resid: np.ndarray) -> np.ndarray:
    """Top-K right singular vectors, (K, dim)."""
    return np.linalg.svd(resid, full_matrices=False)[2][:K]


def overlap(Va: np.ndarray, Vb: np.ndarray) -> float:
    return float(((Va @ Vb.T) ** 2).sum() / K)


def boot_ci(vals: np.ndarray, rng: np.random.Generator):
    meds = [float(np.median(vals[idx]))
            for idx in rng.integers(0, len(vals), (N_BOOT, len(vals)))]
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    out = {"exploratory": True, "k": K, "n_boot": N_BOOT, "layers": {}}
    for L in LAYERS:
        roles, clouds, factors, _ = load_role_clouds("prompt_avg", L)
        within, between = [], []
        prev_Va = None
        with small_matrix_ops():
            for r in roles:
                instr, quest = factors[r]
                Va = topk(half_residual(clouds[r], instr, quest, quest % 2 == 0))
                Vb = topk(half_residual(clouds[r], instr, quest, quest % 2 == 1))
                within.append(overlap(Va, Vb))
                if prev_Va is not None:
                    between.append(overlap(prev_Va, Va))
                prev_Va = Va
        within, between = np.array(within), np.array(between)
        margin = within[1:] - between          # role i's within vs (i-1,i) between
        out["layers"][str(L)] = {
            "n_roles": len(roles),
            "within_median": float(np.median(within)),
            "within_ci": boot_ci(within, rng),
            "between_median": float(np.median(between)),
            "between_ci": boot_ci(between, rng),
            "margin_median": float(np.median(margin)),
            "margin_ci": boot_ci(margin, rng),
        }
        s = out["layers"][str(L)]
        print(f"L{L}: within={s['within_median']:.4f} {s['within_ci']}  "
              f"between={s['between_median']:.4f} {s['between_ci']}  "
              f"margin={s['margin_median']:.4f} {s['margin_ci']}")

    p = Path(args.outdir)
    p.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p / "results.json", "w"), indent=2)
    print(f"wrote {p / 'results.json'}")

    run_dir = os.environ.get("LABBOOK_RUN_DIR")
    if run_dir:
        flat = {f"L{L}_{k}": v for L in LAYERS
                for k, v in out["layers"][str(L)].items()
                if isinstance(v, float)}
        json.dump(flat, open(Path(run_dir) / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
