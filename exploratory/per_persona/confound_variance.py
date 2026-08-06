"""CONFOUND — how much of a role's internal variation is the extraction grid?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 6)

Each role's points are a complete two-factor grid: n_i instruction phrasings x
n_q shared questions, one point per cell. A two-way ANOVA on that grid — done on
VECTORS, so every sum of squares is a Frobenius sum over all 2048 dimensions at
once — splits each role's within-role variance three ways:

    instruction phrasing   forced by the design
    question               forced by the design
    interaction            the ONLY persona-specific term

Instruction and question are the two axes of the grid, so a manifold measured on
gridded points reports the grid unless interaction carries real variance. The
three sum to exactly 1 because the grid is balanced, which makes the instruction
and question effects orthogonal (md/METHODS.md S1).

Run on BOTH clouds, so the contrast is reproduced from live numbers rather than
from the plan's prose: on the 25-point prompt cloud 99.4% of within-role variance
is the grid (interaction 0.6%), while on the 200-point response cloud interaction
carries 17.7%. That ~30x difference is the reason per-role geometry is measurable
here and was not measurable before.

Produces `prompt_vs_response_L<L>.json` -> fig09.

Usage:
    .venv/bin/python exploratory/per_persona/confound_variance.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from common import (load_role_clouds, design_fractions, grid_shape,
                    small_matrix_ops)

TERMS = ("instr_frac", "quest_frac", "interaction_frac")


def contrast_experiment(run_dir: Path, L: int):
    """Variance decomposition of the prompt cloud vs this response cloud."""
    rows = {}
    for label, d in (("response_5x40", os.environ.get("MP_ROLE_DIR",
                                                      "data/embeddings_roles_resp40")),
                     ("prompt_5x5", "data/embeddings_roles")):
        prev = os.environ.get("MP_ROLE_DIR")
        os.environ["MP_ROLE_DIR"] = d
        try:
            roles, clouds, factors, _ = load_role_clouds("prompt_avg", None)
            n_i, n_q, add_rank = grid_shape(factors)
            with small_matrix_ops():
                f = pd.DataFrame([design_fractions(clouds[r], *factors[r])
                                  for r in roles])
            rows[label] = {
                "dir": d, "n_roles": len(roles),
                "points_per_role": int(len(next(iter(clouds.values())))),
                "grid": [n_i, n_q], "additive_rank": add_rank,
                **{k: {"median": float(f[k].median()),
                       "q25": float(f[k].quantile(.25)),
                       "q75": float(f[k].quantile(.75))} for k in TERMS}}
        except Exception as e:  # noqa: BLE001
            rows[label] = {"dir": d, "error": str(e)}
            print(f"  [contrast] {label} unavailable: {e}")
        finally:
            if prev is None:
                os.environ.pop("MP_ROLE_DIR", None)
            else:
                os.environ["MP_ROLE_DIR"] = prev

    out = {"exploratory": True, "clouds": rows}
    if all("error" not in v for v in rows.values()):
        a = rows["prompt_5x5"]["interaction_frac"]["median"]
        b = rows["response_5x40"]["interaction_frac"]["median"]
        out["interaction_fold_change"] = float(b / a) if a else None
    json.dump(out, open(run_dir / "data" / f"prompt_vs_response_L{L}.json", "w"),
              indent=2, default=float)

    print("\n== prompt vs response: within-role variance decomposition ==")
    print(f"  {'cloud':16s} {'grid':>8s} {'rank':>5s} {'instr':>8s} {'quest':>8s} "
          f"{'inter':>8s}")
    for k, v in rows.items():
        if "error" in v:
            print(f"  {k:16s} unavailable")
            continue
        print(f"  {k:16s} {v['grid'][0]:3d}x{v['grid'][1]:<4d} {v['additive_rank']:5d} "
              f"{v['instr_frac']['median']:8.3f} {v['quest_frac']['median']:8.3f} "
              f"{v['interaction_frac']['median']:8.3f}")
    if out.get("interaction_fold_change"):
        print(f"  interaction fold-change response/prompt: "
              f"{out['interaction_fold_change']:.1f}x")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    args = ap.parse_args()
    contrast_experiment(Path(args.outdir), args.label_layer)


if __name__ == "__main__":
    main()
