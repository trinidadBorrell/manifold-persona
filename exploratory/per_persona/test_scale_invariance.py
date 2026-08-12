"""Which metrics survive a pure rescale of the cloud — asserted, not assumed.

WHY THIS IS A TEST AND NOT A COMMENT
------------------------------------
Cloud scale is the study's biggest confound: `log_var` correlates +0.731 with
`axis_proj`, so any metric that secretly carries scale will produce a strong,
meaningless correlation. Three separate claims in this codebase depend on
scale-invariance being true:

  1. `density.py` normalises to unit RMS radius so that `knn_dist_*` and
     `kde_logdens_*` are scale-free BY CONSTRUCTION rather than by a linear
     control applied afterwards.
  2. `curvature.py` does the same for the kNN graph, so `frc_*` (which is
     weight-dependent) is on the same footing as `orc_*` (which is a ratio).
  3. `topology.py`'s `persistence_entropy_*` and `H*_max_lifetime_frac` are
     claimed scale-free while `H*_total_persistence` and `H*_max_lifetime` are
     claimed NOT to be.

All three are easy to break with a one-line change months from now. Multiplying
a cloud by a constant changes its size and nothing else, so this checks them all
at once, and it also documents which metrics are deliberately scale-carrying.

Usage:
    .venv/bin/python exploratory/per_persona/test_scale_invariance.py
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import PCA

from common import load_role_clouds, small_matrix_ops
from metrics import PANEL_COLS, panel_metrics

# Metrics that SHOULD move with the cloud, and are documented as such in
# md/SPACES.md. Everything else in PANEL_COLS must be invariant.
EXPECTED_SCALE_CARRYING = {
    "H0_total_persistence",
    "H1_total_persistence", "H1_max_lifetime",
    "H2_total_persistence", "H2_max_lifetime",
}
SCALE = 2.0
TOL = 1e-6


def main():
    roles, clouds, factors, _ = load_role_clouds("prompt_avg", None)
    role = "poet" if "poet" in clouds else roles[0]
    X = clouds[role]
    instr, quest = factors[role]
    print(f"role={role}  n={len(X)}  rescale x{SCALE}\n")

    with small_matrix_ops():
        a, _, _ = panel_metrics(X, instr, quest)
        b, _, _ = panel_metrics(X * SCALE, instr, quest)

    bad_invariant, bad_carrying = [], []
    print(f"{'metric':30s} {'original':>12s} {'rescaled':>12s}  {'':<10s}")
    for c in PANEL_COLS:
        va, vb = a.get(c), b.get(c)
        if va is None or vb is None or not np.isfinite([va, vb]).all():
            print(f"{c:30s} {'n/a':>12s} {'n/a':>12s}  SKIPPED (non-finite)")
            continue
        moved = abs(va - vb) > TOL * max(1.0, abs(va))
        want_move = c in EXPECTED_SCALE_CARRYING
        if moved and not want_move:
            bad_invariant.append(c)
            tag = "*** LEAKS SCALE ***"
        elif not moved and want_move:
            bad_carrying.append(c)
            tag = "*** expected to scale, did not ***"
        else:
            tag = "scales (expected)" if moved else "invariant"
        print(f"{c:30s} {va:12.4f} {vb:12.4f}  {tag}")

    print()
    if bad_invariant:
        print(f"FAIL: {len(bad_invariant)} metric(s) leak cloud scale: {bad_invariant}")
    if bad_carrying:
        print(f"FAIL: {len(bad_carrying)} metric(s) were expected to scale but did "
              f"not: {bad_carrying}")
    if bad_invariant or bad_carrying:
        sys.exit(1)
    n_inv = len(PANEL_COLS) - len(EXPECTED_SCALE_CARRYING)
    print(f"PASS: {n_inv} metrics invariant, "
          f"{len(EXPECTED_SCALE_CARRYING)} scale-carrying as documented.")


if __name__ == "__main__":
    main()
