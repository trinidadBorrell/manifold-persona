"""Legacy invariant check: old-design pins vs committed baseline artifacts.

The pins in this directory were measured on the pre-integration panel
(commit acef6c0 era). That panel code is gone from this branch, so this
checker does NOT rerun it. It recomputes every artifact-backed number
directly from the committed baseline files and compares against the pins.

Self-contained on purpose: only numpy/pandas/scipy, no imports from
exploratory/. The partial-correlation math is copied verbatim from the
old stats_utils.partial_corr_multi so the numbers reproduce exactly.

Pins that need raw clouds (variance fracs, fold_change) are reported as
SKIP here. For a full legacy rerun, check out feature/invariant-checks
and run its robustness/check_invariants.py.

Usage:
  .venv/bin/python robustness/legacy/check_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"
PINS_FILE = HERE / "invariants.json"
L = 19

# Old ladder conventions, frozen (study_ladder.py at feature/invariant-checks).
PREDICTOR = "axis_proj"
CTRL_ALL = ["log_var", "mean_norm"]
LADDER_METRICS = ["MLE", "TwoNN", "PCA_participation_ratio",
                  "PCA_dim_90pct", "H1_total_persistence"]

# Fallback tolerances when the pins file has no "tol" block for a key.
FALLBACK_TOL = {"pr_median_real": 1.0, "pr_median_null": 1.0,
                "mle_worst_calib_error": 0.05,
                **{f"r.{m}": 0.05 for m in LADDER_METRICS}}

# Pins the baseline artifacts cannot back — they need the raw clouds.
NOT_ARTIFACT_BACKED = ["prompt.instr_frac", "prompt.quest_frac",
                       "prompt.interaction_frac", "response.instr_frac",
                       "response.quest_frac", "response.interaction_frac",
                       "fold_change"]


def partial_corr_multi(x, y, Z) -> float:
    """Pearson r between x and y with the columns of Z linearly removed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float)
    A = np.column_stack([np.ones(len(x)), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def measure() -> dict:
    values = {}

    panel = pd.read_csv(BASELINE / f"per_role_panel_L{L}.csv")
    null = json.load(open(BASELINE / f"design_null_L{L}.json"))
    calib = json.load(open(BASELINE / f"calibration_L{L}.json"))

    values["pr_median_real"] = float(
        panel["PCA_participation_ratio"].median())
    values["pr_median_null"] = float(
        null["design_null"]["PCA_participation_ratio"]["median"])
    values["mle_worst_calib_error"] = float(
        calib["verdict"]["worst_relative_error_d_le_10"]["MLE"])

    # Old ladder rule: `default` defines the axis, so it is excluded.
    d = panel[panel["role"] != "default"].dropna(
        subset=[PREDICTOR] + CTRL_ALL)
    x, Z = d[PREDICTOR], d[CTRL_ALL]
    for m in LADDER_METRICS:
        values[f"r.{m}"] = partial_corr_multi(x, d[m], Z)
    return values


def main() -> None:
    pins = json.loads(PINS_FILE.read_text())
    exp, tols, signs = pins["values"], pins.get("tol", {}), pins["signs"]
    values = measure()
    rows, n_fail = [], 0

    for key, got in values.items():
        tol = tols.get(key, FALLBACK_TOL[key])
        want = exp[key]
        ok = abs(got - want) <= tol
        n_fail += not ok
        rows.append(("exact", key, f"{want:.4g}", f"{got:.4g}",
                     f"±{tol:g}", ok))
    for key in signs:
        got, want = values[key], signs[key]
        ok = (got > 0) == (want > 0)
        n_fail += not ok
        rows.append(("sign", key, "+" if want > 0 else "−",
                     "+" if got > 0 else "−", "", ok))
    structural = [
        ("pr_median_real > pr_median_null + 1.0",
         values["pr_median_real"] > values["pr_median_null"] + 1.0),
        ("mle_worst_calib_error > 0.20  (MLE stays untrusted)",
         values["mle_worst_calib_error"] > 0.20),
    ]
    for desc, ok in structural:
        n_fail += not ok
        rows.append(("struct", desc, "", "", "", ok))

    w = max(len(r[1]) for r in rows)
    print(f"{'kind':7s} {'invariant':{w}s} {'pinned':>10s} {'now':>10s} "
          f"{'tol':>8s} verdict")
    for kind, name, want, got, tol, ok in rows:
        print(f"{kind:7s} {name:{w}s} {want:>10s} {got:>10s} "
              f"{tol:>8s} {'PASS' if ok else '** FAIL **'}")
    for key in NOT_ARTIFACT_BACKED:
        if key in exp:
            print(f"{'skip':7s} {key:{w}s} {exp[key]:>10.4g} "
                  f"{'—':>10s} {'':>8s} SKIP (needs raw clouds)")

    print(f"\npinned {pins['pinned_at']} at {pins['git_commit'][:9]}; "
          f"full legacy rerun: feature/invariant-checks branch")
    if n_fail:
        print(f"\n{n_fail} FAIL. Baseline artifacts or pins changed — "
              f"neither should on this branch. Investigate the diff.")
    else:
        print("\nAll artifact-backed legacy invariants hold.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
