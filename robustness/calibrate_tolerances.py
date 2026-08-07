"""Derive pin tolerances from measured spread instead of judgment.

Two perturbations, matched to what can actually move each number:

  role bootstrap   every pinned statistic is computed over ~276 roles.
                   Resampling roles (2000 draws) measures how much the
                   statistic wobbles under sampling. A drift smaller
                   than that wobble is not a meaningful change.
  seed sweep       the calibration gate plants SEEDED synthetic
                   manifolds. Different seeds -> different planted
                   clouds -> spread in the recovered error. Run
                   separately via calib_estimators.py --seed; this
                   script collects the results.

Tolerance rule per pin:  tol = max(2 * SD, floor), rounded up.
The floor keeps a tolerance from collapsing to ~0 for statistics that
barely wobble (a deterministic pipeline would otherwise pin float
noise). Results are written into invariants.json under "tol", with
the evidence under "tol_provenance".

Usage:
  .venv/bin/python robustness/calibrate_tolerances.py \
      --calib-glob '<scratch>/calib_seed_*/data/calibration_L19.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PP = REPO / "exploratory" / "per_persona"
BASELINE = Path(__file__).resolve().parent / "baseline"
PINS_FILE = Path(__file__).resolve().parent / "invariants.json"
sys.path.insert(0, str(PP))

from common import load_role_clouds, design_fractions, small_matrix_ops  # noqa: E402
from stats_utils import partial_corr_multi  # noqa: E402

L = 19
N_BOOT = 2000
TERMS = ("instr_frac", "quest_frac", "interaction_frac")
LADDER_METRICS = ["MLE", "TwoNN", "PCA_participation_ratio",
                  "PCA_dim_90pct", "H1_total_persistence"]
CTRL_ALL = ["log_var", "mean_norm"]

# Floors: below this, a tolerance would only be pinning float noise.
FLOORS = {"frac": 0.005, "fold": 1.0, "median": 0.3, "r": 0.02,
          "calib": 0.02}


def per_role_fractions(role_dir: str, allow_unclean: bool) -> pd.DataFrame:
    prev = dict(os.environ)
    os.environ["MP_ROLE_DIR"] = role_dir
    if allow_unclean:
        os.environ["MP_ALLOW_UNCLEAN"] = "1"
    try:
        roles, clouds, factors, _ = load_role_clouds("prompt_avg", None)
        with small_matrix_ops():
            df = pd.DataFrame([{"role": r, **design_fractions(clouds[r],
                                                              *factors[r])}
                               for r in roles])
    finally:
        os.environ.clear()
        os.environ.update(prev)
    return df.set_index("role")


def round_up_2sig(x: float) -> float:
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    return math.ceil(x / 10 ** (e - 1)) * 10 ** (e - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-glob", default="",
                    help="glob of calibration_L19.json files from the "
                         "--seed sweep")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sd: dict[str, float] = {}
    floor_of: dict[str, str] = {}

    # ---- per-role fractions, both clouds --------------------------------
    print("loading per-role design fractions (both clouds) ...")
    fr_prompt = per_role_fractions("data/embeddings_roles", allow_unclean=True)
    fr_resp = per_role_fractions("data/embeddings_roles_resp_40q",
                                 allow_unclean=False)
    shared = fr_prompt.index.intersection(fr_resp.index)
    print(f"  prompt {len(fr_prompt)} roles, response {len(fr_resp)}, "
          f"shared {len(shared)}")

    # ---- panel + ladder inputs ------------------------------------------
    panel = pd.read_csv(BASELINE / f"per_role_panel_L{L}.csv")
    d = panel[panel["role"] != "default"].dropna(
        subset=["axis_proj"] + CTRL_ALL).reset_index(drop=True)
    x = d["axis_proj"].to_numpy(float)
    Z = d[CTRL_ALL].to_numpy(float)
    Ys = {m: d[m].to_numpy(float) for m in LADDER_METRICS}
    pr_all = panel["PCA_participation_ratio"].to_numpy(float)

    # ---- role bootstrap --------------------------------------------------
    print(f"role bootstrap, {args.n_boot} draws ...")
    boot: dict[str, list[float]] = {k: [] for k in
                                    [f"prompt.{t}" for t in TERMS]
                                    + [f"response.{t}" for t in TERMS]
                                    + ["fold_change", "pr_median_real"]
                                    + [f"r.{m}" for m in LADDER_METRICS]}
    p_shared = fr_prompt.loc[shared, "interaction_frac"].to_numpy(float)
    r_shared = fr_resp.loc[shared, "interaction_frac"].to_numpy(float)
    for _ in range(args.n_boot):
        ip = rng.integers(0, len(fr_prompt), len(fr_prompt))
        ir = rng.integers(0, len(fr_resp), len(fr_resp))
        for t in TERMS:
            boot[f"prompt.{t}"].append(
                float(np.median(fr_prompt[t].to_numpy(float)[ip])))
            boot[f"response.{t}"].append(
                float(np.median(fr_resp[t].to_numpy(float)[ir])))
        js = rng.integers(0, len(shared), len(shared))
        a, b = np.median(p_shared[js]), np.median(r_shared[js])
        if a > 0:
            boot["fold_change"].append(float(b / a))
        ik = rng.integers(0, len(pr_all), len(pr_all))
        boot["pr_median_real"].append(float(np.median(pr_all[ik])))
        il = rng.integers(0, len(x), len(x))
        for m in LADDER_METRICS:
            r_b = partial_corr_multi(x[il], Ys[m][il], Z[il])[0]
            if np.isfinite(r_b):
                boot[f"r.{m}"].append(float(r_b))

    for k, v in boot.items():
        sd[k] = float(np.std(v))
        floor_of[k] = ("frac" if k.endswith("_frac") else
                       "fold" if k == "fold_change" else
                       "median" if k == "pr_median_real" else "r")

    # null median: only the summary is stored; no per-draw values to
    # resample. Keep the real-median tolerance for it, honestly labeled.
    sd["pr_median_null"] = sd["pr_median_real"]
    floor_of["pr_median_null"] = "median"

    # ---- calibration seed sweep -----------------------------------------
    calib_errs = []
    for f in sorted(glob.glob(args.calib_glob)) if args.calib_glob else []:
        c = json.load(open(f))
        calib_errs.append(
            float(c["verdict"]["worst_relative_error_d_le_10"]["MLE"]))
    if calib_errs:
        sd["mle_worst_calib_error"] = float(np.std(calib_errs))
        floor_of["mle_worst_calib_error"] = "calib"
        print(f"calibration sweep: {len(calib_errs)} seeds, MLE worst error "
              f"min {min(calib_errs):.3f} max {max(calib_errs):.3f} "
              f"sd {sd['mle_worst_calib_error']:.4f}")
    else:
        print("no calibration sweep files given; calib tolerance unchanged")

    # ---- derive + write --------------------------------------------------
    pins = json.loads(PINS_FILE.read_text())
    tol = {k: round_up_2sig(max(2 * s, FLOORS[floor_of[k]]))
           for k, s in sd.items()}
    pins["tol"] = tol
    pins["tol_provenance"] = {
        "date": date.today().isoformat(),
        "method": f"max(2*SD, floor), SD from {args.n_boot}-draw role "
                  f"bootstrap; calibration SD from seed sweep",
        "floors": FLOORS,
        "calib_seeds_used": len(calib_errs),
        "calib_errors": calib_errs,
        "sd": {k: round(v, 6) for k, v in sd.items()},
    }
    PINS_FILE.write_text(json.dumps(pins, indent=2) + "\n")

    w = max(len(k) for k in tol)
    print(f"\n{'pin':{w}s} {'pinned':>10s} {'SD':>9s} {'tol':>8s}")
    for k in tol:
        print(f"{k:{w}s} {pins['values'][k]:10.4g} {sd[k]:9.4f} "
              f"{tol[k]:8.3g}")
    print(f"\nwritten to {PINS_FILE}")


if __name__ == "__main__":
    main()
