"""CALIBRATION — do the estimators recover a dimension we already know?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 0)

This is the run's one hard gate. It runs FIRST and exits non-zero on failure, so
no downstream script can quote a dimension from an estimator that cannot measure
one at this sample size.

WHY IT IS NEEDED
----------------
`id_vs_axis.py` reports MLE = 4.59 for `default`. Nothing established that MLE
*recovers dimension* at N=200 in 2048 ambient dims, and `compute_budget.py`
already showed the bias is large at small N (a true d=15 reads 8.1 at N=25). So
we plant manifolds of KNOWN dimension, calibrated to this cloud's own radius and
noise, and measure what the estimators return.

The radius sweep is the load-bearing part. If recovered dimension depends on
cloud RADIUS at fixed true dimension, then the ID-vs-axis correlation could be
the bias curve reading out cloud size — and the linear partial on log
within-role variance cannot remove a nonlinear bias. Three radii spanning the
roles' observed range is what makes that visible.

Planted manifolds are deliberately CURVED (a random Fourier embedding, not a
linear subspace): a linear subspace lets lPCA succeed trivially and never
stresses the neighbour-based estimators, which are the ones carrying the result.

Topology gets controls in BOTH directions — a noisy circle must give Betti-1 = 1
and a Gaussian blob must give Betti-1 = 0 under the same threshold. Without the
second, the lifetime rule manufactures loops and every role looks interesting.

Produces `calibration_L<L>.json` -> fig00, fig01.

Usage:
    .venv/bin/python exploratory/per_persona/calib_estimators.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from manifold.idim import ESTIMATORS

from common import load_role_clouds, resolve_run_dir, small_matrix_ops
from metrics import ID_COLS, SEED, panel_metrics

# CALIBRATED = recovers a planted dimension within 20% for d<=10 in the PCA-50
# working space. THREE estimators clear that bar, not one:
#
#   PCA_dim_90pct            worst 20%, bias +0.11, spread 1.00  (exact at d=3, 8)
#   MLE                      worst 17%, bias -0.34, spread 0.31  (smoothest, but
#                                                                 under-reads)
#   PCA_participation_ratio  worst 18%, bias +0.22, spread 1.40
#
# CORRECTION (2026-07-31): this list previously held MLE alone. That was chosen
# from the RAW 2048-dim diagnostic, where MLE really was the only survivor, and
# was not revisited after amendment A2 moved the panel to PCA-50. Denoising
# rescues the two PCA-threshold measures; dim_90pct is in fact the most accurate
# estimator in the panel. No run was mis-gated (MLE passed, so the gate never
# fired wrongly), but the "uncalibrated" label on those two was wrong.
#
# The three failures are genuine and unchanged: lPCA 67%, TwoNN 77%,
# PCA_dim_95pct 120%. They stay in the panel at the user's request but must
# never be quoted as a dimension. See md/CALIBRATION.md, md/CALIBRATION-RESULTS.md.
GATED_ESTIMATORS = ("MLE", "PCA_participation_ratio", "PCA_dim_90pct")
UNCALIBRATED = ("TwoNN", "lPCA", "PCA_dim_95pct")

PLANTED_DIMS = (3, 5, 8, 12)
TOLERANCE = 0.20     # relative error a gated estimator may not exceed
MAX_D_GATED = 10     # ...and the planted dimension up to which the gate applies


# --------------------------------------------------------------------------- #
# planted clouds                                                               #
# --------------------------------------------------------------------------- #
def plant_manifold(d: int, n: int, ambient: int, radius: float, noise: float,
                   seed: int = SEED, n_freq: int = 64,
                   freq_scale: float = 0.5) -> np.ndarray:
    """`n` points on a smooth CURVED d-manifold embedded in `ambient` dims.

    phi(u) = sum_k a_k * sin(omega_k . u + b_k) with random frequencies. Curved
    on purpose: a linear subspace would let lPCA read the answer straight off the
    rank and would never test whether the neighbour-based estimators — the ones
    carrying the ID-vs-axis result — can follow a bent manifold at this sample
    size.

    The cloud is rescaled so its RMS radius equals `radius`, then isotropic
    Gaussian noise of per-dimension scale `noise` is added, so the planted cloud
    sits at the real data's own signal-to-noise rather than at an arbitrary one.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u *= rng.uniform(0, 1, (n, 1)) ** (1.0 / d)          # uniform in the d-ball
    # freq_scale controls how bent the embedding is. It must stay SMALL: at
    # scale 2.0 the sin() features fold the manifold so tightly that its local
    # structure fills more than d dimensions, and the estimators then correctly
    # report a dimension above d — which would make the control fail for a
    # reason that is about the control, not the estimators. 0.5 keeps phi smooth
    # over the unit ball while still being genuinely curved (not a linear
    # subspace, which would let lPCA read d straight off the rank).
    omega = rng.standard_normal((n_freq, d)) * freq_scale
    b = rng.uniform(0, 2 * np.pi, n_freq)
    A = rng.standard_normal((n_freq, ambient)) / np.sqrt(n_freq)
    Xm = np.sin(u @ omega.T + b) @ A
    Xm -= Xm.mean(0)
    cur = np.sqrt((Xm ** 2).sum() / n)
    Xm *= radius / max(cur, 1e-12)
    return Xm + rng.standard_normal((n, ambient)) * noise


def plant_circle(n: int, ambient: int, radius: float, noise: float,
                 seed: int = SEED) -> np.ndarray:
    """A noisy 1-manifold WITH one hole, randomly rotated into `ambient` dims.
    The positive control for Betti-1: this must come back as exactly one loop."""
    rng = np.random.default_rng(seed + 1)
    t = rng.uniform(0, 2 * np.pi, n)
    circ = np.c_[np.cos(t), np.sin(t)]
    Q = np.linalg.qr(rng.standard_normal((ambient, 2)))[0]
    X = circ @ Q.T
    X *= radius / max(np.sqrt((X ** 2).sum() / n), 1e-12)
    return X + rng.standard_normal((n, ambient)) * noise


def plant_blob(n: int, ambient: int, radius: float, seed: int = SEED) -> np.ndarray:
    """An isotropic Gaussian blob — NO hole. The negative control for Betti-1:
    if the lifetime threshold reports a loop here, the threshold invents loops
    and every Betti-1 in the panel is meaningless."""
    rng = np.random.default_rng(seed + 2)
    X = rng.standard_normal((n, ambient))
    return X * radius / max(np.sqrt((X ** 2).sum() / n), 1e-12)


def positive_control(n_per: int, ambient: int, radii: list, noise: float,
                     dims=PLANTED_DIMS) -> dict:
    """Calibration curve (recovered vs planted, at 3 radii) + both topology controls."""
    rows = []
    for r_i, radius in enumerate(radii):
        for d in dims:
            X = plant_manifold(d, n_per, ambient, radius, noise, seed=SEED + 7 * r_i)
            with small_matrix_ops():
                m, _, _ = panel_metrics(X)
            rows.append({"planted_d": d, "radius": radius, "radius_rank": r_i,
                         **{k: m.get(k) for k in ID_COLS}})
            print(f"    planted d={d:2d} r={radius:8.2f} -> "
                  + "  ".join(f"{k}={m.get(k):.2f}" for k in ESTIMATORS
                              if m.get(k) is not None))
    mid = radii[len(radii) // 2]
    with small_matrix_ops():
        m_circ, dg_circ, _ = panel_metrics(
            plant_circle(n_per, ambient, mid, noise), keep_diagrams=True)
        m_blob, dg_blob, _ = panel_metrics(
            plant_blob(n_per, ambient, mid), keep_diagrams=True)
    return {"calibration": rows,
            "circle": {k: m_circ.get(k) for k in
                       ("betti0", "betti1", "H1_max_lifetime", "cloud_diameter")},
            "blob": {k: m_blob.get(k) for k in
                     ("betti0", "betti1", "H1_max_lifetime", "cloud_diameter")},
            "circle_dgms": [d.tolist() for d in dg_circ],
            "blob_dgms": [d.tolist() for d in dg_blob]}


def control_verdict(pc: dict, tol: float = TOLERANCE,
                    max_d: int = MAX_D_GATED) -> dict:
    """Pass/fail: the GATED estimators within `tol` of truth for d<=max_d, and
    both topology controls correct. This is the one hard gate in the run.

    Amended 2026-07-30 (plan amendment A1): the original plan gated on TwoNN as
    well, and TwoNN fails in EVERY working space tested — 77% error at a planted
    d=3 even after PCA-50 denoising, with a scale spread of 2.4-3.3 across radii.
    It is not rescuable by preprocessing, so gating on it would make the run
    impossible rather than making it rigorous. TwoNN and the other uncalibrated
    estimators are still computed and reported; the calibration error of each is
    recorded here so the report can carry it.
    """
    cal = pd.DataFrame(pc["calibration"])
    fails = []
    for est in GATED_ESTIMATORS:
        sub = cal[cal.planted_d <= max_d]
        rel = (sub[est] - sub.planted_d).abs() / sub.planted_d
        worst = float(rel.max())
        if worst > tol:
            bad = sub.loc[rel.idxmax()]
            fails.append(f"{est}: worst relative error {worst:.2f} > {tol} "
                         f"(planted d={int(bad.planted_d)}, radius={bad.radius:.1f}, "
                         f"recovered {bad[est]:.2f})")
    if pc["circle"]["betti1"] != 1:
        fails.append(f"circle control: betti1={pc['circle']['betti1']}, expected 1")
    if pc["blob"]["betti1"] != 0:
        fails.append(f"blob control: betti1={pc['blob']['betti1']}, expected 0")

    # Is the bias scale-dependent? This does not gate the run, but it decides
    # whether a LINEAR partial on log_var is an adequate control.
    spread, worst_err = {}, {}
    for est in ID_COLS:
        if est not in cal:
            continue
        g = cal.groupby("planted_d")[est]
        spread[est] = float((g.max() - g.min()).max())
        sub = cal[cal.planted_d <= max_d]
        worst_err[est] = float(((sub[est] - sub.planted_d).abs() / sub.planted_d).max())
    return {"pass": len(fails) == 0, "failures": fails,
            "gated_on": list(GATED_ESTIMATORS),
            "uncalibrated_reported_anyway": list(UNCALIBRATED),
            "max_recovered_spread_across_radii": spread,
            "worst_relative_error_d_le_10": worst_err}


def real_cloud_scale(clouds) -> tuple:
    """(per-role RMS radii, per-role noise floors) of the real role clouds.

    The RADIUS is the role cloud's RMS norm about its own mean. The NOISE is the
    isotropic floor estimated from the TAIL of the role's eigenvalue spectrum,
    NOT the per-element RMS of the whole cloud: the latter includes the signal
    and overstates the noise ~4x, which would make every planted manifold
    noisier than the data it is supposed to imitate. For n points in p >> n
    dims, an isotropic noise floor of per-element variance s^2 puts each tail
    eigenvalue at ~ p * s^2, hence the /ambient.
    """
    radii, noises = [], []
    for Xr in clouds.values():
        Xc = Xr - Xr.mean(0)
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
        radii.append(np.sqrt((Xc ** 2).sum() / len(Xr)))
        tail = lam[100:] if len(lam) > 110 else lam[-10:]
        noises.append(np.sqrt(tail.mean() / Xc.shape[1]))
    return np.array(radii), np.array(noises)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--label-layer", type=int, default=19,
                    help="layer number used in OUTPUT filenames. The resp40 "
                         "manifest stores primary_layer=0 because it holds a "
                         "single extracted layer; the real depth is 19.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-radii", type=int, default=3)
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.outdir)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    L = args.label_layer
    t0 = time.time()

    roles, clouds, factors, _ = load_role_clouds(args.view, args.layer)
    n_per = len(next(iter(clouds.values())))
    ambient = clouds[roles[0]].shape[1]
    print(f"view={args.view} label_layer={L} roles={len(roles)} "
          f"points/role={n_per} ambient={ambient}")

    print("\nPOSITIVE CONTROL — planted manifolds + calibration curve ...")
    with small_matrix_ops():
        radii_all, noise_all = real_cloud_scale(clouds)
    radii = [float(np.percentile(radii_all, p))
             for p in np.linspace(10, 90, args.n_radii)]
    noise = float(np.median(noise_all))
    print(f"    calibrated to real data: radii {['%.1f' % r for r in radii]}, "
          f"per-dim noise {noise:.4f}")

    pc = positive_control(n_per, ambient, radii, noise)
    verdict = control_verdict(pc)
    pc["verdict"] = verdict
    pc["exploratory"] = True
    pc["calibrated_to"] = {"radii": radii, "per_dim_noise": noise, "n_per": n_per}
    json.dump(pc, open(run_dir / "data" / f"calibration_L{L}.json", "w"),
              indent=2, default=float)

    print(f"\n    circle control: betti1={pc['circle']['betti1']} (expect 1)")
    print(f"    blob   control: betti1={pc['blob']['betti1']} (expect 0)")
    print("    recovered spread across radii: "
          + ", ".join(f"{k}={v:.2f}" for k, v in
                      verdict["max_recovered_spread_across_radii"].items()))
    if not verdict["pass"]:
        print("\n*** POSITIVE CONTROL FAILED — STOPPING PER THE PLAN ***")
        for f in verdict["failures"]:
            print("   ", f)
        json.dump({"status": "STOPPED_CONTROL_FAILED", "failures": verdict["failures"]},
                  open(run_dir / "data" / f"STOP_L{L}.json", "w"), indent=2)
        sys.exit(2)
    print(f"    positive control PASSED  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
