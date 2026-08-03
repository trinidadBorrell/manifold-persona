"""Execute plan 2026-07-22-role-count-sweep.

Re-runs plan #1's manifold pipeline at eight role-set sizes to ask whether a
coarser set of role centroids makes the manifold *stronger*, or merely easier to
draw. Everything downstream of the saved role point cloud — no model, no
re-extraction, no generation.

Order of operations is the plan's, and it matters:
  0a  regression check: the n=276 cell must reproduce plan #1's decider, else STOP
  0b  per-n positive control (synthetic curved k=3 at that cell's own SNR)
  1   decider M1: relative NRE reduction vs a null computed AT THAT n
  2,3 secondary M2 (intrinsic dim vs a matched small-N reference), M3 (curvature)
  4   illustrative spline figures at n=10 and n=25

Usage:
    .venv/bin/python -m manifold.sweep                 # full sweep
    .venv/bin/python -m manifold.sweep --smoke         # n=10,25 x 1 seed, 20 perms
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline as P
from . import sweep_plots as SP
from . import idim
from manifold_persona.runlog import (holm, make_say, new_run_dir, provenance,
                                     timestamp, write_manifest)
from .subsets import cloud_scale, kmeans_medoid_roles, subset_cloud

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = "plans/2026-07-22-role-count-sweep.md"

N_LIST = [10, 25, 50, 75, 100, 150, 200, 276]
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 100                 # decider null, seeds 0-99
N_PERM_POSCTRL = 50
N_REF_DRAWS = 100            # M2 Gaussian reference draws per n
K = P.K_INTRINSIC            # 3, preregistered
FLOOR = 0.30                 # plan #1's effect floor, reused
BAND = 0.10                  # plan's decision band on rel. reduction
ALPHA = 0.05
SPLINE_FIG_NS = [10, 25]     # illustrative figures only

# plan #1 decider, for the 0a regression check
PRIOR = {"r2": 0.655, "null_median": 0.372, "rel_reduction": 0.451}
PRIOR_TOL = 0.01


def run_cell(cloud, n: int, seed: int, n_perm: int, say) -> dict:
    """One (n, seed) sweep cell. Returns a flat metrics row."""
    t0 = time.time()
    sel = kmeans_medoid_roles(cloud, n, seed=seed)
    sub = subset_cloud(cloud, sel["roles"])
    noise, spread = cloud_scale(sub)

    # --- 0b positive control, calibrated to THIS cell -----------------------
    pc = P.positive_control(n_roles=n, per_role=25, k=K, target_radius=spread,
                            noise=noise, seed=0)
    pc_pass = bool(pc["stats"]["p"] < ALPHA and pc["stats"]["rel_reduction"] >= FLOOR)

    # --- 1 decider ----------------------------------------------------------
    dec = P.construction_C_role(sub, k=K)
    null = P.permutation_null(sub, n_perms=n_perm, k=K, seed=0)
    st = P.separation_stats(dec.r2, null)

    # --- 3 curvature gain (plane R² is label-free, so it is shared) ---------
    plane = P.pca_plane_r2(sub, k=K)

    # --- 2 intrinsic dimension of the n role means --------------------------
    ids = idim.id_estimates(sub.role_means)

    row = {
        "n": n, "seed": seed, "n_roles": len(sel["roles"]),
        "n_points": int(sub.raw.shape[0]),
        "default_forced": sel["default_forced"], "n_filled": sel["n_filled"],
        "r2": dec.r2, "nre": dec.nre,
        "null_median": st["null_median"], "null_5pct": st["null_5pct"],
        "p": st["p"], "z": st["z"], "r2_gap": st["r2_gap"],
        "rel_reduction": st["rel_reduction"],
        "plane_r2": plane,
        "curv_gain": dec.r2 - plane,
        "curv_gain_null": st["null_median"] - plane,
        "id_TwoNN": ids["TwoNN"], "id_MLE": ids["MLE"], "id_lPCA": ids["lPCA"],
        "pc_r2": pc["r2"], "pc_null_median": pc["stats"]["null_median"],
        "pc_p": pc["stats"]["p"], "pc_rel_reduction": pc["stats"]["rel_reduction"],
        "pc_pass": pc_pass,
        "role_spread": spread, "within_role_noise": noise,
        "seconds": round(time.time() - t0, 1),
    }
    say(f"    n={n:>3} seed={seed}: R²={dec.r2:.3f} null={st['null_median']:.3f} "
        f"p={st['p']:.3g} relred={st['rel_reduction']:.3f} | plane={plane:.3f} "
        f"gain={row['curv_gain']:.3f} | ID 2NN={ids['TwoNN']} | "
        f"posctrl {'PASS' if pc_pass else 'FAIL'} ({pc['stats']['rel_reduction']:.2f}) "
        f"| {row['seconds']}s")
    return row, sel, sub, null, dec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for wiring checks; NOT a result")
    args = ap.parse_args(argv)

    n_list = [10, 25] if args.smoke else N_LIST
    seeds = [0] if args.smoke else SEEDS
    n_perm = 20 if args.smoke else N_PERM
    n_ref = 10 if args.smoke else N_REF_DRAWS

    t0 = time.time()
    stamp = timestamp()
    run_dir = new_run_dir(REPO_ROOT / "output" / "manifold_h1-2",
                          stamp + ("-smoke" if args.smoke else ""))
    say, _close_log = make_say(run_dir / "logs" / "run.log")

    say(f"run_dir={run_dir}")
    say(f"plan={PLAN}  n_list={n_list}  seeds={seeds}  n_perm={n_perm}  "
        f"D={P.D_AMBIENT}  k={K}  floor={FLOOR}  band={BAND}")

    # ---------------------------------------------------- shared ambient space
    say("\n[0] Loading role cloud (prompt_avg); PCA fit ONCE ...")
    cloud = P.load_cloud(view="prompt_avg", seed=0)
    say(f"    model={cloud.manifest.get('model_name', '?')}  layer={cloud.layer}  "
        f"hidden={cloud.manifest.get('hidden', '?')} -> PCA {P.D_AMBIENT}")
    say(f"    raw={cloud.raw.shape}  roles={len(cloud.role_names)}  "
        f"default_present={'default' in cloud.role_names}")

    # kmeans_medoid_roles returns ALL roles whenever n >= n_available, so any
    # n_list entry above the role count silently becomes the same degenerate
    # cell under a wrong label. Clamp and de-duplicate instead.
    R = len(cloud.role_names)
    clamped = sorted({min(n, R) for n in n_list})
    if clamped != sorted(n_list):
        say(f"    n_list clamped to {R} available roles: {sorted(n_list)} -> {clamped}")
        n_list = clamped

    status = "executed"
    rows, cells = [], {}

    # ------------------------------------------------- 0a regression check
    say("\n[0a] REGRESSION CHECK — n=276 cell must reproduce plan #1's decider ...")
    reg_row, reg_sel, reg_sub, reg_null, reg_dec = run_cell(cloud, 276, 0, n_perm, say)
    reg = {k: {"got": reg_row[k], "expected": v,
               "delta": round(reg_row[k] - v, 4),
               "ok": bool(abs(reg_row[k] - v) <= PRIOR_TOL)} for k, v in PRIOR.items()}
    reg_ok = all(v["ok"] for v in reg.values()) if not args.smoke else True
    for k, v in reg.items():
        say(f"    {k}: got {v['got']:.4f} vs plan#1 {v['expected']:.4f} "
            f"(delta {v['delta']:+.4f}) -> {'OK' if v['ok'] else 'MISMATCH'}")
    if not reg_ok:
        say("!! REGRESSION CHECK FAILED — shared code no longer reproduces plan #1. "
            "Stopping; nothing downstream is interpretable.")
        write_manifest(run_dir, {"status": "stopped-regression-failed", "regression": reg})
        (run_dir / "REPORT.md").write_text(
            "# STOPPED — regression check failed\n\n"
            "The n=276 cell did not reproduce plan #1's decider "
            f"(`output/manifold_h1-2/2026-07-21T14-03/`).\n\n```\n{json.dumps(reg, indent=2)}\n```\n")
        return
    if not args.smoke:
        rows.append(reg_row)
        cells[(276, 0)] = (reg_sel, reg_sub, reg_null, reg_dec)

    # --------------------------------------------------------- the sweep
    say("\n[1-3] SWEEP ...")
    for n in n_list:
        say(f"  --- n = {n} ---")
        cell_seeds = [0] if n >= len(cloud.role_names) else seeds
        for s in cell_seeds:
            if (n, s) in cells:
                continue
            row, sel, sub, null, dec = run_cell(cloud, n, s, n_perm, say)
            rows.append(row)
            if s == 0:
                cells[(n, s)] = (sel, sub, null, dec)

    df = pd.DataFrame(rows).sort_values(["n", "seed"]).reset_index(drop=True)
    df.to_csv(run_dir / "data" / "sweep_metrics.csv", index=False)

    # ------------------------------------------- M2 small-N reference curves
    say("\n[2] M2 Gaussian small-N reference curves "
        f"({n_ref} draws per n, matched to role-mean covariance) ...")
    refs = {}
    for n in sorted(df["n"].unique()):
        refs[int(n)] = idim.gaussian_reference(cloud.role_means, int(n),
                                               n_ref=n_ref, seed=0)
        say(f"    n={n:>3} ref TwoNN median={refs[int(n)]['TwoNN']['median']}")
    json.dump(refs, open(run_dir / "data" / "id_reference.json", "w"), indent=2)

    # ------------------------------------------------------- per-n artefacts
    say("\n[4] Per-n directories, figures, reports ...")
    from .sweep_report import build_cell_report, build_sweep_report

    for n in sorted(df["n"].unique()):
        n = int(n)
        d = run_dir / f"n-personas-{n}"
        (d / "figures").mkdir(parents=True)
        (d / "data").mkdir(parents=True)
        sub_df = df[df["n"] == n]
        sub_df.to_csv(d / "data" / "metrics.csv", index=False)

        sel, sub, null, dec = cells[(n, 0)]
        json.dump({"n": n, "seed": 0, "roles": sel["roles"],
                   "default_forced": bool(sel["default_forced"]),
                   "n_filled": int(sel["n_filled"])},
                  open(d / "data" / "roles.json", "w"), indent=2)

        mani = P.fit_manifold(sub.role_means, k=K)
        np.savez(d / "data" / "manifold_C_role_seed0.npz",
                 role_names=np.array(sub.role_names), role_means=sub.role_means,
                 control_points=mani.control_points, w=mani._w, poly=mani._poly,
                 cp_min=mani._cp_min, cp_range=mani._cp_range,
                 null_r2=null, pca_components=cloud.pca.components_,
                 pca_mean=cloud.pca.mean_)

        SP.fig06_null_vs_real(dec.r2, null, d / "figures", n)
        SP.fig07_roles_pc123(sub, d / "figures", n)
        if n in SPLINE_FIG_NS:
            SP.fig05_spline_manifold(sub, mani, d / "figures", n, 0)
            SP.fig05_spline_manifold(sub, mani, run_dir / "figures", n, 0)
        (d / "REPORT.md").write_text(
            build_cell_report(n, sub_df, sel, refs.get(n, {}), stamp, FLOOR))

    # ------------------------------------------------------ top-level figures
    SP.fig01_sweep_decider(df, run_dir / "figures", floor=FLOOR)
    SP.fig02_sweep_id(df, refs, run_dir / "figures")
    SP.fig03_sweep_curvature(df, run_dir / "figures")
    SP.fig04_sweep_posctrl(df, run_dir / "figures", floor=FLOOR)

    # ------------------------------------------------------- decision rule
    rr = df.groupby("n")["rel_reduction"].agg(["mean", "min", "max"])
    if 276 not in rr.index:          # --smoke only; the real run always has it
        say("\n[5] DECISION RULE skipped (no n=276 cell — smoke run, not a result).")
        json.dump({"status": "smoke"}, open(run_dir / "data" / "verdict.json", "w"))
        say(f"\nSMOKE DONE in {time.time()-t0:.0f}s -> {run_dir}")
        return
    rr276 = float(rr.loc[276, "mean"])
    def _cell(n):
        return {"mean": float(rr.loc[n, "mean"]), "min": float(rr.loc[n, "min"]),
                "max": float(rr.loc[n, "max"])}
    small = {n: _cell(n) for n in (10, 25) if n in rr.index}
    better = all(
        small[n]["mean"] >= rr276 + BAND and small[n]["min"] > float(rr.loc[276, "max"])
        and float(df[(df["n"] == n)]["p"].max()) < ALPHA
        and bool(df[(df["n"] == n)]["pc_pass"].all())
        for n in small) if small else False
    worse = (10 in rr.index) and (_cell(10)["mean"] <= rr276 - BAND)
    decision = "small-n BETTER (hypothesis falsified)" if better else (
        "small-n WORSE (hypothesis supported)" if worse else
        "FLAT (hypothesis supported: legibility, not evidence)")
    say(f"\n[5] DECISION RULE: RR(276)={rr276:.3f}  "
        + "  ".join(f"RR({n})={small[n]['mean']:.3f}" for n in small)
        + f"  -> {decision}")

    pmed = df.groupby("n")["p"].median().to_dict()
    holm_adj = holm({f"n={int(k)}": float(v) for k, v in pmed.items()})

    verdicts = {"decision": decision, "rr_276": rr276, "band": BAND,
                "rr_by_n": rr.to_dict(orient="index"), "small": small,
                "holm_p_by_n": holm_adj,
                "posctrl_all_pass": bool(df["pc_pass"].all()),
                "posctrl_failed_cells": df[~df["pc_pass"]][["n", "seed"]].to_dict("records")}
    json.dump(verdicts, open(run_dir / "data" / "verdict.json", "w"), indent=2, default=float)

    # ------------------------------------------------------------- report
    (run_dir / "REPORT.md").write_text(
        build_sweep_report(df, refs, verdicts, reg, stamp, run_dir, FLOOR, BAND,
                           n_perm, seeds, n_ref))

    manifest = {
        "run_id": stamp, "plan": PLAN, "context": "RESEARCH.md", "status": status,
        "git": "not-a-git-repo (reproducibility via manifest+seeds)",
        # Read from the cloud's own manifest -- these used to be hardcoded to the
        # 3B/layer-26 run and silently misreported any other cloud.
        "inputs": f"data/embeddings_roles/ (prompt_avg, layer {cloud.layer}; no re-extraction)",
        "model": cloud.manifest.get("model_name"), "view": "prompt_avg",
        "layer": cloud.layer,
        "source_manifest": cloud.manifest,
        "D_ambient": P.D_AMBIENT, "k_intrinsic": K, "n_list": n_list,
        "kmeans_seeds": seeds, "n_perm_decider": n_perm,
        "n_perm_posctrl": N_PERM_POSCTRL, "n_ref_draws": n_ref,
        "effect_floor_rel_reduction": FLOOR, "decision_band": BAND, "alpha": ALPHA,
        "selection": "kmeans medoid on 276 role means in shared D=50 PCA space; "
                     "default force-included into its own cluster's slot",
        "regression_check_vs_plan1": reg,
        "decision": decision,
        "n_roles_total": len(cloud.role_names), "n_points_total": int(cloud.raw.shape[0]),
        "exclusions": "none (default kept everywhere)",
        **provenance(t0, pandas=True),
        "reproduce": ".venv/bin/python -m manifold.sweep",
        "cells": len(df),
    }
    write_manifest(run_dir, manifest, default=str)
    say(f"\nDONE in {time.time()-t0:.0f}s -> {run_dir}")
    _close_log()


if __name__ == "__main__":
    main()
