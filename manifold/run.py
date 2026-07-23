"""Execute plan 2026-07-21-role-manifold-reconstruction (H1).

Creates a timestamped run dir under output/manifold_h1-2/, runs the positive
control FIRST (stops the run if it fails), then the decider + robustness
constructions with permutation nulls, saves metrics/manifolds/manifest, renders
figures, and writes REPORT.md. Never overwrites a run dir.

Usage:
    .venv/bin/python -m manifold.run
"""
from __future__ import annotations

import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline as P
from . import plots
from .tps import SplineManifold, reconstruction

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = "plans/2026-07-21-role-manifold-reconstruction.md"

N_PERM_DECIDER = 100
N_PERM_ROBUST = 40
N_PERM_POSCTRL = 50
FLOOR = 0.30            # effect-size floor: relative reduction in NRE
ALPHA = 0.05


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a name->p dict."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        prev = max(prev, a)
        adj[name] = prev
    return adj


def verdict(stats: dict) -> str:
    if stats["p"] < ALPHA and stats["rel_reduction"] >= FLOOR:
        return "SUPPORTED"
    if stats["p"] < ALPHA:
        return "weak (significant, effect < floor)"
    return "null"


def main() -> None:
    t0 = time.time()
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = REPO_ROOT / "output" / "manifold_h1-2" / stamp
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)  # fail if exists
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    log = open(run_dir / "logs" / "run.log", "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg); log.write(msg + "\n"); log.flush()

    say(f"run_dir={run_dir}")
    say(f"seed=0  D={P.D_AMBIENT}  k={P.K_INTRINSIC}  floor={FLOOR}")

    rows = []          # metrics.csv rows
    report = {}        # summary for REPORT.md

    # ---------------------------------------------------------- load (inert)
    say("\n[0] Loading role cloud (prompt_avg, layer 26) ...")
    cloud = P.load_cloud(view="prompt_avg", seed=0)
    say(f"    raw={cloud.raw.shape}  roles={len(cloud.role_names)}  "
        f"default_kept={'default' in cloud.role_names}")
    # data-scale calibration for the positive control (audit finding 1):
    ridx = {r: i for i, r in enumerate(cloud.role_names)}
    pt_means = cloud.role_means[[ridx[r] for r in cloud.roles]]
    data_noise = float(np.sqrt(np.mean((cloud.raw - pt_means) ** 2)))          # within-role per-dim RMS
    sig = float(np.sqrt(np.mean(np.sum((cloud.role_means - cloud.global_mean) ** 2, axis=1))))
    say(f"    data-scale: within-role noise/dim={data_noise:.3f}  role-spread={sig:.3f}")

    # -------------------------------------------------------------- controls
    say("\n[1] POSITIVE CONTROL (curved k=3 manifold, calibrated to data; pass or STOP) ...")
    pc = P.positive_control(k=P.K_INTRINSIC, target_radius=sig, noise=data_noise, seed=0)
    pc_pass = pc["stats"]["p"] < ALPHA and pc["stats"]["rel_reduction"] >= FLOOR
    say(f"    pos-ctrl R2={pc['r2']:.3f} null_med={pc['stats']['null_median']:.3f} "
        f"p={pc['stats']['p']:.3g} rel_red={pc['stats']['rel_reduction']:.2f} "
        f"-> {'PASS' if pc_pass else 'FAIL'}")
    report["posctrl"] = {"r2": pc["r2"], **pc["stats"], "pass": bool(pc_pass)}
    rows.append({"construction": "positive_control", "r2": pc["r2"],
                 "null_median": pc["stats"]["null_median"], "p": pc["stats"]["p"],
                 "rel_reduction": pc["stats"]["rel_reduction"],
                 "verdict": "PASS" if pc_pass else "FAIL", "exploratory": True})
    plots.fig_posctrl(pc, run_dir / "figures")

    if not pc_pass:
        say("!! POSITIVE CONTROL FAILED — pipeline cannot detect a manifold that "
            "is really there. Stopping; no role result is interpretable.")
        _write_report(run_dir, report, rows, stamp, stopped=True)
        _write_manifest(run_dir, stamp, t0, {}, status="stopped-posctrl-failed")
        return

    # -------------------------------------------------------------- thresholds
    taus = P.cosine_percentile_thresholds(cloud.role_means)
    say(f"    tau percentiles (cos-sim): {taus}")

    # ---------------------------------------------------------------- decider
    say("\n[2] DECIDER  C_role (prompt_avg, k=3) + 100-perm null ...")
    dec = P.construction_C_role(cloud)
    null = P.permutation_null(cloud, n_perms=N_PERM_DECIDER, seed=0)
    st = P.separation_stats(dec.r2, null)
    dec_verdict = verdict(st)
    say(f"    C_role R2={dec.r2:.3f} null_med={st['null_median']:.3f} p={st['p']:.3g} "
        f"z={st['z']:.1f} rel_red={st['rel_reduction']:.2f} gap={st['r2_gap']:.3f} "
        f"-> {dec_verdict}")
    report["decider"] = {"r2": dec.r2, **st, "verdict": dec_verdict}
    report["null_decider"] = null.tolist()
    rows.append({"construction": "C_role", "n_anchors": dec.n_anchors,
                 "n_manifolds": 1, "r2": dec.r2, "nre": dec.nre,
                 "null_median": st["null_median"], "null_5pct": st["null_5pct"],
                 "p": st["p"], "z": st["z"], "r2_gap": st["r2_gap"],
                 "rel_reduction": st["rel_reduction"], "verdict": dec_verdict,
                 "exploratory": False})

    # ---------------------------------------------------------------- robustness
    say("\n[3] ROBUSTNESS (exploratory) ...")
    robust_nulls = {}
    # C_tau 1/2/3
    tau_results = {}
    for pct, tau in taus.items():
        name = f"C_tau{pct}"
        cr = P.construction_C_tau(cloud, tau, name)
        rnull = P.permutation_null_tau(cloud, tau, n_perms=N_PERM_ROBUST, seed=0)
        rs = P.separation_stats(cr.r2, rnull)
        tau_results[pct] = {"tau": tau, "n_manifolds": cr.n_manifolds,
                            "r2": cr.r2, **rs,
                            "component_sizes": cr.details["component_sizes"]}
        robust_nulls[name] = rnull
        say(f"    {name} tau={tau:.3f} #manifolds={cr.n_manifolds} R2={cr.r2:.3f} "
            f"null_med={rs['null_median']:.3f} p={rs['p']:.3g} rel_red={rs['rel_reduction']:.2f}")
        rows.append({"construction": name, "n_anchors": cr.n_anchors,
                     "n_manifolds": cr.n_manifolds, "r2": cr.r2, "nre": cr.nre,
                     "null_median": rs["null_median"], "p": rs["p"], "z": rs["z"],
                     "r2_gap": rs["r2_gap"], "rel_reduction": rs["rel_reduction"],
                     "verdict": verdict(rs), "exploratory": True,
                     "tau": tau})
    report["tau"] = tau_results

    # C_raw (0a, 5-fold CV) — report R2 only (no permutation null; robustness)
    say("    C_raw (0a, 5-fold CV, anchor cap) ...")
    craw = P.construction_C_raw(cloud, seed=0)
    say(f"    C_raw R2={craw.r2:.3f} (CV)")
    rows.append({"construction": "C_raw", "n_anchors": craw.n_anchors,
                 "n_manifolds": 5, "r2": craw.r2, "nre": craw.nre,
                 "verdict": "cv-only", "exploratory": True})
    report["c_raw_r2"] = craw.r2

    # context: PCA-plane baseline
    plane = P.pca_plane_r2(cloud)
    report["plane_r2"] = plane
    say(f"    PCA-plane baseline R2={plane:.3f}")
    rows.append({"construction": "PCA_plane(k=3)", "r2": plane,
                 "verdict": "baseline", "exploratory": True})

    # context: prompt_last replication of the decider
    say("    prompt_last replication of C_role ...")
    cloud_last = P.load_cloud(view="prompt_last", seed=0)
    dec_last = P.construction_C_role(cloud_last)
    null_last = P.permutation_null(cloud_last, n_perms=N_PERM_ROBUST, seed=0)
    st_last = P.separation_stats(dec_last.r2, null_last)
    say(f"    C_role[prompt_last] R2={dec_last.r2:.3f} null_med={st_last['null_median']:.3f} "
        f"p={st_last['p']:.3g} rel_red={st_last['rel_reduction']:.2f}")
    robust_nulls["C_role_last"] = null_last
    rows.append({"construction": "C_role[prompt_last]", "r2": dec_last.r2,
                 "nre": dec_last.nre, "null_median": st_last["null_median"],
                 "p": st_last["p"], "rel_reduction": st_last["rel_reduction"],
                 "verdict": verdict(st_last), "exploratory": True})
    report["prompt_last"] = {"r2": dec_last.r2, **st_last}

    # context: k=2 robustness
    say("    k=2 robustness of C_role ...")
    dec_k2 = P.construction_C_role(cloud, k=2)
    null_k2 = P.permutation_null(cloud, n_perms=N_PERM_ROBUST, k=2, seed=0)
    st_k2 = P.separation_stats(dec_k2.r2, null_k2)
    say(f"    C_role[k=2] R2={dec_k2.r2:.3f} null_med={st_k2['null_median']:.3f} "
        f"rel_red={st_k2['rel_reduction']:.2f}")
    rows.append({"construction": "C_role[k=2]", "r2": dec_k2.r2, "nre": dec_k2.nre,
                 "null_median": st_k2["null_median"], "p": st_k2["p"],
                 "rel_reduction": st_k2["rel_reduction"], "verdict": verdict(st_k2),
                 "exploratory": True})
    report["k2"] = {"r2": dec_k2.r2, **st_k2}

    # Holm across the EXPLORATORY constructions only. The decider C_role is a
    # single preregistered test and is NOT multiple-comparison-corrected against
    # exploratory robustness runs — its verdict stands on its raw p.
    pmap = {}
    for pct in taus:
        pmap[f"C_tau{pct}"] = tau_results[pct]["p"]
    pmap["C_role[prompt_last]"] = st_last["p"]
    pmap["C_role[k=2]"] = st_k2["p"]
    holm_adj = holm(pmap)
    for r in rows:
        if r["construction"] in holm_adj:
            r["holm_p"] = holm_adj[r["construction"]]
    report["holm"] = holm_adj

    # ---------------------------------------------------------------- save
    say("\n[4] Saving metrics, manifolds, figures ...")
    dfm = pd.DataFrame(rows)
    dfm.to_csv(run_dir / "data" / "metrics.csv", index=False)

    # save the fitted decider manifold + viz coords for plan #2
    mani = P.fit_manifold(cloud.role_means)
    comp_tau2 = P.cosine_components(cloud.role_means, taus[50])
    np.savez(run_dir / "data" / "manifolds_C_role.npz",
             role_names=np.array(cloud.role_names),
             role_means=cloud.role_means, control_points=mani.control_points,
             w=mani._w, poly=mani._poly, cp_min=mani._cp_min,
             cp_range=mani._cp_range, component_tau50=comp_tau2,
             pca_components=cloud.pca.components_, pca_mean=cloud.pca.mean_)

    # figures
    plots.fig01_null_vs_real(report, {"C_role": null, **robust_nulls,
                                      **{f"C_tau{p}": robust_nulls[f"C_tau{p}"]
                                         for p in taus}}, run_dir / "figures")
    plots.fig02_fig03_manifold(cloud, comp_tau2, mani, run_dir / "figures")
    plots.fig04_manifolds_vs_tau(tau_results, run_dir / "figures")
    plots.fig05_spline_vs_plane(report, rows, run_dir / "figures")

    # ---------------------------------------------------------------- report
    _write_report(run_dir, report, rows, stamp, stopped=False)
    _write_manifest(run_dir, stamp, t0, {
        "taus": taus,
        "posctrl_calibration": {"noise_per_dim": data_noise, "role_spread": sig,
                                "manifold_intrinsic_dim": P.K_INTRINSIC},
        "other_params": {"min_component": P.MIN_COMPONENT, "anchor_cap": 600,
                         "reproduce": ".venv/bin/python -m manifold.run"},
    }, status="executed")
    say(f"\nDONE in {time.time()-t0:.0f}s. Verdict (C_role): {dec_verdict}")
    log.close()


def _write_manifest(run_dir, stamp, t0, extra, status):
    manifest = {
        "run_id": stamp, "plan": PLAN, "status": status,
        "git": "not-a-git-repo (reproducibility via manifest+seeds)",
        "seed": 0, "D_ambient": P.D_AMBIENT, "k_intrinsic": P.K_INTRINSIC,
        "n_perm_decider": N_PERM_DECIDER, "n_perm_robust": N_PERM_ROBUST,
        "effect_floor_rel_reduction": FLOOR, "alpha": ALPHA,
        "view": "prompt_avg", "layer": 26, "model": "Qwen/Qwen2.5-3B-Instruct",
        "inputs": "data/embeddings_roles/ (prompt_avg, layer 26; no re-extraction)",
        "n_roles": 276, "n_points": 6900, "exclusions": "none (default role kept)",
        "python": platform.python_version(), "numpy": np.__version__,
        "started": datetime.datetime.fromtimestamp(t0).isoformat(),
        "finished": datetime.datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - t0, 1), **extra,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _write_report(run_dir, report, rows, stamp, stopped):
    from .report import build_report
    (run_dir / "REPORT.md").write_text(build_report(report, rows, stamp, stopped))


if __name__ == "__main__":
    main()
