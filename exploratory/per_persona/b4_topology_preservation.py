"""B4 — persona-cloud TOPOLOGY preservation across OLMo-2-7B training stages.

Question: as OLMo-2-7B goes base -> dpo -> rlvr, does the topology of the
276-role persona cloud stay fixed while the assistant axis rotates?

The cloud here is BETWEEN roles: aggregate each stage to 276 role centroids
(layer 17 = compact depth index 1), then measure the shape of that 276-point
cloud with the repo's own persistence panel recipe (PCA-50 -> ripser), plus the
participation ratio. Reuses topology.topology_metrics, common.pca_stats and
manifold_persona.common.aggregate_by_role / assistant_axis — nothing here
reimplements them.

Three readouts:
  1. per-stage topology signatures (Betti/H0/H1 persistence + entropy + PR)
  2. preservation: cross-stage spread vs a within-stage bootstrap noise band
     (resample the 200 examples per role, re-estimate every centroid), plus a
     diameter-normalised bottleneck/Wasserstein distance between stage diagrams
  3. axis rotation for contrast: cos(default_centroid - mean(centroids)) between
     stages, per depth (layers 9/17/23)

Run:
    .venv/bin/python exploratory/per_persona/b4_topology_preservation.py \
        --outdir output/b4_topology_preservation
"""
from __future__ import annotations

import os
# H0/H1 only. Set BEFORE importing topology (it reads MAXDIM at import time).
# The task asks for Betti/H0/H1; maxdim=1 also keeps the bootstrap cheap.
os.environ.setdefault("MP_RIPSER_MAXDIM", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import persim
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # sibling modules: topology, common

from manifold_persona.common import aggregate_by_role, assistant_axis
from topology import topology_metrics   # per_persona/topology.py (PCA-space PH)
from common import pca_stats            # per_persona/common.py (participation ratio)
from manifold_persona.provenance import write_stamp

STAGES = ["base", "dpo", "rlvr"]
DEPTHS = {0: 9, 1: 17, 2: 23}          # compact depth index -> OLMo layer
L_DEPTH = 1                            # layer 17
D_PCA = 50
SEED = 0
B_BOOT = 20                           # bootstrap resamples per stage
CLOUD = "output/olmo7b_hf_inputs_clean/compact/{}"
NPY = CLOUD + "/prompt_avg_depths.npy"
META = CLOUD + "/metadata.parquet"         # the rows of NPY, written with it

# Scale-free signatures the preservation verdict rests on. Absolute
# H0/H1_total_persistence and cloud_diameter carry the cloud's overall scale,
# so they are reported but NOT used to judge topology (a norm change is not a
# shape change).
SCALE_FREE = ["PCA_participation_ratio_full", "PCA_participation_ratio",
              "persistence_entropy_H0", "H0_total_persistence_norm",
              "H1_total_persistence_norm", "betti0", "betti1"]


def signatures(C: np.ndarray) -> tuple:
    """Topology + participation-ratio signatures of a [n_roles, hidden] cloud.

    Matches metrics.panel_metrics: PCA-50 first (PCA centres; no z-score), then
    ripser + participation ratio in that space. Returns (dict, H0 dgm, H1 dgm).
    """
    C = np.asarray(C, dtype=np.float64)
    d = min(D_PCA, min(C.shape) - 1)
    P = PCA(n_components=d, random_state=SEED).fit_transform(C)
    topo, dgms = topology_metrics(P)
    pr = pca_stats(P)                                  # PR on the PCA-50 cloud
    Xc = C - C.mean(0)                                 # full-spectrum PR too
    eig = np.linalg.svd(Xc, compute_uv=False) ** 2
    diam = topo["cloud_diameter"]
    out = {
        "betti0": topo["betti0"], "betti1": topo["betti1"],
        "H0_total_persistence": topo["H0_total_persistence"],
        "H1_total_persistence": topo["H1_total_persistence"],
        "H0_total_persistence_norm": topo["H0_total_persistence"] / diam,
        "H1_total_persistence_norm": topo["H1_total_persistence"] / diam,
        "persistence_entropy_H0": topo["persistence_entropy_H0"],
        "persistence_entropy_H1": topo["persistence_entropy_H1"],
        "H1_max_lifetime_frac": topo["H1_max_lifetime_frac"],
        "PCA_participation_ratio": pr["PCA_participation_ratio"],
        "PCA_participation_ratio_full": float(eig.sum() ** 2 / (eig ** 2).sum()),
        "cloud_diameter": diam,
    }
    return out, dgms[0], dgms[1]


def clean_dgm(dgm: np.ndarray, diam: float) -> np.ndarray:
    """Cap the essential (infinite) class at the diameter, then divide by it, so
    a bottleneck/Wasserstein distance between two stages is scale-free."""
    d = np.asarray(dgm, float).copy()
    d[~np.isfinite(d[:, 1]), 1] = diam
    return d / diam if diam > 0 else d


def bootstrap_band(Xf: np.ndarray, groups: list, rng) -> pd.DataFrame:
    """Resample each role's rows (with replacement), re-estimate all centroids,
    recompute signatures. B_BOOT rows — the within-stage sampling wobble."""
    rows = []
    for _ in range(B_BOOT):
        C = np.empty((len(groups), Xf.shape[1]), np.float32)
        for i, g in enumerate(groups):
            C[i] = Xf[rng.choice(g, size=len(g), replace=True)].mean(0)
        rows.append(signatures(C)[0])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output/b4_topology_preservation")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    point, boot, dgms_by_stage, axes = {}, {}, {}, {}

    for st in STAGES:
        meta = pd.read_parquet(META.format(st))
        arr = np.load(NPY.format(st), mmap_mode="r")     # [55200,3,4096] fp16
        if len(meta) != arr.shape[0]:
            raise SystemExit(f"{st}: {len(meta)} metadata rows but {arr.shape[0]} array rows")

        # axis per depth (default_centroid - mean(centroids), unit norm)
        axes[st] = {}
        C17 = None
        for di, layer in DEPTHS.items():
            C, meta_role = aggregate_by_role(np.asarray(arr[:, di, :]), meta)
            axes[st][layer] = assistant_axis(C, meta_role)   # already unit-norm
            if di == L_DEPTH:
                C17 = C
        # point-estimate signatures at layer 17
        sig, h0, h1 = signatures(C17)
        point[st] = sig
        dgms_by_stage[st] = (h0, h1)

        # bootstrap band at layer 17
        Xf = np.asarray(arr[:, L_DEPTH, :], dtype=np.float32)
        roles = sorted(meta["role"].unique())
        groups = [np.where(meta["role"].values == r)[0] for r in roles]
        boot[st] = bootstrap_band(Xf, groups, np.random.default_rng(SEED))
        del Xf, arr
        print(f"  {st:5s} done ({time.time()-t0:.0f}s)  PR_full="
              f"{sig['PCA_participation_ratio_full']:.2f}  "
              f"Sent_H0={sig['persistence_entropy_H0']:.3f}  diam={sig['cloud_diameter']:.1f}")

    # ---- assemble tables --------------------------------------------------
    pt = pd.DataFrame(point).T                       # stages x signatures
    b_mean = pd.DataFrame({st: boot[st].mean() for st in STAGES}).T
    b_std = pd.DataFrame({st: boot[st].std(ddof=1) for st in STAGES}).T

    # preservation: cross-stage relative range on scale-free signatures, and
    # how big that cross-stage range is vs the mean within-stage bootstrap SD
    # (>~2-3x SD = the stages genuinely differ; ~1x = indistinguishable noise).
    pres = []
    for m in SCALE_FREE:
        v = pt[m].to_numpy(float)
        rng_rel = (v.max() - v.min()) / abs(v.mean()) if v.mean() != 0 else np.nan
        sd = b_std[m].mean()
        pres.append({"signature": m, "base": v[0], "dpo": v[1], "rlvr": v[2],
                     "rel_range": rng_rel, "boot_sd": sd,
                     "range/sd": (v.max() - v.min()) / sd if sd > 0 else np.inf})
    pres = pd.DataFrame(pres)

    # diameter-normalised diagram distances between stages
    diam = {st: point[st]["cloud_diameter"] for st in STAGES}
    ddist = []
    for a, b in [("base", "dpo"), ("dpo", "rlvr"), ("base", "rlvr")]:
        h0a, h1a = dgms_by_stage[a]; h0b, h1b = dgms_by_stage[b]
        ddist.append({
            "pair": f"{a}->{b}",
            "bottleneck_H0": persim.bottleneck(clean_dgm(h0a, diam[a]), clean_dgm(h0b, diam[b])),
            "bottleneck_H1": persim.bottleneck(clean_dgm(h1a, diam[a]), clean_dgm(h1b, diam[b])),
            "wasserstein_H1": persim.wasserstein(clean_dgm(h1a, diam[a]), clean_dgm(h1b, diam[b])),
        })
    ddist = pd.DataFrame(ddist)

    # axis rotation, per depth
    axrows = []
    for layer in DEPTHS.values():
        row = {"layer": layer}
        for a, b in [("base", "dpo"), ("dpo", "rlvr"), ("base", "rlvr")]:
            c = float(np.dot(axes[a][layer], axes[b][layer]))
            row[f"cos_{a}_{b}"] = c
            row[f"deg_{a}_{b}"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        axrows.append(row)
    axdf = pd.DataFrame(axrows)

    # ---- print ------------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== PER-STAGE TOPOLOGY SIGNATURES (layer 17, 276 role centroids) ===")
    print(pt.round(4).T.to_string())
    print("\n=== PRESERVATION: scale-free signatures vs bootstrap noise (B={}) ===".format(B_BOOT))
    print(pres.round(4).to_string(index=False))
    print("\n=== CROSS-STAGE DIAGRAM DISTANCE (diameter-normalised) ===")
    print(ddist.round(4).to_string(index=False))
    print("\n=== AXIS ROTATION per depth (cos + degrees) ===")
    print(axdf.round(4).to_string(index=False))

    inputs = {st: {"npy": NPY.format(st), "meta": META.format(st),
                   "manifest_sha256": hashlib.sha256(
                       (Path(CLOUD.format(st)) / "manifest.json").read_bytes()).hexdigest()}
              for st in STAGES}
    payload = {"settings": {"layer_depth_index": L_DEPTH, "depths": DEPTHS,
                            "d_pca": D_PCA, "seed": SEED, "b_boot": B_BOOT,
                            "ripser_maxdim": os.environ["MP_RIPSER_MAXDIM"]},
               "inputs": inputs,
               "point": point, "boot_mean": b_mean.to_dict(),
               "boot_sd": b_std.to_dict(), "preservation": pres.to_dict("records"),
               "diagram_dist": ddist.to_dict("records"),
               "axis": axdf.to_dict("records")}
    out = out_dir / "b4_topology.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    write_stamp(out_dir, data_dirs=[CLOUD.format(st) for st in STAGES])
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
