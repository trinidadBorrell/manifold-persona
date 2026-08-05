"""Diagnostic 02 — how much did the attention sink change the published numbers?

Compares two point clouds built from the SAME model, roles, questions and seed,
differing only in whether attention-sink positions were excluded from
``prompt_avg``:

    data/embeddings_roles_full_raw     sink_factor = null   (pre-fix behaviour)
    data/embeddings_roles_full_fixed   sink_factor = 5.0    (post-fix)

Reports, for each, the quantities the repo's claims rest on: PC1 share, whether
PC1 is sequence length, the intrinsic-dimension estimates, the Assistant-Axis
alignment, and the between/within-role separation.

Build the two clouds with:
    .venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 \
        --out_dir data/embeddings_roles_full_fixed
    .venv/bin/python -m extraction.build_and_extract_roles --n_questions 3 \
        --keep_sinks --out_dir data/embeddings_roles_full_raw

Then:
    .venv/bin/python diagnostics/02_sink_impact.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import skdim

from manifold_persona.common import aggregate_by_role
from manifold_persona.io import load_layer

ESTIMATORS = {
    "TwoNN":   (lambda: skdim.id.TwoNN(),   "global"),
    "MLE":     (lambda: skdim.id.MLE(K=20), "pw_mean"),
    "lPCA":    (lambda: skdim.id.lPCA(),    "global"),
    "MOM":     (lambda: skdim.id.MOM(),     "global"),
    "TLE":     (lambda: skdim.id.TLE(),     "global"),
    "CorrInt": (lambda: skdim.id.CorrInt(), "global"),
}


def intrinsic_dims(X):
    out = {}
    for name, (factory, how) in ESTIMATORS.items():
        try:
            est = factory().fit(X)
            out[name] = (float(np.nanmean(est.dimension_pw_))
                         if how == "pw_mean" else float(est.dimension_))
        except Exception as exc:                       # estimator-specific failures
            out[name] = None
            print(f"      ({name} failed: {type(exc).__name__})")
    return out


def pc_spectrum(X):
    Xc = X - X.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    return u * s, ev, vt


def analyse(in_dir: Path, layer: int | None):
    X, meta, manifest = load_layer(view="prompt_avg", layer=layer, in_dir=in_dir)
    X = np.asarray(X, dtype=np.float64)
    lay = layer if layer is not None else manifest["primary_layer"]
    T = meta["n_tokens"].to_numpy(float)

    res = {"dir": str(in_dir), "sink_factor": manifest.get("sink_factor"),
           "layer": lay, "n_records": int(X.shape[0]),
           "n_roles": int(meta["role"].nunique())}

    # ---- raw record level: is PC1 sequence length? -------------------------
    scores, ev, _ = pc_spectrum(X)
    res["raw"] = {
        "pc1_var": float(ev[0]),
        "pc1_3_var": float(ev[:3].sum()),
        "abs_r_pc1_inv_T": abs(float(np.corrcoef(scores[:, 0], 1.0 / T)[0, 1])),
        "abs_r_pc1_T": abs(float(np.corrcoef(scores[:, 0], T)[0, 1])),
    }

    # ---- role-centroid level: what the exploratory stage actually uses -----
    C, cmeta = aggregate_by_role(X, meta)
    C = np.asarray(C, dtype=np.float64)
    roles = cmeta["role"].to_numpy()
    Tbar = np.array([T[meta["role"].to_numpy() == r].mean() for r in roles])

    cscores, cev, _ = pc_spectrum(C)
    is_def = roles == "default"
    Cc = C - C.mean(0, keepdims=True)
    axis = Cc[is_def].mean(0) - Cc.mean(0)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    pc1_dir = np.linalg.svd(Cc, full_matrices=False)[2][0]

    # between/within role separation on the raw points
    rmeta = meta["role"].to_numpy()
    cent = np.stack([X[rmeta == r].mean(0) for r in roles])
    between = float(((cent - cent.mean(0)) ** 2).sum(1).mean())
    within = float(np.mean([((X[rmeta == r] - X[rmeta == r].mean(0)) ** 2).sum(1).mean()
                            for r in roles]))

    res["roles"] = {
        "n": int(C.shape[0]),
        "pc1_var": float(cev[0]),
        "pc1_3_var": float(cev[:3].sum()),
        "pc_for_90pct": int(np.searchsorted(np.cumsum(cev), 0.90) + 1),
        "abs_r_pc1_inv_Tbar": abs(float(np.corrcoef(cscores[:, 0], 1.0 / Tbar)[0, 1])),
        "abs_cos_pc1_assistant_axis": abs(float(pc1_dir @ axis)),
        "between": between, "within": within, "between_over_within": between / within,
        "intrinsic_dim": intrinsic_dims(C),
    }
    return res


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/embeddings_roles_full_raw")
    ap.add_argument("--fixed", default="data/embeddings_roles_full_fixed")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for tag, d in (("raw", args.raw), ("fixed", args.fixed)):
        print(f"\n### {tag}: {d}")
        results[tag] = analyse(Path(d), args.layer)

    R, F = results["raw"], results["fixed"]
    line = "=" * 78
    print(f"\n{line}\nSINK IMPACT  (model={F['n_roles']} roles, layer {F['layer']}, "
          f"{F['n_records']} records)\n{line}")
    print(f"{'quantity':<44}{'RAW':>16}{'FIXED':>16}")
    print("-" * 78)
    rows = [
        ("per-record  PC1 share of variance",      R["raw"]["pc1_var"],  F["raw"]["pc1_var"]),
        ("per-record  |r(PC1, 1/T)|",              R["raw"]["abs_r_pc1_inv_T"], F["raw"]["abs_r_pc1_inv_T"]),
        ("role-mean   PC1 share of variance",      R["roles"]["pc1_var"], F["roles"]["pc1_var"]),
        ("role-mean   PC1-3 share",                R["roles"]["pc1_3_var"], F["roles"]["pc1_3_var"]),
        ("role-mean   |r(PC1, 1/mean_T)|",         R["roles"]["abs_r_pc1_inv_Tbar"], F["roles"]["abs_r_pc1_inv_Tbar"]),
        ("role-mean   |cos(PC1, Assistant Axis)|", R["roles"]["abs_cos_pc1_assistant_axis"],
                                                   F["roles"]["abs_cos_pc1_assistant_axis"]),
        ("between/within role ratio",              R["roles"]["between_over_within"],
                                                   F["roles"]["between_over_within"]),
    ]
    for name, a, b in rows:
        print(f"{name:<44}{fmt(a):>16}{fmt(b):>16}")
    print(f"{'role-mean   PCs for 90% variance':<44}"
          f"{R['roles']['pc_for_90pct']:>16}{F['roles']['pc_for_90pct']:>16}")
    print("-" * 78)
    print(f"{'INTRINSIC DIMENSION (276 role centroids)':<44}{'RAW':>16}{'FIXED':>16}")
    for k in ESTIMATORS:
        print(f"  {k:<42}{fmt(R['roles']['intrinsic_dim'][k], 2):>16}"
              f"{fmt(F['roles']['intrinsic_dim'][k], 2):>16}")
    print(line)

    out = Path(args.out) if args.out else Path("output") / "sink_impact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
