"""Build the three steering strategies for the three study pairs and export them for the artifact.

Strategies, all sharing the same start and end centroid:
  linear       the straight chord A -> B (the paper's intervention)
  manifold_v1  16 knots, one per equal-WIDTH span of the chord, unbounded eps
  manifold_v2  16 knots, one per equal-COUNT bin of the same candidates, unbounded eps

eps is unbounded in both manifold strategies because widening the tube never changes an
already-filled span's pick -- the pick is argmin r, so extra candidates only ever land further from
the chord. Widening only fills EMPTY spans. So "largest eps" maximises knot count and detour at
once, and it makes the partition the only difference between v1 and v2.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from steering.manifold_paths import (LinearPath, PersonaPath, chord_coords, chord_frame,
                                     chunk_diagnostics, END_MARGIN)

PAIRS = [("validator", "vampire", "far"), ("validator", "bard", "far"),
         ("validator", "amateur", "mid")]
KS = (8, 16)          # k is a study variable, not a constant
EPS = 1e9          # unbounded: see module docstring
NSAMP = 160


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geom-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=19)
    a = ap.parse_args()

    d = np.load(a.geom_cache, allow_pickle=True)
    C = np.asarray(d["C"], float)
    names = [str(x) for x in d["names"]]
    axis_proj = np.asarray(d["axis_proj"], float)
    idx = {n: i for i, n in enumerate(names)}

    mu = C.mean(0)
    _, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = (S ** 2) / max(len(C) - 1, 1)
    basis, evr = Vt[:3], (var[:3] / var.sum())

    def pr(X):
        return [[round(float(v), 3) for v in r]
                for r in ((np.atleast_2d(X) - mu) @ basis.T)]

    out = {"layer": a.layer, "evr3": [round(float(v), 4) for v in evr],
           "roles": names, "cloud": pr(C),
           "axis_proj": [round(float(v), 2) for v in axis_proj],
           "ks": list(KS), "n_samp": NSAMP, "pairs": []}

    ts = np.linspace(0, 1, NSAMP)
    for A, B, kind in PAIRS:
        P0, P1 = C[idx[A]], C[idx[B]]
        _, L = chord_frame(P0, P1)
        u_all, r_all = chord_coords(C, P0, P1)
        rec = {"A": A, "B": B, "kind": kind, "chord": round(float(L), 2),
               "axA": round(float(axis_proj[idx[A]]), 2),
               "axB": round(float(axis_proj[idx[B]]), 2),
               "strategies": {}}

        lin = LinearPath(P0, P1)
        rec["strategies"]["linear"] = {
            "label": "linear - straight chord", "chunk": None, "k": 0, "family": "linear",
            "path": pr(lin.at_alpha(ts)), "knots": [], "knot_roles": [],
            "knot_u": [], "knot_r": [], "bins": [],
            "detour": 1.0, "polyline": 1.0, "overshoot": 0.0, "excursion": 1.0,
            "knot_error": 0.0, "n_knots": 0}

        for K in KS:
          for tag0, chunk in (("manifold_v1", "distance"), ("manifold_v2", "density")):
            tag = "%s_k%d" % (tag0, K)
            p = PersonaPath(P0, P1, C, EPS, k=K, lam=0.0, mode="absolute",
                            param="centripetal", chunk=chunk)
            rp = p.report(names)
            recs, _radius = chunk_diagnostics(C, P0, P1, EPS, k=K, mode="absolute", chunk=chunk)
            bins = []
            for b in recs:
                ci = b["cand_idx"]
                bins.append({
                    "lo": (None if not np.isfinite(b["u_lo"]) else round(float(b["u_lo"]), 4)),
                    "hi": (None if not np.isfinite(b["u_hi"]) else round(float(b["u_hi"]), 4)),
                    "n": int(len(ci)),
                    "picked": (None if not len(ci) else names[int(b["picked"])])})
            rec["strategies"][tag] = {
                "label": ("manifold v1 - equal-width spans, k=%d" % K if chunk == "distance"
                          else "manifold v2 - equal-count bins, k=%d" % K),
                "chunk": chunk, "k": K, "family": tag0,
                "path": pr(p.at_alpha(ts)),
                "knots": pr(p.knots[1:-1]) if p.spline is not None else [],
                "knot_roles": rp["centroid_roles"] or [],
                "knot_u": [round(float(u_all[i]), 4) for i in p.centroid_idx],
                "knot_r": [round(float(r_all[i]), 3) for i in p.centroid_idx],
                "detour": round(float(rp["detour_ratio"]), 3),
                "polyline": round(float(rp["polyline_ratio"]), 3),
                "overshoot": round(float(rp["overshoot"]), 4),
                "excursion": round(float(rp["excursion"]), 3),
                "knot_error": float(rp["knot_error"]),
                "n_knots": int(rp["n_centroids"]),
                "bins": bins}

        rel = C - P0
        dv = P1 - P0
        cos = 1.0 - (rel @ dv) / (np.linalg.norm(rel, axis=1) * np.linalg.norm(dv) + 1e-12)
        keep = np.nonzero((u_all > END_MARGIN) & (u_all < 1 - END_MARGIN))[0]
        keep = keep[np.argsort(u_all[keep])]
        rec["cands"] = [[round(float(u_all[i]), 4), round(float(r_all[i]), 3),
                         round(float(cos[i]), 4), int(i)] for i in keep]
        out["pairs"].append(rec)

    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote %s  %.0f KB" % (a.out, os.path.getsize(a.out) / 1024))
    print("PCA-3 explained variance: %s" % [round(float(v), 3) for v in evr])
    for p in out["pairs"]:
        print("  %-22s chord %6.2f  axis %5.2f -> %5.2f  (%s)"
              % (p["A"] + ">" + p["B"], p["chord"], p["axA"], p["axB"], p["kind"]))
        for t in p["strategies"]:
            s = p["strategies"][t]
            print("      %-18s knots %2d  detour %6.3f  excursion %5.2f  knot_err %.2e"
                  % (t, s["n_knots"], s["detour"], s["excursion"], s["knot_error"]))


if __name__ == "__main__":
    main()
