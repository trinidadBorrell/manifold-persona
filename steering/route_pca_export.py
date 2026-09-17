"""PCA-3 coordinates for the routes an A->B run actually steered along.

`geometry_check --export-json` sweeps eps and k over the first few cases; this
exports exactly the routes a given run used, at that run's frozen (eps, k), so
the 3D view in the results artifact is the geometry the generations came from
rather than a neighbouring variant.
"""
import argparse, json
import numpy as np
from steering.geometry import load_geometry
from steering.path_cases import load_cases, EPS, K, LAM, MODE, PARAM, ALPHAS
from steering.manifold_paths import LinearPath, PersonaPath, chord_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--routes", required=True,
                    help="comma list of A>B, e.g. summarizer>scout,validator>veteran")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--layer", type=int, default=19)
    a = ap.parse_args()

    geom = load_geometry(resp_dir=a.cloud, labels_path=a.labels)
    cs = load_cases(geom)
    C, names = cs.C, cs.names

    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = (S ** 2) / max(len(C) - 1, 1)
    basis, evr = Vt[:3], var[:3] / var.sum()
    pr = lambda X: [[round(float(v), 3) for v in r]
                    for r in ((np.atleast_2d(X) - mu) @ basis.T)]

    want = [tuple(r.split(">")) for r in a.routes.split(",")]
    idx = {nm: i for i, nm in enumerate(names)}
    t = np.linspace(0, 1, 140)
    out = {"evr3": [float(v) for v in evr], "roles": names,
           "cloud": [[round(float(v), 3) for v in r] for r in ((C - mu) @ basis.T)],
           "eps": a.eps, "k": a.k, "alphas": [float(x) for x in ALPHAS],
           "layer": a.layer, "routes": []}
    for A, B in want:
        if A not in idx or B not in idx:
            print("skip %s->%s (not a fully-role-playing centroid)" % (A, B)); continue
        P0, P1 = C[idx[A]], C[idx[B]]
        lp = LinearPath(P0, P1)
        mp = PersonaPath(P0, P1, C, a.eps, k=a.k, lam=LAM, mode=MODE, param=PARAM)
        out["routes"].append({
            "A": A, "B": B, "chord_len": round(float(chord_frame(P0, P1)[1]), 2),
            "detour": round(mp.detour_ratio, 3), "n_centroids": mp.n_centroids,
            "waypoints": [names[int(i)] for i in mp.centroid_idx],
            "waypoint_xyz": pr(C[mp.centroid_idx]) if mp.n_centroids else [],
            "linear": pr(lp.at_alpha(t)), "manifold": pr(mp.at_alpha(t)),
            "linear_stops": pr(lp.at_alpha(np.array(ALPHAS))),
            "manifold_stops": pr(mp.at_alpha(np.array(ALPHAS))),
            "A_xyz": pr(P0)[0], "B_xyz": pr(P1)[0],
        })
    open(a.out, "w").write(json.dumps(out, separators=(",", ":")))
    print("wrote %s (%d routes, evr3=%s)" % (a.out, len(out["routes"]),
                                             [round(float(v), 3) for v in evr]))


if __name__ == "__main__":
    main()
