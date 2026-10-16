"""G1 -- consolidate E1 and build the PersonaManifold geometry per layer (plan 2026-10-08, 1.2-1.3).

    python -m pm_addition_round.build --e1 <E1 dir> --out <G1 dir> [--consolidate]

Writes, per hidden-state layer L:
  G1/geom_L<L>.npz   roles, X (eq. 1, last-token view), PCA mean/V, d*, k, and for each artifact
                     route: geodesic path personas and the eq. 6 deltas of both paper arms at t grid
  G1/summary.json    per layer: D', d* (LB, MP), k, held-out persona identification, path-length /
                     straight-line ratios (routes and all pairs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pm_geometry import geometry as G

ROUTES = ["assistant>bard", "assistant>vampire", "assistant>leviathan",
          "grader>tree", "examiner>criminal", "scheduler>void"]          # the artifact's routes
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
NEUTRAL = "__neutral__"


def consolidate(e1: Path):
    """chunks/*.npz -> layer-major fp16 memmaps last_L<L>.npy / mean_L<L>.npy and meta.csv."""
    info = json.loads((e1 / "units.json").read_text())
    layers, n = info["layers"], info["n_units"]
    chunks = sorted((e1 / "chunks").glob("u*.npz"))
    metas = [pd.read_csv(c.with_suffix(".csv")) for c in chunks]
    meta = pd.concat(metas, ignore_index=True).sort_values("unit")
    if len(meta) != n or not (meta.unit.to_numpy() == np.arange(n)).all():
        raise ValueError("E1 incomplete: %d of %d units" % (len(meta), n))
    out = {}
    width = int(np.load(chunks[0])["last"].shape[-1])          # 4096 for Qwen3-8B, 5120 for Qwen3-14B
    for view in ("last", "mean"):
        for li, L in enumerate(layers):
            out[(view, L)] = np.lib.format.open_memmap(e1 / ("%s_L%d.npy" % (view, L)), mode="w+",
                                                       dtype=np.float16, shape=(n, width))
    for c, m in zip(chunks, metas):
        z = np.load(c)
        for view in ("last", "mean"):
            A = z[view]
            for li, L in enumerate(layers):
                out[(view, L)][m.unit.to_numpy()] = A[:, li]
    for a in out.values():
        a.flush()
    meta.to_csv(e1 / "meta.csv", index=False)
    print("consolidated %d units, %d layers" % (n, len(layers)))


def load(e1: Path, view: str, L: int):
    meta = pd.read_csv(e1 / "meta.csv")
    H = np.load(e1 / ("%s_L%d.npy" % (view, L)), mmap_mode="r")
    return meta, H


def heldout_id(meta, H, H0, q0, n_test=8):
    """Layer proxy [ours; the paper uses BST TCR]: role centroids from the first 32 questions,
    each answer to the last 8 questions (minus its neutral answer) assigned to the nearest
    centroid. Returns accuracy over 276 roles (chance 1/276)."""
    qs = sorted(set(q0))
    test = set(qs[-n_test:])
    pr = meta.role != NEUTRAL
    tr = (pr & ~meta.question_idx.isin(test)).to_numpy()
    te = (pr & meta.question_idx.isin(test)).to_numpy()
    roles, C = G.persona_vectors(H[tr], meta.role[tr].to_numpy(), meta.question_idx[tr].to_numpy(), H0, q0)
    pos = {q: i for i, q in enumerate(q0)}
    Y = H[te].astype(np.float64) - H0[[pos[q] for q in meta.question_idx[te]]]
    d = ((Y ** 2).sum(1)[:, None] - 2 * Y @ C.T + (C ** 2).sum(1)[None])
    pred = np.asarray(roles)[d.argmin(1)]
    return float((pred == meta.role[te].to_numpy()).mean())


def all_pair_ratios(g):
    """Euclidean length of the geodesic polyline / straight-line distance, every pair."""
    Z, n, out = g.Z, len(g.Z), []
    for a in range(n):
        for b in range(a + 1, n):
            if not np.isfinite(g.dist[a, b]):
                continue
            p = G.shortest_path(g, a, b)
            out.append(np.linalg.norm(np.diff(Z[p], axis=0), axis=1).sum() / np.linalg.norm(Z[a] - Z[b]))
    return np.asarray(out)


def build_layer(e1: Path, out: Path, L: int):
    meta, H = load(e1, "last", L)
    neu = (meta.role == NEUTRAL).to_numpy()
    q0 = meta.question_idx[neu].to_numpy()
    H0 = H[neu].astype(np.float64)
    pr = ~neu
    roles, X = G.persona_vectors(H[pr], meta.role[pr].to_numpy(), meta.question_idx[pr].to_numpy(), H0, q0)
    pca = G.PCA99.fit(X)
    Z = pca.to_pca(X)
    idim = G.intrinsic_dim(Z)
    g = G.build_graph(Z, idim["d_star"])
    if not np.isfinite(g.dist).all():
        print("L%d: graph disconnected (%d unreachable pairs)" % (L, int((~np.isfinite(g.dist)).sum())))
    ratios = all_pair_ratios(g)
    routes, arrays = {}, {}
    for r in ROUTES:
        s, t = r.split(">")
        R = G.route(pca, g, roles, s, t, TS)
        arrays["geo|" + r] = G.steering_deltas(pca, R["geo"], TS).astype(np.float32)
        arrays["lin|" + r] = G.steering_deltas(pca, R["lin"], TS).astype(np.float32)
        routes[r] = dict(path_personas=R["nodes"], path_len_over_straight=R["d_g_euclid_len"] / R["d_E"],
                         d_g_metric=R["d_g"], d_E=R["d_E"],
                         delta_norm_geo=[float(x) for x in np.linalg.norm(arrays["geo|" + r], axis=1)],
                         delta_norm_lin=[float(x) for x in np.linalg.norm(arrays["lin|" + r], axis=1)])
    np.savez(out / ("geom_L%d.npz" % L), roles=np.asarray(roles), X=X.astype(np.float32),
             pca_mean=pca.mean.astype(np.float32), pca_V=pca.V.astype(np.float32), ts=np.asarray(TS),
             route_keys=np.asarray(list(arrays)), route_deltas=np.stack(list(arrays.values())))
    rec = dict(layer=L, n_personas=len(roles), D_prime=int(pca.V.shape[0]), **idim, k=g.k,
               heldout_id_acc=heldout_id(meta, H, H0, q0),
               all_pairs_ratio=dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                        [float(x) for x in np.percentile(ratios, [10, 25, 50, 75, 90])]),
                                    frac_above_1_2=float((ratios > 1.2).mean())),
               routes=routes, hit_cap_frac=float(meta.hit_cap.mean()))
    print(json.dumps({k: v for k, v in rec.items() if k != "routes"}), flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="12-32")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--consolidate-only", action="store_true")
    a = ap.parse_args(argv)
    e1, out = Path(a.e1), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.consolidate or not (e1 / "meta.csv").exists():
        consolidate(e1)
    if a.consolidate_only:
        return
    lo, hi = (int(x) for x in a.layers.split("-"))
    for L in range(lo, hi + 1):
        if not (out / ("summary_L%d.json" % L)).exists():
            (out / ("summary_L%d.json" % L)).write_text(json.dumps(build_layer(e1, out, L)))
    recs = [json.loads(f.read_text()) for f in sorted(out.glob("summary_L*.json"), key=lambda f: int(f.stem[9:]))]
    (out / "summary.json").write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main()
