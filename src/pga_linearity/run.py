"""Part A of plans/2026-10-29: how linear are the Fletcher PGA directions, and how much do they
differ from the PCA directions? Axis level only, one layer per job.

Data: the 276 persona vectors of `$PM/E1` (view `last`, persona = mean over its 40 answers of
answer minus the no-prompt answer to the same question), as in `pm_geometry.components`.

Everything is computed in the exact PCA coordinates of the 276 vectors (SVD of the centred
matrix, all non-zero components kept, so distances are unchanged). In these coordinates the PCA
directions are the coordinate axes e_j, and the PGA directions are the columns of V from
`pm_geometry.components.pga` (Fletcher 2004 on the k = 10 kNN graph, Frechet mean, 2-hop log map).
So every comparison below is between the columns of V and the coordinate axes.

Measures (plan section 6):
  A1  principal angles between span(PCA 1..k) and span(PGA 1..k): mean cos^2 and Grassmann distance
  A2  per PGA direction u_k: max_j |cos(u_k, e_j)| and j; share of u_k inside span(PCA 1..K), K = 10, 20, 50
  A3  per PGA direction: max_j |Spearman rho| between persona scores on PGA k and on PCA j, j <= 50
  A4  the principal geodesic itself as a polyline on the graph (shortest paths from the Frechet mean
      to the two most extreme personas inside a 30 degree cone around +-u_k), and the personas in the
      cone ordered by their coordinate: straight-line R^2 (PC1 share of the polyline's points),
      arc length / chord, max distance from the chord / chord, mean turning angle between segments
  A5  R^2 of the 276 vectors reconstructed in span(PGA 1..k) against span(PCA 1..k) (PCA is optimal)
  A6  the same on the two question-half persona sets A and B (stability)
  A7  log-map fidelity: for each persona, cos between its log-map vector (the direction of the first two
      hops of the shortest path from the Frechet mean, as Fletcher's Log_mu) and its actual displacement
      z_i - z_mu. If the geodesic to the persona were straight the two would coincide (cos = 1).
      Reported as the distribution over personas, and against |PGA 1 coordinate| (are the extreme personas
      the ones whose paths detour?)
Nulls: Gaussian and shuffled clouds with the same per-axis variances (`curvature_null.run.null`),
graph and PGA refit on each draw.

    python -m pga_linearity.run --layer 25 [--view last] [--n-null 20] --out $PM/PGALIN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from curvature_null.run import null
from pm_geometry.components import persona_sets, knn_graph, pga, path

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
K_MAX = 20          # subspace sizes and directions reported
K_PROJ = (10, 20, 50)
K_RHO = 50
K_GEO = 10          # directions whose geodesic polyline is measured
CONE_DEG = 30.0


# ---------------------------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------------------------

def pca_frame(X):
    """Centred X -> exact PCA coordinates Z (N, D') and the frame (mean, Vt) to map other sets."""
    mean = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    keep = s > s[0] * 1e-8
    return (U[:, keep] * s[keep]), mean, Vt[keep], (s[keep] ** 2) / len(X)


def to_frame(X, mean, Vt):
    return (X - mean) @ Vt.T


# ---------------------------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------------------------

def subspace_overlap(V, kmax=K_MAX):
    """A1. Columns of V = PGA directions in PCA coordinates; PCA k = e_k."""
    out = []
    for k in range(1, min(kmax, V.shape[1]) + 1):
        s = np.linalg.svd(V[:k, :k], compute_uv=False)        # cos of the principal angles
        s = np.clip(s, 0, 1)
        out.append(dict(k=k, mean_cos2=float((s ** 2).mean()),
                        grassmann=float(np.sqrt((np.arccos(s) ** 2).sum())),
                        min_cos=float(s.min())))
    return out


def direction_content(V, kmax=K_MAX):
    """A2. For each PGA direction: the closest PCA axis and the share inside span(PCA 1..K)."""
    out = []
    for k in range(min(kmax, V.shape[1])):
        u = V[:, k]
        out.append(dict(k=k + 1, max_cos=float(np.abs(u).max()), best_pca=int(np.abs(u).argmax() + 1),
                        share_in_pca={str(K): float((u[:K] ** 2).sum()) for K in K_PROJ}))
    return out


def coordinate_agreement(P, Z, kmax=K_MAX, krho=K_RHO):
    """A3. Spearman between persona scores on PGA k and PCA j."""
    K = min(kmax, P["coords"].shape[1])
    J = min(krho, Z.shape[1])
    rho = np.abs(spearmanr(P["coords"][:, :K], Z[:, :J])[0][:K, K:])
    return [dict(k=k + 1, max_rho=float(rho[k].max()), best_pca=int(rho[k].argmax() + 1)) for k in range(K)]


def polyline_measures(Pz):
    """Straight-line R^2, arc / chord, max chord deviation / chord, mean turning angle (deg)."""
    Pz = np.asarray(Pz, dtype=np.float64)
    if len(Pz) < 3:
        return dict(n=int(len(Pz)), line_r2=None, arc_over_chord=None, max_dev_over_chord=None, turning_deg=None)
    seg = np.diff(Pz, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    chord_v = Pz[-1] - Pz[0]
    chord = float(np.linalg.norm(chord_v))
    C = Pz - Pz.mean(0)
    lam = np.linalg.svd(C, compute_uv=False) ** 2
    line_r2 = float(lam[0] / lam.sum()) if lam.sum() > 0 else None
    if chord > 0:
        d = chord_v / chord
        rel = Pz - Pz[0]
        dev = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
        max_dev = float(dev.max() / chord)
        arc_over_chord = float(seglen.sum() / chord)
    else:
        max_dev = arc_over_chord = None
    ok = seglen[:-1] * seglen[1:] > 0
    cosang = (seg[:-1] * seg[1:]).sum(1)[ok] / (seglen[:-1] * seglen[1:])[ok]
    turning = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean()) if ok.any() else None
    return dict(n=int(len(Pz)), line_r2=line_r2, arc_over_chord=arc_over_chord,
                max_dev_over_chord=max_dev, turning_deg=turning)


def geodesic_curves(P, Z, g, roles=None, kgeo=K_GEO, cone_deg=CONE_DEG):
    """A4. For each PGA direction: the graph polyline through the Frechet mean between the two most
    extreme personas inside the cone, and the cone personas ordered by coordinate."""
    mu, U, V = P["mu"], P["U"], P["V"]
    norms = np.linalg.norm(U, axis=1)
    out = []
    for k in range(min(kgeo, V.shape[1])):
        t = U @ V[:, k]
        cosang = np.divide(t, norms, out=np.zeros_like(t), where=norms > 0)
        cone = np.flatnonzero((np.abs(cosang) >= np.cos(np.radians(cone_deg))) & (np.arange(len(t)) != mu))
        pool = cone if len(cone) >= 2 else np.flatnonzero(np.arange(len(t)) != mu)
        i_plus, i_minus = int(pool[np.argmax(t[pool])]), int(pool[np.argmin(t[pool])])
        nodes = path(g, mu, i_minus)[::-1] + path(g, mu, i_plus)[1:]
        poly = polyline_measures(Z[nodes])
        order = pool[np.argsort(t[pool])]
        cone_m = polyline_measures(Z[order])
        rec = dict(k=k + 1, n_cone=int(len(cone)), n_path_nodes=int(len(nodes)),
                   polyline=poly, cone_ordered=cone_m)
        if roles is not None:
            rec["ends"] = dict(minus=roles[i_minus], plus=roles[i_plus], frechet_mean=roles[mu])
            rec["path_roles"] = [roles[i] for i in nodes]
        out.append(rec)
    return out


def reconstruction_gap(V, Z, var, kmax=K_MAX):
    """A5. R^2 in span(PGA 1..k) vs span(PCA 1..k)."""
    tot = float((Z ** 2).sum())
    out = []
    for k in range(1, min(kmax, V.shape[1]) + 1):
        Q, _ = np.linalg.qr(V[:, :k])
        r2_pga = float(((Z @ Q) ** 2).sum() / tot)
        r2_pca = float(var[:k].sum() / var.sum())
        out.append(dict(k=k, r2_pga=r2_pga, r2_pca=r2_pca, gap=r2_pca - r2_pga))
    return out


def logmap_fidelity(P, Z, g, roles=None):
    """A7. cos(log-map vector, true displacement from the Frechet mean) per persona, plus path hops."""
    mu = P["mu"]
    disp = Z - Z[mu]
    nu, nd = np.linalg.norm(P["U"], axis=1), np.linalg.norm(disp, axis=1)
    ok = (nu > 0) & (nd > 0)
    cos = np.full(len(Z), np.nan)
    cos[ok] = (P["U"][ok] * disp[ok]).sum(1) / (nu[ok] * nd[ok])
    hops = np.array([len(path(g, mu, i)) - 1 if i != mu else 0 for i in range(len(Z))])
    c = cos[ok]
    rec = dict(mean=float(c.mean()), median=float(np.median(c)), p10=float(np.percentile(c, 10)),
               frac_below_0_5=float((c < 0.5).mean()), frac_negative=float((c < 0).mean()),
               hops_mean=float(hops[ok].mean()), hops_max=int(hops.max()),
               spearman_abs_pga1_vs_cos=float(spearmanr(np.abs(P["coords"][ok, 0]), c)[0]),
               mean_cos_by_hops={str(h): float(c[hops[ok] == h].mean()) for h in np.unique(hops[ok])})
    if roles is not None:
        order = np.argsort(np.where(ok, cos, 2))
        rec["lowest"] = [dict(role=roles[i], cos=float(cos[i]), hops=int(hops[i])) for i in order[:10]]
        rec["per_persona_cos"] = [None if not ok[i] else float(cos[i]) for i in range(len(Z))]
        rec["per_persona_hops"] = hops.tolist()
    return rec


def all_measures(Z, var, roles=None):
    g = knn_graph(Z)
    P = pga(Z, g)
    return dict(graph_k=int(g["k"]), frechet_mean=(roles[P["mu"]] if roles is not None else int(P["mu"])),
                pga_var_share=(P["lam"] / P["lam"].sum())[:K_MAX].tolist(),
                A1=subspace_overlap(P["V"]), A2=direction_content(P["V"]), A3=coordinate_agreement(P, Z),
                A4=geodesic_curves(P, Z, g, roles), A5=reconstruction_gap(P["V"], Z, var),
                A7=logmap_fidelity(P, Z, g, roles)), P


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(PM / "E1"))
    ap.add_argument("--out", default=str(PM / "PGALIN"))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="last")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    roles, sets, _ = persona_sets(Path(a.e1), a.view, a.layer)
    roles = list(roles)
    X = np.asarray(sets["all"], dtype=np.float64)
    Z, mean, Vt, var = pca_frame(X)
    print(f"L{a.layer} {a.view}: {Z.shape[0]} personas, {Z.shape[1]} exact PCA coordinates", flush=True)

    res = dict(layer=a.layer, view=a.view, n=int(len(Z)), dim=int(Z.shape[1]),
               pca_var_share=(var / var.sum())[:K_MAX].tolist())
    res["real"], P = all_measures(Z, var, roles)
    # the three leading PGA directions written in the PCA frame (for figure F15 and reuse)
    np.savez(out / f"frame_{a.view}_L{a.layer}.npz", Z=Z, V=P["V"][:, :K_MAX], U=P["U"], mu=P["mu"],
             coords=P["coords"][:, :K_MAX], roles=np.array(roles), var=var)

    # A6: question halves, in the same frame
    res["halves"] = {}
    PH = {}
    for h in ("A", "B"):
        Zh = to_frame(np.asarray(sets[h], dtype=np.float64), mean, Vt)
        varh = (Zh ** 2).sum(0) / len(Zh)
        res["halves"][h], PH[h] = all_measures(Zh, varh, roles)
    cos3 = np.linalg.svd(PH["A"]["V"][:, :3].T @ PH["B"]["V"][:, :3], compute_uv=False)
    res["halves"]["pga_top3_principal_cos_A_vs_B"] = [float(c) for c in np.clip(cos3, 0, 1)]
    res["halves"]["pga1_cos_A_vs_B"] = float(abs(PH["A"]["V"][:, 0] @ PH["B"]["V"][:, 0]))

    # nulls
    res["null"] = {}
    for kind in ("gauss", "shuffle"):
        draws = []
        for d in range(a.n_null):
            Zn = null(Z.astype(np.float32), kind, rng).astype(np.float64)
            Zn = Zn - Zn.mean(0)
            varn = (Zn ** 2).sum(0) / len(Zn)
            m, _ = all_measures(Zn, varn, None)
            draws.append(m)
            print(f"  null {kind} {d + 1}/{a.n_null}", flush=True)
        res["null"][kind] = draws

    f = out / f"pgalin_{a.view}_L{a.layer}.json"
    f.write_text(json.dumps(res, indent=1))
    print("wrote", f)


if __name__ == "__main__":
    main()
