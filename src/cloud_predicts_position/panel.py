"""Stage 1: the per-persona cloud metric panel (plan 2026-10-29, section 2).

One job = one layer, one view, a chunk of roles; both centrings (question-centred `q`, raw `raw`) per role.

Cloud of persona p: x_{s,q}, s = 5 prompts, q = 240 questions, 1,200 points in R^4096.
  q-centred  x - d_{s,q}, d = the `default` role's answer to the same question under the same prompt index
             (PersonaManifold eq. 1 with `default` as the neutral)
  raw        x as stored
Metrics that need a low-dimensional space (ID, density, topology, Ricci curvature, bend, path trend, Hopkins)
use the cloud's own PCA-50; the rest use the full space. Subsample sizes are fixed so every cloud gets the
same operation (n matters for ID and persistence).

Families (column prefix in brackets): scale [S], intrinsic dimension [D], spectral shape [P], density [N],
topology [T], curvature [C], anisotropy relative to the assistant axis [A], design decomposition [G],
alignment with the default cloud [R, raw only: the default cloud is zero after q-centring], consistency [I].
`spline_R2` / `curvature_gain` of the August panel are not computed: they were traced to the question grid
(per_persona_geometry/metrics.py, 2026-08-04). `silhouette_vs_all` is a separate stage (`--silhouette`).

    python -m cloud_predicts_position.panel --layer 25 --view avg --chunk 0 --n-chunks 69
    python -m cloud_predicts_position.panel --layer 25 --view avg --collect
    python -m cloud_predicts_position.panel --layer 25 --view avg --silhouette
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.neighbors import NearestNeighbors

from cloud_predicts_position import data as D
from curvature_null.run import bends, graph_stats, null
from role_manifold.idim import id_estimates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "per_persona_geometry"))
from curvature import curvature_metrics            # noqa: E402  (flat script module)
from density import density_metrics                # noqa: E402
from topology import persistence, betti_from_diagrams, persistence_entropy  # noqa: E402

D_PCA = 50
K_NN = 10
N_TOPO, R_TOPO = 400, 3        # ripser subsample size and repeats (400 points, maxdim 2: a few seconds)
N_LOCAL, K_LOCAL = 200, 50     # local TwoNN: 200 anchor points, 50 neighbours each
N_CORR = 600                   # correlation dimension subsample
N_HOPKINS = 200
N_BEND_NULL = 2
SEED = 0

FAMILIES = {
    "S": ["log_total_var", "centroid_norm", "rms_radius"],
    "D": ["TwoNN", "MLE", "PCA_dim_95pct", "PCA_participation_ratio", "local_id_mean", "local_id_cv", "corr_dim"],
    "P": ["effective_rank", "eig_decay_exponent", "top1_share", "top10_share", "spectral_gap"],
    "N": ["knn_dist_mean", "knn_dist_cv", "kde_logdens_mean", "kde_logdens_sd", "hubness", "hopkins"],
    "T": ["H0_total_persistence", "H1_total_persistence", "H2_total_persistence", "H1_max_lifetime", "H1_max_lifetime_frac",
          "H2_max_lifetime", "H2_max_lifetime_frac", "persistence_entropy_H0", "persistence_entropy_H1", "persistence_entropy_H2",
          "betti1_at_median"],
    "C": ["orc_mean", "orc_sd", "frc_mean", "frc_sd", "bend_R2", "bend_R2_raw", "path_ratio_trend", "path_ratio_p50"],
    "A": ["var_frac_along_axis", "cos_pc1_axis", "cos_pc1_centroid", "subspace_overlap_k10"],
    "G": ["instr_frac", "quest_frac", "interaction_frac"],
    "R": ["cka_linear", "mknn_align", "rsa_spearman", "procrustes_resid"],
    "I": ["split_half_cos_prompts", "split_half_cos_questions", "between_prompt_disp", "silhouette_vs_all"],
}
PRIMARY = ["MLE", "effective_rank", "knn_dist_mean", "H0_total_persistence", "orc_mean", "var_frac_along_axis",
           "interaction_frac", "cka_linear", "between_prompt_disp"]


# ---------------------------------------------------------------------------------------------
# metric blocks
# ---------------------------------------------------------------------------------------------

def spectrum(s2):
    """Covariance eigenvalues (descending) -> spectral-shape and PCA-count metrics."""
    s2 = np.asarray(s2, float)
    s2 = s2[s2 > 0]
    if len(s2) < 2:
        return dict(effective_rank=np.nan, eig_decay_exponent=np.nan, PCA_dim_95pct=np.nan, PCA_participation_ratio=np.nan,
                    top1_share=np.nan, top10_share=np.nan, spectral_gap=np.nan)
    p = s2 / s2.sum()
    k = np.arange(1, min(50, len(s2)) + 1)
    return dict(effective_rank=float(np.exp(-(p * np.log(p)).sum())),
                eig_decay_exponent=float(np.polyfit(np.log(k), np.log(s2[:len(k)]), 1)[0]),
                PCA_dim_95pct=int(np.searchsorted(np.cumsum(p), 0.95) + 1),
                PCA_participation_ratio=float(s2.sum() ** 2 / (s2 ** 2).sum()),
                top1_share=float(p[0]), top10_share=float(p[:10].sum()),
                spectral_gap=float((s2[0] - s2[1]) / s2[0]))


def local_id(P, rng):
    import skdim
    nn = NearestNeighbors(n_neighbors=K_LOCAL + 1).fit(P)
    anchors = rng.choice(len(P), min(N_LOCAL, len(P)), replace=False)
    _, idx = nn.kneighbors(P[anchors])
    vals = []
    for row in idx:
        try:
            vals.append(float(skdim.id.TwoNN().fit(P[row]).dimension_))
        except Exception:
            pass
    v = np.array([x for x in vals if np.isfinite(x) and x > 0])
    return dict(local_id_mean=float(v.mean()) if len(v) else np.nan,
                local_id_cv=float(v.std() / v.mean()) if len(v) and v.mean() > 0 else np.nan)


def corr_dim(P, rng):
    from scipy.spatial.distance import pdist
    sub = P[rng.choice(len(P), min(N_CORR, len(P)), replace=False)]
    d = np.sort(pdist(sub))
    lo, hi = np.percentile(d, 5), np.percentile(d, 50)
    r = np.exp(np.linspace(np.log(lo), np.log(hi), 12))
    C = np.searchsorted(d, r) / len(d)
    ok = C > 0
    return float(np.polyfit(np.log(r[ok]), np.log(C[ok]), 1)[0]) if ok.sum() >= 3 else np.nan


def hubness(P):
    nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(P)
    _, idx = nn.kneighbors(P)
    counts = np.bincount(idx[:, 1:].ravel(), minlength=len(P))
    return float(skew(counts))


def hopkins(P10, rng):
    n = min(N_HOPKINS, len(P10))
    lo, hi = P10.min(0), P10.max(0)
    probes = rng.uniform(lo, hi, size=(n, P10.shape[1]))
    nn = NearestNeighbors(n_neighbors=2).fit(P10)
    u = nn.kneighbors(probes, n_neighbors=1)[0][:, 0]
    w = nn.kneighbors(P10[rng.choice(len(P10), n, replace=False)], n_neighbors=2)[0][:, 1]
    return float(u.sum() / (u.sum() + w.sum()))


def topology(P, rng):
    """Mean over R_TOPO subsamples of N_TOPO points, maxdim 2, plus H1 bars alive at the median distance."""
    from scipy.spatial.distance import pdist
    acc = []
    for _ in range(R_TOPO):
        sub = P[rng.choice(len(P), min(N_TOPO, len(P)), replace=False)]
        pz = persistence(sub)
        m = betti_from_diagrams(pz["dgms"], pz["diameter"])
        med = float(np.median(pdist(sub)))
        h1 = pz["dgms"][1] if len(pz["dgms"]) > 1 else np.zeros((0, 2))
        m["betti1_at_median"] = int(((h1[:, 0] <= med) & (h1[:, 1] > med)).sum()) if len(h1) else 0
        acc.append({k: v for k, v in m.items() if k in FAMILIES["T"]})
    df = pd.DataFrame(acc)
    return {k: float(df[k].mean()) for k in df.columns}


def bend(P10, rng):
    real = max(bends(P10, rng))
    nulls = [max(bends(null(P10.astype(np.float32), "shuffle", rng).astype(np.float64), rng)) for _ in range(N_BEND_NULL)]
    return dict(bend_R2_raw=float(real), bend_R2=float(real - np.mean(nulls)))


def design(Xc, instr, quest):
    """METHODS.md section 1: additive two-factor fit on the balanced grid."""
    tot = float((Xc ** 2).sum())
    A = np.stack([Xc[instr == i].mean(0) for i in np.unique(instr)])
    B = np.stack([Xc[quest == q].mean(0) for q in np.unique(quest)])
    fit = A[np.searchsorted(np.unique(instr), instr)] + B[np.searchsorted(np.unique(quest), quest)]
    inter = float(((Xc - fit) ** 2).sum()) / tot
    return dict(instr_frac=float((A ** 2).sum() * (len(Xc) / len(A))) / tot,
                quest_frac=float((B ** 2).sum() * (len(Xc) / len(B))) / tot, interaction_frac=inter)


def anisotropy(Xc, Vt10, axis, centroid_dir, default_top10):
    tot = float((Xc ** 2).sum())
    pc1 = Vt10[0]
    s = np.linalg.svd(Vt10 @ default_top10.T, compute_uv=False)
    return dict(var_frac_along_axis=float(((Xc @ axis) ** 2).sum() / tot),
                cos_pc1_axis=float(abs(pc1 @ axis)),
                cos_pc1_centroid=float(abs(pc1 @ centroid_dir)),
                subspace_overlap_k10=float((np.clip(s, 0, 1) ** 2).mean()))


def _gram(M):
    Mc = M - M.mean(0)
    return Mc @ Mc.T


def alignment(Qp, Qd):
    """Persona question means (240, 4096) against the default role's."""
    Kp, Kd = _gram(Qp), _gram(Qd)
    cka = float((Kp * Kd).sum() / (np.linalg.norm(Kp) * np.linalg.norm(Kd)))
    k = min(K_NN, len(Qp) - 1)                            # Experiment 2 passes as few as 5 question means
    nnp = NearestNeighbors(n_neighbors=k + 1).fit(Qp).kneighbors(Qp)[1][:, 1:]
    nnd = NearestNeighbors(n_neighbors=k + 1).fit(Qd).kneighbors(Qd)[1][:, 1:]
    mk = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nnp, nnd)]))
    from scipy.spatial import procrustes
    from scipy.spatial.distance import pdist
    rsa = float(spearmanr(pdist(Qp), pdist(Qd))[0])
    # Procrustes in the joint span (480 dims hold both sets exactly)
    try:
        J = np.vstack([Qp - Qp.mean(0), Qd - Qd.mean(0)])
        _, _, Vt = np.linalg.svd(J, full_matrices=False)
        _, _, disp = procrustes((Qd - Qd.mean(0)) @ Vt.T, (Qp - Qp.mean(0)) @ Vt.T)
        disp = float(disp)
    except np.linalg.LinAlgError:                          # one L32 cloud: "SVD did not converge" (chunk 49)
        disp = np.nan
    return dict(cka_linear=cka, mknn_align=mk, rsa_spearman=rsa, procrustes_resid=disp)


def consistency(X, instr, quest, rms):
    c_a = X[instr <= 2].mean(0); c_b = X[instr >= 3].mean(0)
    q_a = X[quest % 2 == 0].mean(0); q_b = X[quest % 2 == 1].mean(0)
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    pc = np.stack([X[instr == i].mean(0) for i in np.unique(instr)])
    iu = np.triu_indices(len(pc), 1)
    disp = float(np.linalg.norm(pc[iu[0]] - pc[iu[1]], axis=1).mean() / rms)
    return dict(split_half_cos_prompts=cos(c_a, c_b), split_half_cos_questions=cos(q_a, q_b), between_prompt_disp=disp)


# ---------------------------------------------------------------------------------------------

def cloud_metrics(X, instr, quest, ax, centring, rng, Qd_raw=None):
    """All metrics of one cloud X (1200, 4096) float64, already centred the way `centring` says."""
    t0 = time.time()
    out = {}
    Xc = X - X.mean(0)
    out.update(log_total_var=float(np.log((Xc ** 2).sum())), centroid_norm=float(np.linalg.norm(X.mean(0))),
               rms_radius=float(np.sqrt((Xc ** 2).sum() / len(X))))
    try:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _svd
        U, s, Vt = _svd(Xc, full_matrices=False, lapack_driver="gesvd")
    s2 = s ** 2 / len(X)
    out.update(spectrum(s2))
    P = (U[:, :D_PCA] * s[:D_PCA])                        # the cloud's own PCA-50
    out.update({k: v for k, v in id_estimates(P).items() if k in ("TwoNN", "MLE")})
    out.update(local_id(P, rng))
    out["corr_dim"] = corr_dim(P, rng)
    dens, _ = density_metrics(P, k=K_NN)
    out.update({k: v for k, v in dens.items() if k in FAMILIES["N"]})
    out["hubness"] = hubness(P)
    out["hopkins"] = hopkins(P[:, :10], rng)
    out.update(topology(P, rng))
    curv, _ = curvature_metrics(P, k=K_NN)
    out.update({k: v for k, v in curv.items() if k in FAMILIES["C"]})
    out.update(bend(P[:, :10].astype(np.float64), rng))
    gs = graph_stats(P.astype(np.float32), rng, k=K_NN)
    out.update(path_ratio_trend=float(gs["trend"]), path_ratio_p50=float(gs["p50"]))
    centroid_dir = X.mean(0) - (ax["default_centroid"] if centring == "raw" else 0.0)
    centroid_dir = centroid_dir / (np.linalg.norm(centroid_dir) + 1e-12)
    out.update(anisotropy(Xc, Vt[:10], ax["axis"].astype(np.float64), centroid_dir, ax["default_top10"].astype(np.float64)))
    out.update(design(Xc, instr, quest))
    if Qd_raw is not None:
        out.update(alignment(Qd_raw[0], Qd_raw[1]))
    out.update(consistency(X, instr, quest, out["rms_radius"]))
    out["seconds"] = time.time() - t0
    return out


def run_chunk(a):
    meta = D.metadata(a.layer)
    A = D.open_layer(a.layer, a.view)
    ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
    roles = D.roles(meta)
    mine = roles[a.chunk::a.n_chunks]
    Xd, instr_d, quest_d = D.cloud(A, meta, D.DEFAULT)
    Xd = Xd.astype(np.float64)
    Qd = ax["default_qmeans"].astype(np.float64)
    outdir = D.OUT / "panel" / f"{a.view}_L{a.layer}"
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"chunk_{a.chunk:03d}.csv"
    if f.exists() and not a.force:
        print("exists", f); return
    rows = []
    for r in mine:
        X, instr, quest = D.cloud(A, meta, r)
        X = X.astype(np.float64)
        assert (instr == instr_d).all() and (quest == quest_d).all()
        Qp = np.stack([X[quest == q].mean(0) for q in np.unique(quest)])
        for centring in ("q", "raw"):
            if centring == "q" and r == D.DEFAULT:
                continue                                   # the default role minus itself is a zero cloud
            rng = np.random.default_rng(SEED)
            Xc_ = X - Xd if centring == "q" else X
            m = cloud_metrics(Xc_, instr, quest, ax, centring, rng, Qd_raw=(Qp, Qd))
            m.update(role=r, centring=centring, layer=a.layer, view=a.view)
            rows.append(m)
            print(f"{r:20s} {centring:4s} {m['seconds']:6.1f}s  MLE {m['MLE']:.2f} effrank {m['effective_rank']:.1f} "
                  f"H0 {m['H0_total_persistence']:.2f} orc {m['orc_mean']:.3f} cka {m['cka_linear']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    print("wrote", f)


def collect(a):
    outdir = D.OUT / "panel" / f"{a.view}_L{a.layer}"
    df = pd.concat([pd.read_csv(f) for f in sorted(outdir.glob("chunk_*.csv"))], ignore_index=True)
    ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
    # centroid_norm_raw = |raw centroid|, the August study's "mean vector length"; under question-centring the panel's
    # own centroid_norm is the distance to the default centroid, which is almost the target (Spearman -0.94 at L25)
    pos = pd.DataFrame(dict(role=ax["roles"], axis_proj=ax["axis_proj"], response_chars=ax["response_chars"],
                            centroid_norm_raw=np.linalg.norm(ax["centroids"].astype(np.float64), axis=1)))
    df = df.merge(pos, on="role", how="left")
    sil = D.OUT / f"silhouette_{a.view}_L{a.layer}.csv"
    if sil.exists():
        df = df.merge(pd.read_csv(sil), on=["role", "centring"], how="left")
    f = D.OUT / f"panel_{a.view}_L{a.layer}.csv"
    df.to_csv(f, index=False)
    print("wrote", f, df.shape, "roles", df.role.nunique())


def silhouette(a, n_per=100):
    """Silhouette of each persona's points against all other personas, in a global PCA-50 of a pooled subsample."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples
    meta = D.metadata(a.layer)
    A = D.open_layer(a.layer, a.view)
    roles = D.roles(meta)
    Xd, _, _ = D.cloud(A, meta, D.DEFAULT)
    rng = np.random.default_rng(SEED)
    rows = []
    for centring in ("q", "raw"):
        pts, lab = [], []
        for r in roles:
            X, _, _ = D.cloud(A, meta, r)
            if centring == "q":
                X = X - Xd
            pick = rng.choice(len(X), n_per, replace=False)
            pts.append(X[pick].astype(np.float32)); lab += [r] * n_per
        Z = PCA(n_components=D_PCA, random_state=SEED).fit_transform(np.vstack(pts).astype(np.float64))
        s = silhouette_samples(Z, np.array(lab))
        df = pd.DataFrame(dict(role=lab, s=s)).groupby("role").s.mean()
        rows.append(pd.DataFrame(dict(role=df.index, centring=centring, silhouette_vs_all=df.values)))
        print(centring, "done", flush=True)
    f = D.OUT / f"silhouette_{a.view}_L{a.layer}.csv"
    pd.concat(rows).to_csv(f, index=False)
    print("wrote", f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="avg")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=69)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--silhouette", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--roles", default="", help="comma-separated roles for a smoke test")
    a = ap.parse_args(argv)
    if a.collect:
        collect(a)
    elif a.silhouette:
        silhouette(a)
    elif a.roles:
        meta = D.metadata(a.layer); A = D.open_layer(a.layer, a.view)
        ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
        Xd, _, _ = D.cloud(A, meta, D.DEFAULT); Xd = Xd.astype(np.float64)
        Qd = ax["default_qmeans"].astype(np.float64)
        for r in a.roles.split(","):
            X, instr, quest = D.cloud(A, meta, r); X = X.astype(np.float64)
            Qp = np.stack([X[quest == q].mean(0) for q in np.unique(quest)])
            for c in ("q", "raw"):
                m = cloud_metrics(X - Xd if c == "q" else X, instr, quest, ax, c, np.random.default_rng(SEED), (Qp, Qd))
                print(r, c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    else:
        run_chunk(a)


if __name__ == "__main__":
    main()
