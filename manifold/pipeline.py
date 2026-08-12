"""Plan-1 pipeline: constructions, reconstruction metric, permutation null.

Implements the plan `plans/2026-07-21-role-manifold-reconstruction.md`.
Everything downstream of the saved role point cloud; no model, no re-extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from manifold_persona.common import load_points  # the study's loader/aggregator

from .tps import SplineManifold, reconstruction

D_AMBIENT = 50          # PCA ambient reduction (plan: D=50)
K_INTRINSIC = 3         # TPS intrinsic_dim (plan: k=3 primary, k=2 robustness)
MIN_COMPONENT = 3       # a component needs >=3 anchors to define a surface


# --------------------------------------------------------------------------- #
# Data                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Cloud:
    raw: np.ndarray              # (N, D) PCA-reduced raw points
    roles: np.ndarray            # (N,) role label per raw point
    role_names: list             # sorted unique roles
    role_means: np.ndarray       # (R, D) mean per role (in PCA space)
    global_mean: np.ndarray      # (D,) mean of raw points (fixed TSS anchor)
    pca: PCA
    layer: int = -1              # hidden_states index actually read
    manifest: dict = field(default_factory=dict)   # source cloud's manifest.json


def load_cloud(view: str = "prompt_avg", layer: int | None = None,
               d_ambient: int = D_AMBIENT, seed: int = 0) -> Cloud:
    """Load raw role cloud, PCA-reduce to d_ambient, and derive role means.

    PCA is fit once on the raw points (labels irrelevant), so the ambient space
    is identical across permutations.
    """
    Xraw, meta, manifest = load_points(view=view, layer=layer, aggregate="none")
    eff_layer = layer if layer is not None else manifest.get("primary_layer", -1)
    Xraw = np.asarray(Xraw, dtype=np.float64)
    pca = PCA(n_components=d_ambient, random_state=seed)
    raw = pca.fit_transform(Xraw)                       # (N, d)
    roles = meta["role"].to_numpy()
    role_names = sorted(pd.unique(roles).tolist())
    idx = {r: i for i, r in enumerate(role_names)}
    role_means = np.zeros((len(role_names), raw.shape[1]))
    for r in role_names:
        role_means[idx[r]] = raw[roles == r].mean(0)
    return Cloud(raw=raw, roles=roles, role_names=role_names,
                 role_means=role_means, global_mean=raw.mean(0), pca=pca,
                 layer=eff_layer, manifest=manifest)


# --------------------------------------------------------------------------- #
# Manifold fit through a set of anchors                                         #
# --------------------------------------------------------------------------- #
def fit_manifold(anchors: np.ndarray, k: int = K_INTRINSIC,
                 seed: int = 0) -> SplineManifold:
    """Fit a TPS surface through `anchors` using unsupervised PCA intrinsic
    coordinates (control_points = PCA(anchors, k)). This is the deliberate,
    documented extension over causalab, which is *handed* intrinsic coords by a
    causal model; roles have none, so we derive them from the data itself."""
    kk = min(k, anchors.shape[1], anchors.shape[0] - 1)
    control = PCA(n_components=kk, random_state=seed).fit_transform(anchors)
    return SplineManifold(control_points=control, target_points=anchors)


# --------------------------------------------------------------------------- #
# Constructions                                                                 #
# --------------------------------------------------------------------------- #
def unit_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def cosine_percentile_thresholds(role_means: np.ndarray,
                                 pcts=(25, 50, 75)) -> dict:
    """tau_1<tau_2<tau_3 at cosine-sim percentiles of the role-mean pairs."""
    U = unit_normalize(role_means)
    sims = 1.0 - pdist(U, metric="cosine")     # cosine similarities of all pairs
    return {p: float(np.percentile(sims, p)) for p in pcts}


def cosine_components(role_means: np.ndarray, tau: float) -> np.ndarray:
    """Connected components of the graph {edge if cos-sim >= tau}. Returns a
    component label per role."""
    U = unit_normalize(role_means)
    S = 1.0 - squareform(pdist(U, metric="cosine"))
    A = (S >= tau).astype(int)
    np.fill_diagonal(A, 0)
    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    return labels


@dataclass
class ConstructionResult:
    name: str
    n_anchors: int
    n_manifolds: int
    r2: float
    nre: float
    exploratory: bool
    details: dict = field(default_factory=dict)


def _score_components(cloud: Cloud, role_component: np.ndarray, k: int) -> tuple:
    """Fit one manifold per component (>=MIN_COMPONENT roles), score that
    component's raw points against it, aggregate SSR/TSS over all points.
    Roles in tiny components contribute their raw points' distance-to-nearest
    role-mean (degenerate 'manifold' = a point) so nothing is silently dropped.
    """
    role_idx = {r: i for i, r in enumerate(cloud.role_names)}
    point_comp = np.array([role_component[role_idx[r]] for r in cloud.roles])
    ssr, n_manifolds = 0.0, 0
    tss = float(np.sum(np.sum((cloud.raw - cloud.global_mean) ** 2, axis=1)))
    for c in np.unique(role_component):
        roles_in = [cloud.role_names[i] for i in np.where(role_component == c)[0]]
        pts_mask = point_comp == c
        pts = cloud.raw[pts_mask]
        if len(roles_in) >= MIN_COMPONENT:
            anchors = cloud.role_means[[role_idx[r] for r in roles_in]]
            mani = fit_manifold(anchors, k=k)
            res = reconstruction(pts, mani, cloud.global_mean)
            ssr += res.ssr
            n_manifolds += 1
        else:  # degenerate: nearest anchor point
            anchors = cloud.role_means[[role_idx[r] for r in roles_in]]
            d = np.min(((pts[:, None, :] - anchors[None]) ** 2).sum(-1), axis=1)
            ssr += float(d.sum())
    nre = ssr / tss
    return nre, 1.0 - nre, n_manifolds


def construction_C_role(cloud: Cloud, k: int = K_INTRINSIC) -> ConstructionResult:
    """DECIDER (0b): one surface through all role means, scored on raw points."""
    mani = fit_manifold(cloud.role_means, k=k)
    res = reconstruction(cloud.raw, mani, cloud.global_mean)
    return ConstructionResult("C_role", cloud.role_means.shape[0], 1,
                              res.r2, res.nre, exploratory=False)


def construction_C_tau(cloud: Cloud, tau: float, name: str,
                       k: int = K_INTRINSIC) -> ConstructionResult:
    labels = cosine_components(cloud.role_means, tau)
    nre, r2, n_manifolds = _score_components(cloud, labels, k)
    sizes = np.bincount(labels).tolist()
    return ConstructionResult(name, cloud.role_means.shape[0], n_manifolds,
                              r2, nre, exploratory=True,
                              details={"tau": tau, "component_sizes": sizes})


def construction_C_raw(cloud: Cloud, k: int = K_INTRINSIC, n_folds: int = 5,
                       anchor_cap: int = 600, seed: int = 0) -> ConstructionResult:
    """0a robustness: raw points as anchors, 5-fold CV. Anchors within a fold are
    capped (random subsample) so the TPS solve stays tractable — a documented
    approximation, robustness only."""
    rng = np.random.default_rng(seed)
    N = cloud.raw.shape[0]
    fold = rng.integers(0, n_folds, size=N)
    ssr, tss = 0.0, 0.0
    for f in range(n_folds):
        train = cloud.raw[fold != f]
        test = cloud.raw[fold == f]
        if len(train) > anchor_cap:
            sel = rng.choice(len(train), anchor_cap, replace=False)
            anchors = train[sel]
        else:
            anchors = train
        mani = fit_manifold(anchors, k=k)
        res = reconstruction(test, mani, cloud.global_mean)
        ssr += res.ssr
        tss += res.tss
    nre = ssr / tss
    return ConstructionResult("C_raw", N, n_folds, 1.0 - nre, nre,
                              exploratory=True, details={"anchor_cap": anchor_cap})


def pca_plane_r2(cloud: Cloud, k: int = K_INTRINSIC) -> float:
    """Context baseline: flat k-dim PCA reconstruction R^2 of the raw points."""
    p = PCA(n_components=k, random_state=0).fit(cloud.raw)
    recon = p.inverse_transform(p.transform(cloud.raw))
    ssr = float(np.sum((cloud.raw - recon) ** 2))
    tss = float(np.sum((cloud.raw - cloud.global_mean) ** 2))
    return 1.0 - ssr / tss


# --------------------------------------------------------------------------- #
# Permutation null (negative control) for the decider                          #
# --------------------------------------------------------------------------- #
def permutation_null(cloud: Cloud, n_perms: int = 100, k: int = K_INTRINSIC,
                     seed: int = 0) -> np.ndarray:
    """Null R^2 distribution: shuffle role labels over raw points, rebuild role
    means, refit C_role, rescore. Points & TSS fixed; only the surface moves."""
    rng = np.random.default_rng(seed)
    R = len(cloud.role_names)
    counts = pd.Series(cloud.roles).value_counts()[cloud.role_names].to_numpy()
    r2s = np.zeros(n_perms)
    for p in range(n_perms):
        perm = rng.permutation(cloud.raw.shape[0])
        Xp = cloud.raw[perm]
        # rebuild "role" means from contiguous shuffled blocks matching real sizes
        means = np.zeros((R, cloud.raw.shape[1]))
        start = 0
        for i, c in enumerate(counts):
            means[i] = Xp[start:start + c].mean(0)
            start += c
        mani = fit_manifold(means, k=k)
        res = reconstruction(cloud.raw, mani, cloud.global_mean)
        r2s[p] = res.r2
    return r2s


def coordinate_null(cloud: Cloud, n_draws: int = 100, k: int = K_INTRINSIC,
                    seed: int = 0) -> np.ndarray:
    """Structure-preserving null R^2 distribution for the decider.

    Each draw permutes every coordinate of the REAL role means independently
    across roles (a fresh permutation per coordinate per draw), reattaches each
    role's real within-role residuals, refits C_role and rescores. That destroys
    the joint (manifold) structure of the role means but keeps their
    per-coordinate marginals, the role-mean spread and the within-role noise.

    `permutation_null` cannot do this: its fake role means average ~25 randomly
    chosen points, so they shrink ~4x toward the global mean and the null only
    measures between-role spread. A surface fitted to correctly-scaled but
    structureless anchors already reaches the real R^2, so this is the null the
    decider must beat.
    """
    rng = np.random.default_rng(seed)
    role_idx = {r: i for i, r in enumerate(cloud.role_names)}
    point_role = np.array([role_idx[r] for r in cloud.roles])
    resid = cloud.raw - cloud.role_means[point_role]     # real within-role noise
    R, D = cloud.role_means.shape
    r2s = np.zeros(n_draws)
    for p in range(n_draws):
        means = np.empty_like(cloud.role_means)
        for j in range(D):
            means[:, j] = cloud.role_means[rng.permutation(R), j]
        raw = resid + means[point_role]                  # points follow their anchor
        mani = fit_manifold(means, k=k)
        res = reconstruction(raw, mani, raw.mean(0))
        r2s[p] = res.r2
    return r2s


def covgauss_null(cloud: Cloud, n_draws: int = 100, k: int = K_INTRINSIC,
                  seed: int = 0) -> np.ndarray:
    """Covariance-matched Gaussian null R^2 distribution.

    Each draw replaces the role means with samples from N(mu, Sigma) — the
    Gaussian matching the REAL role means' mean and covariance — reattaches each
    role's real within-role residuals, refits C_role and rescores. That keeps
    ALL linear structure of the role means (their spread and every low-dim
    subspace concentration) and destroys only what a Gaussian cannot carry:
    curvature. `coordinate_null` destroys the linear structure too, so it tests
    "any joint structure"; this null is the rung between it and the real data,
    and beating it is what the "curved manifold" claim specifically requires
    (Theiler-style constrained surrogate; cf. Elsayed & Cunningham 2017).
    """
    rng = np.random.default_rng(seed)
    role_idx = {r: i for i, r in enumerate(cloud.role_names)}
    point_role = np.array([role_idx[r] for r in cloud.roles])
    resid = cloud.raw - cloud.role_means[point_role]     # real within-role noise
    R = cloud.role_means.shape[0]
    mu = cloud.role_means.mean(0)
    cov = np.cov(cloud.role_means, rowvar=False)
    r2s = np.zeros(n_draws)
    for p in range(n_draws):
        means = rng.multivariate_normal(mu, cov, size=R)
        raw = resid + means[point_role]                  # points follow their anchor
        mani = fit_manifold(means, k=k)
        res = reconstruction(raw, mani, raw.mean(0))
        r2s[p] = res.r2
    return r2s


def permutation_null_tau(cloud: Cloud, tau: float, n_perms: int = 40,
                         k: int = K_INTRINSIC, seed: int = 0) -> np.ndarray:
    """Null R^2 for a C_tau construction: shuffle role labels over raw points,
    rebuild role means, recompute cosine components at the *same* tau, rescore."""
    rng = np.random.default_rng(seed)
    R = len(cloud.role_names)
    counts = pd.Series(cloud.roles).value_counts()[cloud.role_names].to_numpy()
    r2s = np.zeros(n_perms)
    for p in range(n_perms):
        perm = rng.permutation(cloud.raw.shape[0])
        Xp = cloud.raw[perm]
        means = np.zeros((R, cloud.raw.shape[1]))
        start = 0
        for i, c in enumerate(counts):
            means[i] = Xp[start:start + c].mean(0)
            start += c
        # relabel the raw points to the shuffled blocks, so points follow means
        shuffled_roles = np.empty(cloud.raw.shape[0], dtype=object)
        start = 0
        for i, c in enumerate(counts):
            shuffled_roles[perm[start:start + c]] = cloud.role_names[i]
            start += c
        tmp = Cloud(raw=cloud.raw, roles=shuffled_roles,
                    role_names=cloud.role_names, role_means=means,
                    global_mean=cloud.global_mean, pca=cloud.pca)
        labels = cosine_components(means, tau)
        nre, r2, _ = _score_components(tmp, labels, k)
        r2s[p] = r2
    return r2s


def separation_stats(real_r2: float, null_r2: np.ndarray) -> dict:
    p = float((np.sum(null_r2 >= real_r2) + 1) / (len(null_r2) + 1))
    z = float((real_r2 - null_r2.mean()) / (null_r2.std() + 1e-12))
    nre_real = 1 - real_r2
    nre_null_med = 1 - float(np.median(null_r2))
    rel_reduction = (nre_null_med - nre_real) / (nre_null_med + 1e-12)
    return {"p": p, "z": z, "r2_gap": real_r2 - float(np.median(null_r2)),
            "rel_reduction": float(rel_reduction),
            "null_median": float(np.median(null_r2)),
            "null_5pct": float(np.percentile(null_r2, 5))}


# --------------------------------------------------------------------------- #
# Positive control                                                              #
# --------------------------------------------------------------------------- #
def positive_control(d_ambient: int = D_AMBIENT, n_roles: int = 276,
                     per_role: int = 25, k: int = K_INTRINSIC,
                     target_radius: float = 11.0, noise: float = 1.0,
                     seed: int = 0) -> dict:
    """Synthetic curved **k-dimensional** manifold in d_ambient dims + isotropic
    Gaussian noise, both calibrated to the real data (target_radius = role-mean
    spread, noise = within-role per-dim RMS).

    The control's intrinsic dim MATCHES the pipeline's k, so it is a fair test of
    the k-dim TPS at the data's signal-to-noise: the pipeline MUST return R^2
    above its permutation null by >= the floor, else STOP (a null role result
    would be uninterpretable, and even the SUPPORTED result loses its guarantee).

    The manifold is the image of a smooth nonlinear map of k intrinsic coordinates
    (trig + low-order polynomial features) into a random d_ambient embedding —
    genuinely curved and genuinely k-dimensional.
    """
    rng = np.random.default_rng(seed)
    U = rng.random((n_roles, k))                       # k intrinsic coords in [0,1]
    feats = [U[:, j] for j in range(k)]
    for j in range(k):
        feats += [np.cos(np.pi * (1 + 0.5 * j) * U[:, j]),
                  np.sin(np.pi * (1 + 0.5 * j) * U[:, j])]
    for a in range(k):                                 # pairwise products (curvature)
        for b in range(a + 1, k):
            feats.append(U[:, a] * U[:, b])
    F = np.stack(feats, axis=1)                        # (n_roles, m)
    B = rng.standard_normal((F.shape[1], d_ambient))
    centers = F @ B
    centers -= centers.mean(0)
    cur_rad = np.sqrt(np.mean(np.sum(centers ** 2, axis=1)))
    centers *= target_radius / (cur_rad + 1e-12)       # match real role-mean spread
    roles, pts = [], []
    for i in range(n_roles):
        roles += [f"r{i}"] * per_role
        pts.append(centers[i] + noise * rng.standard_normal((per_role, d_ambient)))
    raw = np.vstack(pts)
    roles = np.array(roles)
    role_names = [f"r{i}" for i in range(n_roles)]
    means = np.vstack([raw[roles == r].mean(0) for r in role_names])
    cloud = Cloud(raw=raw, roles=roles, role_names=role_names, role_means=means,
                  global_mean=raw.mean(0), pca=None)
    dec = construction_C_role(cloud, k=k)
    null = permutation_null(cloud, n_perms=50, k=k, seed=seed)
    stats = separation_stats(dec.r2, null)
    return {"r2": dec.r2, "null": null, "stats": stats, "cloud": cloud,
            "centers": centers, "theta": U[:, 0]}
