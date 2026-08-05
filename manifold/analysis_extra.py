"""Post-hoc (exploratory) structure analysis added after the 14-03 run, to answer:
  - what is the intrinsic dimension of the role manifold? (we ASSUMED k=3)
  - how many manifolds N are there, decided data-drivenly (spectral eigengap +
    persistence over the cosine-threshold sweep), not by one hand-picked tau?
  - an MST 'skeleton' that visibly connects every centroid, and a 3-D intrinsic-
    coordinate 'volume' view of the manifold.

Reuses the saved role cloud only. Writes figs + a markdown note into a run dir.
Nothing here changes the H1 verdict; all outputs are exploratory.

Usage:
    .venv/bin/python -m manifold.analysis_extra output/manifold_h1-2/2026-07-21T14-03
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from manifold_persona.runlog import save_fig  # noqa: E402

from . import pipeline as P  # noqa: E402


def _save(fig, path):
    save_fig(fig, path)          # dpi=300


# --------------------------------------------------------------------------- #
# 1. Intrinsic dimension — answers "how do we know it's 3-D?" (we don't; measure)
# --------------------------------------------------------------------------- #
def intrinsic_dimension(cloud, out):
    import skdim
    ests = {"TwoNN": skdim.id.TwoNN, "MLE": skdim.id.MLE, "lPCA": skdim.id.lPCA}
    res = {}
    for name, cls in ests.items():
        res[name] = {
            "role_means": float(cls().fit(cloud.role_means).dimension_),
            "raw_points": float(cls().fit(cloud.raw).dimension_),
        }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ests)); w = 0.38
    ax.bar(x - w/2, [res[n]["role_means"] for n in ests], w, label="role means (the manifold)",
           color="#2e7d32")
    ax.bar(x + w/2, [res[n]["raw_points"] for n in ests], w, label="raw points (+within-role noise)",
           color="#90caf9")
    ax.axhline(3, ls="--", color="k", alpha=0.6, label="assumed k=3")
    ax.set_xticks(x); ax.set_xticklabels(list(ests))
    ax.set_ylabel("estimated intrinsic dimension")
    ax.set_title("Intrinsic dimension of the role manifold (ambient = PCA-50)")
    ax.legend(fontsize=8)
    _save(fig, out / "fig12_intrinsic_dimension.png")
    return res


# --------------------------------------------------------------------------- #
# 2. Spectral eigengap — data-driven N from the cosine graph
# --------------------------------------------------------------------------- #
def spectral_eigengap(cloud, out, knn=10, n_show=15):
    C = cloud.role_means
    S = 1.0 - squareform(pdist(P.unit_normalize(C), metric="cosine"))   # cosine sim
    np.fill_diagonal(S, 0.0)
    # symmetric kNN graph (keep each node's top-knn neighbours, symmetrise)
    W = np.zeros_like(S)
    idx = np.argsort(-S, axis=1)[:, :knn]
    for i in range(len(C)):
        W[i, idx[i]] = np.clip(S[i, idx[i]], 0, None)
    W = np.maximum(W, W.T)
    d = W.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(np.clip(d, 1e-12, None)))
    L = np.eye(len(C)) - Dinv @ W @ Dinv               # normalised Laplacian
    ev = np.sort(np.linalg.eigvalsh(L))[:n_show]
    gaps = np.diff(ev)
    n_est = int(np.argmax(gaps[:8]) + 1)               # eigengap heuristic (search small N)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, n_show + 1), ev, "o-", color="#1565c0")
    ax.axvline(n_est + 0.5, ls="--", color="#d1495b",
               label=f"largest eigengap → N≈{n_est}")
    ax.set_xlabel("eigenvalue index (sorted)"); ax.set_ylabel("Laplacian eigenvalue")
    ax.set_title(f"Spectral eigengap on the cosine {knn}-NN graph of role means")
    ax.legend()
    _save(fig, out / "fig10_spectral_eigengap.png")
    return {"eigenvalues": ev.tolist(), "n_estimate": n_est, "knn": knn}


# --------------------------------------------------------------------------- #
# 3. Persistence over the cosine-threshold sweep — how stable is N?
# --------------------------------------------------------------------------- #
def persistence_sweep(cloud, out):
    taus = np.linspace(-0.2, 0.95, 60)
    nman, nsingle = [], []
    for t in taus:
        lab = P.cosine_components(cloud.role_means, float(t))
        sizes = np.bincount(lab)
        nman.append(int((sizes >= 3).sum()))
        nsingle.append(int((sizes < 3).sum()))
    # most-persistent N (widest τ interval at a constant count)
    from itertools import groupby
    runs = [(k, sum(1 for _ in g)) for k, g in groupby(nman)]
    most = max(runs, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(taus, nman, "-", color="#2e7d32", lw=2, label="# manifolds (≥3 roles)")
    ax.plot(taus, nsingle, "-", color="#bbb", lw=2, label="# roles left as singletons")
    ax.axvspan(-0.24, 0.24, color="#ffe082", alpha=0.4, label="preregistered τ range")
    ax.set_xlabel("cosine threshold τ"); ax.set_ylabel("count")
    ax.set_title("Persistence: how #manifolds and dropout move with τ\n"
                 f"(N={most[0]} is the most persistent count, over "
                 f"{most[1]/len(taus)*100:.0f}% of the swept τ range)")
    ax.legend(fontsize=8)
    _save(fig, out / "fig11_components_vs_tau.png")
    return {"most_persistent_N": most[0], "persistence_width_frac": most[1] / len(taus)}


# --------------------------------------------------------------------------- #
# 4. MST skeleton — a line that visibly passes through EVERY centroid
# --------------------------------------------------------------------------- #
def mst_skeleton(cloud, out):
    C = cloud.role_means
    Dm = squareform(pdist(P.unit_normalize(C), metric="cosine"))   # cosine distance
    mst = minimum_spanning_tree(csr_matrix(Dm)).tocoo()
    is_def = np.array([r == "default" for r in cloud.role_names])
    for dim, tag in [(2, "2d"), (3, "3d")]:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d") if dim == 3 else fig.add_subplot(111)
        coords = C[:, :dim]
        for i, j in zip(mst.row, mst.col):
            seg = np.stack([coords[i], coords[j]])
            ax.plot(*[seg[:, d] for d in range(dim)], color="#90a4ae", lw=0.7, alpha=0.8, zorder=1)
        ax.scatter(*[coords[:, d] for d in range(dim)], s=14, c="#1565c0", zorder=2)
        ax.scatter(*[coords[is_def, d] for d in range(dim)], marker="*", s=300,
                   facecolor="gold", edgecolor="k", zorder=3)
        ax.set_title(f"MST skeleton (cosine) over all 276 role centroids — PCA {tag.upper()}\n"
                     "every centroid is a node the skeleton passes through")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if dim == 3:
            ax.set_zlabel("PC3")
        _save(fig, out / f"fig09_mst_skeleton_{tag}.png")


# --------------------------------------------------------------------------- #
# 5. Intrinsic-coordinate 'volume' — the manifold unrolled into its own 3 dims
# --------------------------------------------------------------------------- #
def intrinsic_volume(cloud, out):
    from sklearn.manifold import Isomap
    emb = Isomap(n_components=3, n_neighbors=10).fit_transform(cloud.role_means)
    # colour by projection onto the assistant axis (default - mean), for orientation
    ax_dir = cloud.role_means[[r == "default" for r in cloud.role_names]].mean(0) - cloud.role_means.mean(0)
    proj = cloud.role_means @ (ax_dir / (np.linalg.norm(ax_dir) + 1e-9))
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=proj, cmap="coolwarm", s=18)
    fig.colorbar(p, ax=ax, shrink=0.6, label="projection on Assistant axis")
    ax.set_title("The role manifold 'unrolled' into 3 intrinsic coordinates (Isomap)\n"
                 "a genuine 3-D volume view — vs the PCA scatter which is a 50-D shadow")
    ax.set_xlabel("iso-1"); ax.set_ylabel("iso-2"); ax.set_zlabel("iso-3")
    _save(fig, out / "fig13_intrinsic_volume.png")


def umap_views(cloud, out):
    """UMAP versions of the role-scatter (fig02/03) and the MST skeleton (fig09).
    Dimension-agnostic: no k assumption. Coloured by Assistant-axis projection
    (continuous, meaningful) since the preregistered threshold gives one
    component (colouring by component would be monochrome)."""
    import umap
    C = cloud.role_means
    is_def = np.array([r == "default" for r in cloud.role_names])
    ax_dir = C[is_def].mean(0) - C.mean(0)
    proj = C @ (ax_dir / (np.linalg.norm(ax_dir) + 1e-9))
    Dm = squareform(pdist(P.unit_normalize(C), metric="cosine"))
    mst = minimum_spanning_tree(csr_matrix(Dm)).tocoo()
    for nd, tag in [(2, "2d"), (3, "3d")]:
        emb = umap.UMAP(n_components=nd, random_state=0,
                        n_neighbors=15, min_dist=0.1).fit_transform(C)
        # fig02u: coloured scatter
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d") if nd == 3 else fig.add_subplot(111)
        p = ax.scatter(*[emb[:, d] for d in range(nd)], c=proj, cmap="coolwarm", s=18)
        ax.scatter(*[emb[is_def, d] for d in range(nd)], marker="*", s=320,
                   facecolor="gold", edgecolor="k", zorder=6)
        fig.colorbar(p, ax=ax, shrink=0.6, label="projection on Assistant axis")
        ax.set_title(f"UMAP {tag.upper()} of 276 role means "
                     f"(one connected manifold; coloured by Assistant-axis proj.)")
        _save(fig, out / f"fig02u_roles_umap_{tag}.png")
        # fig09u: MST skeleton over UMAP coords
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d") if nd == 3 else fig.add_subplot(111)
        for i, j in zip(mst.row, mst.col):
            seg = np.stack([emb[i], emb[j]])
            ax.plot(*[seg[:, d] for d in range(nd)], color="#90a4ae", lw=0.7, alpha=0.8)
        ax.scatter(*[emb[:, d] for d in range(nd)], s=14, c="#1565c0")
        ax.scatter(*[emb[is_def, d] for d in range(nd)], marker="*", s=300,
                   facecolor="gold", edgecolor="k", zorder=6)
        ax.set_title(f"MST skeleton (cosine, 50-D) over UMAP {tag.upper()} — "
                     f"through every centroid")
        _save(fig, out / f"fig09u_mst_skeleton_umap_{tag}.png")


def main(run_dir: str):
    figs = Path(run_dir) / "figures"; figs.mkdir(parents=True, exist_ok=True)
    cloud = P.load_cloud(seed=0)
    print("== intrinsic dimension =="); idr = intrinsic_dimension(cloud, figs)
    for n, v in idr.items():
        print(f"   {n:6s} role_means={v['role_means']:.2f}  raw={v['raw_points']:.2f}")
    print("== spectral eigengap =="); sp = spectral_eigengap(cloud, figs)
    print("   eigenvalues[:8]:", [round(x, 3) for x in sp["eigenvalues"][:8]],
          "-> N≈", sp["n_estimate"])
    print("== persistence sweep =="); ps = persistence_sweep(cloud, figs)
    print("   most-persistent N =", ps["most_persistent_N"],
          f"(over {ps['persistence_width_frac']*100:.0f}% of the τ range)")
    print("== MST skeleton =="); mst_skeleton(cloud, figs)
    print("== intrinsic volume =="); intrinsic_volume(cloud, figs)
    print("== UMAP views (fig02u/fig09u) =="); umap_views(cloud, figs)

    note = Path(run_dir) / "POSTHOC-manifold-structure.md"
    note.write_text(_note(idr, sp, ps))
    print("wrote", note)


def _n_verdict(n_spec: int, n_pers: int) -> str:
    """What the two component-count estimates jointly say — read off their values."""
    if n_spec != n_pers:
        return (f"The two estimates **disagree**: the eigengap says N={n_spec} and the "
                f"persistence sweep says N={n_pers}. The component count is unresolved by "
                "this note, so neither a single manifold nor a fixed number of them is "
                "established here.")
    if n_spec == 1:
        return ("Both point at **N=1**: the roles are best described as one connected "
                "manifold, and any larger component count at high τ is over-fragmentation "
                "of that single manifold, not evidence of several.")
    return (f"Both point at **N={n_spec}**, so this run does *not* describe the roles as "
            f"one connected manifold — {n_spec} components are indicated by both estimates.")


def _note(idr, sp, ps):
    L = ["# Post-hoc: how many manifolds, and what dimension?\n",
         "_Exploratory. Added after the run to answer follow-up questions; does not "
         "affect the H1 verdict._\n",
         "## Intrinsic dimension (we ASSUMED k=3 — here it is measured)\n",
         "| estimator | role means (the manifold) | raw points (+noise) |",
         "|---|---|---|"]
    for n, v in idr.items():
        L.append(f"| {n} | {v['role_means']:.2f} | {v['raw_points']:.2f} |")
    L += ["", "The manifold's intrinsic dimension is estimated on the **role means** "
          "(the object we fit); the raw-point estimate is inflated by within-role noise. "
          "k=3 was a preregistered *assumption*; these numbers say whether it was "
          "reasonable.\n",
          "## How many manifolds N (data-driven)\n",
          f"- **Spectral eigengap** (cosine {sp['knn']}-NN graph): largest gap at "
          f"**N≈{sp['n_estimate']}**. First eigenvalues: "
          f"{[round(x,3) for x in sp['eigenvalues'][:6]]}.",
          f"- **Persistence over τ**: the most stable component count is "
          f"**N={ps['most_persistent_N']}**, holding over {ps['persistence_width_frac']*100:.0f}% "
          f"of the swept τ range.",
          "", _n_verdict(int(sp["n_estimate"]), int(ps["most_persistent_N"])) + "\n",
          "## Figures\n",
          "- `fig09_mst_skeleton_2d/3d.png` — MST skeleton through every centroid.",
          "- `fig10_spectral_eigengap.png` — Laplacian spectrum + eigengap.",
          "- `fig11_components_vs_tau.png` — persistence of N across τ.",
          "- `fig12_intrinsic_dimension.png` — ID estimates vs the assumed k=3.",
          "- `fig13_intrinsic_volume.png` — the manifold unrolled into 3 intrinsic dims (Isomap)."]
    return "\n".join(L)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "output/manifold_h1-2/2026-07-21T14-03")
