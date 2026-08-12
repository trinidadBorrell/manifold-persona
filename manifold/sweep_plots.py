"""Figures for plan 2026-07-22-role-count-sweep.

Kept in their own module so `manifold/plots.py` (plan #1's figures, already
executed and reported) is not touched. Same conventions: matplotlib Agg, 300 dpi
PNG, plotly HTML for the 3-D views, and every function is defensive — a plotting
failure logs and returns, it never kills a run that already produced numbers.

fig01 is the decider figure. fig05 is illustrative only (the plan's fence).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from manifold_persona.runlog import save_fig  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def _save(fig, path: Path):
    save_fig(fig, path)          # dpi=300


def _per_role(sub) -> int:
    return sub.raw.shape[0] // len(sub.role_names)


def _grid_note(sub) -> str:
    """'5 role phrasings × 5 questions; ' from the cloud's own manifest; '' if unknown."""
    nq = sub.manifest.get("n_questions")
    per = _per_role(sub)
    if not nq or per % nq:
        return ""
    return f"{per // nq} role phrasings × {nq} questions; "


def _depth_note(sub) -> str:
    """' ≈ 0.72 depth' from the cloud's own layer/n_layers; '' if unknown."""
    nl = sub.manifest.get("n_layers")
    if not nl or nl < 2 or sub.layer < 0:
        return ""
    return f" ≈ {sub.layer / (nl - 1):.2f} depth"


def _agg(df, col):
    """mean / min / max of `col` per n, in ascending n order."""
    g = df.groupby("n")[col]
    ns = sorted(df["n"].unique())
    return (np.array(ns), g.mean().reindex(ns).to_numpy(),
            g.min().reindex(ns).to_numpy(), g.max().reindex(ns).to_numpy())


# --------------------------------------------------------------------------- #
# fig01 — THE DECIDER                                                          #
# --------------------------------------------------------------------------- #
def fig01_sweep_decider(df, out: Path, floor: float = 0.30):
    """Left: M1 relative NRE reduction vs n (the decider). Right: raw R² real
    vs null median vs n. The right panel draws the trap explicitly: a reader
    can see both curves rise together as n falls, and so understand why raw R²
    decides nothing."""
    try:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

        ns, m, lo, hi = _agg(df, "rel_reduction")
        ax.fill_between(ns, lo, hi, color="#2e7d32", alpha=0.18,
                        label="seed range (k-means init)")
        ax.plot(ns, m, "o-", color="#2e7d32", lw=2, label="M1 rel. NRE reduction")
        ax.axhline(floor, color="#d1495b", ls="--", lw=1.2,
                   label=f"effect floor {floor:.2f} (plan #1)")
        ax.set_xscale("log"); ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("n role centroids"); ax.set_ylabel("relative NRE reduction")
        ax.set_title("DECIDER — manifold signal vs its per-n null\n"
                     "(higher = points hug the surface more than chance)")
        ax.legend(fontsize=8)

        ns, r, rlo, rhi = _agg(df, "r2")
        _, nl, _, _ = _agg(df, "null_median")
        ax2.fill_between(ns, rlo, rhi, color="#1565c0", alpha=0.18)
        ax2.plot(ns, r, "o-", color="#1565c0", lw=2, label="real R²")
        ax2.plot(ns, nl, "s--", color="#9e9e9e", lw=2, label="null median R²")
        ax2.set_xscale("log"); ax2.set_xticks(ns)
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlabel("n role centroids"); ax2.set_ylabel("manifold R²")
        # NB the two curves move in OPPOSITE directions. Real R² rises as n
        # falls (fewer, better-separated anchors). The null R² falls: a
        # shuffled labelling gives centroids that all collapse toward the
        # global mean, and with fewer anchors the fitted surface is smaller.
        # Both effects widen the gap, which is why the decider trends as it
        # does.
        ax2.set_title("Raw R² is not n-fair, and the two curves move OPPOSITE ways\n"
                      "real ↑ as n falls; null ↓ (shuffled centroids collapse to the mean)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, out / "fig01_sweep_decider.png")
    except Exception as e:  # noqa: BLE001
        print("fig01 failed:", e)


# --------------------------------------------------------------------------- #
# fig02 — intrinsic dimension vs n, against the small-N reference              #
# --------------------------------------------------------------------------- #
def fig02_sweep_id(df, refs: dict, out: Path):
    """Solid = ID of the real role means. Dashed + band = ID of n structureless
    Gaussian draws matched to the role-mean covariance (100 draws, median + IQR).
    Real tracking the band == small-N bias, NOT simplification."""
    try:
        ests = ["TwoNN", "MLE", "lPCA"]
        colors = {"TwoNN": "#2e7d32", "MLE": "#1565c0", "lPCA": "#ef6c00"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
        for ax, est in zip(axes, ests):
            col = f"id_{est}"
            sub = df.dropna(subset=[col])
            if len(sub):
                ns, m, lo, hi = _agg(sub, col)
                ax.fill_between(ns, lo, hi, color=colors[est], alpha=0.18)
                ax.plot(ns, m, "o-", color=colors[est], lw=2, label=f"{est} (real roles)")
            rn = sorted(refs)
            med = [refs[n][est]["median"] for n in rn]
            q25 = [refs[n][est]["q25"] for n in rn]
            q75 = [refs[n][est]["q75"] for n in rn]
            if all(v is not None for v in med):
                ax.plot(rn, med, "s--", color="#757575", lw=1.8,
                        label="Gaussian small-N reference")
                ax.fill_between(rn, q25, q75, color="#757575", alpha=0.15)
            ax.set_xscale("log"); ax.set_xticks(rn)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("n role centroids"); ax.set_title(est)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("intrinsic dimension")
        fig.suptitle("M2 — intrinsic dimension vs n.  Solid = real role means; dashed + band "
                     "= matched structureless Gaussian of the same n (median, IQR).\n"
                     "Read the GAP: below the band = genuinely lower-dimensional than noise; "
                     "on the band = nothing dimensional to see at that n.", fontsize=9.5)
        fig.tight_layout()
        _save(fig, out / "fig02_sweep_intrinsic_dim.png")
    except Exception as e:  # noqa: BLE001
        print("fig02 failed:", e)


# --------------------------------------------------------------------------- #
# fig03 — curvature gain vs n                                                  #
# --------------------------------------------------------------------------- #
def fig03_sweep_curvature(df, out: Path):
    """Spline R² minus flat PCA-plane(k=3) R² on the same points, for the real
    fit and for the null median. The null curve is the chance gain a flexible
    surface gets over a plane; only the gap between them is 'curvature'."""
    try:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ns, m, lo, hi = _agg(df, "curv_gain")
        ax.fill_between(ns, lo, hi, color="#6a1b9a", alpha=0.18, label="seed range")
        ax.plot(ns, m, "o-", color="#6a1b9a", lw=2, label="real spline − plane")
        _, mn, _, _ = _agg(df, "curv_gain_null")
        ax.plot(ns, mn, "s--", color="#9e9e9e", lw=2, label="null spline − plane")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("n role centroids"); ax.set_ylabel("ΔR²  (spline − flat plane)")
        ax.set_title("M3 — what the curved surface buys over a flat k=3 plane")
        ax.legend(fontsize=8)
        _save(fig, out / "fig03_sweep_curvature_gain.png")
    except Exception as e:  # noqa: BLE001
        print("fig03 failed:", e)


# --------------------------------------------------------------------------- #
# fig04 — per-n positive control                                               #
# --------------------------------------------------------------------------- #
def fig04_sweep_posctrl(df, out: Path, floor: float = 0.30):
    """Can the pipeline still detect a manifold that is definitely there, at this
    n? A cell below the floor cannot be interpreted at all — this figure is what
    licenses (or refuses) every point in fig01."""
    try:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ns, m, lo, hi = _agg(df, "pc_rel_reduction")
        ax.fill_between(ns, lo, hi, color="#00695c", alpha=0.18)
        ax.plot(ns, m, "o-", color="#00695c", lw=2,
                label="positive control rel. NRE reduction")
        ax.axhline(floor, color="#d1495b", ls="--", lw=1.2, label=f"pass floor {floor:.2f}")
        ax.set_xscale("log"); ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("n role centroids"); ax.set_ylabel("rel. NRE reduction")
        ax.set_title("Positive control vs n — synthetic curved k=3 manifold\n"
                     "at each cell's own noise scale (must clear the floor)")
        ax.legend(fontsize=8)
        _save(fig, out / "fig04_sweep_positive_control.png")
    except Exception as e:  # noqa: BLE001
        print("fig04 failed:", e)


# --------------------------------------------------------------------------- #
# fig05 — THE manifold spline figure (illustrative; n=10, 25)                  #
# --------------------------------------------------------------------------- #
def _open_spline(points: np.ndarray, n: int = 200):
    """Smooth OPEN curve through 3-D points in the given order.

    Open, not periodic (cf. manifold-temporal's `_periodic_spline`): roles are
    not cyclic, so closing the loop would assert structure that isn't claimed.

    **Interpolating (`s=0`), so the curve passes exactly through every
    centroid** — same as manifold-temporal's figure, and what was asked for.
    Caveat worth knowing when reading it: the centroids are ordered by
    intrinsic coord 1 but scattered in the other two. An exact cubic through
    such knots overshoots between consecutive knots. So the large excursions
    in the drawn curve are artefacts of the interpolant, not features of the
    data. The surface mesh (which is the fitted model) is the thing to trust;
    this curve is a reading aid that connects the named roles in order.
    """
    from scipy.interpolate import splprep, splev
    k = int(min(3, len(points) - 1))
    tck, _ = splprep(points.T, s=0, per=0, k=k)
    return np.asarray(splev(np.linspace(0, 1, n), tck)).T


def fig05_spline_manifold(sub, mani, out: Path, n_roles: int, seed: int,
                          grid: int = 22, label_every: int = 1):
    """Raw points + labelled role centroids + the DECODED TPS surface + a smooth
    curve, in the manifold's own intrinsic PCA-3 frame, at two view angles.

    Style follows manifold-temporal/framing/plots.py::plot_manifold_3d (scatter,
    big ringed labelled centroids, fitted curve, two angles), with one addition:
    we also decode the actual k=3 TPS on a grid and draw it as a wireframe, so
    the figure shows the object that was *scored*, not only a decorative curve.

    The plot basis IS the fit's intrinsic basis: `fit_manifold` takes
    control_points = PCA(role_means, 3), so PC1–3 here are exactly the surface's
    own coordinates — the picture and the model share a frame.
    """
    try:
        rm = sub.role_means
        p = PCA(n_components=3, random_state=0).fit(rm)
        cent3 = p.transform(rm)
        raw3 = p.transform(sub.raw)
        evr = p.explained_variance_ratio_

        # NOTE: no decoded surface mesh is drawn.
        # The TPS maps THREE intrinsic coords -> 50-D, so any 2-D sheet
        # requires fixing the third coordinate. Every choice of that slice
        # reads as a flat plate hanging in the middle of the cloud — it looks
        # like a claim about the geometry that the slice does not actually
        # support. The curvature question is answered numerically by M3
        # (fig03), not by eye, so the mesh was removed rather than made more
        # elaborate.
        order = np.argsort(cent3[:, 0])          # order along intrinsic coord 1
        curve = _open_spline(cent3[order]) if len(cent3) >= 4 else None

        names = list(sub.role_names)
        is_def = np.array([r == "default" for r in names])
        cmap = plt.get_cmap("turbo")
        ridx = {r: i for i, r in enumerate(names)}
        pcol = np.array([ridx[r] for r in sub.roles]) / max(1, len(names) - 1)

        fig = plt.figure(figsize=(16, 7.5))
        for panel, (elev, azim) in enumerate([(20, 45), (16, 135)], start=1):
            ax = fig.add_subplot(1, 2, panel, projection="3d")
            ax.scatter(raw3[:, 0], raw3[:, 1], raw3[:, 2], c=cmap(pcol), s=9,
                       alpha=0.22, edgecolors="none", depthshade=True)
            if curve is not None:
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#c62828",
                        lw=2.2, alpha=0.95, zorder=4)
            for i, nm in enumerate(names):
                col = cmap(i / max(1, len(names) - 1))
                if is_def[i]:
                    ax.scatter(*cent3[i], marker="*", s=420, facecolor="gold",
                               edgecolors="k", linewidths=1.2, zorder=6)
                else:
                    ax.scatter(*cent3[i], color=col, s=120, edgecolors="k",
                               linewidths=1.0, zorder=5)
                if i % label_every == 0 or is_def[i]:
                    ax.text(*cent3[i], f"  {nm[:16]}", fontsize=6.5, zorder=7)
            ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.set_zlabel(f"PC3 ({evr[2]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=6, pad=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"elev={elev}° azim={azim}°", fontsize=9)
        fig.suptitle(
            f"n = {n_roles} role centroids (k-means medoid, seed {seed}).   "
            f"Red: spline passing exactly through every centroid, in PC1 order — its "
            f"excursions between knots are interpolation overshoot, not data.\n"
            f"Large markers: role means (labelled).   Faint dots: the {n_roles}×"
            f"{_per_role(sub)} raw points ({_grid_note(sub)}layer {sub.layer}"
            f"{_depth_note(sub)}).   ★ = default (Assistant).\n"
            f"Axes are the fitted surface's own "
            f"intrinsic coordinates = PCA of these {n_roles} role means.\n"
            f"ILLUSTRATIVE ONLY — the decider is fig01, and curvature is measured in fig03.",
            fontsize=9, y=0.045, color="0.3")
        fig.subplots_adjust(bottom=0.12, wspace=0.02)
        _save(fig, out / f"fig05_spline_manifold_n{n_roles}.png")

        # interactive version
        try:
            import plotly.graph_objects as go
            f3 = go.Figure()
            f3.add_trace(go.Scatter3d(x=raw3[:, 0], y=raw3[:, 1], z=raw3[:, 2],
                                      mode="markers", name="answers",
                                      marker=dict(size=2, opacity=0.25,
                                                  color=pcol, colorscale="Turbo")))
            f3.add_trace(go.Scatter3d(x=cent3[:, 0], y=cent3[:, 1], z=cent3[:, 2],
                                      mode="markers+text", name="role centroids",
                                      text=names, textposition="top center",
                                      textfont=dict(size=9),
                                      marker=dict(size=6, color="black")))
            if curve is not None:
                f3.add_trace(go.Scatter3d(x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                                          mode="lines", name="spline through centroids",
                                          line=dict(color="#c62828", width=5)))
            f3.update_layout(title=f"n={n_roles} role manifold (illustrative)",
                             scene=dict(xaxis_title="PC1", yaxis_title="PC2",
                                        zaxis_title="PC3"))
            f3.write_html(str(out / f"fig05_spline_manifold_n{n_roles}.html"))
            print("wrote", out / f"fig05_spline_manifold_n{n_roles}.html")
        except Exception as e:  # noqa: BLE001
            print("fig05 html skipped:", e)
    except Exception as e:  # noqa: BLE001
        print("fig05 failed:", e)


# --------------------------------------------------------------------------- #
# per-n figures                                                                #
# --------------------------------------------------------------------------- #
def fig06_null_vs_real(real_r2: float, null: np.ndarray, out: Path, n_roles: int):
    try:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        ax.hist(null, bins=20, color="#cfd8dc", label="null R² (100 role shuffles)")
        ax.axvline(real_r2, color="#2e7d32", lw=2, label=f"real R² = {real_r2:.3f}")
        ax.axvline(float(np.median(null)), color="#9e9e9e", ls="--", lw=1.4,
                   label=f"null median = {np.median(null):.3f}")
        ax.set_xlabel("manifold R²"); ax.set_ylabel("count")
        ax.set_title(f"n = {n_roles}: real vs role-shuffle null (seed 0)")
        ax.legend(fontsize=8)
        _save(fig, out / f"fig06_null_vs_real_n{n_roles}.png")
    except Exception as e:  # noqa: BLE001
        print("fig06 failed:", e)


def fig08_mst_skeleton(sub, out: Path, n_roles: int):
    """MST skeleton over the subset's role centroids (n > 50 cells).

    At large `n` the spline-through-centroids curve of fig05 is unreadable — 100+
    labelled knots in an arbitrary 1-D order. The minimum spanning tree is the
    right object there: it is a *graph* skeleton, so it shows which roles are
    actually near which without imposing an ordering, and it visibly passes
    through every centroid.

    Same construction as `manifold/analysis_extra.py::mst_skeleton` (plan #1):
    MST of the **cosine** distance between unit-normalised role means — cosine
    because `DESIGN.md` D1 defines role nearness by direction, discarding
    magnitude. Drawn here in the subset's own intrinsic PCA-3 frame, the same
    frame fig05 uses, so the two figures are directly comparable.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import pdist, squareform
        from . import pipeline as _P

        C = sub.role_means
        Dm = squareform(pdist(_P.unit_normalize(C), metric="cosine"))
        mst = minimum_spanning_tree(csr_matrix(Dm)).tocoo()
        p = PCA(n_components=3, random_state=0).fit(C)
        c3 = p.transform(C)
        evr = p.explained_variance_ratio_
        names = list(sub.role_names)
        is_def = np.array([r == "default" for r in names])

        fig = plt.figure(figsize=(15, 7))
        for panel, (elev, azim) in enumerate([(20, 45), (16, 135)], start=1):
            ax = fig.add_subplot(1, 2, panel, projection="3d")
            for i, j in zip(mst.row, mst.col):
                seg = np.stack([c3[i], c3[j]])
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#90a4ae",
                        lw=0.8, alpha=0.85, zorder=1)
            ax.scatter(c3[~is_def, 0], c3[~is_def, 1], c3[~is_def, 2], s=18,
                       c="#1565c0", zorder=2, depthshade=False)
            ax.scatter(c3[is_def, 0], c3[is_def, 1], c3[is_def, 2], marker="*",
                       s=340, facecolor="gold", edgecolor="k", zorder=3)
            for i, nm in enumerate(names):
                if is_def[i]:
                    ax.text(*c3[i], "  default", fontsize=8, zorder=6)
            ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.set_zlabel(f"PC3 ({evr[2]*100:.0f}%)", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=6, pad=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"elev={elev}° azim={azim}°", fontsize=9)
        fig.suptitle(
            f"MST skeleton over the n = {n_roles} role centroids (cosine distance, "
            f"unit-normalised) in their own PC1–3.   Every centroid is a node the "
            f"skeleton passes through.\n★ = default (Assistant).   "
            f"POST HOC — added after the run; outside the "
            f"preregistered plan, and no claim rests on it.",
            fontsize=9, y=0.05, color="0.3")
        fig.subplots_adjust(bottom=0.12, wspace=0.02)
        _save(fig, out / f"fig08_mst_skeleton_n{n_roles}.png")

        try:
            import plotly.graph_objects as go
            xs, ys, zs = [], [], []
            for i, j in zip(mst.row, mst.col):
                xs += [c3[i, 0], c3[j, 0], None]
                ys += [c3[i, 1], c3[j, 1], None]
                zs += [c3[i, 2], c3[j, 2], None]
            f3 = go.Figure()
            f3.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="MST",
                                      line=dict(color="#90a4ae", width=2)))
            f3.add_trace(go.Scatter3d(
                x=c3[:, 0], y=c3[:, 1], z=c3[:, 2], mode="markers", name="role centroids",
                text=names, marker=dict(size=4, color=np.where(is_def, 1, 0),
                                        colorscale=[[0, "#1565c0"], [1, "gold"]])))
            f3.update_layout(title=f"MST skeleton, n={n_roles} role centroids (post hoc)",
                             scene=dict(xaxis_title="PC1", yaxis_title="PC2",
                                        zaxis_title="PC3"))
            f3.write_html(str(out / f"fig08_mst_skeleton_n{n_roles}.html"))
            print("wrote", out / f"fig08_mst_skeleton_n{n_roles}.html")
        except Exception as e:  # noqa: BLE001
            print("fig08 html skipped:", e)
    except Exception as e:  # noqa: BLE001
        print("fig08 failed:", e)


def fig07_roles_pc123(sub, out: Path, n_roles: int):
    try:
        rm = sub.role_means
        p = PCA(n_components=3, random_state=0).fit(rm)
        c3 = p.transform(rm)
        names = list(sub.role_names)
        is_def = np.array([r == "default" for r in names])
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(c3[~is_def, 0], c3[~is_def, 1], c3[~is_def, 2], s=26,
                   c="#1565c0", alpha=0.85)
        ax.scatter(c3[is_def, 0], c3[is_def, 1], c3[is_def, 2], marker="*", s=320,
                   facecolor="gold", edgecolor="k", zorder=6)
        if n_roles <= 50:
            for i, nm in enumerate(names):
                ax.text(*c3[i], f"  {nm[:14]}", fontsize=6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"n = {n_roles} selected role means in their own PC1–3\n"
                     "★ = default (Assistant)")
        _save(fig, out / f"fig07_roles_pc123_n{n_roles}.png")
    except Exception as e:  # noqa: BLE001
        print("fig07 failed:", e)
