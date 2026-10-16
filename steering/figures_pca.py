"""Where steering actually took the model, drawn in persona space.

One panel per prompted persona. Background: the 275 role vectors in a PCA fitted
ON THOSE VECTORS, so the axes mean "directions of persona variation" rather than
directions of whatever the trajectories happen to do. Trajectories are projected
into that fixed basis afterwards; they never influence it.

Each trajectory point is the mean layer-19 activation of that role's 20 responses
at one dose, recovered by `capture_activations.py` (a hookless replay, so the
coordinates are the unsteered model's and are comparable to the cloud). The 20
individual responses are drawn as faint dots behind their mean, because a mean of
a scattered cloud sits where nothing actually is.

TWO HONESTY NOTES ON THE FIGURE ITSELF:
  * PC1/PC2 explain a stated fraction of the variance; two points close on screen
    can be far apart in 4096-D. The fraction is printed on the figure.
  * Nearest-role labels are computed in the FULL space, not in the 2-D
    projection, so "it passed near oracle" is a claim about the real geometry.

    python -m steering.figures_pca --acts acts.npy --judged judged.parquet \\
        --cloud <cloud_dir> --labels role_labels.parquet --outdir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

ARM_STYLE = {"linear_axis": ("-", "o", "#4C72B0", "linear"),
             "manifold_axis": ("--", "s", "#C44E52", "manifold"),
             "linear_target": ("-", "o", "#4C72B0", "linear (target)"),
             "manifold_target": ("--", "s", "#C44E52", "manifold (target)"),
             # Journey arms: matched on DESTINATION, not dose. `alpha` holds the
             # fraction f of the way from the activation to the target centroid,
             # so at f = 1 both arms are pinned to exactly the same point and the
             # only thing left that can differ is the route.
             "linear_journey": ("-", "o", "#4C72B0", "linear journey"),
             "manifold_journey": ("--", "s", "#C44E52", "manifold journey"),
             # Without these the contrast arms fell through to the default style
             # and both were drawn in the same grey, so the two routes could not
             # be told apart at all.
             "linear_contrast": ("-", "o", "#2E7D32", "linear contrast"),
             "manifold_contrast": ("--", "s", "#8C6BB1", "manifold contrast")}


def intended_path(arm, h0, fracs, g, c_target=None):
    """Where the PUSH puts the activation, BEFORE the model turns it into words.

    The measured trace says where the resulting text landed; this says where the
    intervention aimed. Drawn together, the gap between them is the finding:
    moving the state by X does not move the LANGUAGE by X.

    Straight by construction for `linear_journey` (h + f(c_T - h)) and for the
    axis arms (h + alpha*N_bar*a_hat); curved by construction for the manifold
    arms, which is the point of drawing them in the same basis.
    """
    from steering import interventions as _IV
    a_hat = _IV.unit(g.axis_unit)
    h0 = np.asarray(h0, dtype=np.float64)
    S = g.spline.evaluate
    out = []
    for f in fracs:
        if arm == "linear_journey":
            out.append(h0 + f * (np.asarray(c_target, dtype=np.float64) - h0))
        elif arm == "manifold_journey":
            c_t = np.asarray(c_target, dtype=np.float64)
            u_t = float(c_t @ a_hat)
            r_t = c_t - S(np.array([u_t]))[0]
            u_h = float(h0 @ a_hat)
            S_h = S(np.array([u_h]))[0]
            step = S(np.array([u_h + f * (u_t - u_h)]))[0] - S_h
            out.append(h0 + step + f * (r_t - (h0 - S_h)))
        else:
            out.append(h0 + f * float(g.n_bar) * a_hat)
    return np.stack(out)


def load_geometry_bits(cloud: str, labels: str):
    """Role vectors + the PCA basis they define, plus the axis for orientation."""
    from steering.geometry import load_geometry
    g = load_geometry(resp_dir=cloud, labels_path=labels)
    C = np.asarray(g.centroids, dtype=np.float64)
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return g, mu, Vt[:2], var[:2]


def project(X, mu, basis):
    return (np.asarray(X, dtype=np.float64) - mu) @ basis.T


def nearest_roles(v, C, roles, k=3):
    d = np.linalg.norm(C - v[None, :], axis=1)
    return [roles[i] for i in np.argsort(d)[:k]]


def trajectories(acts, meta, role, arms):
    """(arm -> [(alpha, mean_vec, member_vecs)]), ordered from 0 down to -1."""
    out = {}
    base = meta[(meta.role == role) & (meta.arm == "unsteered")]
    for arm in arms:
        rows = meta[(meta.role == role) & (meta.arm == arm)]
        if rows.empty:
            continue
        pts = []
        for a in sorted(set(rows.alpha) | ({0.0} if len(base) else set())):
            idx = (base.index if np.isclose(a, 0.0) and len(base)
                   else rows.index[np.isclose(rows.alpha, a)])
            if not len(idx):
                continue
            V = acts[np.asarray(idx)]
            V = V[np.linalg.norm(V, axis=1) > 0]
            if len(V):
                pts.append((float(a), V.mean(0), V))
        out[arm] = sorted(pts, key=lambda t: -t[0])
    return out


def _references(ax, g, mu, basis, C2):
    """The Assistant Axis, its origin, and the fitted curve — drawn in every panel.

    Without these a reader cannot tell whether a trajectory follows the axis or
    merely wanders: every trace is a curve on screen, including a perfectly
    straight push in 4096-D. The dashed line is the thing the linear arm moves
    along; the green curve is the thing the manifold arm moves along.
    """
    from steering import interventions as _IV
    a_hat = _IV.unit(g.axis_unit)
    C = np.asarray(g.centroids)
    pc = C @ a_hat
    lo, hi = float(pc.min()), float(pc.max())
    pad = 0.18 * (hi - lo)

    if "default" in g.roles:
        i = g.roles.index("default")
        origin = C[i] - float(C[i] @ a_hat) * a_hat
        ts = np.linspace(lo - pad, hi + pad, 60)
        L = project(origin[None, :] + ts[:, None] * a_hat[None, :], mu, basis)
        ax.plot(L[:, 0], L[:, 1], ls=(0, (6, 4)), lw=1.6, color="#111111",
                zorder=1, label="Assistant Axis")
        ax.scatter([C2[i, 0]], [C2[i, 1]], s=70, marker="D", c="#111111",
                   zorder=5, label="default (Assistant)")

    us = np.linspace(float(g.spline.x[0]), float(g.spline.x[-1]), 220)
    S = project(g.spline.evaluate(us), mu, basis)
    ax.plot(S[:, 0], S[:, 1], lw=2.0, color="#2E7D32", alpha=0.9, zorder=1,
            label="fitted manifold")


def steering_line(ax, g, mu, basis, role, target, fracs, h0=None):
    """THE STEERING ITSELF: the predefined segment from initialization to end.

    delta = f*(c_E - c_I) is fixed before a single token is generated, so the
    intervention is this straight line and f is just position along it. Nothing
    about it is discovered point by point.

    The wandering series elsewhere in this panel is a different object: the
    MEASURED landing of the text at each f, one independent generation per f.
    Drawing only that made the steering look like a path that had to be searched
    out, which it is not.
    """
    if target not in g.roles or role not in g.roles:
        return
    C = np.asarray(g.centroids, dtype=np.float64)
    # Anchor at the UNSTEERED LANDING when we have it, not the cloud centroid.
    # The two sit 30.49 apart for doctor -- larger than the whole journey -- because
    # the extraction cloud and the eval run are different corpora. PC1+PC2 carry
    # 51% of the variance, so that offset is invisible on screen and a line drawn
    # from the centroid silently starts in the wrong place.
    c_i = C[g.roles.index(role)] if h0 is None else np.asarray(h0, dtype=np.float64)
    c_e = C[g.roles.index(target)]
    P = np.stack([c_i + f * (c_e - c_i) for f in fracs])
    P2 = project(P, mu, basis)
    ax.plot(P2[:, 0], P2[:, 1], "-", lw=2.6, color="#111111", alpha=0.85,
            zorder=7, label="steering line (I -> E)")
    ax.scatter(P2[:, 0], P2[:, 1], s=22, marker="|", color="#111111", zorder=8)
    i2 = project(c_i[None, :], mu, basis)[0]
    ax.scatter([i2[0]], [i2[1]], s=120, marker="o", facecolors="none",
               edgecolors="#111111", lw=1.8, zorder=8,
               label="I = %s%s" % (role, "" if h0 is None else " (unsteered)"))


def panel(ax, role, traj, g, mu, basis, C2, fs=8, refs=False, target=None,
          fracs=None):
    ax.scatter(C2[:, 0], C2[:, 1], s=7, c="#CCCCCC", lw=0, zorder=0)
    if refs:
        _references(ax, g, mu, basis, C2)
    seen = set()
    for arm, pts in traj.items():
        if not pts:
            continue
        ls, mk, col, lab = ARM_STYLE.get(arm, ("-", "o", "#333333", arm))
        P = project(np.stack([p[1] for p in pts]), mu, basis)
        for (a, m, V), xy in zip(pts, P):
            M = project(V, mu, basis)
            ax.scatter(M[:, 0], M[:, 1], s=4, color=col, alpha=0.16, lw=0, zorder=1)
            seen.update(nearest_roles(m, np.asarray(g.centroids), g.roles, 2))
        ax.plot(P[:, 0], P[:, 1], ls, marker=mk, ms=5, lw=1.9, color=col,
                label=lab, zorder=3)
        # Label every other point, and alternate the offset above/below the
        # line: with 8 fractions the ends bunch together and a label on every
        # marker is unreadable. The endpoints are always labelled, because
        # "where it started" and "where it ended" are the two the reader needs.
        last = len(pts) - 1
        for i, ((a, _, _), xy) in enumerate(zip(pts, P)):
            if i % 2 and i != last:
                continue
            dy = 4 if i % 4 == 0 else -9
            ax.annotate("%.2f" % a if a >= 0 else "%+.2f" % a, xy,
                        fontsize=fs - 2.5, color=col, xytext=(3, dy),
                        textcoords="offset points", zorder=4)
    # Neighbouring roles the trajectories actually pass, named in FULL space.
    for r in sorted(seen)[:14]:
        xy = C2[g.roles.index(r)]
        ax.annotate(r, xy, fontsize=fs - 2.0, color="#555555",
                    xytext=(2, -8), textcoords="offset points", zorder=2)
    if target is not None and fracs is not None:
        # the f = 0 point of any arm is the shared unsteered landing
        h0 = None
        for pts in traj.values():
            for a_, mvec, _ in pts:
                if abs(a_) < 1e-9:
                    h0 = mvec
                    break
            if h0 is not None:
                break
        steering_line(ax, g, mu, basis, role, target, fracs, h0=h0)
    if target is not None and target in g.roles:
        # The destination. In the journey arms every trace is SUPPOSED to end
        # here, so a trace that stops short is the result, not a drawing error.
        xy = C2[g.roles.index(target)]
        ax.scatter([xy[0]], [xy[1]], s=190, marker="*", c="#F0A202",
                   edgecolors="#6B4700", lw=0.8, zorder=6, label="target: %s" % target)
    ax.set_title(role, fontsize=fs + 2)
    ax.tick_params(labelsize=fs - 2)
    ax.grid(alpha=0.2, lw=0.5)
    # Frame on the CLOUD and the measured landings. The steering line is drawn
    # over the whole f grid, which for the contrast arms reaches f = 3 and runs
    # to PC1 = -60; letting that set the limits squeezed every measured point
    # into one corner. The line is allowed to leave the frame instead.
    pts = [C2] + [project(np.stack([q[1] for q in pv]), mu, basis)
                  for pv in traj.values() if pv]
    A = np.vstack(pts)
    for lim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1)):
        lo, hi = float(A[:, i].min()), float(A[:, i].max())
        pad = 0.08 * max(hi - lo, 1e-6)
        lim(lo - pad, hi + pad)


def make_figure(acts, meta, roles, g, mu, basis, var, out: Path, arms, title,
                refs=False, target=None, fracs=None):
    C2 = project(np.asarray(g.centroids), mu, basis)
    n = len(roles)
    ncol = 3 if n > 2 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow), squeeze=False)
    for k, role in enumerate(roles):
        ax = axes[k // ncol][k % ncol]
        panel(ax, role, trajectories(acts, meta, role, arms), g, mu, basis, C2,
              refs=refs, target=target, fracs=fracs)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=15, y=0.995)
    # Two lines, not one: on a 2-panel figure the single-line version ran off
    # the canvas and the last clause was cut mid-word.
    fig.text(0.5, 0.010,
             "PCA fitted on the 275 role vectors (PC1 %.1f%%, PC2 %.1f%% of variance); "
             "trajectories projected in, never fitted.\n"
             "Faint dots = the 20 individual responses behind each mean. "
             "Labels = alpha, which for the journey arms is f, the fraction of the "
             "way to the target." % (100 * var[0], 100 * var[1]),
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, help="steered_acts.npy")
    ap.add_argument("--meta", required=True, help="steered_acts.meta.parquet")
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--roles", required=True, help="comma-separated")
    ap.add_argument("--arms", default="linear_axis,manifold_axis")
    ap.add_argument("--tag", default="")
    ap.add_argument("--target", default=None,
                    help="keep only rows steered toward this role (journey/ablation "
                         "arms have two targets per role; blending them would average "
                         "two different destinations into one meaningless point)")
    ap.add_argument("--with-axis", action="store_true",
                    help="draw the Assistant Axis, the `default` point and the fitted "
                         "manifold in every panel, as a sanity reference")
    a = ap.parse_args()

    acts = np.load(a.acts)
    meta = pd.read_parquet(a.meta)
    if a.target:
        # Filter WITHOUT reindexing: meta's index is the row index into `acts`.
        if "target_role" not in meta.columns:
            raise SystemExit("--target given but meta has no target_role column")
        meta = meta[(meta.target_role == a.target) | (meta.arm == "unsteered")]
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    arms = [x.strip() for x in a.arms.split(",")]
    g, mu, basis, var = load_geometry_bits(a.cloud, a.labels)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    p = make_figure(acts, meta, roles, g, mu, basis, var,
                    out / ("fig07_pca_trajectories%s_L19.png" % a.tag), arms,
                    ("Where steering takes each persona — measured, layer 19"
                     + ("" if not a.target else " (toward %s)" % a.target)
                    + ("\ndashed = Assistant Axis · diamond = default · green = fitted manifold"
                        if a.with_axis else "")), refs=a.with_axis, target=a.target,
                    fracs=sorted(meta.loc[meta.arm != "unsteered", "alpha"]
                                 .dropna().unique()) if a.target else None)
    print("wrote", p)
    json.dump({"pc1_var": float(var[0]), "pc2_var": float(var[1]),
               "roles": roles, "arms": arms},
              open(out / ("fig07_pca%s_meta.json" % a.tag), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
