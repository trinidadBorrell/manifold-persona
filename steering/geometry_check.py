"""Experiment 1: does the cylinder+spline construction produce sensible paths?

Plan: plans/2026-09-08-steering-rebuild.md (Design, experiment 1).

Pure geometry. No generation, no GPU, no judge calls. It sweeps the cylinder
radius eps, builds the linear and manifold paths for both cases, and draws the
two figures that gate everything downstream:

  fig01  eps ablation   PCA small multiples per eps, plus eps vs bending energy
                        and eps vs number of interior centroids
  fig02  PCA paths      chord and spline overlaid on the role cloud, markers at
                        each of the 8 alpha stops

WHY THIS RUNS FIRST. Every number after this depends on the two arms being two
different interventions. If the manifold path is visually indistinguishable from
the chord, or wanders somewhere absurd, that is knowable now for the price of
some CPU rather than after a GPU run and a judge bill.

CENTROIDS ARE FULLY-ROLE-PLAYING ONLY. `Geometry.centroids` silently falls back
fully -> somewhat -> unfiltered. The paper's axis is defined against the `fully`
vectors, and a curve anchored on vectors the >=10 rule discards is the same
class of error this rebuild exists to fix. Roles without a `fully` centroid are
dropped from A/B eligibility AND from the spline centroids.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from steering.geometry import load_geometry  # noqa: E402
from steering.manifold_paths import (  # noqa: E402
    LinearPath, PersonaPath, chord_frame, chord_coords,
    select_cylinder, endpoint_drift,
)

ALPHAS = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0])
# absolute tube radius in activation units. Mean centroid norm ~45, mean
# near->far chord ~38, near->midway ~5.8, so this brackets the useful range.
EPS_GRID = [0.0, 8.0, 12.0, 16.0, 20.0]

C_CLOUD = "#CCCCCC"
C_LINEAR = "#4C72B0"
C_MANIFOLD = "#C44E52"
C_INSIDE = "#F0A202"
MODE = "absolute"
KFIG = 5          # k for the figures; the artifact sweeps k itself
PLAM = 0.0        # lam=0 -> the curve passes through every chosen persona
PARAM = "centripetal"   # knot abscissa; see manifold_paths.PersonaPath
# lam is the zigzag control. geometry.py Observations O2: lam=0 interpolates and
# the curve zigzags between adjacent centroids (tortuosity 361x); lam>0
# makes it a trend near them. GCV picks lam to FIT the points, and nothing in it
# penalises route length, so it lands at ~1e-6 and the zigzag survives.
LAM_GRID = ['gcv']


# --------------------------------------------------------------------------

def _git_sha():
    """Sha + dirty flag. eps and k are frozen from these figures for the
    generation run, so this file is the only link from the chosen eps to the
    code that justified it (steering/README.md: a run whose manifest is lost is
    unreproducible)."""
    import subprocess
    try:
        root = str(Path(__file__).resolve().parent.parent)
        sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                      text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return {"sha": sha, "dirty": dirty}
    except Exception as exc:
        return {"sha": None, "error": str(exc)}


def fully_only(geom):
    """Indices of roles whose centroid is a genuine fully-role-playing vector.

    HARD-FAILS rather than falling through. The previous version read
    `roles_targeted_by_unfiltered` (a key `geometry.py` never writes) and
    `geom.target_category` (a local in `_role_vectors`, not a Geometry field,
    so always None). It therefore dropped only `default` while claiming to
    filter -- the same class of silent fallback this rebuild exists to remove.

    `geometry.py` exposes exactly two usable signals: the per-role
    `roles_targeted_by_somewhat` list, and the count
    `n_centroids_unfiltered_in_curve`. If any centroid fell back to unfiltered
    we cannot say WHICH from the report alone, so we refuse to proceed.
    """
    rep = geom.axis_report or {}
    counts = rep.get("target_category_counts")
    if not counts or not counts.get("fully"):
        raise SystemExit(
            "axis_report has no fully-role-playing counts (%r). Pass --labels "
            "pointing at a role_labels.parquet; without it load_geometry falls "
            "back to unfiltered centroids and this run would claim a filter it "
            "did not apply." % (counts,))
    # `default` ALWAYS takes the unfiltered fallback -- geometry.py:349, "none
    # when the >=10 rule dropped the role (or for `default`)". It is the
    # reference, not a role, and is excluded by name below, so exactly one
    # unfiltered centroid is the expected baseline. More than one means a real
    # role fell back, and axis_report does not record WHICH, so we refuse
    # rather than quietly anchor the curve on a vector 2.1.2 says to discard.
    n_unfiltered = int(rep.get("n_centroids_unfiltered_in_curve", 0) or 0)
    expected = 1 if "default" in geom.roles else 0
    if n_unfiltered > expected:
        raise SystemExit(
            "%d centroids fell back to unfiltered (expected %d, for `default`), "
            "and axis_report does not say which. Re-run steering.rolefilter so "
            "every role has a `fully` vector, or extend geometry.py to record "
            "roles_targeted_by_unfiltered." % (n_unfiltered, expected))
    bad = set(rep.get("roles_targeted_by_somewhat", []) or [])
    keep = [i for i, r in enumerate(geom.roles) if r not in bad and r != "default"]
    return np.array(keep, dtype=int), sorted(bad)


def pca_basis(C, k=2):
    """SVD-PCA fitted on the centroids only. Returns mu, basis (k,H), evr (k,)."""
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = (S ** 2) / max(len(C) - 1, 1)
    return mu, Vt[:k], var[:k] / var.sum()


def project(X, mu, basis):
    return (np.atleast_2d(X) - mu) @ basis.T


def pick_endpoints(C, names, axis_proj, axis_unit):
    """Case definitions.

    The AXIS case travels along the Assistant Axis itself -- a segment of the
    line through the cloud centre in direction `axis_unit`, spanning the same
    axis extent the role centroids reach. It is deliberately NOT the chord
    between the two extreme centroids: that is just another A->B pair, and using
    it made the axis panel identical to summarizer->leviathan.

    The PAIR cases are 3 near-Assistant sources x (3 rank-midway + 2 far).
    """
    order = np.argsort(axis_proj)
    n_near = [int(i) for i in order[-3:][::-1]]
    mid_c = len(order) // 2
    n_mid = [int(i) for i in order[mid_c - 1:mid_c + 2]]
    n_far = [int(i) for i in order[:2]]

    centre = C.mean(0)
    a = axis_unit / np.linalg.norm(axis_unit)
    off = centre @ a
    P_hi = centre + (axis_proj.max() - off) * a      # Assistant end
    P_lo = centre + (axis_proj.min() - off) * a      # far end

    cases = [("axis", P_hi, P_lo, "axis+", "axis-")]
    for i in n_near:
        for j in n_mid + n_far:
            cases.append(("pair", C[i], C[j], names[i], names[j]))
    return cases, dict(near=[names[i] for i in n_near],
                       mid=[names[i] for i in n_mid],
                       far=[names[i] for i in n_far])


# --------------------------------------------------------------------------

def sweep_k(C, names, cases, eps, k_grid, lam):
    """PersonaPath: k real persona centroids, interpolated exactly.

    Sweeps the knot abscissa too. `polyline` is the floor -- the length of the
    straight-line path through the same knots -- so `overshoot` isolates what
    the spline adds on top of an unavoidable cost.
    """
    rows = []
    for kind, P0, P1, na, nb in cases:
      for pm in ("projection", "length", "centripetal"):
        for k in k_grid:
            w = PersonaPath(P0, P1, C, eps, k=k, lam=lam, mode=MODE, param=pm)
            rows.append(dict(case=kind, A=na, B=nb, eps=float(eps), k=int(k),
                             param=pm,
                             n_centroids=w.n_centroids,
                             detour_ratio=w.detour_ratio,
                             polyline=w.polyline_ratio,
                             overshoot=w.overshoot,
                             knot_error=w.knot_error(),
                             excursion=w.excursion(),
                             bending_energy=w.bending_energy(),
                             lam=w.lam,
                             centroids=" ".join(names[int(i)] for i in w.centroid_idx)))
    return rows


def sweep(C, names, cases, eps_grid):
    rows = []
    for kind, P0, P1, na, nb in cases:
        _, L = chord_frame(P0, P1)
        for eps in eps_grid:
          for lam_req in LAM_GRID:
            mp = PersonaPath(P0, P1, C, eps, k=KFIG, lam=(0.0 if lam_req == "gcv"
                             else float(lam_req)), mode=MODE, param=PARAM)
            rows.append(dict(
                lam_req=("gcv" if lam_req is None else lam_req),
                case=kind, A=na, B=nb, eps=float(eps),
                chord_len=L, n_centroids=mp.n_centroids,
                detour_ratio=mp.detour_ratio,
                bending_energy=mp.bending_energy(),
                lam=mp.lam,
                endpoint_drift=endpoint_drift(mp, P0, P1),
            ))
    return rows


def fig01(C, names, cases, rows, mu, basis, evr, outdir, layer):
    """eps ablation: PCA small multiples, eps vs bending energy, eps vs n_centroids."""
    show = [c for c in cases if c[0] == "axis"][:1] + [c for c in cases if c[0] == "pair"][:1]
    eps_show = [e for e in EPS_GRID if e > 0][:6]
    P2 = project(C, mu, basis)

    fig = plt.figure(figsize=(15, 4.2 * (len(show) + 1)))
    gs = fig.add_gridspec(len(show) + 1, len(eps_show))

    for r, (kind, P0, P1, na, nb) in enumerate(show):
        for c, eps in enumerate(eps_show):
            ax = fig.add_subplot(gs[r, c])
            mp = PersonaPath(P0, P1, C, eps, k=KFIG, lam=PLAM, mode=MODE, param=PARAM)
            lp = LinearPath(P0, P1)
            ax.scatter(P2[:, 0], P2[:, 1], s=6, c=C_CLOUD, lw=0, zorder=1)
            if mp.n_centroids:
                q = project(C[mp.centroid_idx], mu, basis)
                ax.scatter(q[:, 0], q[:, 1], s=16, c=C_INSIDE, lw=0.3,
                           edgecolor="#6B4700", zorder=3)
                if mp.n_centroids <= 14:
                    for j, ci in enumerate(mp.centroid_idx):
                        ax.annotate(names[ci], (q[j, 0], q[j, 1]), fontsize=4,
                                    color="#6B4700", xytext=(2, 1),
                                    textcoords="offset points", zorder=7)
            t = np.linspace(0, 1, 200)
            for path, col, ls in ((lp, C_LINEAR, "-"), (mp, C_MANIFOLD, "--")):
                p = project(path.at_alpha(t), mu, basis)
                ax.plot(p[:, 0], p[:, 1], color=col, ls=ls, lw=1.6, zorder=4)
            e = project(np.vstack([P0, P1]), mu, basis)
            ax.scatter(e[:, 0], e[:, 1], s=60, c="#111111", marker="D", zorder=5)
            ax.set_title("%s  eps=%g  k=%d/%d  detour=%.3f"
                         % (kind, eps, mp.n_centroids, KFIG, mp.detour_ratio), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.2, lw=0.5)

    axb = fig.add_subplot(gs[len(show), :len(eps_show) // 2])
    axc = fig.add_subplot(gs[len(show), len(eps_show) // 2:])
    for kind, style in (("axis", "-"), ("pair", ":")):
        sub = [r for r in rows if r["case"] == kind]
        if not sub:
            continue
        keys = sorted({(r["A"], r["B"]) for r in sub})
        for A, B in keys[:8]:
            s = sorted([r for r in sub if r["A"] == A and r["B"] == B],
                       key=lambda r: r["eps"])
            x = [r["eps"] for r in s]
            axb.plot(x, [max(r["bending_energy"], 1e-12) for r in s],
                     ls=style, lw=1.1, alpha=0.85)
            axc.plot(x, [r["n_centroids"] for r in s], ls=style, lw=1.1, alpha=0.85)
    axb.set_yscale("log"); axb.set_xlabel("eps (tube radius, activation units)")
    axb.set_ylabel(r"bending energy  $\int\|S''\|^2\,dt$")
    axc.set_xlabel("eps (tube radius, activation units)")
    axc.set_ylabel("interior centroids")
    for a in (axb, axc):
        a.grid(alpha=0.25, lw=0.6)

    fig.suptitle("EXPLORATORY - eps ablation. PC1/PC2 explained variance %.1f%%/%.1f%%. "
                 "Paths are projected into the centroid PCA, never fitted in it."
                 % (100 * evr[0], 100 * evr[1]), fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    p = Path(outdir) / ("fig01_epsilon_ablation_L%s.png" % layer)
    fig.savefig(str(p), dpi=300); plt.close(fig)
    return p


def fig02(C, names, cases, eps, mu, basis, evr, outdir, layer):
    """The two paths on the cloud, with a marker at each alpha stop."""
    show = [c for c in cases if c[0] == "axis"] + [c for c in cases if c[0] == "pair"][:5]
    n = len(show)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
    P2 = project(C, mu, basis)

    for k, (kind, P0, P1, na, nb) in enumerate(show):
        ax = axes[k // ncol][k % ncol]
        mp = PersonaPath(P0, P1, C, eps, k=KFIG, lam=PLAM, mode=MODE, param=PARAM)
        lp = LinearPath(P0, P1)
        ax.scatter(P2[:, 0], P2[:, 1], s=6, c=C_CLOUD, lw=0, zorder=1)
        if mp.n_centroids:
            q = project(C[mp.centroid_idx], mu, basis)
            ax.scatter(q[:, 0], q[:, 1], s=30, c=C_INSIDE, lw=0.4,
                       edgecolor="#6B4700", zorder=3,
                       label="curve passes through these centroids (n=%d)" % mp.n_centroids)
            # Name them. Which roles the curve threads is the whole question --
            # "inside the tube" and "on the way from A to B" are different
            # predicates, and only the names tell you which you got.
            for j, ci in enumerate(mp.centroid_idx):
                ax.annotate(names[ci], (q[j, 0], q[j, 1]), fontsize=5.5,
                            color="#6B4700", xytext=(3, 2),
                            textcoords="offset points", zorder=7)
        t = np.linspace(0, 1, 300)
        for path, col, ls, lab in ((lp, C_LINEAR, "-", "linear"),
                                   (mp, C_MANIFOLD, "--", "manifold")):
            p = project(path.at_alpha(t), mu, basis)
            ax.plot(p[:, 0], p[:, 1], color=col, ls=ls, lw=2.0, zorder=4, label=lab)
            s = project(path.at_alpha(ALPHAS), mu, basis)
            ax.scatter(s[:, 0], s[:, 1], s=26, c=col, zorder=5, edgecolor="white", lw=0.6)
        e = project(np.vstack([P0, P1]), mu, basis)
        ax.scatter(e[:, 0], e[:, 1], s=90, c="#111111", marker="D", zorder=6)
        ax.annotate(na, e[0], fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.annotate(nb, e[1], fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title("%s: %s -> %s   detour %.3f"
                     % (kind, na, nb, mp.detour_ratio), fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("EXPLORATORY - linear vs manifold paths, eps=%.2f. "
                 "PC1/PC2 explained variance %.1f%%/%.1f%%. Markers are the 8 alpha stops "
                 "(arc position). Paths projected in, never fitted."
                 % (eps, 100 * evr[0], 100 * evr[1]), fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    p = Path(outdir) / ("fig02_pca_paths_L%s.png" % layer)
    fig.savefig(str(p), dpi=300); plt.close(fig)
    return p


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True,
                    help="resp_dir: the response-token activation cloud")
    ap.add_argument("--labels", default=None, help="role_labels.parquet")
    ap.add_argument("--outdir", required=True,
                    help="where the figures, CSVs and controls JSON are written")
    ap.add_argument("--layer", type=int, default=19,
                    help="layer LABEL for filenames only")
    ap.add_argument("--layer-index", type=int, default=0,
                    help="index into the .npy. The published clouds are thinned "
                         "to a single layer, so this is 0, not the layer number.")
    ap.add_argument("--eps-fig2", type=float, default=9.0,
                    help="cylinder radius, activation units, for fig02 and the "
                         "JSON exports (default matches the ablation grid)")
    ap.add_argument("--param", default="centripetal",
                    choices=["projection", "length", "centripetal"],
                    help="knot abscissa for the spline")
    ap.add_argument("--k-fig", type=int, default=5,
                    help="k personas for the PNG figures")
    ap.add_argument("--persona-lam", type=float, default=0.0,
                    help="lam for PersonaPath. 0 = interpolate every chosen "
                         "persona centroid exactly (still C2-smooth).")
    ap.add_argument("--export-json", action="store_true",
                    help="dump PCA-3 cloud + paths for the interactive viewer")
    ap.add_argument("--eps-mode", default="absolute",
                    choices=["absolute", "relative"])
    args = ap.parse_args()

    global MODE, KFIG, PLAM, PARAM
    MODE = args.eps_mode
    KFIG = args.k_fig
    PLAM = args.persona_lam
    PARAM = args.param
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("loading geometry...", flush=True)
    geom = load_geometry(resp_dir=args.cloud, layer=args.layer_index,
                         labels_path=args.labels)
    keep, dropped = fully_only(geom)
    print("roles: %d total, %d fully-role-playing, %d dropped"
          % (len(geom.roles), len(keep), len(dropped)), flush=True)
    if len(keep) < 10:
        raise SystemExit("only %d fully-role-playing centroids - labels missing?" % len(keep))

    C = np.asarray(geom.centroids, dtype=np.float64)[keep]
    names = [geom.roles[i] for i in keep]
    axis_proj = np.asarray(geom.axis_proj, dtype=np.float64)[keep]

    cases, picked = pick_endpoints(C, names, axis_proj, np.asarray(geom.axis_unit, dtype=np.float64))
    print("cases: %d (1 axis + %d pairs)" % (len(cases), len(cases) - 1), flush=True)
    print("picked:", json.dumps(picked), flush=True)

    rows = sweep(C, names, cases, EPS_GRID)

    krows = sweep_k(C, names, cases, args.eps_fig2, [3, 4, 5, 6, 8, 10, 12],
                    args.persona_lam)
    kc = outdir / ("persona_k_sweep_L%s.csv" % args.layer)
    kcols = ["case", "A", "B", "eps", "k", "param", "n_centroids",
             "detour_ratio", "polyline", "overshoot", "knot_error", "excursion",
             "bending_energy", "lam", "centroids"]
    with open(kc, "w") as fh:
        fh.write(",".join(kcols) + "\n")
        for r in krows:
            fh.write(",".join('"%s"' % r[c] if c == "centroids" else str(r[c])
                              for c in kcols) + "\n")
    print("wrote", kc, flush=True)
    mu, basis, evr = pca_basis(C, 2)
    print("PCA evr: %.3f %.3f" % (evr[0], evr[1]), flush=True)

    p1 = fig01(C, names, cases, [r for r in rows if r["lam_req"] == "gcv"],
               mu, basis, evr, outdir, args.layer)
    print("wrote", p1, flush=True)
    p2 = fig02(C, names, cases, args.eps_fig2, mu, basis, evr, outdir, args.layer)
    print("wrote", p2, flush=True)

    # csv sidecar - the plan requires the data behind every figure
    csv = outdir / ("fig01_epsilon_ablation_L%s.csv" % args.layer)
    cols = ["case", "A", "B", "eps", "lam_req", "chord_len", "n_centroids",
            "detour_ratio", "bending_energy", "lam", "endpoint_drift"]
    with open(csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print("wrote", csv, flush=True)

    # eps x k ablation grid + PCA-3 export for the interactive viewer
    if args.export_json:
        EPSG = [6.0, 9.0, 12.0, 16.0, 20.0]
        KG = [2, 3, 4, 5, 6, 8]
        mu3, basis3, evr3 = pca_basis(C, 3)
        P3 = (C - mu3) @ basis3.T
        t = np.linspace(0, 1, 120)
        def pr(X):
            return [[round(float(v), 3) for v in r]
                    for r in ((np.atleast_2d(X) - mu3) @ basis3.T)]

        # one axis case + every pair, but cap the pairs so the payload stays sane
        export_cases = [c for c in cases if c[0] == "axis"] + \
                       [c for c in cases if c[0] == "pair"][:8]
        grid = {"evr3": [float(v) for v in evr3], "roles": names,
                "cloud": [[round(float(v), 3) for v in r] for r in P3],
                "eps_grid": EPSG, "k_grid": KG, "layer": int(args.layer),
                "alphas": [float(a) for a in ALPHAS], "cases": []}
        for kind, P0, P1, na, nb in export_cases:
            lp = LinearPath(P0, P1)
            entry = {"kind": kind, "A": na, "B": nb,
                     "chord_len": round(float(chord_frame(P0, P1)[1]), 2),
                     "linear": pr(lp.at_alpha(t)),
                     "linear_stops": pr(lp.at_alpha(ALPHAS)),
                     "variants": {}}
            for e in EPSG:
                for k in KG:
                    w = PersonaPath(P0, P1, C, e, k=k, lam=args.persona_lam,
                                    mode=MODE, param=PARAM)
                    entry["variants"]["%g|%d" % (e, k)] = {
                        "n": w.n_centroids,
                        "detour": round(w.detour_ratio, 3),
                        "excursion": round(w.excursion(), 3),
                        "knot_err": float("%.2e" % w.knot_error()),
                        "floor": round(w.polyline_ratio, 3),
                        "overshoot": round(w.overshoot, 3),
                        "idx": [int(i) for i in w.centroid_idx],
                        "seq": [names[int(i)] for i in w.centroid_idx],
                        "path": pr(w.at_alpha(t)),
                        "stops": pr(w.at_alpha(ALPHAS)),
                    }
            grid["cases"].append(entry)
        # supply: how many personas the tube actually contains per route per
        # eps, and where they sit along the chord. The k you ask for is capped
        # by this, and 275 centroids in 4096 dims is sparse.
        supply = []
        for kind, P0, P1, na, nb in export_cases:
            row = {"kind": kind, "A": na, "B": nb, "by_eps": {}}
            for e in EPSG:
                idx, u_all, r_all = select_cylinder(C, P0, P1, e, MODE)
                row["by_eps"]["%g" % e] = {
                    "n": int(len(idx)),
                    "u": [round(float(u_all[i]), 4) for i in idx],
                    "r": [round(float(r_all[i]), 2) for i in idx],
                    "roles": [names[int(i)] for i in idx],
                }
            supply.append(row)
        grid["supply"] = supply

        gp = outdir / ("ablation3d_L%s.json" % args.layer)
        gp.write_text(json.dumps(grid, separators=(",", ":")))
        print("wrote", gp, "(%.0f KB)" % (gp.stat().st_size / 1024), flush=True)

    # PCA-3 export for the interactive viewer
    if args.export_json:
        mu3, basis3, evr3 = pca_basis(C, 3)
        P3 = (C - mu3) @ basis3.T
        t = np.linspace(0, 1, 160)
        exp = {"evr3": [float(v) for v in evr3],
               "roles": names,
               "cloud": [[round(float(v), 4) for v in row] for row in P3],
               "axis_proj": [round(float(v), 4) for v in axis_proj],
               "eps": float(args.eps_fig2), "layer": int(args.layer), "cases": []}
        for kind, P0, P1, na, nb in cases:
            mp = PersonaPath(P0, P1, C, args.eps_fig2, k=KFIG, lam=PLAM, mode=MODE, param=PARAM)
            lp = LinearPath(P0, P1)
            def pr(X):
                return [[round(float(v), 4) for v in r] for r in ((np.atleast_2d(X) - mu3) @ basis3.T)]
            exp["cases"].append({
                "kind": kind, "A": na, "B": nb,
                "n_centroids": mp.n_centroids,
                "detour": round(mp.detour_ratio, 4),
                "chord_len": round(float(chord_frame(P0, P1)[1]), 3),
                "centroid_idx": [int(i) for i in mp.centroid_idx],
                "centroid_roles": [names[int(i)] for i in mp.centroid_idx],
                "linear": pr(lp.at_alpha(t)),
                "manifold": pr(mp.at_alpha(t)),
                "linear_stops": pr(lp.at_alpha(ALPHAS)),
                "manifold_stops": pr(mp.at_alpha(ALPHAS)),
            })
        jp = outdir / ("paths3d_L%s.json" % args.layer)
        jp.write_text(json.dumps(exp))
        print("wrote", jp, "(%.1f KB)" % (jp.stat().st_size / 1024), flush=True)

    # controls
    zero_path = [PersonaPath(P0, P1, C, 0.0, k=KFIG, lam=PLAM, mode=MODE,
                             param=PARAM) for _, P0, P1, _, _ in cases]
    lin = [LinearPath(P0, P1) for _, P0, P1, _, _ in cases]
    aa = np.array([0.0, 0.5, 1.0])
    z = [r for r in rows if r["eps"] == 0.0]
    ctl = {
        # Measured on PersonaPath -- the class the generation run will use.
        # Deriving the controls from any other construction certifies something
        # the experiment does not do.
        "personapath_eps0_is_chord": float(max(
            np.abs(z0.at_alpha(aa) - l0.at_alpha(aa)).max()
            for z0, l0 in zip(zero_path, lin))),
        "personapath_endpoint_drift": float(max(
            endpoint_drift(PersonaPath(P0, P1, C, args.eps_fig2, k=KFIG,
                                       lam=PLAM, mode=MODE, param=PARAM), P0, P1)
            for _, P0, P1, _, _ in cases)),
        "eps0_n_centroids_all_zero": all(r["n_centroids"] == 0 for r in z),
        "eps0_detour_max_dev": max(abs(r["detour_ratio"] - 1.0) for r in z),
        "endpoint_drift_max": max(r["endpoint_drift"] for r in rows),
        "pca_evr": [float(evr[0]), float(evr[1])],
        "n_fully_centroids": int(len(keep)),
        "n_dropped": int(len(dropped)),
        "roles_picked": picked,
        "axis_definition": (geom.axis_report or {}).get("definition"),
        "args": {k: (str(v) if not isinstance(v, (int, float, bool, type(None)))
                     else v) for k, v in vars(args).items()},
        "git_sha": _git_sha(),
        "cloud": args.cloud,
        "labels": args.labels,
        "centroid_roles_at_fig2_eps": {
            ("%s->%s" % (na, nb)): [names[int(i)] for i in
                PersonaPath(P0, P1, C, args.eps_fig2, k=KFIG,
                            lam=PLAM, mode=MODE, param=PARAM).centroid_idx]
            for kind, P0, P1, na, nb in cases},
    }
    (outdir / ("fig01_epsilon_ablation_L%s.json" % args.layer)).write_text(
        json.dumps(ctl, indent=2))
    print("CONTROLS", json.dumps(ctl, indent=2), flush=True)


if __name__ == "__main__":
    main()
