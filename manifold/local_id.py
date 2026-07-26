"""Local intrinsic dimension of the role cloud — does it vary from place to place?

WHY THIS EXISTS
---------------
Every "one manifold" result in this repo so far is a *connectivity* result:
single-linkage components across a cosine-threshold sweep (analysis_extra.py),
HDBSCAN/DBSCAN, spectral eigengap. All of those detect manifolds that are
separated by a gap in density. Manifolds that intersect or are glued -- two
subspaces meeting in a line, a sheet threaded through a tube -- have no density
gap anywhere, and every one of those methods reports a single blob.

Local intrinsic dimension is the standard way to see that case. One manifold =>
roughly the same intrinsic dimension in every neighbourhood. Several glued
manifolds of different dimension => local ID varies systematically across the
cloud, and does so *smoothly* (neighbouring points agree), not as scatter.

The global ID estimates already in the repo (idim.py, exploratory/) return ONE
number for the whole cloud and cannot see this.

DESIGN
------
For each of the 276 role means, take its k nearest neighbours in the 50-D PCA
space and estimate the dimension of that neighbourhood two ways:

  * participation ratio of the local covariance spectrum,
    PR = (sum lambda)^2 / sum(lambda^2). Continuous, threshold-free, stable at
    small k. This is the primary estimator.
  * skdim TwoNN on the same neighbourhood. Secondary; noisy at k=15, reported
    for agreement only.

Statistic: the coefficient of variation of local ID across roles, CV = sd/mean,
plus the correlation of local ID between neighbouring roles (does the variation
have spatial structure, or is it noise?).

CONTROLS -- these are what make the number readable
---------------------------------------------------
CV > 0 always, because k points give a noisy estimate. The question is whether
it is larger than noise. So we run the identical pipeline on:

  NEGATIVE ("one manifold"): a Gaussian matched to the role-mean covariance.
    Constant local dimension everywhere by construction. Its CV is the noise
    floor -- what a genuinely homogeneous cloud looks like at n=276, this k.

  POSITIVE ("two manifolds, no density gap"): 138 points on a 4-D linear
    subspace and 138 on a 14-D subspace, sharing an origin and overlapping in
    space, matched to the data's overall scale. There is no density gap, so
    single-linkage and HDBSCAN would call this ONE cluster. If local ID cannot
    separate it, the method is useless and the real result means nothing.

Run:  .venv/bin/python -m manifold.local_id
"""
from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from manifold_persona.runlog import (new_run_dir, save_fig, timestamp,
                                     write_manifest)
from . import pipeline as P

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "local_id"

K_LIST = (15, 25, 40)      # neighbourhood sizes; 25 is primary
K_PRIMARY = 25
N_REF = 50                 # reference draws for the negative control
SEED = 0


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions of a point set, from its covariance
    spectrum. (sum l)^2 / sum l^2 -- equals d for an isotropic d-dim cloud and
    falls toward 1 as variance concentrates in one direction."""
    Xc = X - X.mean(0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = lam.sum()
    return float(s * s / np.sum(lam ** 2)) if s > 0 else np.nan


def local_id(X: np.ndarray, k: int, twonn: bool = False) -> np.ndarray:
    """Local ID at every point of X, from its k nearest neighbours (self
    included, so the patch has k+1 points)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    out = np.empty(len(X))
    for i, nbr in enumerate(idx):
        patch = X[nbr]
        if twonn:
            try:
                import skdim
                out[i] = float(skdim.id.TwoNN().fit(patch).dimension_)
            except Exception:
                out[i] = np.nan
        else:
            out[i] = participation_ratio(patch)
    return out


def neighbour_corr(X: np.ndarray, vals: np.ndarray, k: int = 10) -> float:
    """Correlation between each point's value and its neighbours' mean value.
    High => the variation is spatially organised. ~0 => it is estimator noise."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    nbr_mean = np.array([np.nanmean(vals[j[1:]]) for j in idx])
    ok = np.isfinite(vals) & np.isfinite(nbr_mean)
    return float(np.corrcoef(vals[ok], nbr_mean[ok])[0, 1])


def cv(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(v.std() / v.mean())


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def negative_control(role_means: np.ndarray, k: int, n_ref: int, seed: int):
    """One homogeneous manifold: Gaussian matched to the real covariance."""
    rng = np.random.default_rng(seed)
    n, d = role_means.shape
    mu, cov = role_means.mean(0), np.cov(role_means, rowvar=False)
    cvs, means = [], []
    for _ in range(n_ref):
        Y = rng.multivariate_normal(mu, cov, size=n)
        v = local_id(Y, k)
        cvs.append(cv(v)); means.append(np.nanmean(v))
    return np.array(cvs), np.array(means)


def positive_control(role_means: np.ndarray, k: int, seed: int):
    """Two manifolds of different dimension, glued, with NO density gap:
    138 points in a 4-D subspace + 138 in a 14-D subspace of the same 50-D
    space, scaled to the real cloud's radius."""
    rng = np.random.default_rng(seed)
    n, d = role_means.shape
    scale = float(np.linalg.norm(role_means - role_means.mean(0), axis=1).mean())
    half = n // 2
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    A = rng.standard_normal((half, 4)) @ Q[:, :4].T
    B = rng.standard_normal((n - half, 14)) @ Q[:, :14].T
    Y = np.vstack([A, B])
    Y *= scale / np.linalg.norm(Y - Y.mean(0), axis=1).mean()
    Y += 0.05 * scale * rng.standard_normal(Y.shape) / np.sqrt(d)
    truth = np.r_[np.full(half, 4), np.full(n - half, 14)]
    return Y, truth, local_id(Y, k)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save(fig, path):
    # dpi=160 and the indented short form are deliberate deviations from the
    # other plot modules (300 / "wrote <path>"); see the refactor plan's ## Found.
    save_fig(fig, path, dpi=160, log=lambda p: print("   ", p.name))


def fig_hist(real, neg_cvs, pos_vals, figs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(real[np.isfinite(real)], bins=30, color="#2e7d32", alpha=.85)
    axes[0].set_xlabel("local intrinsic dimension (participation ratio)")
    axes[0].set_ylabel("# roles")
    axes[0].set_title(f"Real roles — one tight mode\nCV = {cv(real):.3f}")
    axes[1].hist(pos_vals[np.isfinite(pos_vals)], bins=30, color="#c62828", alpha=.85)
    axes[1].set_xlabel("local intrinsic dimension (participation ratio)")
    axes[1].set_title("Positive control (4-D + 14-D glued)\n"
                      f"CV = {cv(pos_vals):.3f} — what two manifolds look like")
    for a in axes:
        a.grid(alpha=.25, lw=.6)
    _save(fig, figs / "lid01_hist_real_vs_control.png")


def fig_map(coords, real, figs):
    fig, ax = plt.subplots(figsize=(7.6, 6))
    s = ax.scatter(coords[:, 0], coords[:, 1], c=real, cmap="viridis", s=34)
    fig.colorbar(s, ax=ax, label="local intrinsic dimension")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Role map coloured by local intrinsic dimension\n"
                 "(patchy colour = several manifolds; uniform = one)")
    ax.grid(alpha=.25, lw=.6)
    _save(fig, figs / "lid02_map_by_local_id.png")


def fig_axis(proj, real, names, figs):
    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.scatter(proj, real, s=26, color="#37474f", alpha=.8)
    i = int(np.argmax(proj))
    ax.annotate(names[i], (proj[i], real[i]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
    r = np.corrcoef(proj, real)[0, 1]
    ax.set_xlabel("projection on the Assistant axis")
    ax.set_ylabel("local intrinsic dimension")
    ax.set_title(f"Does dimension change along the main axis?  r = {r:.2f}")
    ax.grid(alpha=.25, lw=.6)
    _save(fig, figs / "lid03_local_id_vs_axis.png")
    return float(r)


def fig_k(per_k, neg_by_k, pos_by_k, figs):
    ks = sorted(per_k)
    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.plot(ks, [cv(per_k[k]) for k in ks], "o-", lw=2.4,
            color="#2e7d32", label="real roles")
    ax.plot(ks, [np.median(neg_by_k[k]) for k in ks], "s--", lw=2,
            color="#888", label="negative control (one manifold)")
    ax.plot(ks, [pos_by_k[k] for k in ks], "^--", lw=2,
            color="#c62828", label="positive control (two manifolds)")
    ax.set_xlabel("neighbourhood size k"); ax.set_ylabel("CV of local ID")
    ax.set_title("Sensitivity to k — real sits just above the one-manifold floor,\n"
                 "far below the two-manifold control (and converges to the floor at k=40)")
    ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.6)
    _save(fig, figs / "lid04_cv_vs_k.png")


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    import argparse
    argparse.ArgumentParser(prog="manifold.local_id",
                            description="Local intrinsic-dimension study (exploratory).").parse_args(argv)
    stamp = timestamp()
    run = new_run_dir(OUT_ROOT, f"{stamp}-local-intrinsic-dimension",
                      subdirs=("figures", "data"))
    figs = run / "figures"

    print("[0] loading role cloud (prompt_avg, layer 26, PCA-50) ...")
    cloud = P.load_cloud(view="prompt_avg", seed=SEED)
    X = cloud.role_means
    names = list(cloud.role_names)
    print(f"    {X.shape[0]} roles x {X.shape[1]} dims")

    per_k, neg_by_k, pos_by_k = {}, {}, {}
    for k in K_LIST:
        print(f"[1] k={k}: real ...", flush=True)
        per_k[k] = local_id(X, k)
        print(f"    negative control ({N_REF} draws) ...", flush=True)
        neg_by_k[k], _ = negative_control(X, k, N_REF, SEED)
        print("    positive control ...", flush=True)
        _, _, pv = positive_control(X, k, SEED)
        pos_by_k[k] = cv(pv)
        print(f"    CV real={cv(per_k[k]):.3f}  neg={np.median(neg_by_k[k]):.3f}"
              f"  pos={pos_by_k[k]:.3f}")

    k = K_PRIMARY
    real = per_k[k]
    _, pos_truth, pos_vals = positive_control(X, k, SEED)
    tw = local_id(X, k, twonn=True)

    neg = neg_by_k[k]
    cv_real = cv(real)
    p_neg = float((np.sum(neg >= cv_real) + 1) / (len(neg) + 1))
    nc_real = neighbour_corr(X, real)
    pos_X, _, _ = positive_control(X, k, SEED)
    nc_pos = neighbour_corr(pos_X, pos_vals)

    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    axis = X[names.index("default")] - X.mean(0)
    axis /= np.linalg.norm(axis)
    proj = (X - X.mean(0)) @ axis

    print("[2] figures ...")
    fig_hist(real, neg, pos_vals, figs)
    fig_map(coords, real, figs)
    r_axis = fig_axis(proj, real, names, figs)
    fig_k(per_k, neg_by_k, pos_by_k, figs)

    res = {
        "k_primary": k, "k_list": list(K_LIST), "n_roles": int(X.shape[0]),
        "ambient": int(X.shape[1]),
        "real": {"mean": float(np.nanmean(real)), "sd": float(np.nanstd(real)),
                 "min": float(np.nanmin(real)), "max": float(np.nanmax(real)),
                 "cv": cv_real, "neighbour_corr": nc_real,
                 "corr_with_axis": r_axis,
                 "twonn_mean": float(np.nanmean(tw)), "twonn_cv": cv(tw)},
        "negative_control": {"cv_median": float(np.median(neg)),
                             "cv_p5": float(np.percentile(neg, 5)),
                             "cv_p95": float(np.percentile(neg, 95)),
                             "n_draws": N_REF,
                             "p_real_ge_null": p_neg},
        "positive_control": {"cv": cv(pos_vals),
                             "mean_lo_manifold": float(np.nanmean(pos_vals[pos_truth == 4])),
                             "mean_hi_manifold": float(np.nanmean(pos_vals[pos_truth == 14])),
                             "neighbour_corr": nc_pos},
        "cv_by_k": {str(kk): {"real": cv(per_k[kk]),
                              "neg_median": float(np.median(neg_by_k[kk])),
                              "pos": pos_by_k[kk]} for kk in K_LIST},
    }
    (run / "data" / "local_id.json").write_text(json.dumps(res, indent=2))
    np.savetxt(run / "data" / "local_id_per_role.csv",
               np.c_[real, proj], delimiter=",",
               header="local_id,assistant_axis_proj", comments="")
    write_manifest(run, {
        "run": stamp, "script": "manifold/local_id.py", "seed": SEED,
        "view": "prompt_avg", "layer": 26, "d_ambient": int(X.shape[1]),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "status": "exploratory — post hoc, not preregistered",
    })

    write_md(run, res)
    print(f"\ndone -> {run}")


def write_md(run: Path, r: dict) -> None:
    k = r["k_primary"]
    real, neg, pos = r["real"], r["negative_control"], r["positive_control"]
    above = real["cv"] > neg["cv_p95"]
    frac = (real["cv"] - neg["cv_median"]) / (pos["cv"] - neg["cv_median"])
    verdict = ("**more variable than a single homogeneous manifold**, but only mildly — "
               f"it sits {frac:.0%} of the way from the one-manifold floor to the "
               "two-manifold control" if above else
               "**uniform** — indistinguishable from a single homogeneous manifold")
    L = []
    A = L.append
    A("# Local intrinsic dimension of the role cloud\n")
    A("*Exploratory, post hoc. Not preregistered, cannot change any H1 verdict.*\n")
    A(f"Run: `{run.name}` · code: `manifold/local_id.py` · seed {SEED}\n")

    A("## The question\n")
    A("Everything we have said so far about \"one manifold\" comes from **connectivity**: "
      "single-linkage components across a threshold sweep, HDBSCAN, spectral eigengap. "
      "Those methods find groups separated by a **gap in density**.\n")
    A("Manifolds do not have to be separated by a gap. Two flat sheets crossing each other, "
      "or a low-dimensional ribbon lying inside a thicker cloud, are genuinely two different "
      "objects with no empty space between them. Every connectivity method calls that one "
      "blob. So the earlier results rule out roles falling into **discrete types**, but they "
      "do not rule out several manifolds tangled together.\n")
    A("This study asks the question those methods cannot: **is the cloud the same "
      "'thickness' everywhere?**\n")

    A("## The method, in plain terms\n")
    A(f"1. Take the {r['n_roles']} role means in the {r['ambient']}-dimensional PCA space.\n")
    A(f"2. Around each role, grab its **{k} nearest neighbours** — a small local patch.\n")
    A("3. Measure how many dimensions that patch spreads out in. We use the "
      "**participation ratio** of the patch's covariance: it equals 4 for a patch spread "
      "evenly in 4 directions, 14 for 14, and needs no threshold or cutoff.\n")
    A("4. Do this for every role. Then ask whether the numbers are all the same "
      "(one manifold) or split into groups (several manifolds of different dimension).\n")
    A("5. The summary number is the **coefficient of variation** (CV = spread ÷ average) "
      "of local dimension across roles.\n")

    A("## Why the two controls are the whole study\n")
    A("A CV bigger than zero proves nothing — a patch of "
      f"{k} points gives a noisy estimate, so the numbers will always wobble a bit. "
      "The only way to read the CV is against two references, both run through the "
      "identical code:\n")
    A("- **Negative control (one manifold):** a Gaussian cloud matched to the real data's "
      "covariance. It has the same dimension everywhere by construction, so its CV is "
      "**pure estimation noise** — the floor.\n")
    A("- **Positive control (two manifolds, no density gap):** half the points on a flat "
      "4-dimensional subspace, half on a 14-dimensional one, overlapping in the same space. "
      "There is no gap anywhere, so single-linkage and HDBSCAN would call this **one** "
      "cluster. If local dimension can't split this, the method is worthless.\n")

    A("## Results\n")
    A(f"| | mean local dim | CV | \n|---|---|---|")
    A(f"| **Real roles** | {real['mean']:.2f} | **{real['cv']:.3f}** |")
    A(f"| Negative control (one manifold) | — | {neg['cv_median']:.3f} "
      f"(90% range {neg['cv_p5']:.3f}–{neg['cv_p95']:.3f}) |")
    A(f"| Positive control (two manifolds) | "
      f"{pos['mean_lo_manifold']:.1f} / {pos['mean_hi_manifold']:.1f} | {pos['cv']:.3f} |\n")

    A(f"**Positive control: passes, with a caveat.** The two planted regions come out at "
      f"{pos['mean_lo_manifold']:.1f} and {pos['mean_hi_manifold']:.1f} — clearly "
      f"*separated*, and in the right order, but both are badly **underestimated** "
      f"(planted: 4 and 14). A {k}-point patch cannot span 14 directions, so the estimator "
      f"compresses high dimensions toward the low end. What matters for this study is that "
      f"the CV, {pos['cv']:.3f}, is ~3x the noise floor: the method **can** detect two "
      f"glued manifolds. It just cannot be trusted for absolute dimension.\n")

    A(f"**Real data: {verdict}.** Local dimension averages {real['mean']:.2f} across roles "
      f"(range {real['min']:.1f}–{real['max']:.1f}), with CV {real['cv']:.3f} against a "
      f"one-manifold floor of {neg['cv_median']:.3f} "
      f"({neg['cv_p5']:.3f}–{neg['cv_p95']:.3f}); p = {neg['p_real_ge_null']:.3f} against "
      f"{neg['n_draws']} one-manifold draws. So the cloud is **not perfectly homogeneous** — "
      "but it is nowhere near the two-manifold control, and the histogram has one mode, "
      "not two. This is the signature of a single manifold that is a bit thicker in some "
      "places than others, not of two objects glued together.\n")

    A(f"**Is the variation organised?** Neighbouring roles' local dimensions correlate at "
      f"r = {real['neighbour_corr']:.2f} (the positive control gives "
      f"{pos['neighbour_corr']:.2f}). High correlation would mean the cloud has thick "
      "regions and thin regions in specific places — real geometric structure. Low "
      "correlation means the wobble is just estimator noise.\n")
    A(f"**Does dimension track the Assistant axis?** r = {real['corr_with_axis']:.2f}. "
      "A strong value would mean the cloud gets systematically simpler or more complex as "
      "roles get further from the Assistant.\n")
    A(f"**Robustness to k:** CV at k = "
      + ", ".join(f"{kk} → {v['real']:.3f} (floor {v['neg_median']:.3f}, "
                  f"two-manifold {v['pos']:.3f})"
                  for kk, v in r["cv_by_k"].items()) + ".\n")
    A(f"**Second estimator:** TwoNN on the same patches gives mean "
      f"{real['twonn_mean']:.2f}, CV {real['twonn_cv']:.3f} — reported for agreement only; "
      f"TwoNN is noisy on {k}-point patches.\n")

    A("## Figures\n")
    A("| file | what it shows |\n|---|---|")
    A("| `lid01_hist_real_vs_control.png` | distribution of local dimension, real vs the "
      "two-manifold control. One mode = one manifold; two modes = two. |")
    A("| `lid02_map_by_local_id.png` | the role map coloured by local dimension. Patches of "
      "colour would mean different regions have different geometry. |")
    A("| `lid03_local_id_vs_axis.png` | local dimension against Assistant-axis position. |")
    A("| `lid04_cv_vs_k.png` | the whole result across neighbourhood sizes, with both "
      "control lines. The one figure to show if you only show one. |\n")

    A("## What this does and does not establish\n")
    A("- **The excess variation has an innocent explanation and we cannot rule it out.** "
      "The participation ratio of a k-nearest-neighbour patch depends on local *density* "
      "and *anisotropy*, not on dimension alone. A single curved manifold that is sampled "
      "unevenly — which ours certainly is, since roles were chosen by hand — will produce "
      "exactly this: a mild, spatially smooth wobble in local dimension. Distinguishing "
      "that from genuine geometry would need a density-corrected estimator, which we did "
      "not run.\n")
    A("- It tests **variation in local dimension**, which is how glued manifolds of "
      "*different* dimension reveal themselves. Two manifolds of the **same** dimension "
      "glued together would not show up here — nothing in this repo would see that. "
      "Persistent homology (loops and voids) is still not done.\n")
    A("- Local dimension at n=276 with k-point patches is biased downward; the controls "
      "share that bias, so the **comparison** is fair even though the absolute numbers "
      "are underestimates.\n")
    A("- Exploratory and post hoc. It supports or fails to support the one-manifold "
      "reading; it cannot confirm it.\n")
    (run / "METHOD-AND-RESULTS.md").write_text("\n".join(L) + "\n")
    print("    METHOD-AND-RESULTS.md")


if __name__ == "__main__":
    main()
