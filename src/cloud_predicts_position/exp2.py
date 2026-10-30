"""Experiment 2 (plan 2026-10-29, section 4): hold out questions.

For a random ordering of the 240 questions (seed) and a count i: for every persona,
  m_train  = metric from questions Q[:i] (all 5 prompts, 5 i points)
  m_test   = metric from i questions drawn from Q[i:] (matched point count)
  axis_proj from ALL questions (the position does not depend on i; the metric does)
Then across personas: fit axis_proj = a + b m_train and m_train = a' + b' axis_proj; evaluate the held-out
points (axis_proj, m_test) on the same lines (R2_held, RMSE_held / RMSE_train); reliability = Pearson r between
m_train and m_test. Reference curve: the reliability of axis_proj itself from Q[:i] vs Q[i:].

A reduced panel is used (small n makes ripser maxdim 2, 50-neighbour local ID and KDE meaningless at i = 5):
the nine primary metrics, the scale controls, and the cheap n-robust members of each family. H0 total
persistence is computed as the total length of the Euclidean minimum spanning tree (identical to the ripser
H0 total persistence, without ripser).

One job = one (seed, i): writes `$PM/CPP/exp2/metrics_<view>_L<L>_s<seed>_i<i>.csv`.
    python -m cloud_predicts_position.exp2 --layer 25 --view avg --seed 0 --i 40
    python -m cloud_predicts_position.exp2 --layer 25 --view avg --analyse
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from cloud_predicts_position import data as D
from cloud_predicts_position.panel import (D_PCA, K_NN, SEED, FAMILIES, PRIMARY, spectrum, design, anisotropy, alignment,
                                           consistency, curvature_metrics, density_metrics, id_estimates)

I_GRID = (5, 10, 20, 40, 60, 80, 120)
N_SEEDS = 10
SCALE = ["log_total_var", "centroid_norm_raw", "response_chars"]
REDUCED = PRIMARY + ["TwoNN", "PCA_participation_ratio", "top1_share", "knn_dist_cv", "frc_mean", "cos_pc1_axis",
                     "quest_frac", "mknn_align", "rsa_spearman", "split_half_cos_prompts", "log_total_var", "centroid_norm", "rms_radius"]


def mst_total(P):
    W = squareform(pdist(P))
    return float(minimum_spanning_tree(W).sum())


def subset_metrics(X, instr, quest, ax, centring, Qp, Qd):
    """The reduced panel on a cloud of 5 i points (X already centred the way `centring` says)."""
    n = len(X)
    out = {}
    Xc = X - X.mean(0)
    out.update(log_total_var=float(np.log((Xc ** 2).sum())), centroid_norm=float(np.linalg.norm(X.mean(0))),
               rms_radius=float(np.sqrt((Xc ** 2).sum() / n)))
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    s2 = s ** 2 / n
    sp = spectrum(s2)
    out.update({k: sp[k] for k in ("effective_rank", "PCA_participation_ratio", "top1_share")})
    d = min(D_PCA, n - 1)
    P = U[:, :d] * s[:d]
    k = min(K_NN, n - 2)
    ids = id_estimates(P)
    out.update(TwoNN=ids.get("TwoNN"), MLE=ids.get("MLE"))
    dens, _ = density_metrics(P, k=k)
    out.update(knn_dist_mean=dens["knn_dist_mean"], knn_dist_cv=dens["knn_dist_cv"])
    out["H0_total_persistence"] = mst_total(P)
    curv, _ = curvature_metrics(P, k=k)
    out.update(orc_mean=curv["orc_mean"], frc_mean=curv["frc_mean"])
    centroid_dir = X.mean(0) - (ax["default_centroid"] if centring == "raw" else 0.0)
    centroid_dir = centroid_dir / (np.linalg.norm(centroid_dir) + 1e-12)
    an = anisotropy(Xc, Vt[:min(10, len(Vt))], ax["axis"].astype(np.float64), centroid_dir, ax["default_top10"].astype(np.float64))
    out.update(var_frac_along_axis=an["var_frac_along_axis"], cos_pc1_axis=an["cos_pc1_axis"])
    de = design(Xc, instr, quest)
    out.update(interaction_frac=de["interaction_frac"], quest_frac=de["quest_frac"])
    al = alignment(Qp, Qd)
    out.update(cka_linear=al["cka_linear"], mknn_align=al["mknn_align"], rsa_spearman=al["rsa_spearman"])
    co = consistency(X, instr, quest, out["rms_radius"])
    out.update(between_prompt_disp=co["between_prompt_disp"], split_half_cos_prompts=co["split_half_cos_prompts"])
    return out


def run_cell(a):
    meta = D.metadata(a.layer)
    A = D.open_layer(a.layer, a.view)
    ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
    roles = D.roles(meta)
    outdir = D.OUT / "exp2"
    outdir.mkdir(exist_ok=True)
    f = outdir / f"metrics_{a.view}_L{a.layer}_s{a.seed}_i{a.i}.csv"
    if f.exists() and not a.force:
        print("exists", f); return
    rng = np.random.default_rng(1000 + a.seed)
    order = rng.permutation(np.unique(meta.question_idx))      # actual question ids (pool indices), not 0..N-1
    q_train = order[:a.i]
    q_test = rng.choice(order[a.i:], a.i, replace=False)
    Xd, instr_d, quest_d = D.cloud(A, meta, D.DEFAULT)
    Xd = Xd.astype(np.float64)
    C = ax["centroids"].astype(np.float64)
    rows = []
    t0 = time.time()
    for r in roles:
        X, instr, quest = D.cloud(A, meta, r)
        X = X.astype(np.float64)
        for split, qs in (("train", q_train), ("test", q_test)):
            m = np.isin(quest, qs)
            Qp_raw = np.stack([X[quest == q].mean(0) for q in qs])
            Qd_raw = np.stack([Xd[quest_d == q].mean(0) for q in qs])
            # position of this persona from this question subset (for the reference reliability curve)
            sub_centroid = X[m].mean(0)
            for centring in ("q", "raw"):
                if centring == "q" and r == D.DEFAULT:
                    continue
                Xs = (X[m] - Xd[m]) if centring == "q" else X[m]
                met = subset_metrics(Xs, instr[m], quest[m], ax, centring, Qp_raw, Qd_raw)
                met.update(role=r, split=split, centring=centring, seed=a.seed, i=a.i,
                           axis_proj_subset=float((sub_centroid - C.mean(0)) @ ax["axis"].astype(np.float64)))
                rows.append(met)
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"wrote {f} in {time.time() - t0:.0f}s")


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def analyse(a):
    outdir = D.OUT / "exp2"
    ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
    pos = pd.Series(ax["axis_proj"].astype(float), index=ax["roles"])
    files = sorted(outdir.glob(f"metrics_{a.view}_L{a.layer}_s*_i*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.role != D.DEFAULT]
    df["axis_proj"] = df.role.map(pos)
    rows = []
    for (centring, i, seed), g in df.groupby(["centring", "i", "seed"]):
        tr = g[g.split == "train"].set_index("role"); te = g[g.split == "test"].set_index("role")
        common = tr.index.intersection(te.index)
        tr, te = tr.loc[common], te.loc[common]
        y = tr.axis_proj.to_numpy(float)
        # reference: reliability of the position itself
        rows.append(dict(centring=centring, i=i, seed=seed, metric="axis_proj_itself",
                         reliability=float(pearsonr(tr.axis_proj_subset, te.axis_proj_subset)[0]),
                         r2_held_pos=np.nan, r2_train_pos=np.nan, rmse_ratio=np.nan, r2_held_metric=np.nan))
        for c in REDUCED:
            mt, me = tr[c].to_numpy(float), te[c].to_numpy(float)
            ok = np.isfinite(mt) & np.isfinite(me)
            if ok.sum() < 20 or mt[ok].std() == 0:
                continue
            mt, me, yy = mt[ok], me[ok], y[ok]
            a0, b0 = ols(mt, yy)                                  # position from metric
            r2_tr = 1 - ((yy - (a0 + b0 * mt)) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            r2_held = 1 - ((yy - (a0 + b0 * me)) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            rmse_ratio = np.sqrt(((yy - (a0 + b0 * me)) ** 2).mean()) / np.sqrt(((yy - (a0 + b0 * mt)) ** 2).mean())
            a1, b1 = ols(yy, mt)                                  # metric from position
            r2_held_m = 1 - ((me - (a1 + b1 * yy)) ** 2).sum() / ((me - me.mean()) ** 2).sum()
            rows.append(dict(centring=centring, i=i, seed=seed, metric=c, reliability=float(pearsonr(mt, me)[0]),
                             r2_held_pos=float(r2_held), r2_train_pos=float(r2_tr), rmse_ratio=float(rmse_ratio),
                             r2_held_metric=float(r2_held_m)))
    res = pd.DataFrame(rows)
    summ = res.groupby(["centring", "metric", "i"]).agg(reliability=("reliability", "median"),
                                                        reliability_lo=("reliability", lambda v: np.percentile(v, 5)),
                                                        reliability_hi=("reliability", lambda v: np.percentile(v, 95)),
                                                        r2_held_pos=("r2_held_pos", "median"), r2_train_pos=("r2_train_pos", "median"),
                                                        rmse_ratio=("rmse_ratio", "median"), r2_held_metric=("r2_held_metric", "median"),
                                                        n_seeds=("seed", "nunique")).reset_index()
    # smallest i that passes the rule: reliability >= 0.8 and rmse_ratio <= 1.2
    rule = summ[(summ.reliability >= 0.8) & ((summ.rmse_ratio <= 1.2) | summ.metric.eq("axis_proj_itself"))]
    first = rule.groupby(["centring", "metric"]).i.min().rename("smallest_i_passing").reset_index()
    tag = f"{a.view}_L{a.layer}"
    res.to_csv(outdir / f"exp2_{tag}_cells.csv", index=False)
    summ.merge(first, how="left").to_csv(outdir / f"exp2_{tag}_summary.csv", index=False)
    print(summ[summ.metric.isin(PRIMARY + ["axis_proj_itself"]) & (summ.centring == "q")]
          .pivot(index="metric", columns="i", values="reliability").round(2).to_string())
    print("wrote", outdir / f"exp2_{tag}_summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="avg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i", type=int, default=40)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.analyse:
        analyse(a)
    else:
        run_cell(a)


if __name__ == "__main__":
    main()
