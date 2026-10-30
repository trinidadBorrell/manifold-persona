"""Experiment 1 (plan 2026-10-29, section 3): hold out personas.

Input: `$PM/CPP/panel_<view>_L<L>.csv` (one row per role x centring, from `panel.py --collect`).
Target: `axis_proj` = <centroid_p - mean centroid, assistant axis>, recomputed on each training fold from
the stored centroids (`axis_<view>_L<L>.npz`), so held-out personas never enter the axis.

For every metric m: random orderings of the 275 personas (the `default` role is excluded, it defines the axis),
fit on the first i, test on the rest.
  univariate      axis_proj = a + b m                                   -> R2_test, RMSE_test, Spearman_test
  reverse         m = a' + b' axis_proj                                 -> R2_test in the metric direction
  scale-adjusted  axis_proj = a + b m + g1 log_total_var + g2 centroid_norm_raw + g3 response_chars
                  (centroid_norm_raw = norm of the raw centroid; the question-centred centroid norm is the distance to the
                  default centroid, i.e. nearly the target itself, so it is not a scale covariate)
                  reported as delta R2_test over the scale-only model
  ridge (exploratory) over the whole standardised panel, lambda by inner 5-fold CV
Baselines: training mean (R2 = 0), permutation null (axis_proj shuffled over personas), scale-only model.
Decision rule at i = 200: median R2_test >= 0.25, above the 95th percentile of the permutation null,
delta R2 over scale-only >= 0.05.

    python -m cloud_predicts_position.exp1 --layer 25 --view avg [--centring q] [--n-orders 100] [--n-perm 1000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cloud_predicts_position import data as D
from cloud_predicts_position.panel import FAMILIES, PRIMARY

I_GRID = (25, 50, 100, 150, 200, 250)
SCALE = ["log_total_var", "centroid_norm_raw", "response_chars"]   # raw centroid norm: see panel.collect
I_RULE = 200
RULE = dict(r2=0.25, delta=0.05)


def metric_columns(df):
    cols = [c for fam in FAMILIES.values() for c in fam if c in df.columns and c not in SCALE]
    return [c for c in cols if df[c].notna().mean() > 0.9 and df[c].std() > 0]


def axis_proj_fold(C, train_idx, i_default, all_idx):
    """Lu et al. axis on the training fold (default centroid minus the mean of the training roles' centroids)."""
    a = C[i_default] - C[train_idx].mean(0)
    a /= np.linalg.norm(a)
    return (C[all_idx] - C[train_idx].mean(0)) @ a


def ols_fit_predict(Xtr, ytr, Xte):
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.hstack([np.ones((len(Xte), 1)), Xte]) @ w


def r2(y, yhat, ref):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - ref) ** 2).sum())


def ridge_fit_predict(Xtr, ytr, Xte, rng, alphas=np.logspace(-2, 3, 11)):
    mx, sx, my = Xtr.mean(0), Xtr.std(0) + 1e-12, ytr.mean()
    Z, Zte = (Xtr - mx) / sx, (Xte - mx) / sx

    def fit(A, b, lam):
        return np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ b)

    folds = np.array_split(rng.permutation(len(ytr)), 5)
    best, best_err = None, np.inf
    for lam in alphas:
        err = 0
        for t in folds:
            tr = np.setdiff1d(np.arange(len(ytr)), t)
            w = fit(Z[tr], ytr[tr] - my, lam)
            err += ((Z[t] @ w + my - ytr[t]) ** 2).sum()
        if err < best_err:
            best, best_err = lam, err
    return Zte @ fit(Z, ytr - my, best) + my, best


def run(df, C, roles, cols, n_orders, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    i_default = roles.index(D.DEFAULT)
    df = df.set_index("role").loc[[r for r in roles if r != D.DEFAULT]]
    idx = np.array([roles.index(r) for r in df.index])              # into C
    M = df[cols].to_numpy(float)
    M = np.where(np.isnan(M), np.nanmedian(M, 0), M)
    S = df[SCALE].to_numpy(float)
    n = len(df)
    rec = {c: {i: dict(r2=[], rmse=[], rho=[], r2_rev=[], r2_adj=[], r2_scale=[]) for i in I_GRID} for c in cols}
    ridge = {i: dict(r2=[], lam=[]) for i in I_GRID}
    perm = {i: [] for i in I_GRID}
    for o in range(n_orders):
        order = rng.permutation(n)
        for i in I_GRID:
            tr, te = order[:i], order[i:]
            y = axis_proj_fold(C, idx[tr], i_default, idx)
            ytr, yte = y[tr], y[te]
            ref = ytr.mean()
            ysc = ols_fit_predict(S[tr], ytr, S[te])
            r2_scale = r2(yte, ysc, ref)
            for j, c in enumerate(cols):
                yhat = ols_fit_predict(M[tr, j:j + 1], ytr, M[te, j:j + 1])
                mhat = ols_fit_predict(ytr[:, None], M[tr, j], yte[:, None])
                yadj = ols_fit_predict(np.hstack([M[tr, j:j + 1], S[tr]]), ytr, np.hstack([M[te, j:j + 1], S[te]]))
                R = rec[c][i]
                R["r2"].append(r2(yte, yhat, ref)); R["rmse"].append(float(np.sqrt(((yte - yhat) ** 2).mean())))
                R["rho"].append(float(spearmanr(yte, yhat)[0]) if len(te) > 2 else np.nan)
                R["r2_rev"].append(r2(M[te, j], mhat, M[tr, j].mean()))
                R["r2_adj"].append(r2(yte, yadj, ref)); R["r2_scale"].append(r2_scale)
            yr, lam = ridge_fit_predict(M[tr], ytr, M[te], rng)
            ridge[i]["r2"].append(r2(yte, yr, ref)); ridge[i]["lam"].append(float(lam))
        if o % 10 == 0:
            print("ordering", o, flush=True)
    # permutation null: shuffle the target over personas, one random ordering per permutation, univariate R2 max over metrics
    # and per metric at I_RULE
    perm_metric = {c: [] for c in cols}
    for p in range(n_perm):
        order = rng.permutation(n)
        i = I_RULE
        tr, te = order[:i], order[i:]
        y = axis_proj_fold(C, idx[tr], i_default, idx)
        y = rng.permutation(y)
        ytr, yte = y[tr], y[te]
        vals = []
        for j, c in enumerate(cols):
            yhat = ols_fit_predict(M[tr, j:j + 1], ytr, M[te, j:j + 1])
            v = r2(yte, yhat, ytr.mean()); vals.append(v); perm_metric[c].append(v)
        perm[i].append(max(vals))
    return rec, ridge, perm, perm_metric, df


def summarise(rec, ridge, perm, perm_metric, cols, centring):
    rows = []
    p95_any = float(np.percentile(perm[I_RULE], 95))
    for c in cols:
        fam = next(k for k, v in FAMILIES.items() if c in v)
        R = rec[c][I_RULE]
        med = {k: float(np.median(v)) for k, v in R.items()}
        lo, hi = np.percentile(R["r2"], [2.5, 97.5])
        p95 = float(np.percentile(perm_metric[c], 95))
        delta = med["r2_adj"] - med["r2_scale"]
        verdict = ("predicts position" if (med["r2"] >= RULE["r2"] and med["r2"] > p95 and delta >= RULE["delta"])
                   else "proxy for scale" if (med["r2"] >= RULE["r2"] and med["r2"] > p95)
                   else "no")
        rows.append(dict(metric=c, family=fam, primary=c in PRIMARY, centring=centring,
                         r2_test_median=med["r2"], r2_test_lo=float(lo), r2_test_hi=float(hi), rmse_test=med["rmse"],
                         spearman_test=med["rho"], r2_reverse=med["r2_rev"], r2_scale_only=med["r2_scale"],
                         r2_metric_plus_scale=med["r2_adj"], delta_r2_over_scale=delta,
                         perm_p95_this_metric=p95, perm_p95_max_over_metrics=p95_any, verdict=verdict,
                         **{f"r2_i{i}": float(np.median(rec[c][i]["r2"])) for i in I_GRID}))
    tab = pd.DataFrame(rows).sort_values("r2_test_median", ascending=False)
    rid = {str(i): dict(r2_median=float(np.median(ridge[i]["r2"])), r2_lo=float(np.percentile(ridge[i]["r2"], 2.5)),
                        r2_hi=float(np.percentile(ridge[i]["r2"], 97.5)), lambda_median=float(np.median(ridge[i]["lam"])))
           for i in I_GRID}
    return tab, rid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="avg")
    ap.add_argument("--centring", default="q,raw")
    ap.add_argument("--n-orders", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args(argv)
    panel = pd.read_csv(D.OUT / f"panel_{a.view}_L{a.layer}.csv")
    ax = np.load(D.OUT / f"axis_{a.view}_L{a.layer}.npz", allow_pickle=True)
    roles, C = list(ax["roles"]), ax["centroids"].astype(np.float64)
    outdir = D.OUT / "exp1"
    outdir.mkdir(exist_ok=True)
    for centring in a.centring.split(","):
        df = panel[panel.centring == centring].copy()
        df["response_chars"] = df["response_chars"].astype(float)
        cols = metric_columns(df)
        print(centring, len(cols), "metrics", flush=True)
        rec, ridge, perm, perm_metric, dfi = run(df, C, roles, cols, a.n_orders, a.n_perm)
        tab, rid = summarise(rec, ridge, perm, perm_metric, cols, centring)
        tag = f"{a.view}_L{a.layer}_{centring}"
        tab.to_csv(outdir / f"exp1_{tag}.csv", index=False)
        curves = {c: {str(i): dict(r2=rec[c][i]["r2"], r2_scale=rec[c][i]["r2_scale"], r2_adj=rec[c][i]["r2_adj"]) for i in I_GRID}
                  for c in cols}
        (outdir / f"exp1_{tag}_curves.json").write_text(json.dumps(dict(curves=curves, ridge=rid, perm_max=perm[I_RULE],
                                                                          perm_metric=perm_metric, i_grid=list(I_GRID))))
        print(tab[["metric", "family", "r2_test_median", "delta_r2_over_scale", "perm_p95_this_metric", "verdict"]].head(15).to_string())
        print("ridge", rid)


if __name__ == "__main__":
    main()
