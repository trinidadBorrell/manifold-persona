"""NESTED REGRESSION: does geometry predict assistant-likeness beyond cloud scale?

Plan: plans/2026-07-30-manifold-geometry-vs-assistant-axis.md (Experiment 4)

THE QUESTION THIS ANSWERS THAT THE LADDER CANNOT
------------------------------------------------
The ladder reports one correlation per metric. It cannot say how much of
assistant-likeness the panel explains *jointly*, and — more importantly — it
cannot say whether the panel beats the most boring alternative available:

    a role's cloud is simply bigger, or its mean vector simply longer.

Both are size statements, not shape statements, and both move a role along the
axis on their own. `mean_norm` in particular is a factor of the target by
construction (axis_proj = mean_norm x cos_axis), so it is the strongest honest
baseline in the data. If a scale-only model already explains most of the
variance, then "geometry tracks assistant-likeness" is really "size tracks
assistant-likeness" and the whole panel is decoration.

Four nested models, axis_proj as the dependent variable:

    M0  intercept only                        sanity; CV R2 should be ~0
    M1  scale only  (log_var, mean_norm)      THE BASELINE TO BEAT
    M2  geometry only  (the ~19 panel metrics)
    M3  both

The comparison that matters is M3 vs M1, not M2 vs M0.

The earlier baseline here was response length (`mean_tokens`, `trunc_rate`).
It was dropped on 2026-08-03: these are MEAN-pooled activations, so the token
count is divided out before the role's vector is formed, and length was never
a mechanism by which a longer answer moved a role along the axis.

WHY CROSS-VALIDATED R2 AND NOT IN-SAMPLE
----------------------------------------
19 predictors on n=275 roles will fit in-sample no matter what; OLS R2 here is
close to meaningless as a measure of signal. Both are reported side by side and
the prose uses the CV number. Ridge is used for the CV fit because the panel is
heavily collinear by construction (several dimension estimators measuring
related things), with alpha chosen by an INNER CV on the training folds only, so
no test fold ever informs the penalty.

VIF is reported for the OLS fit so the collinearity is visible rather than
implied: a coefficient with VIF 40 is not interpretable on its own, and readers
should be told that rather than left to assume otherwise.

Produces `regression_L<L>.json` -> fig05.

Usage:
    .venv/bin/python exploratory/per_persona/study_regression.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import geometry_columns

SEED = 0
N_FOLDS = 5
ALPHAS = np.logspace(-3, 4, 40)
SCALE_COLS = ["log_var", "mean_norm"]
TARGET = "axis_proj"

# The three design fractions sum to EXACTLY 1 by construction (the grid is
# balanced -- md/METHODS.md S1), so including all three makes the design matrix
# singular: VIF came back at 1e12 and no coefficient among them is identifiable.
# Drop one, as one would with a categorical's reference level. quest_frac is the
# one dropped because instr_frac and interaction_frac are the interpretable pair
# (phrasing sensitivity, persona-specific structure) and the dropped one is
# recoverable as 1 - the other two, so no information is lost. Ridge tolerated
# the singularity; OLS and VIF do not.
DROPPED_SIMPLEX = "quest_frac"


def cv_r2(X, y, n_boot: int = 2000):
    """Out-of-fold R2 of a standardised ridge, plus a bootstrap CI over roles.

    R2 is computed on the pooled out-of-fold predictions rather than averaged
    per fold, so it is a single honest number on the same scale as the
    in-sample R2 it is compared against.
    """
    if X.shape[1] == 0:                       # intercept-only model
        pred = np.full(len(y), np.nan)
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        for tr, te in kf.split(y):
            pred[te] = y[tr].mean()
    else:
        model = make_pipeline(StandardScaler(),
                              RidgeCV(alphas=ALPHAS))       # inner CV for alpha
        pred = cross_val_predict(model, X, y,
                                 cv=KFold(N_FOLDS, shuffle=True,
                                          random_state=SEED))
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        sr = float(((y[i] - pred[i]) ** 2).sum())
        st = float(((y[i] - y[i].mean()) ** 2).sum())
        if st > 0:
            boots.append(1 - sr / st)
    boots = np.asarray(boots)
    return r2, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def vif(X: np.ndarray, names) -> dict:
    """Variance inflation factor per predictor, from R2 of each on the others."""
    out = {}
    for j, nm in enumerate(names):
        others = np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            out[nm] = 1.0
            continue
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        out[nm] = float(1.0 / max(1e-12, 1 - r2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    args = ap.parse_args()
    run_dir = Path(args.outdir)
    L = args.label_layer

    df = pd.read_csv(run_dir / "data" / f"per_role_panel_L{L}.csv")
    panel = geometry_columns(df)
    if DROPPED_SIMPLEX in panel and {"instr_frac", "interaction_frac"} <= set(panel):
        panel = [c for c in panel if c != DROPPED_SIMPLEX]

    d = df[df["role"] != "default"].dropna(
        subset=panel + SCALE_COLS + [TARGET]).reset_index(drop=True)
    y = d[TARGET].to_numpy(float)
    print(f"n={len(d)} roles (default excluded), target={TARGET}, "
          f"{len(panel)} geometry predictors, {len(SCALE_COLS)} scale predictors")

    models = {
        "M0_intercept": [],
        "M1_scale_only": SCALE_COLS,
        "M2_geometry_only": panel,
        "M3_geometry_plus_scale": panel + SCALE_COLS,
    }
    out = {"exploratory": True, "target": TARGET, "n": int(len(d)),
           "n_folds": N_FOLDS, "seed": SEED, "models": {}}
    print(f"\n{'model':26s} {'CV R2':>8s} {'95% CI':>18s} {'in-sample R2':>13s} {'k':>4s}")
    for name, cols in models.items():
        X = d[cols].to_numpy(float) if cols else np.empty((len(d), 0))
        r2, lo, hi = cv_r2(X, y)
        ins = (LinearRegression().fit(X, y).score(X, y) if len(cols) else 0.0)
        out["models"][name] = {"predictors": cols, "cv_r2": r2,
                               "cv_r2_ci": [lo, hi], "in_sample_r2": ins,
                               "n_predictors": len(cols)}
        print(f"{name:26s} {r2:8.3f} [{lo:7.3f},{hi:7.3f}] {ins:13.3f} {len(cols):4d}")

    # The comparison the plan singles out.
    m1, m2 = out["models"]["M1_scale_only"], out["models"]["M2_geometry_only"]
    m3 = out["models"]["M3_geometry_plus_scale"]
    out["headline"] = {
        "cv_r2_scale_only": m1["cv_r2"],
        "cv_r2_geometry_only": m2["cv_r2"],
        "cv_r2_both": m3["cv_r2"],
        "gain_of_geometry_over_scale": m3["cv_r2"] - m1["cv_r2"],
        "clears_plan_bar_0.25": bool(m3["cv_r2"] >= 0.25),
    }
    print(f"\n  geometry's gain over scale alone: "
          f"{out['headline']['gain_of_geometry_over_scale']:+.3f} CV R2")
    print(f"  plan's 'would matter' bar (CV R2 >= 0.25): "
          f"{'MET' if out['headline']['clears_plan_bar_0.25'] else 'NOT MET'}")

    # OLS coefficients + VIF on the full model, for the collinearity picture.
    names = panel + SCALE_COLS
    Xs = StandardScaler().fit_transform(d[names].to_numpy(float))
    ols = LinearRegression().fit(Xs, y)
    v = vif(Xs, names)
    out["ols_full_model"] = {
        "standardized_coefficients": {n: float(c) for n, c in zip(names, ols.coef_)},
        "vif": v,
        "note": "VIF > 10 means the coefficient is not interpretable alone; "
                "read it beside the univariate ladder, never instead of it."}
    hi_vif = {k: round(x, 1) for k, x in sorted(v.items(), key=lambda t: -t[1])[:6]}
    print(f"\n  highest VIFs: {hi_vif}")

    json.dump(out, open(run_dir / "data" / f"regression_L{L}.json", "w"),
              indent=2, default=float)
    print(f"\nwrote regression_L{L}.json")


if __name__ == "__main__":
    main()
