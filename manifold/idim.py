"""Metric M2: intrinsic dimension vs n, with a matched small-N reference.

Why this module exists instead of reusing
``exploratory/assistant_axis/01_intrinsic_dimension.py::estimate_id_only``:
that helper hardcodes ``skdim.id.MLE(K=20)``, which stops being LOCAL for a
cloud of n <= 21 points. skdim clamps the neighbourhood to n-1, so every
point's "neighbourhood" is the whole cloud. That covers the entire small-n end
of this sweep (n=10, n=25), which is the part the question is about. Here the
neighbourhood adapts as ``n_neighbors = min(10, n-2)`` (the plan's k=10,
clipped to what n allows). The SAME size is used for the reference draw at
that n, so the comparison stays fair. Note it is passed to ``fit``, not to the
constructor: in skdim 0.3.6 ``MLE(K=...)`` is read only when
``neighborhood_based=False``, and the default is True, so a constructor K is
silently ignored.

The reference is the load-bearing part. ID estimators are biased *downward*
when N is small (they need N >> 2^d). So "ID falls as n falls" is the null
expectation, not a finding. We therefore estimate ID on `n_ref` draws of n
points from a Gaussian matched to the full role-mean covariance — a cloud with
NO manifold structure, only the data's second-order statistics — and report
real ID against that band. Real ID tracking the band = small-N bias. Real ID
below the band = genuine simplification.
"""
from __future__ import annotations

import numpy as np
import skdim

ESTIMATORS = ("TwoNN", "MLE", "lPCA")


def _build(name: str, n: int):
    """(estimator, how to read it, fit kwargs).

    The fit kwargs exist for MLE: its neighbourhood size is a ``fit`` argument
    (``n_neighbors``, default 20), not the constructor's ``K``. TwoNN and lPCA
    are global estimators here — their ``fit`` takes no neighbourhood at all.
    """
    if name == "TwoNN":
        return skdim.id.TwoNN(), "global", {}
    if name == "MLE":
        return skdim.id.MLE(), "pw_mean", {"n_neighbors": max(2, min(10, n - 2))}
    if name == "lPCA":
        return skdim.id.lPCA(), "global", {}
    raise ValueError(name)


def _read(est, how):
    v = float(np.nanmean(est.dimension_pw_)) if how == "pw_mean" else float(est.dimension_)
    return None if (v != v or v <= 0) else v


def id_estimates(X: np.ndarray) -> dict:
    """TwoNN / MLE / lPCA on a point cloud. None where the estimator fails."""
    n = X.shape[0]
    out = {}
    for name in ESTIMATORS:
        try:
            est, how, fit_kw = _build(name, n)
            out[name] = _read(est.fit(np.asarray(X, dtype=np.float64), **fit_kw), how)
        except Exception as e:  # noqa: BLE001  estimators are brittle at small n
            out[name] = None
            print(f"    [ID {name}] n={n} failed: {e}")
    return out


def gaussian_reference(role_means: np.ndarray, n: int, n_ref: int = 100,
                       seed: int = 0) -> dict:
    """ID of `n` points drawn from N(mean, cov) of the FULL role-mean cloud.

    Structureless by construction, matched in n and in second-order statistics.
    Returns {est: {"median":, "q25":, "q75":}} over `n_ref` draws.
    """
    rng = np.random.default_rng(seed)
    mu = role_means.mean(0)
    cov = np.cov(role_means, rowvar=False)
    # symmetric sqrt via eigendecomposition (cov is PSD; clip tiny negatives)
    w, V = np.linalg.eigh(cov)
    A = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    draws = {k: [] for k in ESTIMATORS}
    for _ in range(n_ref):
        Z = rng.standard_normal((n, role_means.shape[1]))
        vals = id_estimates(mu + Z @ A.T)
        for k, v in vals.items():
            if v is not None:
                draws[k].append(v)
    out = {}
    for k, vs in draws.items():
        if vs:
            a = np.asarray(vs)
            out[k] = {"median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
                      "q75": float(np.percentile(a, 75)), "n_ok": len(vs)}
        else:
            out[k] = {"median": None, "q25": None, "q75": None, "n_ok": 0}
    return out
