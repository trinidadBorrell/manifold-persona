"""Shared helpers for the PER-PERSONA manifold study.

The assistant-axis study collapses each role to ONE mean point and studies the
276-point cloud *between* roles. This study does the opposite: it keeps the raw
cloud and asks, for each role separately, what the geometry *within* that role
looks like — one manifold, one intrinsic dimension and one clustering per role.

THE STRUCTURAL CONSTRAINT (read this before interpreting any number here)
------------------------------------------------------------------------
Each role owns exactly 25 raw points: 5 instruction phrasings x 5 shared
questions (``data/embeddings_roles/manifest.json``: n_questions=5). Those 25
points are a complete 5x5 factorial grid, so before any model geometry enters,
the cloud's rank is capped by the design:

    additive (no-interaction) rank = (5-1) + (5-1) = 8

That is not a bound we impose, it is what a two-factor grid *is*. Any ID
estimate on 25 gridded points is therefore reporting the grid unless the
interaction term carries real variance. :func:`design_fractions` measures
exactly that, and every script here reports it alongside the ID so the number
cannot be read without its caveat.

Two nulls are provided, and the second is the load-bearing one:

  GAUSSIAN null  — 25 points from a Gaussian matched to the pooled within-role
    covariance. Structureless. Answers "is the role cloud lower-dimensional
    than noise at this n?" It always says yes, because of the grid.

  DESIGN null    — 25 points synthesised as ``a[i] + b[j]`` from random
    instruction/question effects with the data's empirical effect covariances.
    Same 5x5 grid, same additive rank, NO persona-specific geometry. This is
    the null that matters: if a role's real ID sits inside this band, the
    per-persona "manifold" is the experiment's design and nothing else.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np

from manifold_persona.config import REPO_ROOT
from manifold_persona.common import load_points

FIGURES_DIR = REPO_ROOT / "exploratory" / "per_persona" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, the palette the rest of the repo's figures already use.
C_REAL, C_DESIGN, C_GAUSS = "#0072B2", "#D55E00", "#999999"
C_INSTR, C_QUEST, C_INTER = "#009E73", "#56B4E9", "#CC79A7"


def single_threaded():
    """Context manager pinning BLAS/OpenMP to one thread.

    Every loop in this study runs the same estimator over hundreds of TINY
    clouds (25 x 2048, and ~25 x 8 after PCA). At that size multithreaded BLAS
    is pure overhead — the threads spend their time synchronising over matrices
    that fit in cache. Measured on this machine, one role's full clustering
    suite takes 40 ms single-threaded and 182 ms with the default 10 threads,
    i.e. the whole 376-cloud sweep goes from ~68 s to ~15 s by using FEWER
    cores. Wrap the per-cloud loops, not the whole program: the one genuinely
    large operation here (eigh of a 2048x2048 covariance, once per null) does
    want its threads.
    """
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def timestamp() -> str:
    return datetime.datetime.now().strftime("%d-%b-%Y-%H%M")


def resolve_run_dir(outdir: str = None) -> Path:
    if outdir:
        p = Path(outdir)
    elif os.environ.get("MP_RUN_DIR"):
        p = Path(os.environ["MP_RUN_DIR"])
    else:
        p = FIGURES_DIR / timestamp()
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, name: str, run_dir: Path):
    out = Path(run_dir) / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    return out


def load_role_clouds(view: str = "prompt_avg", layer: int = None):
    """Return (roles, clouds, factors, manifest) for the RAW per-example cloud.

    ``aggregate="none"`` is mandatory here — the default role-mean collapse is
    precisely what this study is trying to look inside.

    clouds[role]  -> float64 [25, hidden]
    factors[role] -> (instr [25], quest [25]) contiguous 0..4 codes. The stored
        ``question_idx`` indexes the 240-question pool, not the 5 sampled ones,
        so it is re-coded here; ``instruction_idx`` is already 0..4 but is
        re-coded the same way so both factors are handled identically.
    """
    X, meta, manifest = load_points(view=view, layer=layer, aggregate="none")
    roles = sorted(meta["role"].unique())
    rv = meta["role"].values
    ii, qi = meta["instruction_idx"].values, meta["question_idx"].values
    clouds, factors = {}, {}
    for r in roles:
        m = rv == r
        clouds[r] = X[m].astype(np.float64)   # fp16 -> fp64: summing 25 fp16 rows overflows
        factors[r] = (np.unique(ii[m], return_inverse=True)[1],
                      np.unique(qi[m], return_inverse=True)[1])
    return roles, clouds, factors, manifest


def design_fractions(Xr: np.ndarray, instr: np.ndarray, quest: np.ndarray) -> dict:
    """Two-way ANOVA of a role's 25 points: how much is the 5x5 grid?

    Fits the additive model ``x_ij = grand + A_i + B_j`` and splits the total
    centred sum-of-squares into the instruction effect, the question effect and
    the residual (= interaction, the only term that is NOT forced by the grid).
    Fractions are of the total centred SS and the additive ones need not sum to
    exactly 1 minus interaction unless the design is balanced — it is (5x5).
    """
    Xc = Xr - Xr.mean(0)
    tot = float((Xc ** 2).sum())
    A = np.stack([Xc[instr == u].mean(0) for u in range(instr.max() + 1)])
    B = np.stack([Xc[quest == u].mean(0) for u in range(quest.max() + 1)])
    resid = Xc - (A[instr] + B[quest])
    return {"instr_frac": float((A[instr] ** 2).sum() / tot),
            "quest_frac": float((B[quest] ** 2).sum() / tot),
            "interaction_frac": float((resid ** 2).sum() / tot)}


def pca_stats(Xr: np.ndarray) -> dict:
    """Participation ratio + variance-threshold dims of one role's cloud."""
    Xc = Xr - Xr.mean(0)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eig = s ** 2
    cum = np.cumsum(eig) / eig.sum()
    return {"PCA_participation_ratio": float(eig.sum() ** 2 / (eig ** 2).sum()),
            "PCA_dim_90pct": int(np.searchsorted(cum, 0.90) + 1),
            "PCA_dim_95pct": int(np.searchsorted(cum, 0.95) + 1)}


def _effect_covariances(clouds, factors):
    """Empirical covariance of the instruction effects and of the question
    effects, pooled over roles. Used to build the DESIGN null."""
    A_all, B_all = [], []
    for r, Xr in clouds.items():
        instr, quest = factors[r]
        Xc = Xr - Xr.mean(0)
        A_all.append(np.stack([Xc[instr == u].mean(0) for u in range(instr.max() + 1)]))
        B_all.append(np.stack([Xc[quest == u].mean(0) for u in range(quest.max() + 1)]))
    return np.concatenate(A_all), np.concatenate(B_all)


def _gauss(rng, n: int, S: np.ndarray) -> np.ndarray:
    """`n` samples with covariance S @ S.T, i.e. ``randn(n, h) @ S.T``.

    Wrapped because numpy 2.0.2 on this machine's Apple Accelerate BLAS raises
    spurious divide-by-zero / overflow / invalid RuntimeWarnings on a matmul
    against a 2048x2048 factor even when every output value is finite (verified:
    |S| max ~3, all outputs finite). The flags come from the BLAS kernel, not
    from the arithmetic, so they are suppressed here — and the assertion below
    makes sure a genuine numerical failure still stops the run rather than
    hiding behind the suppression.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = rng.standard_normal((n, S.shape[0])) @ S.T
    assert np.isfinite(out).all(), "non-finite draw from the null model"
    return out


def _sqrt_cov(M: np.ndarray) -> np.ndarray:
    """Symmetric square root of cov(M) via eigendecomposition (clip tiny negatives).

    Same construction as ``manifold.idim.gaussian_reference``, but the factor is
    formed by column-scaling ``V * sqrt(w)`` instead of ``V @ diag(sqrt(w))``.
    Identical result; it avoids materialising and multiplying a dense 2048x2048
    diagonal, which on this machine's BLAS (numpy 2.0.2 on Apple Accelerate)
    also raised spurious divide-by-zero/overflow RuntimeWarnings on the zero
    entries even though every output value was finite.
    """
    w, V = np.linalg.eigh(np.cov(M, rowvar=False))
    return V * np.sqrt(np.clip(w, 0, None))


def design_null_draws(clouds, factors, n_draws: int = 100, seed: int = 0):
    """Yield ``n_draws`` synthetic 25-point role clouds with the grid but no persona.

    Each draw picks 5 random instruction effects and 5 random question effects
    from Gaussians matched to the pooled empirical effect covariances, then
    forms the full 5x5 additive grid ``a[i] + b[j]``. Interaction is exactly
    zero by construction, so a real role that scores like this band has no
    within-role structure beyond its design.

    Returns (generator of (X [25,h], instr, quest)).
    """
    A, B = _effect_covariances(clouds, factors)
    SA, SB = _sqrt_cov(A), _sqrt_cov(B)
    h = A.shape[1]
    rng = np.random.default_rng(seed)
    instr = np.repeat(np.arange(5), 5)
    quest = np.tile(np.arange(5), 5)

    def gen():
        for _ in range(n_draws):
            a, b = _gauss(rng, 5, SA), _gauss(rng, 5, SB)
            yield a[instr] + b[quest], instr, quest
    return gen()


def gaussian_null_draws(clouds, n_draws: int = 100, seed: int = 0):
    """Yield ``n_draws`` structureless 25-point clouds matched to the pooled
    WITHIN-role covariance (each role centred before pooling)."""
    W = np.concatenate([Xr - Xr.mean(0) for Xr in clouds.values()])
    S = _sqrt_cov(W)
    rng = np.random.default_rng(seed)

    def gen():
        for _ in range(n_draws):
            yield _gauss(rng, 25, S)
    return gen()


def band(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None and v == v], dtype=float)
    if a.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "n": int(a.size)}
