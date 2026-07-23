"""Shared helpers for the assistant-axis exploratory study.

Same shape as exploratory/persona_vectors/common.py, but the point cloud is the
276 character-role personas (data/embeddings_roles/), there is no polarity, and
we add an "assistant-axis" helper: a direction computed *in our own embedding
space* as mean(default-role points) - mean(all-role points), the analogue of
the paper's axis = mean(default) - mean(roles).
"""
from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# exploratory/assistant_axis/common.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from manifold_persona.config import ROLE_EMBEDDINGS_DIR  # noqa: E402
from manifold_persona.io import load_layer  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLOR = "#111111"     # the neutral Assistant baseline ("default" role)


def aggregate_by_role(X: np.ndarray, meta) -> tuple:
    """Collapse the per-example cloud to ONE mean vector per role.

    This is the paper's role-vector definition (arXiv:2601.10387 §2.1.2–2.1.3):
    each role vector is the mean over all its (system-prompt × extraction-question)
    rollouts, which averages away the question-content confound so that the
    remaining variation is *between roles*. Here each raw point is a
    (role, instruction, question) prompt activation; we average the 25 rows per
    role (5 instructions × 5 questions) into a single role-mean point.

    Returns (X_role [n_roles, hidden] float32, meta_role DataFrame[role, is_default]).
    Row order = sorted(unique role), so it is identical across all scripts.
    """
    roles = sorted(meta["role"].unique())
    role_vals = meta["role"].values
    Xr = np.zeros((len(roles), X.shape[1]), dtype=np.float32)
    for i, r in enumerate(roles):
        # upcast to float32 BEFORE averaging: the raw cloud is fp16 and summing
        # ~25 rows overflows fp16 (max ~65504) → inf. Accumulate in float32.
        Xr[i] = X[role_vals == r].astype(np.float32).mean(0)
    meta_role = pd.DataFrame({"role": roles,
                              "is_default": [r == "default" for r in roles]})
    return Xr, meta_role


def load_points(view: str = "prompt_avg", layer: int = None,
                in_dir=ROLE_EMBEDDINGS_DIR, aggregate: str = None):
    """Return (X [N,hidden] float32, meta DataFrame, manifest dict).

    By default (``aggregate="role"``, overridable via the ``MP_AGGREGATE`` env
    var) the raw per-example cloud is collapsed to **one mean point per role**
    (mean across answers) — see :func:`aggregate_by_role`. Pass
    ``aggregate="none"`` for the raw 6,900-point cloud.
    """
    # MP_ROLE_DIR lets the whole exploratory stack point at the response-token
    # cloud (data/embeddings_roles_resp) instead of the prompt-token default,
    # with no per-script flags — used by the paper-matched recompute.
    in_dir = os.environ.get("MP_ROLE_DIR", in_dir)
    X, meta, manifest = load_layer(view=view, layer=layer, in_dir=in_dir)
    if aggregate is None:
        aggregate = os.environ.get("MP_AGGREGATE", "role")
    if aggregate == "role":
        X, meta = aggregate_by_role(X, meta)
    return X, meta, manifest


def ward_families(C: np.ndarray, k: int = 15):
    """Ward-linkage hierarchical families of role centroids.

    Returns (labels [n], Z linkage). Deterministic, so every script that calls
    this on the same standardized role means gets the same family partition.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(C, method="ward")
    return fcluster(Z, t=k, criterion="maxclust"), Z


def clusters_path(view: str, layer: int, in_dir=ROLE_EMBEDDINGS_DIR) -> Path:
    """Path of the role-mean cluster-assignment parquet (kept distinct from the
    old per-example ``clusters_<view>_L<layer>.parquet`` so old runs stay valid)."""
    return Path(in_dir) / f"clusters_rolemean_{view}_L{layer}.parquet"


def center(X: np.ndarray) -> np.ndarray:
    """Mean-centre the role means (subtract the mean vector across roles).

    This is the paper's exact normalization (arXiv:2601.10387 §2.1.3:
    "standardized these role vectors by subtracting the mean vector across
    roles") — i.e. **covariance** PCA. We deliberately do NOT z-score: dividing
    by per-feature std reweights all 2048 residual-stream features (which share
    one natural scale) and destroys the covariance geometry the Assistant Axis
    lives in — empirically it drops |cos(PC1, axis)| from ~0.87 to ~0.19.
    Returns float64 (also avoids a spurious fp32 matmul overflow in sklearn SVD).
    """
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(0, keepdims=True)


def assistant_axis(X: np.ndarray, meta) -> np.ndarray:
    """axis = mean(default points) - mean(all role points), unit-normalized.

    Points with role == 'default' are the neutral Assistant baseline. Positive
    projection => more Assistant-like (toward default); negative => more in-role.
    """
    is_default = meta["role"].values == "default"
    if is_default.sum() == 0:
        raise ValueError("No 'default' role points found to define the axis.")
    ax = X[is_default].mean(0) - X.mean(0)
    n = np.linalg.norm(ax)
    return ax / (n + 1e-8)


def project(X: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return X @ axis


def role_centroids(X: np.ndarray, meta):
    """Return (roles list, C [n_roles, hidden]) mean vector per role."""
    roles = sorted(meta["role"].unique())
    C = np.stack([X[meta["role"].values == r].mean(0) for r in roles])
    return roles, C


def distinct_colors(n: int):
    """n visually distinct RGBA colors by concatenating tab20 variants."""
    import matplotlib.pyplot as plt
    maps = ["tab20", "tab20b", "tab20c"]
    cols = []
    for m in maps:
        cols.extend(list(plt.get_cmap(m).colors))
    if n <= len(cols):
        return cols[:n]
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


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


def savefig(fig, name: str, run_dir: Path = FIGURES_DIR):
    out = Path(run_dir) / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    return out
