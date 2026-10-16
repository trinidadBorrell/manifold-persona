"""Four ways to ask "how close is this role to the Assistant?".

Every result in this study is a correlation against closeness to `default`. If
all the correlations come from ONE definition of closeness, the study is really
a study of that definition. So four are computed, and the ladder reports all
four side by side. They disagree enough to be worth the panels.

    axis_proj      mean(role) . axis_unit        the paper's, a projection
    cos_centroid   cos(mean(role), mean(default))  direction between centroids
    mknn_align     shared nearest neighbours     do the two clouds order the
                                                 40 questions the same way?
    cka            linear CKA                    do the two clouds have the
                                                 same similarity structure?

THE FIRST TWO ARE ABOUT WHERE A ROLE SITS. THE LAST TWO ARE ABOUT ITS SHAPE.
----------------------------------------------------------------------------
`axis_proj` and `cos_centroid` collapse a role to one point and ask where that
point is. `mknn_align` and `cka` never collapse it: they compare the role's
whole 40-point response cloud against `default`'s, question by question. A role
can sit far from `default` and still answer the 40 questions in the same
relative order, and only the last two can see that.

PAIRING (chosen 2026-08-03)
---------------------------
mKNN and CKA both need row-aligned samples. All 276 roles were asked the SAME
40 questions, so the clouds are paired on question id after averaging each
role's 5 instruction phrasings:

    role R   -> 40 points, one per question
    default  -> 40 points, one per question

The alternative — all 200 points paired by (instruction slot, question) — was
rejected: instruction slot 0 of `poet` and slot 0 of `default` are unrelated
sentences, so that pairing would be partly arbitrary. n = 40 is smallish for
CKA; this is recorded rather than hidden.

WHY cos_centroid IS MEAN-CENTRED
--------------------------------
Raw activation vectors share a large common component, so the raw cosine
between any two role centroids lives in [0.873, 0.999] with sd 0.013 — it
technically varies but is dominated by what every role has in common.
Subtracting the mean over the 276 roles first spreads it to [-0.721, +0.904],
sd 0.466. Both are computed; the mean-centred one is the predictor and
`cos_centroid_raw` is kept beside it. They correlate at only r = 0.63, so this
is a real choice and not a rescaling.
"""
from __future__ import annotations

import numpy as np

MKNN_K = 10          # neighbours per point, of the 40 available
SEED = 0


def question_means(X: np.ndarray, quest: np.ndarray) -> np.ndarray:
    """Collapse a role's cloud to one point per question, in question order.

    Rows come back sorted by question code, which is what makes two roles'
    outputs row-aligned: every role was asked the same 40 questions.
    """
    return np.stack([X[quest == q].mean(0) for q in range(int(quest.max()) + 1)])


def cos_to_reference(M: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine between every row of `M` and the reference vector."""
    n = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return n @ (ref / max(float(np.linalg.norm(ref)), 1e-12))


def mknn_align(A: np.ndarray, B: np.ndarray, k: int = MKNN_K) -> float:
    """Shared-nearest-neighbour alignment of two row-aligned clouds.

    For each question i, take its k nearest neighbours among the other
    questions in cloud A, and again in cloud B, and score the overlap:

        align = mean_i |knn_A(i) & knn_B(i)| / k

    1.0 means the two roles order the 40 questions identically; ~k/(n-1) is
    what random agreement gives (0.256 at k=10, n=40). Neighbours are found by
    COSINE distance, matching `cos_centroid` — the two clouds sit at different
    radii and Euclidean neighbours would partly read that difference.

    This is the alignment measure from the representation-similarity literature
    (Huh et al. 2024); it is rank-based, so it is invariant to any monotone
    rescaling of distances and to rotation of either space.
    """
    def knn(M):
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        d = 1.0 - Mn @ Mn.T
        np.fill_diagonal(d, np.inf)          # a point is not its own neighbour
        return np.argsort(d, axis=1)[:, :k]
    ka, kb = knn(A), knn(B)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ka, kb)]))


def _gram_centred(M: np.ndarray, normalise_rows: bool = False) -> np.ndarray:
    if normalise_rows:
        M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    M = M - M.mean(0)
    K = M @ M.T
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka(A: np.ndarray, B: np.ndarray, normalise_rows: bool = False) -> float:
    """Centred Kernel Alignment between two row-aligned clouds.

        CKA = <K, L>_F / (||K||_F ||L||_F)

    where K and L are the doubly-centred Gram matrices. This asks whether the
    two roles find the SAME PAIRS of questions similar, ignoring rotation and
    isotropic scaling of either representation. It is computed through the
    40x40 Gram matrices rather than the 2048x2048 covariances, which is both
    cheaper and the standard formulation.

    ``normalise_rows=True`` gives the COSINE-kernel variant the user asked
    about: L2-normalising the rows first makes the linear kernel a cosine
    kernel, so the answer depends only on the angles between question vectors
    and not on their lengths. It is a legitimate CKA — the kernel is still
    positive semi-definite — and it is reported beside the linear one rather
    than instead of it, because the two answer slightly different questions.
    """
    K = _gram_centred(A, normalise_rows)
    L = _gram_centred(B, normalise_rows)
    denom = np.linalg.norm(K) * np.linalg.norm(L)
    return float((K * L).sum() / denom) if denom > 0 else np.nan


def cloud_closeness(clouds: dict, factors: dict, roles: list,
                    reference: str = "default", k: int = MKNN_K) -> dict:
    """mKNN alignment and CKA of every role's response cloud against `reference`.

    Returns {role: {"mknn_align": ..., "cka": ..., "cka_cosine": ...}}. The
    reference role scores 1.0 against itself by construction and is excluded
    from every fit downstream, exactly as it already is for `axis_proj`.
    """
    qm = {r: question_means(clouds[r], factors[r][1]) for r in roles}
    ref = qm[reference]
    return {r: {"mknn_align": mknn_align(qm[r], ref, k),
                "cka": cka(qm[r], ref),
                "cka_cosine": cka(qm[r], ref, normalise_rows=True)}
            for r in roles}
