"""Pod C map statistics: role maps from prompt cells, cross-half agreement, paired bootstrap draws, the
one-map null, reliability-normalised agreement, CKA, and the G5 synthetic cases.

Design: THRESHOLDS.md here ("Inference units", "Calibration", D1, D2, G5) and pod A THRESHOLDS.md (D2).
- Cells: replicates are averaged inside each prompt (role, question, wording) over valid rows; a prompt with no
  valid row is missing.
- Role vector on one half side: the mean over its present prompts, with questions weighted by their bootstrap
  multiplicity and wordings by the side mask; then centred across the roles.
- Map: cosine distances between role vectors; Spearman r over role pairs (distinct sampled roles in a
  bootstrap draw, the E3 rule).
- Cross-half r(a, b): a on side A against b on side B, then the reverse, averaged; then averaged over the
  halves. Reliability rel(a) = r(a, a).
- Draws: 2,000 paired draws resample roles and questions once and serve every checkpoint; draw 0 is the data.
- Null: one checkpoint's role identities permuted (9,999), the same permutation in both directions and halves.
Everything runs on inner products (Gram matrices), so any feature width works.
"""
import json

import numpy as np
from scipy.stats import rankdata

N_BOOT, N_PERM, SEED = 2000, 9999, 20261029
CHUNK = 100


# ---- cells and halves

def cells(X, keys, roles, questions, mask=None, n_wordings=5):
    """Prompt means S [roles, questions, wordings, width] (0 where missing) and presence N (1 or 0)."""
    ri, qi = {r: k for k, r in enumerate(roles)}, {q: k for k, q in enumerate(questions)}
    idx, rows = [], []
    for j, key in enumerate(keys):
        role, _, w, q, _ = json.loads(key)
        if role in ri and (mask is None or mask[j]) and np.isfinite(X[j]).all():
            idx.append((ri[role], qi[q], w))
            rows.append(j)
    S = np.zeros((len(roles), len(questions), n_wordings, X.shape[1]))
    cnt = np.zeros(S.shape[:3])
    if rows:
        c, q, w = np.array(idx).T
        np.add.at(S, (c, q, w), np.asarray(X[rows], np.float64))
        np.add.at(cnt, (c, q, w), 1.0)
    N = (cnt > 0).astype(float)
    return S / np.maximum(cnt, 1)[..., None], N


def halves_from(fixed, question_only=False):
    """Sides ((qA, wA), (qB, wB)) from the fixed (qa, wa) masks; question_only keeps all wordings on both sides."""
    out = []
    for qa, wa in fixed:
        qa, wa = np.asarray(qa, bool), np.asarray(wa, bool)
        w_all = np.ones_like(wa)
        out.append(((qa, w_all), (~qa, w_all)) if question_only else ((qa, wa), (~qa, ~wa)))
    return out


def common_mask(N, qmask, wmask):
    """Questions on a side where every role has a present prompt (the F4 half-support rule)."""
    return qmask & (N[:, :, wmask].sum(2) > 0).all(0)


# ---- Gram matrices

class Side:
    """One checkpoint on one half side: inner products of its question sums over the side's questions only, so a
    draw costs no feature work."""

    def __init__(self, S, N, qmask, wmask):
        q = np.where(qmask)[0]
        self._set(q, S[:, q][:, :, wmask].sum(2).reshape(S.shape[0] * len(q), S.shape[-1]), N[:, q][:, :, wmask].sum(2))

    @classmethod
    def from_sums(cls, q, T, Nq):
        """q: the side's question indices; T: question sums [roles * len(q), width], dense or sparse; Nq: prompts
        per role and question [roles, len(q)]."""
        self = cls.__new__(cls)
        self._set(q, T, Nq)
        return self

    def _set(self, q, T, Nq):
        T = T.astype(np.float64)  # hidden states share a large common part; float32 sums would blur role differences
        K = T @ T.T
        K = K.toarray() if hasattr(K, "toarray") else K
        self.q, self.Nq = np.asarray(q, int), np.asarray(Nq, float)
        C, Q = len(self.Nq), len(self.q)
        self.K = np.asarray(K, float).reshape(C, Q, C * Q)

    def grams(self, W):
        """Centred role Gram matrices [b, C, C] for question weights W [b, all questions]; NaN where a role has
        no prompt (or the side has no question)."""
        C, Q = self.Nq.shape
        w = W[:, self.q]
        den = w @ self.Nq.T
        with np.errstate(divide="ignore", invalid="ignore"):
            Wn = w[:, None, :] / den[:, :, None]
        tmp = np.einsum("bcq,cqk->bck", Wn, self.K)
        G = centre(np.einsum("bceq,beq->bce", tmp.reshape(len(W), C, C, Q), Wn))  # exact symmetry: mirrored pairs tie
        G[(den == 0).any(1)] = np.nan
        return G


def dist(G):
    d = np.sqrt(np.einsum("...cc->...c", G))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 - G / (d[..., :, None] * d[..., None, :])


def corr(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1))


def cka(K, L):
    """Linear CKA from two centred Gram matrices over the last two axes."""
    def c(M):
        return M - M.mean(-1, keepdims=True) - M.mean(-2, keepdims=True) + M.mean((-1, -2), keepdims=True)
    K, L = c(K), c(L)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (K * L).sum((-1, -2)) / np.sqrt((K * K).sum((-1, -2)) * (L * L).sum((-1, -2)))


# ---- draws and statistics

def make_draws(n_roles, n_q, n=N_BOOT, seed=SEED):
    """Role indices [n+1, C] and question weights [n+1, Q]; draw 0 is the observed data."""
    rng = np.random.default_rng(seed)
    rr = np.vstack([np.arange(n_roles), rng.integers(0, n_roles, (n, n_roles))])
    qq = rng.integers(0, n_q, (n, n_q))
    W = np.vstack([np.ones(n_q), np.array([np.bincount(x, minlength=n_q) for x in qq], float).reshape(n, n_q)])
    return rr, W


def pairs(rr):
    """A draw's distinct roles, and its position pairs (i < j) with two different roles mapped onto them, so a
    duplicated role contributes its pairs once per copy (the E3 rule, 261008 part2_analysis.map_boot)."""
    uniq, pos = np.unique(rr, return_inverse=True)
    i, j = np.triu_indices(len(rr), 1)
    keep = rr[i] != rr[j]
    return uniq, pos[i[keep]], pos[j[keep]]


def centre(G):
    """Double-centre Gram matrices over their last two axes, kept exactly symmetric."""
    G = G - G.mean(-1, keepdims=True) - G.mean(-2, keepdims=True) + G.mean((-1, -2), keepdims=True)
    return (G + np.swapaxes(G, -1, -2)) / 2


def build_sides(S, N, halves, support=None):
    """[(side A, side B)] per half; support "common" keeps only each side's common questions (F4)."""
    return [tuple(Side(S, N, common_mask(N, q, w) if support == "common" else q, w) for q, w in h) for h in halves]


def panel(sides, rr, W, comparisons, chunk=CHUNK, strict=False):
    """Cross-half statistics on every draw.

    sides: {name: build_sides(...)}; comparisons: [(a, b)] (a == b gives reliability).
    A half is undefined in a draw when a side has no sampled question for some role. The draw's value is the
    mean over its defined halves (count in "undefined_halves"); strict (F4 support) makes the draw undefined.
    Returns {(a, b): {"r": [draws], "cka": [draws], "undefined_halves": [draws]}}.
    """
    mean = np.mean if strict else _nanmean
    out = {c: dict(r=np.full(len(W), np.nan), cka=np.full(len(W), np.nan), undefined_halves=np.zeros(len(W), int))
           for c in comparisons}
    for s in range(0, len(W), chunk):
        sl = slice(s, s + chunk)
        G = {k: np.array([[side.grams(W[sl]) for side in h] for h in v]) for k, v in sides.items()}  # [H, 2, b, C, C]
        for b in range(sl.start, min(sl.stop, len(W))):
            uniq, i, j = pairs(rr[b])
            bi = b - s
            # role vectors centred over the draw's distinct roles (draw 0 has every role once: map unchanged)
            sub = {k: centre(g[:, :, bi][..., uniq, :][..., uniq]) for k, g in G.items()}  # [H, 2, U, U]
            ranks = {k: rankdata(dist(g)[..., i, j], axis=-1) for k, g in sub.items()}  # [H, 2, P]
            for a, c in comparisons:
                Ra, Rc = ranks[a], ranks[c]
                per_half = (corr(Ra[:, 0], Rc[:, 1]) + corr(Rc[:, 0], Ra[:, 1])) / 2
                out[(a, c)]["r"][b] = mean(per_half)
                out[(a, c)]["undefined_halves"][b] = int((~np.isfinite(per_half)).sum())
                out[(a, c)]["cka"][b] = mean((cka(sub[a][:, 0], sub[c][:, 1]) + cka(sub[c][:, 0], sub[a][:, 1])) / 2)
    return out


def _nanmean(x):
    x = np.asarray(x)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def null(sides_a, sides_b, n_q, n_perm=N_PERM, seed=SEED + 1):
    """One-map null for r(a, b) on the observed data: b's role identities permuted. Returns the null r values."""
    ones = np.ones((1, n_q))
    Da, Db = (np.array([[dist(side.grams(ones))[0] for side in h] for h in s]) for s in (sides_a, sides_b))  # [H, 2, C, C]
    C = Da.shape[-1]
    i, j = np.triu_indices(C, 1)
    Ra = rankdata(Da[..., i, j], axis=-1)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm)
    for s in range(0, n_perm, 500):
        perms = np.array([rng.permutation(C) for _ in range(min(500, n_perm - s))])
        Rb = rankdata(Db[:, :, perms[:, i], perms[:, j]], axis=-1)  # [H, 2, n, P]
        per_half = (corr(Ra[:, 0, None], Rb[:, 1]) + corr(Rb[:, 0], Ra[:, 1, None])) / 2  # [H, n]
        with np.errstate(invalid="ignore"):  # the observed r skips undefined halves; so does the null
            vals[s:s + len(perms)] = np.nanmean(np.where(np.isfinite(per_half), per_half, np.nan), axis=0)
    return vals


def p_value(obs, null_vals):
    """Permutation p over the finite null values only; NaN when the observed value or every null value is
    undefined (an undefined permutation is never evidence)."""
    null_vals = np.asarray(null_vals, float)
    ok = np.isfinite(null_vals)
    if not np.isfinite(obs) or not ok.any():
        return float("nan")
    return float((1 + np.sum(null_vals[ok] >= obs)) / (1 + ok.sum()))


def summary(draws):
    """Point value (draw 0) and the 95% percentile interval over draws 1..n; no interval if any draw is not
    finite. Draws repeat questions and sit below the point; that shift (median of draws minus the point) is
    reported, and the interval is not moved (owner, 30/10)."""
    d = np.asarray(draws, float)
    ok = np.isfinite(d[1:]).all()
    if not ok:
        return dict(point=float(d[0]), lo=None, hi=None, n_nonfinite=int((~np.isfinite(d[1:])).sum()))
    q_lo, med, q_hi = np.percentile(d[1:], [2.5, 50, 97.5])
    return dict(point=float(d[0]), lo=float(q_lo), hi=float(q_hi), n_nonfinite=0, draw_shift=float(med - d[0]))


def r_norm(stats, a, b, floor=0.5):
    """Reliability-normalised r(a, b) = r / sqrt(rel_a rel_b) under the same-anchor rule (THRESHOLDS.md)."""
    r, ra, rb = stats[(a, b)]["r"], stats[(a, a)]["r"], stats[(b, b)]["r"]
    rel = {a: summary(ra), b: summary(rb)}
    if any(s["lo"] is None or s["lo"] < floor for s in rel.values()):
        return dict(available=False, why="a reliability lower bound is below the floor or not finite",
                    raw=summary(r), rel=rel), None
    with np.errstate(invalid="ignore"):
        vals = r / np.sqrt(ra * rb)
    s = summary(vals)
    if (ra[1:] <= 0).any() or (rb[1:] <= 0).any():  # draws withheld, so no difference rebuilds an interval
        s.update(lo=None, hi=None, why="a draw has reliability <= 0")
        return dict(available=True, value=s, raw=summary(r), rel=rel), None
    return dict(available=True, value=s, raw=summary(r), rel=rel), vals


# ---- G5 synthetic cases

def synthetic_cells(role_vecs, q_eff, noise, rng, wording_dirs=None, k=None, shared=None):
    """Prompt means S [C, Q, W, d]: role vector + question effect + noise (+ shared row noise) (+ k_w E_c)."""
    C, d = role_vecs.shape
    Q, W = q_eff.shape[0], 5
    S = role_vecs[:, None, None] + q_eff[None, :, None] + noise * rng.standard_normal((C, Q, W, d))
    if shared is not None:
        S = S + shared
    if wording_dirs is not None:
        S = S + 0.2 * np.asarray(k, float)[None, None, :, None] * wording_dirs[:, None, None, :]
    return S, np.ones((C, Q, W))
