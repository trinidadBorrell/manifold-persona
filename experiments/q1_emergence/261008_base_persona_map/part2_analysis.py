"""E3 pilot Part 2: role-specific information in base-model reply states. Implements THRESHOLDS.md v2
(and its 09/10 "Changes" entries). No judge output and no human label is used here.

Inputs (per run dir): replies.jsonl, reply_mean_acts.npy (rows in replies.jsonl order, float16,
[n, 33 hidden states, 4096]).

Per condition (desc, ex2, ex2_other, ex2_filler); primary set = valid replies of the 20 pilot roles on the
18 shared questions; hidden state 17.
  decoding   multinomial logistic regression (L2, C=1, class-balanced, lbfgs); standardization fit on the
             training fold; 50 fixed splits that hold out questions AND wordings (train 3 wordings x 9
             questions, test the other 2 wordings x other 9 questions); macro accuracy.
  CI         crossed bootstrap (2,000): per (role, question) held-out cell accuracy pooled over the 50 splits;
             each draw resamples roles and questions, averages a role's sampled question cells, then roles.
  null       role labels permuted as whole blocks within each question (a role's wordings and donors move
             together); the full pipeline is refit on the first 10 of the 50 splits; the statistic (mean
             over those 10) is the same for the observed data and every permutation.
  baselines  TF-IDF, sentence embedding, length + stop reason (each + LR), same splits.
  gain       [activations, text features] minus text, for all three baselines; the best baseline is
             re-selected inside every bootstrap draw (and for the point estimate) by its draw accuracy.
  map        50 fixed halves that split questions AND wordings; mean-centred role vectors; cosine distance
             matrices; Spearman r of upper triangles, mean over the 50 halves. Null: block permutations on
             the same 50 halves. CI: crossed bootstrap that recomputes the 50-half mean on each draw, using
             distinct-role pairs only. Same for the text-embedding geometry.
  geometry   top-3 PCs per half: principal angles; eigengaps of both halves.
  axis       desc only: mean of the default_ai and default_llm frame means (equal weight) minus the mean
             of the role vectors; split-half cosine; axis_interpretable = cosine >= 0.80.
  donor      ex2_other: donor decoding, descriptive and exploratory only.
  layers 9, 23: decoding estimate only, exploratory.
Raw intersection-union p-values are written; Holm over the 4 primary decisions is applied in the report.

    OMP_NUM_THREADS=1 python part2_analysis.py --run run_main --out part2 --jobs 240
    python part2_analysis.py --selftest
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROLES = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
         "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
         "pirate", "vampire", "robot", "void"]
CONDS = ["desc", "ex2", "ex2_other", "ex2_filler"]
LAYER, EXPLORATORY_LAYERS = 17, (9, 23)
N_SPLITS, N_SPLITS_NULL, N_BOOT, SEED = 50, 10, 2000, 20261009
EMBED_MODEL = ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130")


class AnalysisError(RuntimeError):
    pass


def cache_path(out, run_dir, tag, n):
    """Cache file whose name carries the full identity: run fingerprint, code, tag (cond/feature), draw count, seed."""
    import hashlib
    h = hashlib.sha256()
    h.update((run_dir / "fingerprint.txt").read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(f"{tag}|{n}|{SEED}|{N_SPLITS}|{N_SPLITS_NULL}".encode())
    return out / f"cache_{tag}_{h.hexdigest()[:16]}.npy"


def load_cache(path, n):
    if path is None or not path.exists():
        return None
    a = np.load(path)
    if len(a) != n:
        raise AnalysisError(f"cache {path.name} has {len(a)} rows, want {n}")
    return a


def finite(x, what):
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise AnalysisError(f"non-finite values in {what}")
    return x


# ---------------------------------------------------------------- data

SLIM = "reply_mean_acts_slim.json"  # slim copy: only the hidden states this analysis reads


def load(run_dir):
    rows = [json.loads(l) for l in open(run_dir / "replies.jsonl")]
    if (run_dir / "reply_mean_acts.npy").exists():
        acts = np.load(run_dir / "reply_mean_acts.npy", mmap_mode="r")
        col = {L: L for L in (LAYER,) + EXPLORATORY_LAYERS}
    else:
        meta = json.load(open(run_dir / SLIM))
        acts = np.load(run_dir / meta["file"], mmap_mode="r")
        if acts.shape[1] != len(meta["layers"]) or not set((LAYER,) + EXPLORATORY_LAYERS) <= set(meta["layers"]):
            raise AnalysisError("slim activations do not hold the analysed hidden states")
        col = {L: meta["layers"].index(L) for L in (LAYER,) + EXPLORATORY_LAYERS}
    if acts.shape[0] != len(rows):
        raise AnalysisError("activation rows != replies")
    X = {L: np.asarray(acts[:, c, :], dtype=np.float32) for L, c in col.items()}
    return rows, X


def usable(r):
    return bool(r["valid"] and r["reply"].strip())


def subset(rows, cond):
    return np.array([k for k, r in enumerate(rows) if r["cond"] == cond and r["role"] in ROLES and usable(r)
                     and r["shared_q"]])


def check_design(y, questions, wordings, n_roles=20, n_q=18, n_w=5):
    """The frozen design: every role on the same 18 questions and 5 wordings."""
    if len(np.unique(y)) != n_roles or len(np.unique(questions)) != n_q or len(np.unique(wordings)) != n_w:
        raise AnalysisError(f"design is not {n_roles} roles x {n_q} questions x {n_w} wordings")
    for c in range(n_roles):
        m = y == c
        if len(np.unique(questions[m])) != n_q or len(np.unique(wordings[m])) != n_w:
            raise AnalysisError(f"role {c} does not cover all questions and wordings")


def make_splits(y, questions, wordings, rng, n):
    qs, ws = np.unique(questions), np.unique(wordings)
    out = []
    for _ in range(n):
        q_tr = rng.choice(qs, len(qs) // 2, replace=False)
        w_tr = rng.choice(ws, 3, replace=False)
        tr = np.where(np.isin(questions, q_tr) & np.isin(wordings, w_tr))[0]
        te = np.where(~np.isin(questions, q_tr) & ~np.isin(wordings, w_tr))[0]
        for part, name in ((tr, "train"), (te, "test")):
            if len(np.unique(y[part])) != y.max() + 1:
                raise AnalysisError(f"a role has no {name} replies in a split")
        out.append((tr, te))
    return out


def make_halves(y, questions, wordings, rng, n):
    """Half A: half the questions x 2 or 3 of the 5 wordings; half B: the other questions x other wordings."""
    qs, ws = np.unique(questions), np.unique(wordings)
    out = []
    for _ in range(n):
        qa = rng.choice(qs, len(qs) // 2, replace=False)
        wa = rng.choice(ws, rng.choice([len(ws) // 2, len(ws) - len(ws) // 2]), replace=False)
        out.append((np.isin(qs, qa), np.isin(ws, wa)))  # masks over the sorted question / wording ids
    return out


# ---------------------------------------------------------------- decoding

def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def macro_acc(y, p, n_classes):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(n_classes)]))


JOBS = 1  # workers for decode(); set from --jobs. Each split is fit independently, so results do not depend on it.


def decode(feat, y, splits, featurize=None):
    from joblib import Parallel, delayed
    n_c = y.max() + 1

    def one(tr, te):
        Xtr, Xte = featurize(tr, te) if featurize else (feat[tr], feat[te])
        return fit_predict(Xtr, y[tr], Xte)
    ps = Parallel(n_jobs=JOBS)(delayed(one)(tr, te) for tr, te in splits)
    preds = [(te, p) for (_, te), p in zip(splits, ps)]
    return float(np.mean([macro_acc(y[te], p, n_c) for te, p in preds])), preds


def cell_acc(y, questions, preds, n_classes):
    """Held-out accuracy per (role, question) cell, pooled over splits. NaN where a cell was never tested."""
    qs = np.unique(questions)
    qi = np.searchsorted(qs, questions)
    hit = np.zeros((n_classes, len(qs)))
    cnt = np.zeros_like(hit)
    for te, p in preds:
        np.add.at(hit, (y[te], qi[te]), (p == y[te]).astype(float))
        np.add.at(cnt, (y[te], qi[te]), 1.0)
    if np.any(cnt == 0):
        raise AnalysisError("a (role, question) cell was never in a test fold")
    return hit / cnt


def boot_draws(n_roles, n_q, rng, n=N_BOOT):
    return [(rng.integers(0, n_roles, n_roles), rng.integers(0, n_q, n_q)) for _ in range(n)]


def draw_stat(C, rr, qq):
    """Average a role's sampled question cells, then the sampled roles."""
    return float(C[np.ix_(rr, qq)].mean(1).mean())


def gain_draws(base_C, comb_C, draws):
    """Per draw: pick the text baseline with the best draw accuracy, then (activations + it) minus it."""
    out = []
    for rr, qq in draws:
        bs = {k: draw_stat(base_C[k], rr, qq) for k in base_C}
        kb = max(bs, key=bs.get)
        out.append(draw_stat(comb_C[kb], rr, qq) - bs[kb])
    return np.array(out)


def block_permute(y, questions, rng):
    yp = y.copy()
    for q in np.unique(questions):
        m = questions == q
        yp[m] = rng.permutation(y.max() + 1)[y[m]]
    return yp


def perm_test(feat, y, questions, splits, n_perm, jobs, seed, ckpt=None):
    from joblib import Parallel, delayed
    n_c = y.max() + 1
    obs = float(np.mean([macro_acc(y[te], fit_predict(feat[tr], y[tr], feat[te]), n_c) for tr, te in splits]))

    def one(k):
        yp = block_permute(y, questions, np.random.default_rng(seed + 1 + k))
        return float(np.mean([macro_acc(yp[te], fit_predict(feat[tr], yp[tr], feat[te]), n_c) for tr, te in splits]))
    null = load_cache(ckpt, n_perm)
    if null is None:
        null = np.array(Parallel(n_jobs=jobs, batch_size=4)(delayed(one)(k) for k in range(n_perm)))
        if ckpt is not None:
            np.save(ckpt, null)
    finite(null, "decoding null")
    return obs, null, float((1 + np.sum(null >= obs)) / (1 + n_perm))


# ---------------------------------------------------------------- text features

def tfidf_featurizer(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    def f(tr, te):
        v = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True).fit([texts[i] for i in tr])
        return v.transform([texts[i] for i in tr]).toarray(), v.transform([texts[i] for i in te]).toarray()
    return f


def embed(texts, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL[0], revision=EMBED_MODEL[1], device=device)
    return m.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)


def length_feats(rows_sub):
    ends = ["marker", "limit", "eos", "pad"]
    return np.array([[r["reply_tokens"], np.log1p(r["reply_tokens"])] + [r["ended_by"] == e for e in ends]
                     for r in rows_sub], dtype=np.float32)


# ---------------------------------------------------------------- map (cell sums)

def cell_sums(X, y, questions, wordings, n_classes):
    qs, ws = np.unique(questions), np.unique(wordings)
    qi, wi = np.searchsorted(qs, questions), np.searchsorted(ws, wordings)
    S = np.zeros((n_classes, len(qs), len(ws), X.shape[1]), dtype=np.float64)
    N = np.zeros((n_classes, len(qs), len(ws)))
    np.add.at(S, (y, qi, wi), X)
    np.add.at(N, (y, qi, wi), 1.0)
    return S, N


def half_vectors(S, N, qmult, wmask):
    """Role vectors from cell sums: questions weighted by multiplicity, wordings by mask; mean-centred."""
    num = np.einsum("cqwd,q,w->cd", S, qmult, wmask.astype(float))
    den = np.einsum("cqw,q,w->c", N, qmult, wmask.astype(float))
    if np.any(den == 0):
        raise AnalysisError("a role has no replies in a map half")
    V = num / den[:, None]
    return V - V.mean(0)


def cosdist(V):
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    if np.any(nrm == 0):
        raise AnalysisError("zero-length role vector")
    Vn = V / nrm
    return 1 - Vn @ Vn.T


def map_stat(S, N, halves, qmult=None, pairs=None, geometry=False):
    from scipy.stats import spearmanr
    n_c = S.shape[0]
    if qmult is None:
        qmult = np.ones(S.shape[1])
    i, j = pairs if pairs is not None else np.triu_indices(n_c, 1)
    rs, angles, gaps_a, gaps_b = [], [], [], []
    for qa, wa in halves:
        VA = half_vectors(S, N, qmult * qa, wa)
        VB = half_vectors(S, N, qmult * ~qa, ~wa)
        rs.append(spearmanr(cosdist(VA)[i, j], cosdist(VB)[i, j])[0])
        if geometry:
            ua, sa, _ = np.linalg.svd(VA.T, full_matrices=False)
            ub, sb, _ = np.linalg.svd(VB.T, full_matrices=False)
            c = np.linalg.svd(ua[:, :3].T @ ub[:, :3], compute_uv=False)
            angles.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            gaps_a.append([(sa[k] ** 2 - sa[k + 1] ** 2) / (sa ** 2).sum() for k in range(3)])
            gaps_b.append([(sb[k] ** 2 - sb[k + 1] ** 2) / (sb ** 2).sum() for k in range(3)])
    r = float(np.mean(finite(rs, "map correlations")))
    if not geometry:
        return r
    return r, np.mean(angles, 0).tolist(), np.mean(gaps_a, 0).tolist(), np.mean(gaps_b, 0).tolist()


def permute_cells(S, N, rng):
    """Block permutation on cell sums: within each question, role blocks get a random permutation."""
    Sp, Np = np.empty_like(S), np.empty_like(N)
    for q in range(S.shape[1]):
        perm = rng.permutation(S.shape[0])
        Sp[perm, q], Np[perm, q] = S[:, q], N[:, q]
    return Sp, Np


def map_null(S, N, halves, n_perm, seed, jobs, ckpt=None):
    from joblib import Parallel, delayed
    cached = load_cache(ckpt, n_perm)
    if cached is not None:
        return finite(cached, "map null")

    def one(k):
        Sp, Np = permute_cells(S, N, np.random.default_rng(seed + 7 + k))
        return map_stat(Sp, Np, halves)
    null = np.array(Parallel(n_jobs=jobs, batch_size=8)(delayed(one)(k) for k in range(n_perm)))
    if ckpt is not None:
        np.save(ckpt, null)
    return finite(null, "map null")


def map_boot(S, N, n_draws, seed, jobs):
    """Crossed bootstrap that recomputes the 50-half mean on each draw; duplicate questions stay in one half.
    Role vectors are built and centred over the DISTINCT drawn roles (centring over duplicated rows would
    give every half the same duplicate-driven structure); pairs of draw positions with different roles are
    then mapped onto those distinct roles, so a duplicated role contributes its pairs once per copy."""
    from joblib import Parallel, delayed
    n_c, n_q, n_w = S.shape[:3]

    def one(k):
        rng = np.random.default_rng(seed + 11 + k)
        rr, qq = rng.integers(0, n_c, n_c), rng.integers(0, n_q, n_q)
        qmult = np.bincount(qq, minlength=n_q).astype(float)
        uq = np.where(qmult > 0)[0]
        hv = []
        for _ in range(N_SPLITS):
            qa = np.isin(np.arange(n_q), rng.choice(uq, len(uq) // 2, replace=False))
            wa = np.isin(np.arange(n_w), rng.choice(n_w, rng.choice([n_w // 2, n_w - n_w // 2]), replace=False))
            hv.append((qa, wa))
        uniq, pos = np.unique(rr, return_inverse=True)
        i, j = np.triu_indices(n_c, 1)
        keep = rr[i] != rr[j]
        return map_stat(S[uniq], N[uniq], hv, qmult=qmult, pairs=(pos[i[keep]], pos[j[keep]]))
    return finite(np.array(Parallel(n_jobs=jobs, batch_size=4)(delayed(one)(k) for k in range(n_draws))), "map boot")


# ---------------------------------------------------------------- axis

def axis_split_half(Xr, yr, qr, Xd, fd, qd, halves_q, qs, n_classes):
    """Axis per half: mean of the two default-frame means (equal weight) minus the mean of role vectors."""
    cs = []
    for qa in halves_q:
        ax = []
        for side in (qa, ~qa):
            qset = qs[side]
            mr, md = np.isin(qr, qset), np.isin(qd, qset)
            roles_mean = np.stack([Xr[mr & (yr == c)].mean(0) for c in range(n_classes)]).mean(0)
            frames = []
            for f in ("default_ai", "default_llm"):
                m = md & (fd == f)
                if not m.any():
                    raise AnalysisError(f"no {f} replies in an axis half")
                frames.append(Xd[m].mean(0))
            ax.append(np.mean(frames, 0) - roles_mean)
        cs.append(float(ax[0] @ ax[1] / np.linalg.norm(ax[0]) / np.linalg.norm(ax[1])))
    return float(np.mean(finite(cs, "axis cosines")))


# ---------------------------------------------------------------- main analysis

def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def analyse_condition(rows, X, cond, args, emb_all, out):
    idx = subset(rows, cond)
    sub = [rows[k] for k in idx]
    nR = len(ROLES)
    y = np.array([ROLES.index(r["role"]) for r in sub])
    qn = np.array([r["question_idx"] for r in sub])
    wn = np.array([r["wording"] for r in sub])
    texts = [r["reply"] for r in sub]
    check_design(y, qn, wn)
    Xa = finite(X[LAYER][idx], "activations")
    E = finite(emb_all[idx], "embeddings")
    rng = np.random.default_rng(SEED)
    splits = make_splits(y, qn, wn, rng, N_SPLITS)
    draws = boot_draws(nR, len(np.unique(qn)), np.random.default_rng(SEED + 1))
    res = dict(cond=cond, n_replies=len(idx),
               n_invalid_in_cond=int(sum(1 for r in rows if r["cond"] == cond and r["role"] in ROLES
                                         and r["shared_q"] and not usable(r))))

    # primary decoding
    acc, preds = decode(Xa, y, splits)
    C = cell_acc(y, qn, preds, nR)
    bt = np.array([draw_stat(C, rr, qq) for rr, qq in draws])
    obs10, null, p = perm_test(Xa, y, qn, splits[:N_SPLITS_NULL], args.n_perm, args.jobs, SEED,
                               ckpt=cache_path(out, HERE / args.run, f"{cond}_decoding_null", args.n_perm))
    res["decoding"] = dict(acc_mean_50=acc, ci95=ci(bt), stat_first10=obs10, null_mean=float(null.mean()),
                           null_p95=float(np.percentile(null, 95)), null_p99=float(np.percentile(null, 99)),
                           p=p, per_role=dict(zip(ROLES, C.mean(1).tolist())))

    # text baselines and gain (best baseline re-selected in every draw)
    tf = tfidf_featurizer(texts)
    L = length_feats(sub)
    feats = {"tfidf": tf, "embedding": lambda tr, te: (E[tr], E[te]), "length": lambda tr, te: (L[tr], L[te])}
    base_C, comb_C, base_acc, comb_acc = {}, {}, {}, {}
    for name, f in feats.items():
        a_b, p_b = decode(None, y, splits, featurize=f)

        def comb(tr, te, f=f):
            a, b = f(tr, te)
            return np.hstack([Xa[tr], a]), np.hstack([Xa[te], b])
        a_c, p_c = decode(None, y, splits, featurize=comb)
        base_acc[name], comb_acc[name] = a_b, a_c
        base_C[name], comb_C[name] = cell_acc(y, qn, p_b, nR), cell_acc(y, qn, p_c, nR)
    best = max(base_acc, key=base_acc.get)
    gains = gain_draws(base_C, comb_C, draws)
    res["baselines"] = {k: dict(acc_mean_50=base_acc[k], combined_with_acts=comb_acc[k]) for k in base_acc}
    res["gain_over_best_text"] = dict(best_text=best, diff=comb_acc[best] - base_acc[best], ci95=ci(gains),
                                      p_one_sided=float((1 + np.sum(gains <= 0)) / (1 + len(gains))))

    # map reliability and geometry: activations and text embedding
    halves = make_halves(y, qn, wn, np.random.default_rng(SEED + 3), N_SPLITS)
    for name, F in (("map_acts", Xa), ("map_text_embedding", E)):
        S, Nc = cell_sums(F, y, qn, wn, nR)
        r, ang, ga, gb = map_stat(S, Nc, halves, geometry=True)
        nul = map_null(S, Nc, halves, args.n_perm_map, SEED, args.jobs,
                       ckpt=cache_path(out, HERE / args.run, f"{cond}_{name}_null", args.n_perm_map))
        bt_m = map_boot(S, Nc, args.n_boot_map, SEED, args.jobs)
        res[name] = dict(spearman_r=r, spearman_brown=float(2 * r / (1 + r)), ci95=ci(bt_m),
                         null_p99=float(np.percentile(nul, 99)), p=float((1 + np.sum(nul >= r)) / (1 + len(nul))),
                         pc_principal_angles_deg=ang, eigengaps_top3_half_a=ga, eigengaps_top3_half_b=gb)

    res["exploratory_layers"] = {str(Lr): decode(finite(X[Lr][idx], f"layer {Lr}"), y, splits)[0]
                                 for Lr in EXPLORATORY_LAYERS}
    if cond == "ex2_other":
        yd = np.array([ROLES.index(r["example_role"]) for r in sub])
        res["donor_decoding_exploratory"] = dict(acc_mean_50=decode(Xa, yd, splits)[0],
                                                 note="descriptive only; no interval or null")
    if cond == "desc":
        d_idx = np.array([k for k, r in enumerate(rows) if r["cond"] == "desc" and usable(r) and r["shared_q"]
                          and r["role"] in ("default_ai", "default_llm")])
        qs = np.unique(qn)
        hq = [qa for qa, _ in halves]
        cosv = axis_split_half(Xa, y, qn, finite(X[LAYER][d_idx], "default activations"),
                               np.array([rows[k]["role"] for k in d_idx]),
                               np.array([rows[k]["question_idx"] for k in d_idx]), hq, qs, nR)
        res["axis"] = dict(split_half_cos=cosv, axis_interpretable=bool(cosv >= 0.80))

    d, g, m = res["decoding"], res["gain_over_best_text"], res["map_acts"]
    res["decision"] = dict(rule1_decoding=bool(d["ci95"][0] >= 0.10 and d["p"] < 0.01),
                           rule2_gain=bool(g["ci95"][0] > 0),
                           information_present=bool(d["ci95"][0] >= 0.10 and d["p"] < 0.01 and g["ci95"][0] > 0),
                           geometry_interpretable=bool(m["ci95"][0] >= 0.5 and m["p"] < 0.01),
                           p_intersection_union=max(d["p"], g["p_one_sided"]),
                           note="raw p; Holm over the 4 primary decisions is applied in the report")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_main")
    ap.add_argument("--out", default="part2")
    ap.add_argument("--n_perm", type=int, default=9999)
    ap.add_argument("--n_perm_map", type=int, default=9999)
    ap.add_argument("--n_boot_map", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embed_only", action="store_true", help="write the text-embedding cache and stop")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    if a.embed_only:
        out = HERE / a.out
        out.mkdir(exist_ok=True)
        rows = [json.loads(l) for l in open(HERE / a.run / "replies.jsonl")]
        ef = cache_path(out, HERE / a.run, "text_embeddings", len(rows))
        if not ef.exists():
            np.save(ef, embed([r["reply"] for r in rows], a.device))
        return print("embeddings:", ef)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1 (one BLAS thread per worker)")
    global JOBS
    JOBS = a.jobs
    t = time.time()
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    rows, X = load(HERE / a.run)
    ef = cache_path(out, HERE / a.run, "text_embeddings", len(rows))
    emb_all = load_cache(ef, len(rows))
    if emb_all is None:
        emb_all = embed([r["reply"] for r in rows], a.device)
        np.save(ef, emb_all)
    results = {}
    for c in a.conds:
        results[c] = analyse_condition(rows, X, c, a, emb_all, out)
        json.dump(results, open(out / f"{a.run}_part2.json", "w"), indent=1)
        print(c, "done", round(time.time() - t), "s", flush=True)
    results["_meta"] = dict(run=a.run, layer=LAYER, n_splits=N_SPLITS, n_splits_null=N_SPLITS_NULL,
                            n_perm=a.n_perm, n_perm_map=a.n_perm_map, n_boot=N_BOOT, n_boot_map=a.n_boot_map,
                            seed=SEED, embed_model=EMBED_MODEL, seconds=round(time.time() - t))
    json.dump(results, open(out / f"{a.run}_part2.json", "w"), indent=1)


# ---------------------------------------------------------------- self-test

def synth(rng, strength, wording_offset, n_r=20, n_q=18, n_w=5, d=64, reps=1):
    role_mu = rng.normal(size=(n_r, d)) * strength
    q_mu = rng.normal(size=(n_q, d))
    w_mu = rng.normal(size=(n_r, n_w, d)) * wording_offset
    y, q, w, X = [], [], [], []
    for r in range(n_r):
        for qi in range(n_q):
            for wi in range(n_w):
                for _ in range(reps):
                    y.append(r); q.append(qi); w.append(wi)
                    X.append(role_mu[r] + q_mu[qi] + w_mu[r, wi] + rng.normal(size=d))
    return np.array(X, dtype=np.float32), np.array(y), np.array(q), np.array(w)


def selftest(a):
    rng = np.random.default_rng(0)
    # split isolation
    X, y, q, w = synth(rng, 0.6, 0.3)
    for tr, te in make_splits(y, q, w, np.random.default_rng(1), 20):
        assert not set(q[tr]) & set(q[te]) and not set(w[tr]) & set(w[te])
    # block permutation keeps (question, role) blocks together
    Xr, yr, qr, wr = synth(rng, 0.6, 0.3, reps=3)
    yp = block_permute(yr, qr, np.random.default_rng(2))
    for qq in range(18):
        for r in range(20):
            m = (qr == qq) & (yr == r)
            assert len(np.unique(yp[m])) == 1
    res = {}
    for name, strength in (("signal", 0.6), ("noise", 0.0)):
        X, y, q, w = synth(np.random.default_rng(3), strength, 0.3)
        splits = make_splits(y, q, w, np.random.default_rng(4), 10)
        acc, preds = decode(X, y, splits)
        C = cell_acc(y, q, preds, 20)
        bt = [draw_stat(C, rr, qq) for rr, qq in boot_draws(20, 18, np.random.default_rng(5), 200)]
        _, null, p = perm_test(X, y, q, splits[:5], 199, a.jobs, 3)
        check_design(y, q, w)
        S, N = cell_sums(X, y, q, w, 20)
        halves = make_halves(y, q, w, np.random.default_rng(6), N_SPLITS)
        r = map_stat(S, N, halves)
        nul = map_null(S, N, halves, 99, 4, a.jobs)
        mb = map_boot(S, N, 50, 8, a.jobs)
        res[name] = (acc, ci(bt), p, r, float(np.percentile(nul, 99)), ci(mb))
        print(f"{name}: acc {acc:.3f} CI {ci(bt)[0]:.3f}-{ci(bt)[1]:.3f} p {p:.3f} | map r {r:.3f} "
              f"null p99 {np.percentile(nul, 99):.3f} boot CI {ci(mb)[0]:.3f}-{ci(mb)[1]:.3f}")
    s, n = res["signal"], res["noise"]
    assert s[1][0] > 0.10 and s[2] < 0.01 and s[3] > s[4], "planted signal must pass"
    assert n[2] > 0.05 and abs(n[0] - 0.05) < 0.03 and n[3] < n[4] + 0.05, "noise must not pass"
    assert n[5][0] < 0 < n[5][1] and n[1][0] < 0.05 < n[1][1], "noise intervals must cover the null value"
    assert s[5][0] < s[3] < s[5][1], "signal map interval must cover the estimate"
    # gain path, through gain_draws with three baselines (best-of-three selection inside each draw)
    rng_g = np.random.default_rng(10)
    Xs, ys, qs_, ws_ = synth(rng_g, 0.6, 0.3)
    Xn = synth(np.random.default_rng(15), 0.0, 0.0)[0]          # activations with no role signal
    weak = synth(np.random.default_rng(16), 0.25, 0.0)[0][:, :16]  # a text feature with weak role signal
    texts = {"noise_a": rng_g.normal(size=(len(ys), 16)).astype(np.float32),
             "noise_b": rng_g.normal(size=(len(ys), 16)).astype(np.float32), "weak": weak}
    sp = make_splits(ys, qs_, ws_, np.random.default_rng(12), 20)
    dr = boot_draws(20, 18, np.random.default_rng(13), 300)
    for label, A, expect_gain in (("acts with signal", Xs, True), ("acts without signal", Xn, False)):
        bC, cC = {}, {}
        for k, T in texts.items():
            bC[k] = cell_acc(ys, qs_, decode(T, ys, sp)[1], 20)
            cC[k] = cell_acc(ys, qs_, decode(None, ys, sp, featurize=lambda tr, te, T=T: (
                np.hstack([A[tr], T[tr]]), np.hstack([A[te], T[te]])))[1], 20)
        g = gain_draws(bC, cC, dr)
        print(f"gain check ({label}): CI {ci(g)[0]:.3f}-{ci(g)[1]:.3f}")
        if expect_gain:
            assert ci(g)[0] > 0, "gain must be positive when activations add signal"
        else:
            assert ci(g)[0] < 0 < ci(g)[1] or ci(g)[1] <= 0, "no-signal activations must not show a positive gain"
    # design check must stop a reduced design
    try:
        check_design(ys[qs_ < 17], qs_[qs_ < 17], ws_[qs_ < 17])
        raise AssertionError("reduced design not caught")
    except AnalysisError:
        pass
    # cache with the wrong draw count must stop
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "c.npy"
        np.save(f, np.zeros(3))
        try:
            load_cache(f, 99)
            raise AssertionError("wrong-size cache not caught")
        except AnalysisError:
            pass
    # non-finite input must stop
    try:
        cosdist(np.zeros((3, 4)))
        raise AssertionError("zero vector not caught")
    except AnalysisError:
        pass
    try:
        finite([1.0, np.nan], "x")
        raise AssertionError("NaN not caught")
    except AnalysisError:
        pass
    # missing role in a half must stop
    Xm, ym, qm, wm = synth(np.random.default_rng(9), 0.6, 0.3)
    S, N = cell_sums(Xm, ym, qm, wm, 20)
    N2 = N.copy(); N2[0] = 0
    try:
        half_vectors(S, N2, np.ones(18), np.ones(5, bool))
        raise AssertionError("missing role not caught")
    except AnalysisError:
        pass
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
