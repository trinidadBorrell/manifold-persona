"""Phase 1 -- offline linear algebra (plan v2 s1 M1/M2, s2, s4, s5 Phase 1). CPU, one condor job.

    python -m persona_axes.phase1 --view mean --layer 15        # primary case
    python -m persona_axes.phase1 --view last --layer 28        # robustness column

Writes $PM/AX21/phase1/results_{view}_L{layer}.json (+ a few csv). All numbers are held-out:
  folds      5-fold over personas x 10 repeats (seeds 0-9); greedy selection and PCA see training folds only
  R2_m       1 - sum|y - P y|^2 / sum|y|^2 over test personas, y = x - mean(train)  ("span + mean")
  R2_0       the same with y = x                                                    ("through origin")
  nulls      B1 isotropic, B2 in span(PCA 1-50 of train), B3 random k kept traits; 200 draws per fold and k
  M2         split-half retrieval: per persona, halves A/B of its 40 answers (10 splits); coordinates
             c = pinv(A) x on the set's first k axes (greedy order on all personas); cosine NN of c^B among c^A
Sets 1-8 are the pre-registered combinations (axes.json families); greedy = forward selection from the kept
trait pool; ablation = union minus one family, k = 8.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from persona_axes import AX, HERE, NEUTRAL, PM

K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50]
N_FOLDS, N_REP, N_NULL, N_SPLITS = 5, 10, 200, 10
FAM_SETS = {"1_OCEAN": "ocean", "2_HEXACO": "A", "3_OCEAN+B": "ocean+B", "4_OCEAN+G": "ocean+G",
            "5_utility": "BCDEF", "6_utility+G": "BCDEFG", "7_TIDE": "H", "8_union": "ABCDEFGH"}


def eff_rank(A):
    G = A.T @ A
    w = np.linalg.eigvalsh(G); w = w[w > 1e-10 * w.max()]; p = w / w.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def unit(A):
    return A / np.linalg.norm(A, axis=0, keepdims=True)


def orth(A):
    Q, R = np.linalg.qr(A)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
    return Q[:, keep]


def res2(Y, Q):
    """squared residual norm of each row of Y after projection on the orthonormal columns Q"""
    if Q.shape[1] == 0:
        return (Y ** 2).sum(1)
    C = Y @ Q
    return (Y ** 2).sum(1) - (C ** 2).sum(1)


def greedy(C, Y, kmax):
    """forward selection of columns of C (4096 x m) to explain the rows of Y; returns the index order"""
    Q = np.zeros((C.shape[0], 0)); order = []; rem = list(range(C.shape[1]))
    for _ in range(min(kmax, C.shape[1])):
        R = C[:, rem] - Q @ (Q.T @ C[:, rem]) if Q.shape[1] else C[:, rem].copy()
        nrm = np.linalg.norm(R, axis=0); ok = nrm > 1e-6
        R = R / np.where(ok, nrm, 1.0)
        gain = ((Y @ R) ** 2).sum(0); gain[~ok] = -1
        j = int(np.argmax(gain)); pick = rem.pop(j); order.append(pick)
        Q = np.concatenate([Q, R[:, [j]]], 1)
    return order


class Data:
    def __init__(self, view, L):
        V = AX / "vectors"
        self.view, self.L = view, L
        self.axes = json.loads((HERE / "axes.json").read_text())["axes"]
        self.anames = [a["name"] for a in self.axes]
        self.fam = {a["name"]: a["family"] for a in self.axes}
        A = np.load(V / ("axes_%s_L%d.npy" % (view, L))).astype(np.float64)
        self.A = unit(A.T)                                               # 4096 x 35
        tnames = json.loads((V / "traits_names.json").read_text())
        kept = json.loads((AX / "filter" / "traits_kept.json").read_text())["kept"]
        T = np.load(V / ("traits_%s_L%d.npy" % (view, L))).astype(np.float64)
        ti = [tnames.index(t) for t in kept]
        self.tnames = kept; self.T = unit(T[ti].T)                       # 4096 x n_kept
        rn = json.loads((V / "roles_names.json").read_text())
        R = np.load(V / ("roles_%s_L%d.npy" % (view, L))).astype(np.float64)
        human = json.loads((HERE / "personas_b_human.json").read_text())["roles"]
        sn = json.loads((V / "seta_names.json").read_text())
        S = np.load(V / ("seta_%s_L%d.npy" % (view, L))).astype(np.float64)
        self.sets = {"b": (human, R[[rn.index(r) for r in human]]),
                     "b_all": ([r for r in rn], R),
                     "a": (sn, S)}
        self.cat = {p["name"]: p["category"] for p in json.loads((HERE / "personas_a.json").read_text())["personas"]}
        self.nuis = [i for i, a in enumerate(self.axes) if a["family"] == "N"]

    def _names(self, spec):
        if spec == "ocean":
            return [a["name"] for a in self.axes if a["family"] == "A" and a["name"] != "honesty_humility"]
        if spec.startswith("ocean+"):
            return self._names("ocean") + self._names(spec[6:])
        return [a["name"] for a in self.axes if a["family"] in set(spec)]

    def axis_set(self, spec):
        return [self.anames.index(n) for n in self._names(spec)]


def folds(n, rep):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(n)
    return [(np.setdiff1d(np.arange(n), perm[f::N_FOLDS]), np.sort(perm[f::N_FOLDS])) for f in range(N_FOLDS)]


def curve_sets(D, X, names, out, log):
    """M1 for every pre-registered set (greedy order inside the set), the pool greedy, PCA, nulls; per persona"""
    n = len(names)
    kgrid = [k for k in K_GRID if k <= max(50, 0)]
    acc = {}                                                            # (cond, k, centred) -> res2 sum per persona

    def add(cond, k, c, te, r):
        key = (cond, k, c)
        if key not in acc:
            acc[key] = np.zeros(n)
        acc[key][te] += r
    y2 = {True: np.zeros(n), False: np.zeros(n)}
    orders = {s: [] for s in FAM_SETS}; orders["greedy_pool"] = []
    null_draws = {"B1": {}, "B2": {}, "B3": {}}
    abl = {}
    for rep in range(N_REP):
        for tr, te in folds(n, rep):
            mu = X[tr].mean(0)
            for c in (True, False):
                Ytr = X[tr] - mu if c else X[tr]
                Yte = X[te] - mu if c else X[te]
                y2[c][te] += (Yte ** 2).sum(1)
                # PCA ceiling (fitted on train)
                _, _, Vt = np.linalg.svd(Ytr, full_matrices=False)
                for k in kgrid:
                    add("PCA", k, c, te, res2(Yte, Vt[:min(k, Vt.shape[0])].T))
                add("mean_only", 0, c, te, (Yte ** 2).sum(1))
                if not c:
                    continue
                # pre-registered sets: greedy order inside the set on the training fold
                for sname, spec in FAM_SETS.items():
                    cols = D.axis_set(spec); C = D.A[:, cols]
                    order = greedy(C, Ytr, len(cols)); orders[sname].append([D.anames[cols[j]] for j in order])
                    for k in kgrid:
                        if k <= len(cols):
                            add(sname, k, c, te, res2(Yte, orth(C[:, order[:k]])))
                    if sname == "8_union":
                        add("8_union_full", len(cols), c, te, res2(Yte, orth(C)))
                # greedy from the kept pool
                order = greedy(D.T, Ytr, 50); orders["greedy_pool"].append([D.tnames[j] for j in order])
                for k in kgrid:
                    if k <= D.T.shape[1]:
                        add("greedy_pool", k, c, te, res2(Yte, orth(D.T[:, order[:k]])))
                add("pool_full", D.T.shape[1], c, te, res2(Yte, orth(D.T)))
                # family ablation at k = 8: union minus one family
                for F in "ABCDEFGH":
                    cols = [i for i in D.axis_set("ABCDEFGH") if D.fam[D.anames[i]] != F]
                    o = greedy(D.A[:, cols], Ytr, 8)
                    add("ablate_" + F, 8, c, te, res2(Yte, orth(D.A[:, [cols[j] for j in o]])))
                # nulls: 200 draws per fold and k, with the fold's centring
                rng = np.random.default_rng(1000 * rep + te[0])
                P50 = Vt[:min(50, Vt.shape[0])].T
                for k in kgrid:
                    for nm in null_draws:
                        vals = []
                        for _ in range(N_NULL):
                            if nm == "B1":
                                Q = orth(rng.standard_normal((X.shape[1], k)))
                            elif nm == "B2":
                                Q = orth(P50 @ rng.standard_normal((P50.shape[1], k)))
                            else:
                                if k > D.T.shape[1]:
                                    break
                                Q = orth(D.T[:, rng.choice(D.T.shape[1], k, replace=False)])
                            r = res2(Yte, Q)
                            vals.append(1 - r.sum() / (Yte ** 2).sum())
                        if vals:
                            null_draws[nm].setdefault(k, []).extend(vals)
        log("  rep %d done" % rep)

    def r2(cond, k, c):
        return float(1 - acc[(cond, k, c)].sum() / y2[c].sum())
    curves = {}
    for (cond, k, c) in acc:
        curves.setdefault(cond, {}).setdefault("centred" if c else "origin", {})[k] = r2(cond, k, c)
    per_persona = {cond: {k: (1 - acc[(cond, k, True)] / y2[True]).tolist() for (cc, k, c) in acc if cc == cond and c}
                   for cond in {cc for (cc, k, c) in acc}}
    nulls = {nm: {k: dict(p05=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                          mean=float(np.mean(v)), draws=sorted(float(x) for x in v)[::max(1, len(v) // 400)])
                  for k, v in d.items()} for nm, d in null_draws.items()}
    # percentile of each set in B3 at each k
    pct = {}
    for cond in list(FAM_SETS) + ["greedy_pool"]:
        pct[cond] = {k: float(np.mean(np.array(null_draws["B3"][k]) < curves[cond]["centred"][k]) * 100)
                     for k in curves[cond]["centred"] if k in null_draws["B3"]}
    ablation = {F: dict(r2_k8=curves["ablate_" + F]["centred"][8], delta_vs_union8=curves["ablate_" + F]["centred"][8] - curves["8_union"]["centred"][8])
                for F in "ABCDEFGH"}
    most_common = {s: pd.Series([tuple(o) for o in v]).value_counts().index[0] for s, v in orders.items() if v}
    out.update(dict(curves=curves, nulls=nulls, b3_percentile=pct, ablation=ablation, per_persona=per_persona,
                    y2=y2[True].tolist(), orders_most_common={s: list(v) for s, v in most_common.items()},
                    greedy_pool_k8_frequency=pd.Series([t for o in orders["greedy_pool"] for t in o[:8]]).value_counts().head(20).to_dict(),
                    names=names))


def m2(D, X_half, names, out, log, pca_vt):
    """split-half identifiability; X_half(seed) -> (XA, XB) role vectors from disjoint halves"""
    conds = {s: [D.anames.index(nm) for nm in out["orders_most_common"][s]] for s in FAM_SETS}
    res = {}
    for seed in range(N_SPLITS):
        XA, XB = X_half(seed)
        for cond, cols in list(conds.items()) + [("greedy_pool", None), ("PCA", None), ("full", None)]:
            for k in K_GRID + [4096]:
                if cond == "full" and k != 4096:
                    continue
                if cond != "full" and k == 4096:
                    continue
                if cond == "greedy_pool":
                    sel = [D.tnames.index(t) for t in out["orders_most_common"]["greedy_pool"][:k]]
                    if k > len(out["orders_most_common"]["greedy_pool"]):
                        continue
                    B = D.T[:, sel]
                elif cond == "PCA":
                    B = pca_vt[:k].T
                elif cond == "full":
                    B = None
                else:
                    if k > len(cols):
                        continue
                    B = D.A[:, cols[:k]]
                if B is not None:
                    Bp = np.linalg.pinv(B)
                    CA, CB = XA @ Bp.T, XB @ Bp.T
                else:
                    CA, CB = XA, XB
                CA = CA / (np.linalg.norm(CA, axis=1, keepdims=True) + 1e-12)
                CB = CB / (np.linalg.norm(CB, axis=1, keepdims=True) + 1e-12)
                sim = CB @ CA.T
                rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
                r = res.setdefault(cond, {}).setdefault(k, dict(top1=[], top5=[], mrr=[]))
                r["top1"].append(float((rank == 1).mean())); r["top5"].append(float((rank <= 5).mean())); r["mrr"].append(float((1 / rank).mean()))
        log("  split %d done" % seed)
    out["m2"] = {c: {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in kd.items()} for c, kd in res.items()}
    out["m2_chance"] = dict(top1=1 / len(names), top5=5 / len(names))


def half_vectors_e1(D, names):
    meta = pd.read_csv(PM / "E1" / "meta.csv", usecols=["role", "question_idx"])
    H = np.load(PM / "E1" / ("%s_L%d.npy" % (D.view, D.L)), mmap_mode="r")
    nsel = (meta.role == NEUTRAL).to_numpy()
    q2 = {q: i for i, q in enumerate(meta.question_idx[nsel])}
    N = np.asarray(H[nsel], dtype=np.float32)
    rows = {r: np.where((meta.role == r).to_numpy())[0] for r in names}
    pre = {r: np.asarray(H[rows[r]], dtype=np.float32) - N[[q2[q] for q in meta.question_idx.iloc[rows[r]]]] for r in names}

    def X_half(seed):
        rng = np.random.default_rng(seed)
        XA, XB = [], []
        for r in names:
            d = pre[r]; p = rng.permutation(len(d)); h = len(d) // 2
            XA.append(d[p[:h]].mean(0)); XB.append(d[p[h:]].mean(0))
        return np.stack(XA).astype(np.float64), np.stack(XB).astype(np.float64)
    return X_half


def half_vectors_seta(D, names):
    meta = pd.read_csv(AX / "seta" / "meta.csv", usecols=["who", "question_idx"])
    H = np.load(AX / "seta" / ("%s_L%d.npy" % (D.view, D.L)), mmap_mode="r")
    me = pd.read_csv(PM / "E1" / "meta.csv", usecols=["role", "question_idx"])
    E = np.load(PM / "E1" / ("%s_L%d.npy" % (D.view, D.L)), mmap_mode="r")
    nsel = (me.role == NEUTRAL).to_numpy()
    q2 = {q: i for i, q in enumerate(me.question_idx[nsel])}
    N = np.asarray(E[nsel], dtype=np.float32)
    pre = {}
    for r in names:
        rows = np.where((meta.who == r).to_numpy())[0]
        pre[r] = np.asarray(H[rows], dtype=np.float32) - N[[q2[q] for q in meta.question_idx.iloc[rows]]]

    def X_half(seed):
        rng = np.random.default_rng(seed)
        XA, XB = [], []
        for r in names:
            d = pre[r]; p = rng.permutation(len(d)); h = len(d) // 2
            XA.append(d[p[:h]].mean(0)); XB.append(d[p[h:]].mean(0))
        return np.stack(XA).astype(np.float64), np.stack(XB).astype(np.float64)
    return X_half


def extras(D, out):
    """effective rank, nuisance cosines, nuisance-projected R2, residual per persona, sparse recipes, stability"""
    sets_er = {}
    for s in FAM_SETS:
        cols = [D.anames.index(n) for n in out["b"]["orders_most_common"][s]]
        sets_er[s] = {k: eff_rank(D.A[:, cols[:k]]) for k in range(1, len(cols) + 1)}
    out["effective_rank"] = sets_er
    Nz = D.A[:, D.nuis]
    out["nuisance_cos"] = {D.anames[i]: {D.anames[j]: float(D.A[:, i] @ D.A[:, j]) for j in D.nuis} for i in range(len(D.anames)) if i not in D.nuis}
    out["axis_gram"] = (D.A.T @ D.A).round(4).tolist()
    # residual per persona on the full kept pool (through origin), every set
    Qp = orth(D.T)
    out["residual_pool"] = {s: {n: float(r) for n, r in zip(names, res2(X, Qp) / (X ** 2).sum(1))} for s, (names, X) in D.sets.items()}
    # nuisance-projected M1 for set 6 and union, k = 8 and full, set (b)
    names, X = D.sets["b"]
    Qn = orth(Nz)
    Xp = X - (X @ Qn) @ Qn.T
    Ap = D.A - Qn @ (Qn.T @ D.A)
    mu = Xp.mean(0); Y = Xp - mu
    out["nuisance_projected_r2_b"] = {}
    for s in ("6_utility+G", "8_union"):
        cols = [D.anames.index(n) for n in out["b"]["orders_most_common"][s]]
        out["nuisance_projected_r2_b"][s] = {k: float(1 - res2(Y, orth(Ap[:, cols[:k]])).sum() / (Y ** 2).sum()) for k in (4, 8, len(cols))}
    # sparse decomposition: NNLS of each persona on the kept pool, # traits for 90% of own pool-R2 by per-persona greedy
    from scipy.optimize import nnls
    sparse = {}
    for s, (names, X) in D.sets.items():
        rows = []
        for n, x in zip(names, X):
            coef, _ = nnls(D.T, x)
            nz = np.where(coef > 1e-3 * coef.max())[0]
            top = [(D.tnames[j], float(coef[j])) for j in np.argsort(-coef)[:5] if coef[j] > 0]
            full = 1 - (res2(x[None], Qp)[0] / (x ** 2).sum())
            order = greedy(D.T, x[None], 30)
            need = next((k for k in range(1, 31) if 1 - res2(x[None], orth(D.T[:, order[:k]]))[0] / (x ** 2).sum() >= 0.9 * full), 31)
            rows.append(dict(persona=n, n_nonzero=int(len(nz)), n_for_90pct=int(need), pool_r2=float(full), top=top))
        sparse[s] = rows
    out["sparse"] = sparse
    out["trait_usage"] = pd.Series([t for s in sparse.values() for r in s for t, _ in r["top"][:3]]).value_counts().head(30).to_dict()
    # axis stability: split halves and neighbouring layers
    V = AX / "vectors"
    def cos_rows(P, Q):
        return (np.sum(P * Q, 1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Q, axis=1) + 1e-12))
    st = {}
    for tag in ("pairs", "q"):
        PA = np.load(V / ("axes_%s_L%d_%sA.npy" % (D.view, D.L, tag))); PB = np.load(V / ("axes_%s_L%d_%sB.npy" % (D.view, D.L, tag)))
        st["split_" + tag] = dict(zip(D.anames, cos_rows(PA, PB).round(3).tolist()))
    A0 = np.load(V / ("axes_%s_L%d.npy" % (D.view, D.L)))
    for dL in (-4, -2, 2, 4):
        L2 = D.L + dL
        if 12 <= L2 <= 32:
            st["layer_%+d" % dL] = dict(zip(D.anames, cos_rows(A0, np.load(V / ("axes_%s_L%d.npy" % (D.view, L2)))).round(3).tolist()))
    out["axis_stability"] = st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="mean")
    ap.add_argument("--layer", type=int, default=15)
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print("%6.0fs %s" % (time.time() - t0, s), flush=True)      # noqa: E731
    D = Data(a.view, a.layer)
    log("data: %d kept traits, %d axes; sets %s" % (D.T.shape[1], D.A.shape[1], {s: len(n) for s, (n, _) in D.sets.items()}))
    out = dict(view=a.view, layer=a.layer, k_grid=K_GRID, n_folds=N_FOLDS, n_rep=N_REP, n_null=N_NULL, kept_traits=D.tnames)
    for s, (names, X) in D.sets.items():
        log("M1 set %s" % s)
        out[s] = {}
        curve_sets(D, X, names, out[s], log)
        if s == "a":
            cats = {}
            for cond in ("6_utility+G", "8_union", "greedy_pool", "PCA"):
                for k in (4, 8):
                    pp = np.array(out[s]["per_persona"][cond][k]) if k in out[s]["per_persona"][cond] else None
                    if pp is None:
                        continue
                    y2 = np.array(out[s]["y2"]); r = (1 - pp) * y2
                    for c in sorted(set(D.cat.values())):
                        idx = [i for i, n in enumerate(names) if D.cat[n] == c]
                        cats.setdefault(cond, {}).setdefault(k, {})[c] = float(1 - r[idx].sum() / y2[idx].sum())
            out[s]["per_category"] = cats
    log("M2 set b")
    names, X = D.sets["b"]
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    m2(D, half_vectors_e1(D, names), names, out["b"], log, Vt)
    log("M2 set a")
    names, X = D.sets["a"]
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    m2(D, half_vectors_seta(D, names), names, out["a"], log, Vt)
    log("extras")
    extras(D, out)
    (AX / "phase1").mkdir(exist_ok=True)
    (AX / "phase1" / ("results_%s_L%d.json" % (a.view, a.layer))).write_text(json.dumps(out))
    c = out["b"]["curves"]
    log("done. set (b) R2_m at k=8: set6 %.3f, union %.3f, greedy %.3f, PCA %.3f; B3 pct of set6: %.0f" % (
        c["6_utility+G"]["centred"][8], c["8_union"]["centred"][8], c["greedy_pool"]["centred"][8], c["PCA"]["centred"][8],
        out["b"]["b3_percentile"]["6_utility+G"][8]))


if __name__ == "__main__":
    main()
