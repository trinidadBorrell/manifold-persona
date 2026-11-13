"""Check 3 (13/11/2026, exploratory): the default on Lu et al.'s 0-1 scale, at each pod C reader and for the words.

Lu et al. report the default 0.03 from one end of PC1 of the role vectors. Check 2 reports position in role SDs, which
hides where the default sits in the role range. Here: PC1 of the 20 centred role means (roles only, all questions),
the default projected on it, position = (default - min role) / (max role - min role), distance = min(x, 1 - x).
Intervals: 1,000 question resamples (counts over the 40 questions), role means re-formed per draw. Text: the same
statistic on check 2's text features (character 3-5-gram TF-IDF, 256-dim SVD). Also saves full-data PC1/PC2
coordinates of the role means and the default for each reader, for figures. State 17, Instruct-written replies.

    .venv/bin/python experiments/q1_emergence/261112_free_checks/check3_lu_scale.py
"""
import json
import time

import numpy as np

import check_role_count as C
import check_default_readers as R

READERS = ("base", "sft", "dpo", "instruct")
SEED = 20261113
N_DRAWS = 1000


def question_means(X, rows, names):
    qs = sorted({r["question"] for r in rows})
    role = np.array([r["role"] for r in rows])
    q = np.array([r["question"] for r in rows])
    P = np.zeros((len(names), len(qs), X.shape[1]))
    for i, n in enumerate(names):
        for j, qq in enumerate(qs):
            P[i, j] = X[(role == n) & (q == qq)].mean(0)
    return P


def lu_position(P, w):
    M = np.tensordot(w, P, axes=([0], [1])) / w.sum()
    roles, d = M[:-1], M[-1]
    c = roles.mean(0)
    u = np.linalg.svd(roles - c, full_matrices=False)[2][0]
    pr, pd = (roles - c) @ u, (d - c) @ u
    x = (pd - pr.min()) / (pr.max() - pr.min())
    return float(min(x, 1 - x)), bool(pd < pr.min() or pd > pr.max())


def coords(P, names):
    M = P.mean(1)
    roles, d = M[:-1], M[-1]
    c = roles.mean(0)
    vt = np.linalg.svd(roles - c, full_matrices=False)[2][:2]
    pr, pd = (roles - c) @ vt.T, (d - c) @ vt.T
    if pd[0] > 0:
        pr[:, 0], pd[0] = -pr[:, 0], -pd[0]
    lo, hi = pr[:, 0].min(), pr[:, 0].max()
    return dict(roles=names[:-1], pc1=pr[:, 0].tolist(), pc2=pr[:, 1].tolist(), default_pc1=float(pd[0]),
                default_pc2=float(pd[1]), pc1_scaled=((pr[:, 0] - lo) / (hi - lo)).tolist(),
                default_scaled=float((pd[0] - lo) / (hi - lo)))


def main():
    t0 = time.time()
    rows = C.grid_order(C.poda_rows())
    names = sorted({r["role"] for r in rows} - {C.DEFAULT}) + [C.DEFAULT]
    nq = len({r["question"] for r in rows})
    rng = np.random.default_rng(SEED)
    W = [np.ones(nq)] + [np.bincount(rng.integers(0, nq, nq), minlength=nq).astype(float) for _ in range(N_DRAWS)]
    out = dict(rules="check 3, exploratory, written 13/11/2026 after check 2", state=C.STATE, seed=SEED,
               draws=N_DRAWS, n_roles=len(names) - 1, readers={}, coords={})
    sources = {r: f"podc:{r}" for r in READERS}
    for name, src in list(sources.items()) + [("text", None)]:
        if src:
            X, rows_x = C.load(src)
            assert [r["role"] for r in rows_x] == [r["role"] for r in rows]
        else:
            T, _ = R.text_features(C.PODA / "rows.jsonl")
            key = lambda r: (r["role"], r["wording"], r["question"], r["replicate"])
            file_keys = [key(r) for r in C.poda_rows()]
            assert len(set(file_keys)) == len(file_keys)
            pos = {k: i for i, k in enumerate(file_keys)}
            X = T[[pos[key(r)] for r in rows]].astype(np.float64)
        P = question_means(X, rows, names)
        del X
        draws = np.array([lu_position(P, w)[0] for w in W])
        point, outside = lu_position(P, W[0])
        out["readers"][name] = dict(distance_to_nearer_end=point, outside_role_range=outside,
                                    interval=C.interval(draws[1:]))
        out["coords"][name] = coords(P, names)
        print(name, round(point, 3), out["readers"][name]["interval"], round(time.time() - t0), "s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (C.HERE / "results").mkdir(exist_ok=True)
    (C.HERE / "results/check3_lu_scale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
