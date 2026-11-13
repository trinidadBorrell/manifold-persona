"""Seed vs question/wording agreement in E3 base-model role maps (offline, no GPU, no judge).

Descriptive input to pod A's run-count discussion (THRESHOLDS.md, "Run count and noise"). Uses the two
saved E3 seeds (run_main, run_rep = seed offset 100000) on the 18 shared questions, 20 pilot roles,
5 wordings, hidden state 17, conditions desc and ex2. Base model, not Instruct. It does not estimate
variance components or forecast interval widths.

Map = Spearman r between the upper triangles of two role-role cosine-distance matrices (role vectors
centred across roles). Halves split questions (9/9) and wordings (2/3) together; one saved list of
halves is shared by both conditions. Seed comparisons average both directions and both sides.

    .venv/bin/python experiments/q1_emergence/261029_instruct_persona_map/e3_noise_check.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
E3 = HERE.parent / "261008_base_persona_map"
STATE = 17
N_SPLITS = 50
SEED = 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load(run):
    rows = [json.loads(l) for l in open(E3 / run / "replies.jsonl")]
    meta = json.load(open(E3 / run / "reply_mean_acts_slim.json"))
    path = E3 / run / meta["file"]
    assert sha256(path) == meta["sha256"], f"{run}: state file hash mismatch"
    acts = np.load(path, mmap_mode="r")
    assert acts.shape[0] == len(rows)
    return rows, acts[:, meta["layers"].index(STATE)], meta["sha256"]


def grid(rows, acts, cond, roles, qs):
    """X[role, wording, question, dim] for one condition (shared questions only); each cell exactly once."""
    X = np.full((len(roles), 5, len(qs), acts.shape[1]), np.nan, np.float32)
    seen = set()
    ri, qi = {r: k for k, r in enumerate(roles)}, {q: k for k, q in enumerate(qs)}
    for j, r in enumerate(rows):
        if r["cond"] == cond and r["shared_q"] and r["valid"] and r["role"] in ri:
            key = (r["role"], r["wording"], r["question_idx"])
            assert key not in seen, f"{cond}: duplicate cell {key}"
            seen.add(key)
            X[ri[r["role"]], r["wording"], qi[r["question_idx"]]] = acts[j]
    assert len(seen) == X.shape[0] * X.shape[1] * X.shape[2], f"{cond}: missing cells"
    assert np.isfinite(X).all(), f"{cond}: non-finite states"
    return X


def role_map(X, w_mask, q_mask):
    v = X[:, w_mask][:, :, q_mask].mean(axis=(1, 2))
    v = v - v.mean(0)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = 1 - v @ v.T
    return d[np.triu_indices(len(v), 1)]


def r(a, b):
    return float(spearmanr(a, b).statistic)


def make_halves():
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_SPLITS):
        q = np.zeros(18, bool)
        q[rng.choice(18, 9, replace=False)] = True
        w = np.zeros(5, bool)
        w[rng.choice(5, int(rng.integers(2, 4)), replace=False)] = True
        out.append((w, q))
    return out


def summarise(vals):
    vals = np.asarray(vals)
    return dict(mean=round(float(vals.mean()), 3), sd=round(float(vals.std(ddof=1)), 3),
                min=round(float(vals.min()), 3), max=round(float(vals.max()), 3))


def ss_shares(Xa, Xb):
    """Observed sums-of-squares shares on the two-seed crossed grid (descriptive; not variance components).

    The seed term is the within-prompt residual of two replies; it holds half the single-reply sampling
    variance in expectation, and the other half stays in the other terms.
    """
    X = np.stack([Xa, Xb], axis=-2)                     # role, wording, question, seed, dim
    g = X.mean(axis=(0, 1, 2, 3))
    tot = ((X - g) ** 2).sum()
    prompt = X.mean(3)
    seed = ((X - prompt[:, :, :, None]) ** 2).sum()
    role = X.mean(axis=(1, 2, 3))
    ques = X.mean(axis=(0, 1, 3))
    rq = X.mean(axis=(1, 3))
    rw = X.mean(axis=(2, 3))
    n_r, n_w, n_q, n_s = X.shape[:4]
    ss_role = n_w * n_q * n_s * ((role - g) ** 2).sum()
    ss_q = n_r * n_w * n_s * ((ques - g) ** 2).sum()
    ss_rq = n_w * n_s * ((rq - role[:, None] - ques[None] + g) ** 2).sum()
    ss_w = n_q * n_s * ((rw - role[:, None]) ** 2).sum()
    out = dict(role=ss_role, question=ss_q, role_x_question=ss_rq, wording_within_role=ss_w,
               two_seed_residual=seed)
    out["other"] = tot - sum(out.values())
    return {k: round(float(v / tot), 4) for k, v in out.items()}


def main():
    (rm, am, ha), (rr, ar, hb) = load("run_main"), load("run_rep")
    roles = sorted({x["role"] for x in rm if not x["role"].startswith("default")})
    qs = sorted({x["question_idx"] for x in rm if x["shared_q"]})
    assert len(roles) == 20 and len(qs) == 18
    halves = make_halves()
    allw, allq = np.ones(5, bool), np.ones(18, bool)
    res = dict(state=STATE, n_splits=N_SPLITS, half_seed=SEED, roles=len(roles), questions=len(qs), wordings=5,
               state_sha256=dict(run_main=ha, run_rep=hb),
               halves=[dict(wordings=np.flatnonzero(w).tolist(), questions=[qs[i] for i in np.flatnonzero(q)])
                       for w, q in halves])
    for cond in ("desc", "ex2"):
        A, B = grid(rm, am, cond, roles, qs), grid(rr, ar, cond, roles, qs)
        P = (A + B) / 2
        out = {"seed_vs_seed_full": round(r(role_map(A, allw, allq), role_map(B, allw, allq)), 3)}
        within_a, within_b, cross, pooled, same_half = [], [], [], [], []
        for w, q in halves:
            h1 = {k: role_map(X, w, q) for k, X in (("A", A), ("B", B), ("P", P))}
            h0 = {k: role_map(X, ~w, ~q) for k, X in (("A", A), ("B", B), ("P", P))}
            within_a.append(r(h1["A"], h0["A"]))
            within_b.append(r(h1["B"], h0["B"]))
            cross.append((r(h1["A"], h0["B"]) + r(h1["B"], h0["A"])) / 2)
            pooled.append(r(h1["P"], h0["P"]))
            same_half.append((r(h1["A"], h1["B"]) + r(h0["A"], h0["B"])) / 2)
        out.update(other_half_seed_main=summarise(within_a), other_half_seed_rep=summarise(within_b),
                   other_half_other_seed=summarise(cross), other_half_two_seeds_averaged=summarise(pooled),
                   same_half_other_seed=summarise(same_half), ss_shares_descriptive=ss_shares(A, B))
        res[cond] = out
    path = HERE / "results" / "e3_noise_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "halves"}, indent=1))


if __name__ == "__main__":
    main()
