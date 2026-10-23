"""Item files for Exp 1, Exp 2 and Exp 4 from scores.npz (all directions it holds). Seeds are fixed and depend on (component, repeat/split)
only, so PCA k and PGA k get the same shuffles of rank positions and the same A/B assignment.
Exp 5 (whole axis, no label): 5 + 5 example roles drawn from the 20 most extreme at each end, then 20 roles, one
from each of 20 equal-count bins of the score over the other 266 roles.
Exp 4 (whole axis) reuses each Exp 2 split's label: 20 roles the labeller did not see, one drawn from each
of 20 equal-count bins of the score over the other 256 roles.

    python -m axis_interpretability.build --dir $PM/AXI_L28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_REP, N_SPLIT, N_RATE, N_END, N_SPREAD = 5, 5, 10, 20, 20


def ends(score, n):
    o = np.argsort(score)
    return o[:n], o[::-1][:n]          # most negative first, most positive first


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    D = Path(a.dir)
    s = np.load(D / "scores.npz")
    roles = [str(r) for r in s["roles"]]
    K = s["pca_score"].shape[1]
    exp1, exp2, exp5 = [], [], []
    for m in ("pca", "pga"):
        for k in range(K):
            ax = "%s%d" % (m.upper(), k + 1)
            score = s["%s_score" % m][:, k]
            minus, plus = ends(score, N_END)
            for r in range(N_REP):
                g = np.random.default_rng(100 * (k + 1) + r)
                pm, pp, flip = g.permutation(N_RATE), g.permutation(N_RATE), bool(g.integers(2))
                lo, hi = [roles[i] for i in minus[:N_RATE][pm]], [roles[i] for i in plus[:N_RATE][pp]]
                A, B = (hi, lo) if flip else (lo, hi)
                exp1.append(dict(id="exp1_%s_r%d" % (ax, r), axis=ax, rep=r, A=A, B=B,
                                 A_is="plus" if flip else "minus"))
            for sp in range(N_SPLIT):
                g = np.random.default_rng(1000 * (k + 1) + sp)
                sm, spl, flip = g.permutation(N_END), g.permutation(N_END), bool(g.integers(2))
                seen_m, held_m = minus[np.sort(sm[:10])], minus[np.sort(sm[10:])]
                seen_p, held_p = plus[np.sort(spl[:10])], plus[np.sort(spl[10:])]
                truth = {roles[i]: "minus" for i in np.r_[seen_m, held_m]}
                truth.update({roles[i]: "plus" for i in np.r_[seen_p, held_p]})
                lo, hi = [roles[i] for i in g.permutation(seen_m)], [roles[i] for i in g.permutation(seen_p)]
                A, B = (hi, lo) if flip else (lo, hi)
                held = [roles[i] for i in g.permutation(np.r_[held_m, held_p])]
                seen = [roles[i] for i in g.permutation(np.r_[seen_m, seen_p])]
                assert not set(held) & set(A + B)
                g4 = np.random.default_rng(5000 * (k + 1) + sp)
                rest = np.setdiff1d(np.arange(len(roles)), np.r_[seen_m, seen_p])
                rest = rest[np.argsort(score[rest])]
                spread = [roles[int(g4.choice(b))] for b in np.array_split(rest, N_SPREAD)]
                spread = [spread[i] for i in g4.permutation(N_SPREAD)]
                assert not set(spread) & set(A + B)
                g5 = np.random.default_rng(7000 * (k + 1) + sp)
                am, ap_ = minus[np.sort(g5.permutation(N_END)[:5])], plus[np.sort(g5.permutation(N_END)[:5])]
                flip5 = bool(g5.integers(2))
                rest5 = np.setdiff1d(np.arange(len(roles)), np.r_[am, ap_])
                rest5 = rest5[np.argsort(score[rest5])]
                spread5 = [roles[int(g5.choice(b))] for b in np.array_split(rest5, N_SPREAD)]
                spread5 = [spread5[i] for i in g5.permutation(N_SPREAD)]
                lo5, hi5 = [roles[i] for i in g5.permutation(am)], [roles[i] for i in g5.permutation(ap_)]
                A5, B5 = (hi5, lo5) if flip5 else (lo5, hi5)
                assert not set(spread5) & set(A5 + B5)
                exp5.append(dict(id="exp5_%s_s%d" % (ax, sp), axis=ax, split=sp, A=A5, B=B5, A_is="plus" if flip5 else "minus",
                                 spread=spread5, spread_score={r: float(score[roles.index(r)]) for r in spread5}))
                exp2.append(dict(id="exp2_%s_s%d" % (ax, sp), axis=ax, split=sp, A=A, B=B,
                                 A_is="plus" if flip else "minus", held=held, seen=seen, truth=truth,
                                 spread=spread, spread_score={r: float(score[roles.index(r)]) for r in spread}))
    (D / "items").mkdir(exist_ok=True)
    (D / "items" / "exp1.json").write_text(json.dumps(exp1, indent=1))
    (D / "items" / "exp2.json").write_text(json.dumps(exp2, indent=1))
    (D / "items" / "exp5.json").write_text(json.dumps(exp5, indent=1))
    print("%d exp1 items, %d exp2 items, %d exp5 items" % (len(exp1), len(exp2), len(exp5)))


if __name__ == "__main__":
    main()
