"""One set of coefficients for every layer at once.

A persona's direction is rebuilt from other personas with a single weight vector
that is reused unchanged at all 37 layers, so the recipe is a property of the
persona and not of the layer. Fitted by orthogonal matching pursuit on the
stacked layers, with the target's nearest neighbours banned so a synonym cannot
stand in for it.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_layers

CACHE = "attr_analysis/_alllayer_cache.npz"
TARGETS = ["ghost", "pirate", "vampire", "musician", "spy"]


def load(layers):
    import os
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        names = list(z["names"].astype(str))
        M = z["M"]
    else:
        means, names = load_layers()
        M = np.stack([np.stack([means[L][p] for L in range(37)]) for p in names]).astype(np.float32)
        np.savez(CACHE, M=M, names=np.array(names))
    return {L: {p: M[i, L] for i, p in enumerate(names)} for L in layers}, names


def omp(y, A, k):
    """Greedy sparse fit of y by columns of A; returns (weights, support)."""
    r, sup = y.copy(), []
    for _ in range(k):
        g = A.T @ r
        g[sup] = 0
        j = int(np.argmax(np.abs(g)))
        sup.append(j)
        w, *_ = np.linalg.lstsq(A[:, sup], y, rcond=None)
        r = y - A[:, sup] @ w
    return w, sup


def r2(y, yh):
    return 1.0 - float(((y - yh) ** 2).sum() / (y ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ban", type=int, default=8, help="nearest neighbours to exclude")
    ap.add_argument("--fit_layers", default="all", help="'all' or e.g. 26,27,28,29,30")
    ap.add_argument("--out", default="attr_analysis/shared_all.json")
    args = ap.parse_args()

    L_ALL = list(range(37))
    means, names = load(L_ALL)
    fit_L = L_ALL if args.fit_layers == "all" else [int(x) for x in args.fit_layers.split(",")]
    pool = [p for p in names if p != "default"]

    # d[p] is the persona's displacement from the assistant, stacked over all layers
    d = {p: np.concatenate([means[L][p] - means[L]["default"] for L in L_ALL]) for p in pool}
    W = 2048
    sl = {L: slice(L * W, (L + 1) * W) for L in L_ALL}
    fit_idx = np.concatenate([np.arange(sl[L].start, sl[L].stop) for L in fit_L])

    report = {"k": args.k, "ban": args.ban, "fit_layers": fit_L, "targets": {}}
    for t in TARGETS:
        y = d[t]
        # ban the target and whatever is nearest to it, so no synonym stands in
        others = [p for p in pool if p != t]
        cos = {p: float(d[p] @ y / (np.linalg.norm(d[p]) * np.linalg.norm(y))) for p in others}
        banned = sorted(cos, key=cos.get, reverse=True)[:args.ban]
        basis = [p for p in others if p not in banned]
        A = np.stack([d[p] for p in basis], 1)

        w, sup = omp(y[fit_idx], A[fit_idx], args.k)
        yh = A[:, sup] @ w
        per_layer = {L: r2(y[sl[L]], yh[sl[L]]) for L in L_ALL}

        # what a separate recipe per layer would buy, refitting weights at each layer
        free = {}
        for L in L_ALL:
            wl, sl_ = omp(y[sl[L]], A[sl[L]], args.k)
            free[L] = r2(y[sl[L]], A[sl[L]][:, sl_] @ wl)

        recipe = sorted(zip([basis[j] for j in sup], w.tolist()),
                        key=lambda kv: -abs(kv[1]))
        report["targets"][t] = {
            "banned": banned, "recipe": recipe,
            "shared_r2": per_layer, "free_r2": free,
            "cos_all": float(yh @ y / (np.linalg.norm(yh) * np.linalg.norm(y))),
            "cos_steer": {L: float(yh[sl[L]] @ y[sl[L]] /
                                   (np.linalg.norm(yh[sl[L]]) * np.linalg.norm(y[sl[L]])))
                          for L in (26, 27, 28, 29, 30)}}

        print(f"\n{t}  (banned: {', '.join(banned[:4])} ...)")
        print("  " + "  ".join(f"{c:+.2f} {n}" for n, c in recipe))
        print(f"  cos over all 37 layers {report['targets'][t]['cos_all']:.3f}")
        print(f"  {'layer':<7}" + "".join(f"{L:>7}" for L in (20, 24, 26, 28, 30, 32, 36)))
        print(f"  {'shared':<7}" + "".join(f"{per_layer[L]:>7.2f}"
                                           for L in (20, 24, 26, 28, 30, 32, 36)))
        print(f"  {'free':<7}" + "".join(f"{free[L]:>7.2f}"
                                         for L in (20, 24, 26, 28, 30, 32, 36)))

    steer = [26, 27, 28, 29, 30]
    print(f"\n{'persona':<10}{'shared R2 (26-30)':>19}{'free R2 (26-30)':>17}{'cost':>8}")
    for t in TARGETS:
        r = report["targets"][t]
        a = np.mean([r["shared_r2"][L] for L in steer])
        b = np.mean([r["free_r2"][L] for L in steer])
        print(f"{t:<10}{a:>19.3f}{b:>17.3f}{b - a:>8.3f}")
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
