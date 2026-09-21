"""Do persona attributes live in reusable directions?

Reads the merged per-response activations, then runs four tests:
  1. ladder ordering   -- do infant..ancient sit in age order on one direction?
  2. curvature         -- is the ladder a line (a direction) or a curve (a spline)?
  3. parallelism       -- is old-young the same vector across occupations?
  4. analogy retrieval -- leave one occupation out, predict it, rank the truth

Test 3 is only meaningful with the assistant axis projected out: every pair of
persona directions already shares cosine ~0.30 through that common axis, so raw
parallelism would look strong even for unrelated attributes.
"""
import argparse
import glob
import json
import itertools

import numpy as np
from scipy.stats import spearmanr

AGE_LADDER = ["infant", "toddler", "adolescent", "teenager", "graduate",
              "parent", "grandparent", "retiree", "elder", "ancient"]
ERA_LADDER = ["era_medieval", "era_victorian", "era_midcentury",
              "era_contemporary", "era_futuristic"]
OCC = ["musician", "spy", "scholar", "soldier", "merchant"]


def load(paths):
    P, names = [], []
    for f in sorted(glob.glob(paths)):
        z = np.load(f, allow_pickle=True)
        P.append(z["prompted"].astype(np.float32))
        names.append(z["persona"].astype(str))
    X = np.concatenate(P, 0)
    who = np.concatenate(names)
    return X, who


def means(X, who, layer):
    return {p: X[who == p, layer, :].mean(0) for p in np.unique(who)}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def deflate(v, axis):
    """Remove the assistant-axis component -- the shared 'being a character' direction."""
    return v - (v @ axis) * axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="attr_acts/*.npz")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="attr_analysis/geometry.json")
    args = ap.parse_args()

    X, who = load(args.acts)
    M = means(X, who, args.layer)
    have = set(M)
    print(f"{len(X)} responses, {len(have)} personas, layer {args.layer}")

    asst = M["default"]
    D = {p: M[p] - asst for p in have if p != "default"}
    # assistant axis, paper-style: assistant mean minus the mean persona vector
    axis = unit(asst - np.mean([M[p] for p in have if p != "default"], 0))

    res = {"layer": args.layer, "n_personas": len(have)}

    # 1+2. ladders: ordering and curvature
    for name, ladder in [("age", AGE_LADDER), ("era", ERA_LADDER)]:
        L = [p for p in ladder if p in have]
        if len(L) < 4:
            continue
        A = np.stack([D[p] for p in L])
        C = A - A.mean(0)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()
        t = C @ Vt[0]
        rho, pv = spearmanr(t, np.arange(len(L)))
        res[name] = {
            "order": L,
            "pc1_var": round(float(var[0]), 3),
            "pc2_var": round(float(var[1]), 3),
            "spearman_pc1_vs_rank": round(float(abs(rho)), 3),
            "spearman_p": float(pv),
            "coord_on_pc1": [round(float(x), 2) for x in t],
            "step_norms": [round(float(np.linalg.norm(D[p])), 2) for p in L],
        }
        print(f"\n[{name}] PC1 explains {var[0]:.2f}, PC2 {var[1]:.2f} | "
              f"|rho| vs true order = {abs(rho):.3f} (p={pv:.3g})")
        print(f"   {' < '.join(L)}")

    # 3+4. attribute offsets across occupations
    for name, lo, hi in [("age", "young", "old"), ("era", "medieval", "futuristic")]:
        pairs = [(f"{lo}_{o}", f"{hi}_{o}", o) for o in OCC]
        pairs = [(a, b, o) for a, b, o in pairs if a in have and b in have]
        if len(pairs) < 3:
            continue
        off = {o: D[b] - D[a] for a, b, o in pairs}
        raw = [unit(off[x]) @ unit(off[y]) for x, y in itertools.combinations(off, 2)]
        dfl = {o: unit(deflate(v, axis)) for o, v in off.items()}
        de = [dfl[x] @ dfl[y] for x, y in itertools.combinations(dfl, 2)]

        # null: random persona-pair differences, same deflation
        rng = np.random.default_rng(0)
        keys = sorted(D)
        null = []
        for _ in range(400):
            a, b, c, d = rng.choice(len(keys), 4, replace=False)
            v1 = unit(deflate(D[keys[b]] - D[keys[a]], axis))
            v2 = unit(deflate(D[keys[d]] - D[keys[c]], axis))
            null.append(v1 @ v2)

        # leave-one-out analogy retrieval
        hits, ranks = 0, []
        cand = sorted(D)
        for a, b, o in pairs:
            others = [off[k] for k in off if k != o]
            pred = D[a] + np.mean(others, 0)
            sims = sorted(((unit(pred) @ unit(D[c]), c) for c in cand), reverse=True)
            order = [c for _, c in sims]
            r = order.index(b) + 1
            ranks.append(r)
            hits += (r == 1)

        res[f"{name}_offset"] = {
            "pairs": [o for _, _, o in pairs],
            "cos_raw_mean": round(float(np.mean(raw)), 3),
            "cos_deflated_mean": round(float(np.mean(de)), 3),
            "null_mean": round(float(np.mean(null)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "top1": f"{hits}/{len(pairs)}",
            "ranks": ranks,
            "median_rank": float(np.median(ranks)),
            "n_candidates": len(cand),
            "offset_norm_mean": round(float(np.mean([np.linalg.norm(v) for v in off.values()])), 2),
        }
        print(f"\n[{name} offset: {hi} - {lo}] across {len(pairs)} occupations")
        print(f"   cosine raw      {np.mean(raw):+.3f}")
        print(f"   cosine deflated {np.mean(de):+.3f}   "
              f"(null {np.mean(null):+.3f}, 95th pct {np.percentile(null, 95):+.3f})")
        print(f"   analogy top-1   {hits}/{len(pairs)}  median rank "
              f"{np.median(ranks):.0f} of {len(cand)}")
        print(f"   |offset|        {np.mean([np.linalg.norm(v) for v in off.values()]):.1f} "
              f"(coherence budget was ~20 at L28)")

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
