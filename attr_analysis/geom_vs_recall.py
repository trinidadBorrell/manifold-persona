"""Does geometry alone predict the judged recall of pinned steering at one layer?

The pinned arm moves the residual stream to a + s (mu_t - a) along the target's own
direction at layer L. That point is scored against every persona mean with a Gaussian
mixture (tied covariance, fitted in a low-dimensional persona subspace), and the scores
are compared with the recall@k the ranking judge measured on the generations.
"""
import argparse
import glob
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "attr_analysis")

CACHE = "attr_analysis/_alllayer_cache.npz"
RAW = ("attr_acts/*.npz", "attrx_acts/*.npz")


def raw_points(layer):
    pts = {}
    for g in RAW:
        for f in sorted(glob.glob(g)):
            z = np.load(f, allow_pickle=True)
            who = z["persona"].astype(str)
            X = z["prompted"][:, layer, :].astype(np.float32)
            for p in np.unique(who):
                pts.setdefault(p, []).append(X[who == p])
            del z, X
    return {p: np.concatenate(v, 0) for p, v in pts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=0.25)
    ap.add_argument("--meta", default="attr_analysis/l23_meta.json")
    ap.add_argument("--strengths", default="3,4,6,8")
    ap.add_argument("--judge", default="attr_analysis/rank_l23_results.json")
    ap.add_argument("--arm", default="L23")
    ap.add_argument("--out", default="attr_analysis/geom_vs_recall_l23.json")
    args = ap.parse_args()
    L = args.layer

    z = np.load(CACHE, allow_pickle=True)
    names = list(z["names"].astype(str))
    M = z["M"][:, L, :].astype(np.float32)
    idx = {p: i for i, p in enumerate(names)}
    a = M[idx["default"]]
    roles = [p for p in names if p != "default"]

    # persona subspace from the between-persona scatter of the means
    D = np.stack([M[idx[p]] - a for p in roles])
    _, _, vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    E = vt[:args.k].T
    var_k = float((np.linalg.svd(D - D.mean(0), compute_uv=False)[:args.k] ** 2).sum()
                  / (np.linalg.svd(D - D.mean(0), compute_uv=False) ** 2).sum())

    # tied within-persona covariance in that subspace, pooled over the personas with raw points
    pts = raw_points(L)
    W = np.zeros((args.k, args.k))
    n_tot = 0
    for p, Y in pts.items():
        if p not in idx:
            continue
        Q = (Y - M[idx[p]]) @ E
        W += Q.T @ Q
        n_tot += len(Q) - 1
    S = W / n_tot
    S = (1 - args.shrink) * S + args.shrink * np.trace(S) / args.k * np.eye(args.k)
    Si = np.linalg.inv(S)
    classes = names
    MU = np.stack([(M[idx[p]] - a) @ E for p in classes])

    def score(x):
        d = MU - x
        q = np.einsum("ij,jk,ik->i", d, Si, d)
        ll = -0.5 * q
        post = np.exp(ll - ll.max())
        post /= post.sum()
        return post, np.sqrt(q)

    meta = json.load(open(args.meta))
    strengths = [float(s) for s in args.strengths.split(",")]
    judge = json.load(open(args.judge))["cells"] if args.judge else {}

    rows = []
    print(f"layer {L}, {args.k}-d subspace keeps {var_k:.2f} of between-persona variance, "
          f"tied covariance from {len(pts)} personas\n")
    print(f"{'persona':<12}{'s':>4}{'disp':>7}{'p(t)':>7}{'p(def)':>8}{'rank':>6}"
          f"{'mah(t)':>8}{'top1':>14}{'judged r@1':>11}{'r@5':>6}{'r@10':>6}")
    for t in meta["targets"]:
        d = M[idx[t]] - a
        for s in strengths:
            x = (s * d) @ E
            post, mah = score(x)
            order = np.argsort(-post)
            rank = int(np.where(np.array(classes)[order] == t)[0][0]) + 1
            key = f"{args.arm}|{t}|{s:g}"
            j = judge.get(key, {})
            row = {"target": t, "s": s, "disp": float(s * np.linalg.norm(d)),
                   "post_target": float(post[idx[t]]), "post_default": float(post[idx["default"]]),
                   "geom_rank": rank, "geom_in5": rank <= 5, "geom_in10": rank <= 10,
                   "mah_target": float(mah[idx[t]]), "top1": classes[order[0]],
                   "top5": [classes[i] for i in order[:5]],
                   "r@1": j.get("r@1"), "r@5": j.get("r@5"), "r@10": j.get("r@10"),
                   "nonsense": j.get("nonsense")}
            rows.append(row)
            print(f"{t:<12}{s:>4g}{row['disp']:>7.1f}{row['post_target']:>7.2f}"
                  f"{row['post_default']:>8.2f}{rank:>6}{row['mah_target']:>8.1f}"
                  f"{row['top1']:>14}"
                  + (f"{row['r@1']:>11.2f}{row['r@5']:>6.2f}{row['r@10']:>6.2f}"
                     if row["r@5"] is not None else f"{'-':>11}{'-':>6}{'-':>6}"))

    have = [r for r in rows if r["r@5"] is not None]
    out = {"layer": L, "k": args.k, "shrink": args.shrink, "var_kept": var_k, "rows": rows}
    if len(have) > 4:
        print(f"\ncorrelation with judged recall over {len(have)} cells "
              f"(persona x strength):")
        print(f"{'predictor':<16}" + "".join(f"{m:>22}" for m in ("r@1", "r@5", "r@10")))
        out["corr"] = {}
        for name in ("post_target", "post_default", "mah_target", "disp", "geom_rank"):
            x = np.array([r[name] for r in have], float)
            line = f"{name:<16}"
            out["corr"][name] = {}
            for m in ("r@1", "r@5", "r@10"):
                y = np.array([r[m] for r in have], float)
                pr, pp = pearsonr(x, y)
                sr, _ = spearmanr(x, y)
                out["corr"][name][m] = {"pearson": float(pr), "p": float(pp),
                                        "spearman": float(sr)}
                line += f"{pr:+.2f} (p={pp:.2g}) {sr:+.2f}".rjust(22)
            print(line)
        # geometric top-5 membership as a hard prediction of judged recall@5 > 0.5
        for k in (5, 10):
            pred = np.array([r[f"geom_in{k}"] for r in have])
            obs = np.array([r[f"r@{k}"] >= 0.5 for r in have])
            acc = float((pred == obs).mean())
            out[f"agree_in{k}"] = acc
            print(f"geometric top-{k} membership agrees with judged r@{k} >= 0.5 in "
                  f"{acc:.2f} of cells (pred positive {pred.mean():.2f}, "
                  f"observed positive {obs.mean():.2f})")
        # per strength: mean judged recall against mean posterior
        print(f"\n{'s':>4}{'mean p(t)':>11}{'mean p(def)':>13}{'mean r@1':>10}{'mean r@5':>10}"
              f"{'nonsense':>10}")
        for s in strengths:
            v = [r for r in have if r["s"] == s]
            if v:
                print(f"{s:>4g}{np.mean([r['post_target'] for r in v]):>11.2f}"
                      f"{np.mean([r['post_default'] for r in v]):>13.2f}"
                      f"{np.mean([r['r@1'] for r in v]):>10.2f}"
                      f"{np.mean([r['r@5'] for r in v]):>10.2f}"
                      f"{np.mean([r['nonsense'] for r in v]):>10.2f}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
