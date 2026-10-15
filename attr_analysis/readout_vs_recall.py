"""Score the read-out activation of each steered answer against the persona clouds.

Each answer's mean response activation (empty system prompt, no steering) is placed in
the persona subspace and ranked against all persona means with the same tied-covariance
Gaussian used for the planned destination. The persona rank of the answer's own point is
compared with the rank the text judge gave the same answer, so the question becomes
whether geometry read off the output predicts what the judge reads off the text.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "attr_analysis")
from geom_vs_recall import CACHE, raw_points
from score_rank import load as load_judge

KS = (1, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=0.25)
    ap.add_argument("--judge", nargs="+", default=["judge_rank_l23", "judge_rank_l23k"])
    ap.add_argument("--arms", default="L23,k8,k32,prompt")
    ap.add_argument("--out", default="attr_analysis/readout_vs_recall.json")
    args = ap.parse_args()
    L = args.layer

    z = np.load(CACHE, allow_pickle=True)
    names = list(z["names"].astype(str))
    M = z["M"][:, L, :].astype(np.float32)
    idx = {p: i for i, p in enumerate(names)}
    a = M[idx["default"]]
    roles = [p for p in names if p != "default"]
    D = np.stack([M[idx[p]] - a for p in roles])
    _, _, vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    E = vt[:args.k].T

    pts = raw_points(L)
    W = np.zeros((args.k, args.k))
    n_tot = 0
    for p, Y in pts.items():
        if p in idx:
            Q = (Y - M[idx[p]]) @ E
            W += Q.T @ Q
            n_tot += len(Q) - 1
    S = W / n_tot
    S = (1 - args.shrink) * S + args.shrink * np.trace(S) / args.k * np.eye(args.k)
    Si = np.linalg.inv(S)
    MU = np.stack([(M[idx[p]] - a) @ E for p in names])

    def rank_of(x, t):
        d = MU - x
        q = np.einsum("ij,jk,ik->i", d, Si, d)
        order = np.argsort(q)
        return int(np.where(np.array(names)[order] == t)[0][0]) + 1, names[order[0]], \
            int(np.where(np.array(names)[order] == "default")[0][0]) + 1

    # judged ranks keyed by (cell, alpha, qi)
    judged = {}
    for d in args.judge:
        try:
            rows, _ = load_judge(d)
        except FileNotFoundError:
            continue
        for r in rows:
            judged[(r["cell"], float(r["alpha"]), int(r["qi"]))] = r

    arms = set(args.arms.split(","))
    per = []
    for f in args.acts:
        za = np.load(f, allow_pickle=True)
        A = za["acts"][:, L, :].astype(np.float32)
        for x, cell, al, qi in zip(A, za["cell"].astype(str), za["alpha"], za["qi"]):
            arm_t = cell.split("|")[1]
            arm, t = (arm_t.split("_", 1) if "_" in arm_t else ("prompt", arm_t))
            if arm not in arms:
                continue
            gr, top, dr = rank_of((x - a) @ E, t)
            j = judged.get((cell, float(al), int(qi)))
            per.append({"arm": arm, "target": t, "s": float(al), "qi": int(qi),
                        "geom_rank": gr, "geom_top1": top, "default_rank": dr,
                        "judge_rank": (j["rank"].index(t) + 1 if j and t in j["rank"] else None),
                        "judged": j is not None,
                        "judge_nonsense": j["nonsense"] if j else None})

    print(f"layer {L}: {len(per)} read-out points, {sum(p['judged'] for p in per)} with a verdict\n")
    print(f"{'arm':<8}{'s':>4}{'n':>5}" + "".join(f"{'geom@' + str(k):>8}" for k in KS)
          + "".join(f"{'judge@' + str(k):>9}" for k in KS) + f"{'agree@5':>9}{'def top1':>10}")
    out = {"layer": L, "k": args.k, "by_arm_strength": {}, "corr": {}}
    cells = defaultdict(list)
    for p in per:
        cells[(p["arm"], p["s"])].append(p)
    for (arm, s), v in sorted(cells.items()):
        row = {"n": len(v)}
        line = f"{arm:<8}{s:>4g}{len(v):>5}"
        for k in KS:
            row[f"geom@{k}"] = float(np.mean([p["geom_rank"] <= k for p in v]))
            line += f"{row[f'geom@{k}']:>8.2f}"
        jv = [p for p in v if p["judged"]]
        for k in KS:
            row[f"judge@{k}"] = (float(np.mean([(p["judge_rank"] or 99) <= k for p in jv]))
                                 if jv else None)
            line += (f"{row[f'judge@{k}']:>9.2f}" if jv else f"{'-':>9}")
        if jv:
            row["agree@5"] = float(np.mean([(p["geom_rank"] <= 5) == ((p["judge_rank"] or 99) <= 5)
                                            for p in jv]))
            line += f"{row['agree@5']:>9.2f}"
        else:
            line += f"{'-':>9}"
        row["default_top1"] = float(np.mean([p["geom_top1"] == "default" for p in v]))
        line += f"{row['default_top1']:>10.2f}"
        out["by_arm_strength"][f"{arm}|{s:g}"] = row
        print(line)

    # per (arm, persona, strength) cell: geometric recall against judged recall
    pc = defaultdict(list)
    for p in per:
        if p["judged"]:
            pc[(p["arm"], p["target"], p["s"])].append(p)
    if len(pc) > 4:
        print(f"\nper cell ({len(pc)} cells), geometric recall against judged recall:")
        for k in KS:
            g = np.array([np.mean([p["geom_rank"] <= k for p in v]) for v in pc.values()])
            j = np.array([np.mean([(p["judge_rank"] or 99) <= k for p in v]) for v in pc.values()])
            pr, pp = pearsonr(g, j)
            sr, _ = spearmanr(g, j)
            out["corr"][f"@{k}"] = {"pearson": float(pr), "p": float(pp), "spearman": float(sr),
                                   "mean_geom": float(g.mean()), "mean_judge": float(j.mean())}
            print(f"  recall@{k:<3} pearson {pr:+.2f} (p={pp:.2g}) spearman {sr:+.2f}   "
                  f"mean geom {g.mean():.2f} mean judge {j.mean():.2f}")
        # per answer: does the geometric rank of this answer track the judge's rank?
        gr = np.array([p["geom_rank"] for v in pc.values() for p in v])
        jr = np.array([p["judge_rank"] or 11 for v in pc.values() for p in v])
        sr, _ = spearmanr(gr, jr)
        out["corr"]["per_answer_rank_spearman"] = float(sr)
        print(f"  per answer, spearman(geometric rank, judge rank) {sr:+.2f}  n={len(gr)}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
