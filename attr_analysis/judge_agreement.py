"""Agreement between two ranking judges on the same answers.

Both directories hold verdicts for the same chunk files. For every answer judged by
both, the script compares the two top-1 names, the overlap of the two top-5 lists, the
nonsense flags, and recall@k against the intended persona under each judge, overall
and per arm and strength.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "attr_analysis")
from score_rank import load, wilson

KS = (1, 5, 10)


def rec(rows, k):
    return float(np.mean([r["target"] in r["rank"][:k] for r in rows])) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--names", default="sonnet,opus")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    na, nb = args.names.split(",")

    A = {(r["cell"], float(r["alpha"]), int(r["qi"])): r for r in load(args.a)[0]}
    B = {(r["cell"], float(r["alpha"]), int(r["qi"])): r for r in load(args.b)[0]}
    keys = sorted(set(A) & set(B))
    pa = [A[k] for k in keys]
    pb = [B[k] for k in keys]
    print(f"{len(keys)} answers judged by both")

    top1 = np.mean([a["rank"][:1] == b["rank"][:1] for a, b in zip(pa, pb)])
    top1_in5 = np.mean([a["rank"][0] in b["rank"][:5] for a, b in zip(pa, pb)])
    top1_in5r = np.mean([b["rank"][0] in a["rank"][:5] for a, b in zip(pa, pb)])
    ov5 = np.mean([len(set(a["rank"][:5]) & set(b["rank"][:5])) for a, b in zip(pa, pb)])
    nons = np.mean([a["nonsense"] == b["nonsense"] for a, b in zip(pa, pb)])
    hit5_agree = np.mean([(a["target"] in a["rank"][:5]) == (b["target"] in b["rank"][:5])
                          for a, b in zip(pa, pb)])
    hit1_agree = np.mean([(a["target"] in a["rank"][:1]) == (b["target"] in b["rank"][:1])
                          for a, b in zip(pa, pb)])
    print(f"same top-1 name                 {top1:.2f}")
    print(f"{na} top-1 inside {nb} top-5     {top1_in5:.2f}")
    print(f"{nb} top-1 inside {na} top-5     {top1_in5r:.2f}")
    print(f"names shared in the two top-5   {ov5:.2f} of 5")
    print(f"same nonsense flag              {nons:.2f}")
    print(f"same hit@1 on the target        {hit1_agree:.2f}")
    print(f"same hit@5 on the target        {hit5_agree:.2f}")

    out = {"n": len(keys), "same_top1": float(top1), "a_top1_in_b_top5": float(top1_in5),
           "b_top1_in_a_top5": float(top1_in5r), "top5_overlap": float(ov5),
           "same_nonsense": float(nons), "same_hit1": float(hit1_agree),
           "same_hit5": float(hit5_agree), "overall": {}, "by_arm_strength": {}}

    print(f"\n{'':<14}" + "".join(f"{na + ' r@' + str(k):>12}{nb + ' r@' + str(k):>12}" for k in KS)
          + f"{na + ' nons':>12}{nb + ' nons':>12}{'n':>6}")
    groups = defaultdict(list)
    for a, b in zip(pa, pb):
        groups[(a["arm"], float(a["alpha"]))].append((a, b))
    rows = [("all", list(zip(pa, pb)))] + [(f"{arm} s={s:g}", v) for (arm, s), v in sorted(groups.items())]
    for name, v in rows:
        ra = [a for a, _ in v]
        rb = [b for _, b in v]
        line = f"{name:<14}"
        rec_ = {}
        for k in KS:
            line += f"{rec(ra, k):>12.2f}{rec(rb, k):>12.2f}"
            rec_[f"{na}@{k}"] = rec(ra, k)
            rec_[f"{nb}@{k}"] = rec(rb, k)
        line += f"{np.mean([a['nonsense'] for a in ra]):>12.2f}{np.mean([b['nonsense'] for b in rb]):>12.2f}{len(v):>6}"
        print(line)
        rec_[f"{na}_nonsense"] = float(np.mean([a["nonsense"] for a in ra]))
        rec_[f"{nb}_nonsense"] = float(np.mean([b["nonsense"] for b in rb]))
        rec_["n"] = len(v)
        (out["overall"] if name == "all" else out["by_arm_strength"]).update(
            {name: rec_} if name != "all" else rec_)

    # per-cell recall@5 from the two judges, for the cells both have judged
    cells = defaultdict(list)
    for a, b in zip(pa, pb):
        cells[(a["arm"], a["target"], float(a["alpha"]))].append((a, b))
    ca = np.array([rec([a for a, _ in v], 5) for v in cells.values()])
    cb = np.array([rec([b for _, b in v], 5) for v in cells.values()])
    r = float(np.corrcoef(ca, cb)[0, 1])
    print(f"\nper-cell recall@5, {len(cells)} cells: r = {r:+.2f}, "
          f"mean |diff| = {np.abs(ca - cb).mean():.2f}")
    out["per_cell_r5_corr"] = r
    out["per_cell_r5_mad"] = float(np.abs(ca - cb).mean())
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
