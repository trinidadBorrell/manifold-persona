"""Score read-outs with the judge classifier, per arm and strength, on a common persona set.

Reference arms come from the overnight read-outs; the new arm from a second set of
read-out files. Only personas present in the new arm are counted, so the arms compare on
the same personas. Also reports where the new arm's answers land along the classifier's
direction and along the persona-mean direction, in the persona's own distance units.
"""
import argparse
import glob
from collections import defaultdict

import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from judge_classifier import JudgeClassifier

CACHE = "attr_analysis/_alllayer_cache.npz"


def parse(c):
    h, t = c.split("|")[:2]
    return ("prompt", t) if h == "prompt" else tuple(t.split("_", 1))


def load(files, L):
    X, cell, al = [], [], []
    for f in files:
        za = np.load(f, allow_pickle=True)
        li = list(za["layers"]).index(L) if "layers" in za else L
        X.append(za["acts"][:, li, :].astype(np.float32))
        cell += list(za["cell"].astype(str))
        al += list(za["alpha"])
    return np.concatenate(X), cell, np.array(al)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attr_analysis/judge_classifier_all_bycell_model.npz")
    ap.add_argument("--ref", nargs="+", default=sorted(glob.glob("l23all_out/acts_*.npz")))
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--dirs", default="attr_analysis/clf_direction_vectors.npz")
    args = ap.parse_args()

    clf = JudgeClassifier.load(args.model)
    cidx, L = clf.cidx, clf.layer

    Xn, cn, an = load(args.new, L)
    Xr, cr, ar = load(args.ref, L)
    personas = {parse(c)[1] for c in cn}
    print(f"new arm: {len(Xn)} answers, {len(personas)} personas")

    v = np.load(args.dirs, allow_pickle=True)
    dirs = {t: d for t, d in zip(v["names"].astype(str), v["dirs"])}
    centre = v["centre"]
    z = np.load(CACHE, allow_pickle=True)
    names = list(z["names"].astype(str))
    M = z["M"][:, L, :].astype(np.float32)
    midx = {p: i for i, p in enumerate(names)}
    a = M[midx["default"]]

    rows = defaultdict(list)
    for X, cells, al in ((Xr, cr, ar), (Xn, cn, an)):
        for i, (c, s) in enumerate(zip(cells, al)):
            arm, t = parse(c)
            if t in personas and t in cidx:
                rows[(arm, float(s))].append((t, X[i]))

    print(f"\n{'arm':>7}{'s':>4}{'mean p(target)':>16}{'p>=0.1':>9}{'top-1':>7}{'in top-5':>10}"
          f"{'r·d* (x|d|)':>13}{'r·u (x|d|)':>12}{'n':>6}")
    for (arm, s) in sorted(rows):
        ts = [t for t, _ in rows[(arm, s)]]
        X = np.stack([x for _, x in rows[(arm, s)]])
        P = clf.predict_proba(X)
        ti = np.array([cidx[t] for t in ts])
        pt = P[np.arange(len(ts)), ti]
        rank = clf.rank_of(P, ti) - 1
        per = defaultdict(list)
        for t, x in zip(ts, X):
            per[t].append(x)
        pd, pu = [], []
        for t, xs in per.items():
            r = np.mean(xs, 0) - centre
            d = M[midx[t]] - a
            n = np.linalg.norm(d)
            pd.append(r @ dirs[t] / n)
            pu.append(r @ (d / n) / n)
        print(f"{arm:>7}{s:>4g}{pt.mean():>16.3f}{(pt >= 0.1).mean():>9.2f}{(rank == 0).mean():>7.2f}"
              f"{(rank < 5).mean():>10.2f}{np.median(pd):>13.2f}{np.median(pu):>12.2f}{len(ts):>6}")


if __name__ == "__main__":
    main()
