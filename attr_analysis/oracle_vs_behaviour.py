"""Does the GMM oracle predict what steering actually does?

The oracle scores a steering vector's destination. The judged hit rate says what the
model actually produced. If the oracle is a planning tool, these should agree.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

M = json.load(open("attr_analysis/mixer_data.json"))
K, L = M["k"], "28"
LD = M["L"][L]
G = LD["gauss"]
personas = M["personas"]

Z = np.load("attr_analysis/diff_fid_vectors.npz")
idx = json.load(open("judge_df/index.json"))
lab = {}
for f in sorted(glob.glob("judge_df/verdict_P_*.json")):
    n = int(Path(f).stem.split("_")[-1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"P{n}:{k}")
        if m:
            lab.setdefault(m["cell"], []).append(v)

# the mixer subspace: rebuild E from the stored 3-d view is not enough, so use the
# oracle's own coordinates by projecting with the same basis the mixer exported
import sys
sys.path.insert(0, "attr_analysis")
from loader import load_all
means, _p, allp = load_all(need_pts=False)
a = means[28]["default"]
Mm = np.stack([means[28][p] - a for p in personas])
_, _, vt = np.linalg.svd(Mm - Mm.mean(0), full_matrices=False)
E = vt[:K].T

def score(vec):
    x = vec @ E
    ll = []
    for p in personas:
        g = G[p]
        d = x - np.array(g["mu"])
        q = float(d @ np.array(g["Si"]) @ d)
        ll.append((-0.5 * (q + g["logdet"]), np.sqrt(max(q, 0)), p))
    arr = np.array([x[0] for x in ll])
    post = np.exp(arr - arr.max()); post /= post.sum()
    order = np.argsort(-arr)
    return {p: (post[i], ll[i][1]) for i, p in enumerate(personas)}, personas[order[0]]

TARGETS = ["ghost", "pirate", "vampire"]
ARMS = ["true", "k1", "k2", "k4", "k8", "k20", "nosyn", "cosmatch", "rand"]
print(f"{'cell':<20}{'oracle post':>12}{'mahal':>8}{'oracle top1':>14}{'judged hit':>12}")
P, H, Mh = [], [], []
for t in TARGETS:
    for arm in ARMS:
        key = f"{t}|{arm}|L28"
        if key not in Z:
            continue
        s, top = score(Z[key])
        post, mah = s[t]
        v = lab.get(f"{t}|{arm}", [])
        hit = sum(1 for x in v if x == t) / len(v) if v else np.nan
        print(f"{t + '|' + arm:<20}{post:>12.3f}{mah:>8.1f}{top:>14}{hit:>12.2f}")
        if not np.isnan(hit):
            P.append(post); H.append(hit); Mh.append(mah)
print()
if len(P) > 4:
    r1, p1 = pearsonr(P, H); r2, p2 = spearmanr(P, H)
    r3, p3 = pearsonr(Mh, H)
    print(f"corr(oracle posterior, judged hit) pearson {r1:+.3f} (p={p1:.3g}) "
          f"spearman {r2:+.3f}")
    print(f"corr(mahalanobis,      judged hit) pearson {r3:+.3f} (p={p3:.3g})   n={len(P)}")
    print("\noracle top-1 agreement with the intended persona:")
    agree = sum(1 for t in TARGETS for arm in ARMS
                if f"{t}|{arm}|L28" in Z and score(Z[f"{t}|{arm}|L28"])[1] == t)
    tot = sum(1 for t in TARGETS for arm in ARMS if f"{t}|{arm}|L28" in Z)
    print(f"  {agree}/{tot} cells where the oracle names the intended persona")
