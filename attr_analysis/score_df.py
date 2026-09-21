"""Does composition matter, or only the angle?

nosyn    reconstruction with every near-synonym banned  (forced composition)
cosmatch a RANDOM vector at the same cosine as the k8 reconstruction
If cosmatch matches nosyn, the behaviour is about angle, not structure.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_df")
idx = json.load(open(D / "index.json"))
rep = json.load(open("attr_analysis/diff_fid_report.json"))
lab = {}
for f in sorted(glob.glob(str(D / "verdict_P_*.json"))):
    n = int(Path(f).stem.split("_")[-1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"P{n}:{k}")
        if m:
            lab.setdefault(m["cell"], []).append(v)

TARGETS = ["ghost", "pirate", "vampire"]
ARMS = ["true", "k1", "k2", "k4", "k8", "k20", "nosyn", "cosmatch", "rand"]
print("hit = fraction of answers the blind judge labelled as the intended persona\n")
print(f"{'arm':<10}" + "".join(f"{t:>22}" for t in TARGETS) + f"{'mean hit':>10}")
rows = {}
for arm in ARMS:
    line = f"{arm:<10}"
    hits = []
    for t in TARGETS:
        v = lab.get(f"{t}|{arm}")
        if not v:
            line += f"{'-':>22}"; continue
        hit = sum(1 for x in v if x == t) / len(v)
        ns = sum(1 for x in v if x == "nonsense") / len(v)
        cos = rep[t].get(arm, {}).get("cos") if isinstance(rep[t].get(arm), dict) else None
        if arm == "cosmatch":
            cos = rep[t]["cosmatch_target"]
        if arm == "true":
            cos = 1.0
        c = f"{cos:.3f}" if cos else "  -  "
        hits.append(hit)
        line += f"   cos{c} hit{hit:>5.2f} ns{ns:>5.2f}"
    rows[arm] = hits
    line += f"{np.mean(hits) if hits else np.nan:>10.2f}"
    print(line)

print("\n=== the decisive comparison: matched cosine, structured vs random ===")
for t in TARGETS:
    a = lab.get(f"{t}|nosyn", []); b = lab.get(f"{t}|cosmatch", [])
    if a and b:
        ha = sum(1 for x in a if x == t) / len(a)
        hb = sum(1 for x in b if x == t) / len(b)
        print(f"  {t:8} nosyn cos={rep[t]['nosyn']['cos']:.3f} hit={ha:.2f}   "
              f"cosmatch cos={rep[t]['cosmatch_target']:.3f} hit={hb:.2f}   diff={ha-hb:+.2f}")
na = [sum(1 for x in lab.get(f'{t}|nosyn', []) if x == t) / max(len(lab.get(f'{t}|nosyn', [1])), 1) for t in TARGETS]
nb = [sum(1 for x in lab.get(f'{t}|cosmatch', []) if x == t) / max(len(lab.get(f'{t}|cosmatch', [1])), 1) for t in TARGETS]
print(f"  mean: nosyn {np.mean(na):.2f} vs cosmatch {np.mean(nb):.2f}")

print("\n=== does hit rate track cosine? ===")
xs, ys = [], []
for t in TARGETS:
    for arm in ["k1", "k2", "k4", "k8", "k20", "nosyn"]:
        v = lab.get(f"{t}|{arm}")
        c = rep[t].get(arm, {}).get("cos")
        if v and c:
            xs.append(c); ys.append(sum(1 for x in v if x == t) / len(v))
if len(xs) > 4:
    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)
    print(f"  corr(cosine, hit rate) = {r:+.3f} (p={p:.3g}, n={len(xs)}) over reconstruction arms")
