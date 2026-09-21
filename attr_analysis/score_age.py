"""Does steering along the age ladder produce the age we aimed at?

control    each rung on its own -- does the judge recover its nominal age?
piecewise  follow the ladder rung to rung
spline     Catmull-Rom through the rungs
linear     straight youngest -> oldest

All three paths aim at the same nominal age at each beta, so error is comparable.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_age")
idx = json.load(open(D / "index.json"))
meta = json.load(open("attr_analysis/age_meta.json"))
LADDER, AGES = meta["ladder"], meta["ages"]
NOM = dict(zip(LADDER, AGES))

lab = {}
for f in sorted(glob.glob(str(D / "verdict_*.json"))):
    n = int(Path(f).stem.split("_")[1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"{n}:{k}")
        if m:
            lab.setdefault((m["cell"], m["alpha"]), []).append(v)

def split(vals):
    nums = [float(v) for v in vals if not isinstance(v, str)]
    ns = sum(1 for v in vals if v == "nonsense")
    asst = sum(1 for v in vals if v == "assistant")
    return nums, ns / len(vals), asst / len(vals)

print(f"{len(lab)} cells, {sum(len(v) for v in lab.values())} judgements\n")
print("=== CONTROL: steer at each rung's own centroid ===")
print(f"{'rung':<13}{'nominal':>8}{'a':>5}{'judged':>9}{'err':>7}{'nonsense':>10}{'assistant':>10}")
ctrl = {}
for a in (1.0, 1.4):
    for p in LADDER:
        v = lab.get((f"control|{p}", a))
        if not v:
            continue
        nums, ns, asst = split(v)
        med = np.median(nums) if nums else float("nan")
        err = abs(med - NOM[p]) if nums else float("nan")
        ctrl.setdefault(a, []).append((err, ns))
        print(f"{p:<13}{NOM[p]:>8.0f}{a:>5}{med:>9.0f}{err:>7.0f}{ns:>10.2f}{asst:>10.2f}")
    e = [x[0] for x in ctrl[a] if not np.isnan(x[0])]
    print(f"  --> alpha={a}: median |error| {np.median(e):.0f}y, "
          f"nonsense {np.mean([x[1] for x in ctrl[a]]):.2f}, "
          f"corr(judged,nominal) computed below\n")

# correlation across rungs
for a in (1.0, 1.4):
    xs, ys = [], []
    for p in LADDER:
        v = lab.get((f"control|{p}", a))
        if v:
            nums, _, _ = split(v)
            if nums:
                xs.append(NOM[p]); ys.append(np.median(nums))
    if len(xs) > 3:
        from scipy.stats import spearmanr, pearsonr
        print(f"  control a={a}: pearson r={pearsonr(xs, ys)[0]:+.3f}  "
              f"spearman={spearmanr(xs, ys)[0]:+.3f}  (n={len(xs)} rungs)")

print("\n=== PATHS: same target age, three ways of getting there ===")
betas = meta["betas"]
for a in (1.0, 1.4):
    print(f"\n--- alpha = {a} ---")
    print(f"{'beta':>5}{'target':>8}" + "".join(f"{arm:>22}" for arm in
                                                ("piecewise", "spline", "linear")))
    agg = {arm: {"err": [], "ns": []} for arm in ("piecewise", "spline", "linear")}
    for b in betas:
        tgt = meta[str(b)]["target_age"]
        row = f"{b:>5}{tgt:>8.0f}"
        for arm in ("piecewise", "spline", "linear"):
            v = lab.get((f"{arm}|{b}", a))
            if not v:
                row += f"{'-':>22}"; continue
            nums, ns, asst = split(v)
            med = np.median(nums) if nums else float("nan")
            err = abs(med - tgt) if nums else float("nan")
            if not np.isnan(err):
                agg[arm]["err"].append(err)
            agg[arm]["ns"].append(ns)
            row += f"{med:>10.0f}{'(e' + f'{err:.0f}' + ')':>7}{ns:>5.2f}"
        print(row)
    print(f"\n{'arm':<12}{'median |err|':>14}{'mean |err|':>12}{'nonsense':>11}{'n':>5}")
    for arm in ("piecewise", "spline", "linear"):
        e, n = agg[arm]["err"], agg[arm]["ns"]
        if e:
            print(f"{arm:<12}{np.median(e):>14.1f}{np.mean(e):>12.1f}"
                  f"{np.mean(n):>11.2f}{len(e):>5}")

# does displacement explain the nonsense rate, rather than the arm?
print("\n=== confound check: nonsense vs steering distance ===")
norms = meta["norms"]
X, Y, A = [], [], []
for (cell, a), v in lab.items():
    if "|" not in cell or cell.startswith("control"):
        continue
    arm, b = cell.split("|")
    k = f"{arm}|{b}|L28"
    if k in norms:
        _, ns, _ = split(v)
        X.append(norms[k] * a); Y.append(ns); A.append(arm)
if len(X) > 5:
    from scipy.stats import pearsonr
    r, p = pearsonr(X, Y)
    print(f"  corr(displacement, nonsense) = {r:+.3f} (p={p:.3g}, n={len(X)})")
    for arm in ("piecewise", "spline", "linear"):
        d = [x for x, aa in zip(X, A) if aa == arm]
        y = [yy for yy, aa in zip(Y, A) if aa == arm]
        print(f"    {arm:<11} mean displacement {np.mean(d):6.1f}  mean nonsense {np.mean(y):.2f}")
