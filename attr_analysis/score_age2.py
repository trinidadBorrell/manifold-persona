"""Age steering, matched displacement (22 units for every target).

Reports the prompt ceiling first: no steering arm can beat what the persona's own
prompt achieves, so that is the yardstick, not perfection.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

D = Path("judge_age2")
idx = json.load(open(D / "index.json"))
meta = json.load(open("attr_analysis/age_meta.json"))
LADDER, AGES = meta["ladder"], meta["ages"]
NOM = dict(zip(LADDER, AGES))
QS = json.load(open("attr_analysis/age_questions.json"))

lab = {}
for f in sorted(glob.glob(str(D / "verdict_*.json"))):
    n = int(Path(f).stem.split("_")[1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"{n}:{k}")
        if m:
            lab.setdefault((m["cell"], m["alpha"]), []).append((v, m["qi"]))

def stats(vals, drop_qi=()):
    v = [x for x, q in vals if q not in drop_qi]
    if not v:
        return None
    nums = [float(x) for x in v if not isinstance(x, str)]
    return {"n": len(v), "med": np.median(nums) if nums else np.nan,
            "nonsense": sum(1 for x in v if x == "nonsense") / len(v),
            "assistant": sum(1 for x in v if x == "assistant") / len(v),
            "aged": len(nums) / len(v)}

# which questions actually let a persona speak?
print("=== per-question: does this probe elicit an age at all? (prompt arm) ===")
byq = {}
for p in LADDER:
    for a in (0.8, 1.0, 1.25):
        for x, q in lab.get((f"{p}|prompt", a), []):
            byq.setdefault(q, []).append(x)
BAD = []
for q in sorted(byq):
    v = byq[q]
    aged = sum(1 for x in v if not isinstance(x, str)) / len(v)
    if aged < 0.5:
        BAD.append(q)
    print(f"  q{q} aged={aged:.2f}  {'<-- DROP' if aged < 0.5 else ''}  {QS[q][:52]}")
print(f"\ndropping questions {BAD} from the age-accuracy numbers below\n")

print("=== PROMPT CEILING vs CONTROL (steer at the rung's own centroid) ===")
print(f"{'rung':<13}{'nom':>5} | {'prompt':>17} | {'control a=0.8':>17}{'a=1.0':>17}{'a=1.25':>17}")
rows = {}
for p in LADDER:
    line = f"{p:<13}{NOM[p]:>5.0f} |"
    for key, a in [("prompt", 1.0), ("control", 0.8), ("control", 1.0), ("control", 1.25)]:
        cell = f"{p}|prompt" if key == "prompt" else f"control|{p}"
        s = stats(lab.get((cell, a), []), BAD)
        if not s:
            line += f"{'-':>17}"; continue
        e = abs(s["med"] - NOM[p]) if not np.isnan(s["med"]) else np.nan
        rows.setdefault((key, a), []).append((e, s["nonsense"], s["aged"]))
        line += f"{s['med']:>7.0f}{'e' + f'{e:.0f}' if not np.isnan(e) else ' -':>5}{s['nonsense']:>5.2f}"
    print(line)
print(f"\n{'arm':<16}{'median |err|':>13}{'nonsense':>10}{'aged frac':>11}")
for (key, a), v in rows.items():
    e = [x[0] for x in v if not np.isnan(x[0])]
    print(f"{key + ' a=' + str(a):<16}{np.median(e) if e else np.nan:>13.1f}"
          f"{np.mean([x[1] for x in v]):>10.2f}{np.mean([x[2] for x in v]):>11.2f}")

for key, a in [("prompt", 1.0), ("control", 1.0), ("control", 1.25)]:
    xs, ys = [], []
    for p in LADDER:
        cell = f"{p}|prompt" if key == "prompt" else f"control|{p}"
        s = stats(lab.get((cell, a), []), BAD)
        if s and not np.isnan(s["med"]):
            xs.append(NOM[p]); ys.append(s["med"])
    if len(xs) > 3:
        print(f"  {key} a={a}: pearson {pearsonr(xs, ys)[0]:+.3f}  "
              f"spearman {spearmanr(xs, ys)[0]:+.3f}  (n={len(xs)})")

print("\n=== THREE PATHS, identical displacement, same target age ===")
for a in (0.8, 1.0, 1.25):
    agg = {arm: {"e": [], "n": [], "aged": []} for arm in ("piecewise", "spline", "linear")}
    print(f"\n--- alpha {a} ---")
    print(f"{'beta':>5}{'target':>7}" + "".join(f"{x:>20}" for x in ("piecewise", "spline", "linear")))
    for b in meta["betas"]:
        tgt = meta[str(b)]["target_age"]
        line = f"{b:>5}{tgt:>7.0f}"
        for arm in ("piecewise", "spline", "linear"):
            s = stats(lab.get((f"{arm}|{b}", a), []), BAD)
            if not s:
                line += f"{'-':>20}"; continue
            e = abs(s["med"] - tgt) if not np.isnan(s["med"]) else np.nan
            if not np.isnan(e):
                agg[arm]["e"].append(e)
            agg[arm]["n"].append(s["nonsense"]); agg[arm]["aged"].append(s["aged"])
            line += f"{s['med']:>9.0f}{'e' + f'{e:.0f}' if not np.isnan(e) else ' -':>6}{s['nonsense']:>5.2f}"
        print(line)
    print(f"\n{'arm':<12}{'median |err|':>13}{'mean |err|':>12}{'nonsense':>10}{'aged':>8}{'n':>4}")
    for arm in ("piecewise", "spline", "linear"):
        g = agg[arm]
        if g["e"]:
            print(f"{arm:<12}{np.median(g['e']):>13.1f}{np.mean(g['e']):>12.1f}"
                  f"{np.mean(g['n']):>10.2f}{np.mean(g['aged']):>8.2f}{len(g['e']):>4}")
    # monotonicity: does judged age rise with beta?
    for arm in ("piecewise", "spline", "linear"):
        xs, ys = [], []
        for b in meta["betas"]:
            s = stats(lab.get((f"{arm}|{b}", a), []), BAD)
            if s and not np.isnan(s["med"]):
                xs.append(meta[str(b)]["target_age"]); ys.append(s["med"])
        if len(xs) > 3:
            print(f"    {arm:<11} spearman(target, judged) = {spearmanr(xs, ys)[0]:+.3f} (n={len(xs)})")
