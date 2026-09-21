"""Did the predicted attribute vector move age without moving occupation?

Two things must both hold for the analogy to have worked behaviourally:
  transfer    -- pred shifts age away from base, toward the real target
  selectivity -- pred keeps the occupation base already had
"""
import glob
import json
from collections import Counter
from pathlib import Path

D = Path("judge_analogy")
idx = json.load(open(D / "index.json"))
lab = {}
for f in sorted(glob.glob(str(D / "verdict_*.json"))):
    n = int(Path(f).stem.split("_")[1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"{n}:{k}")
        if m:
            lab.setdefault((m["cell"], m["alpha"]), []).append(v)

OCC = ["musician", "spy", "scholar", "soldier", "merchant"]
ATTR = {"age": ("young", "old"), "era": ("medieval", "futuristic")}
missing = sum(1 for c in lab.values() for _ in c)
print(f"{len(lab)} cells labelled, {missing} judgements\n")

for attr, (lo, hi) in ATTR.items():
    if attr == "era":
        continue  # era has no age/occ ground truth on this axis
    print(f"=== {attr}: does pred move {lo} -> {hi}? ===")
    print(f"{'occupation':<11} {'arm':<6} {'a':<5} {'age=old':>8} {'age=young':>10} {'occ kept':>9}")
    agg = {}
    for o in OCC:
        for arm in ["base", "real", "pred"]:
            for a in [1.0, 0.6]:
                key = (f"{attr}|{arm}|{o}", a)
                if key not in lab:
                    continue
                v = lab[key]
                old = sum(x["age"] == hi[:3] or x["age"] == "old" for x in v) / len(v)
                yng = sum(x["age"] == "young" for x in v) / len(v)
                kept = sum(x["occ"] == o for x in v) / len(v)
                print(f"{o:<11} {arm:<6} {a:<5} {old:>8.2f} {yng:>10.2f} {kept:>9.2f}")
                agg.setdefault((arm, a), []).append((old, yng, kept))
    print(f"\n{'ARM':<12} {'p(old)':>8} {'p(young)':>10} {'occ kept':>9}")
    for (arm, a), rows in sorted(agg.items()):
        n = len(rows)
        print(f"{arm+' a='+str(a):<12} {sum(r[0] for r in rows)/n:>8.2f} "
              f"{sum(r[1] for r in rows)/n:>10.2f} {sum(r[2] for r in rows)/n:>9.2f}")

print("\n=== occupation retention, all arms (selectivity) ===")
for attr in ATTR:
    for arm in ["base", "real", "pred"]:
        rows = [(k, v) for k, v in lab.items() if k[0].startswith(f"{attr}|{arm}|")]
        if not rows:
            continue
        kept = []
        for (cell, a), v in rows:
            o = cell.split("|")[2]
            kept.append(sum(x["occ"] == o for x in v) / len(v))
        print(f"  {attr}|{arm}: occupation visible in {sum(kept)/len(kept):.2f} of responses")
