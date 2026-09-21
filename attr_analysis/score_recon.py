"""Does a persona rebuilt from other personas behave like the persona?

true  = steer with the real persona vector       (positive control)
recon = steer with its 6-component reconstruction
rand  = steer with a random vector of equal norm (negative control)
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_recon")
idx = json.load(open(D / "index.json"))
lab = {}
for f in sorted(glob.glob(str(D / "verdict_*.json"))):
    n = int(Path(f).stem.split("_")[1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"{n}:{k}")
        if m:
            lab.setdefault((m["cell"], m["alpha"]), []).append(v)

# what each target should look like
WANT = {
    "old_musician":      {"age": "old",   "occ": "musician"},
    "medieval_spy":      {"occ": "spy",   "era": "medieval"},
    "old_spy":           {"age": "old",   "occ": "spy"},
    "futuristic_scholar":{"occ": "scholar", "era": "futuristic"},
    "toddler":           {"age": "young"},
    "elder":             {"age": "old"},
}
print(f"{len(lab)} cells, {sum(len(v) for v in lab.values())} judgements\n")
print(f"{'target':<20}{'a':<6}{'true':>8}{'recon':>8}{'rand':>8}   (fraction of target"
      " attributes hit)")
agg = {}
for t, want in WANT.items():
    for a in (1.25, 1.5):
        row = {}
        for arm in ("true", "recon", "rand"):
            v = lab.get((f"{t}|{arm}", a), [])
            if not v:
                continue
            hits = [np.mean([x.get(k) == val for k, val in want.items()]) for x in v]
            row[arm] = float(np.mean(hits))
            agg.setdefault((arm, a), []).append(row[arm])
        if row:
            print(f"{t:<20}{a:<6}" + "".join(f"{row.get(k, float('nan')):>8.2f}"
                                             for k in ("true", "recon", "rand")))
print(f"\n{'ARM':<12}{'mean hit rate':>14}")
for (arm, a), v in sorted(agg.items()):
    print(f"{arm+' a='+str(a):<12}{np.mean(v):>14.3f}")

# how closely does recon track true, target by target?
print("\ncorrelation of recon with true across targets:")
for a in (1.25, 1.5):
    ts = [t for t in WANT if (f"{t}|true", a) in lab and (f"{t}|recon", a) in lab]
    tv = [np.mean([np.mean([x.get(k) == val for k, val in WANT[t].items()])
                   for x in lab[(f"{t}|true", a)]]) for t in ts]
    rv = [np.mean([np.mean([x.get(k) == val for k, val in WANT[t].items()])
                   for x in lab[(f"{t}|recon", a)]]) for t in ts]
    if len(ts) > 2:
        print(f"  a={a}: r = {np.corrcoef(tv, rv)[0,1]:+.3f} over {len(ts)} targets")
