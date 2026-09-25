"""Does one absolute displacement work for every persona?

Sixteen personas sit between 6.9 and 61.6 units from the assistant at layer 30 and are
pinned to the same three distances. If the working range is a property of the model, the
three distances score alike for near and far personas.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_fd")
META = json.load(open("attr_analysis/fixdist_meta.json"))
IN = {"human_role", "nonhuman", "weird"}

KEYS = {
    "young_scholar": ["scholar", "student", "academ", "research", "phd", "univers"],
    "graduate": ["graduate", "student", "scholar", "academ", "phd", "univers"],
    "scholar": ["scholar", "academ", "research", "professor", "学"],
    "therapist": ["therap", "counsel", "psych"],
    "old_soldier": ["soldier", "veteran", "war", "milit", "old man"],
    "retiree": ["retire", "old", "elder", "grandfather", "grandmother"],
    "young_musician": ["music", "singer", "band", "song", "young"],
    "old_musician": ["music", "singer", "band", "song", "old"],
    "ancient": ["ancient", "old", "elder", "immortal", "god", "sage"],
    "musician": ["music", "singer", "band", "song", "composer"],
    "teenager": ["teen", "young", "adolesc", "child", "kid"],
    "adolescent": ["teen", "young", "adolesc", "child", "kid"],
    "vampire": ["vampire", "blood", "undead", "immortal", "night"],
    "era_victorian": ["victorian", "19th", "gentleman", "lady", "aristocra"],
    "medieval_musician": ["medieval", "bard", "minstrel", "music", "lute"],
    "toddler": ["toddler", "child", "kid", "baby", "young"],
}


def load():
    lab = {}
    idx = json.load(open(D / "index.json"))
    for f in sorted(glob.glob(str(D / "verdict_*.json"))):
        n = int(Path(f).stem.split("_")[1])
        for k, v in json.load(open(f)).items():
            m = idx.get(f"{n}:{k}")
            if m:
                lab.setdefault((m["cell"].split("|")[1], float(m["alpha"])), []).append(v)
    return lab


def main():
    lab = load()
    d30, dists = META["d30"], META["dists"]
    # the strengths the run actually used, in the same order as the target distances
    cells = json.load(open("attr_analysis/fixdist_cells.json"))
    want = {(name.split("|")[1], float(s)): dist
            for name, ss in cells for s, dist in zip(ss, dists)}

    rows = []
    for (p, s), v in lab.items():
        dist = want.get((p, s))
        if dist is None:
            continue
        n = len(v)
        rows.append(dict(
            p=p, dist=dist, s=s, n=n, d30=d30[p],
            inrole=sum(x["role"] in IN for x in v) / n,
            ns=sum(x["role"] == "nonsense" for x in v) / n,
            on=sum(x["role"] != "nonsense"
                   and any(k in str(x.get("char", "")).lower() for k in KEYS[p])
                   for x in v) / n))

    order = sorted(d30, key=d30.get)
    print(f"{'persona':<19}{'|d30|':>7}" + "".join(f"{'D=' + str(int(x)):>22}" for x in dists))
    print(f"{'':<19}{'':>7}" + "".join(f"{'s':>6}{'in':>5}{'on':>5}{'ns':>6}" for _ in dists))
    for p in order:
        line = f"{p:<19}{d30[p]:>7.1f}"
        for dist in dists:
            r = next((r for r in rows if r["p"] == p and r["dist"] == dist), None)
            line += (f"{r['s']:>6.1f}{r['inrole']:>5.2f}{r['on']:>5.2f}{r['ns']:>6.2f}"
                     if r else f"{'-':>22}")
        print(line)

    print(f"\n{'distance':<10}{'in-role':>9}{'on-target':>11}{'nonsense':>10}{'cells':>7}")
    for dist in dists:
        g = [r for r in rows if r["dist"] == dist]
        if g:
            print(f"{int(dist):<10}{np.mean([r['inrole'] for r in g]):>9.2f}"
                  f"{np.mean([r['on'] for r in g]):>11.2f}"
                  f"{np.mean([r['ns'] for r in g]):>10.2f}{len(g):>7}")

    # if the band is a model property, the best distance does not track the persona's own
    best = {}
    for p in order:
        g = [r for r in rows if r["p"] == p]
        if g:
            best[p] = max(g, key=lambda r: (r["on"], r["inrole"] - r["ns"]))["dist"]
    if len(best) > 2:
        x = np.array([d30[p] for p in best])
        y = np.array([best[p] for p in best])
        r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else 0.0
        print(f"\nbest distance vs the persona's own |d30|: pearson r = {r:+.3f}"
              f"  (best distances: {sorted(set(y.tolist()))})")


if __name__ == "__main__":
    main()
