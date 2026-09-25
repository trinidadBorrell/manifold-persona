"""How many other personas does a working reconstruction need?

Four arms pinned to the same displacement at each layer, judged blind together:
the persona's own vector, and the shared all-layer reconstruction built from 2, 8
and 32 other personas.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_ks")
IN = {"human_role", "nonhuman", "weird"}
ARMS = ("true", "k2", "k8", "k32")
TARGETS = ["ghost", "pirate", "vampire", "musician", "spy"]


def load():
    lab = {}
    idx = json.load(open(D / "index.json"))
    for f in sorted(glob.glob(str(D / "verdict_*.json"))):
        n = int(Path(f).stem.split("_")[1])
        for k, v in json.load(open(f)).items():
            m = idx.get(f"{n}:{k}")
            if m:
                arm, t = m["cell"].split("|")[1].split("_", 1)
                lab.setdefault((arm, t, float(m["alpha"])), []).append(v)
    return lab


def score(v, t):
    n = len(v)
    return {"in": sum(x["role"] in IN for x in v) / n,
            "ns": sum(x["role"] == "nonsense" for x in v) / n,
            "on": sum(x["char"] == t and x["role"] != "nonsense" for x in v) / n,
            "n": n}


def main():
    lab = load()
    print(f"{'persona':<10}{'arm':<7}{'s':>6}{'in-role':>9}{'on-target':>11}{'nonsense':>10}{'n':>4}")
    best = {a: {} for a in ARMS}
    for t in TARGETS:
        for a in ARMS:
            rows = sorted((s, score(v, t)) for (arm, tt, s), v in lab.items()
                          if arm == a and tt == t)
            for s, r in rows:
                print(f"{t:<10}{a:<7}{s:>6.2f}{r['in']:>9.2f}{r['on']:>11.2f}"
                      f"{r['ns']:>10.2f}{r['n']:>4}")
            if rows:
                best[a][t] = max(rows, key=lambda r: (r[1]["on"], r[1]["in"] - r[1]["ns"]))
        print()

    print("=== each arm at its best strength ===")
    print(f"{'persona':<10}" + "".join(f"{a:>22}" for a in ARMS))
    for t in TARGETS:
        line = f"{t:<10}"
        for a in ARMS:
            if t in best[a]:
                s, r = best[a][t]
                line += f"  s={s:<5.2f} on{r['on']:>5.2f} ns{r['ns']:>5.2f}"
        print(line)

    print(f"\n{'arm':<8}{'in-role':>9}{'on-target':>11}{'nonsense':>10}")
    for a in ARMS:
        g = [r for _, r in best[a].values()]
        if g:
            print(f"{a:<8}{np.mean([r['in'] for r in g]):>9.2f}"
                  f"{np.mean([r['on'] for r in g]):>11.2f}"
                  f"{np.mean([r['ns'] for r in g]):>10.2f}")

    # the random floor was measured at 0.00 on-target in the paired run
    tr = np.mean([r["on"] for _, r in best["true"].values()])
    print()
    for a in ARMS:
        if a == "true":
            continue
        v = np.mean([r["on"] for _, r in best[a].values()])
        print(f"{a:<5} keeps {v / tr:.0%} of the real vector's on-target score")


if __name__ == "__main__":
    main()
