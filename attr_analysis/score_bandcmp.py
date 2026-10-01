"""Which layers should we steer, and how many?

Five arms over the same strength grid: three single layers and two five-layer
bands. Each arm is read at its own best strength, so no arm is penalised for
needing a different push.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_bc")
IN = {"human_role", "nonhuman", "weird"}
TARGETS = ["ghost", "pirate", "vampire", "musician", "spy"]
ARMS = json.load(open("attr_analysis/bandcmp_arms.json"))
LABEL = {"one23": "layer 23 alone", "one20": "layer 20 alone", "one26": "layer 26 alone",
         "band19": "layers 19-23", "band26": "layers 26-30"}


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


def sc(v, t):
    n = len(v)
    return {"in": sum(x["role"] in IN for x in v) / n,
            "ns": sum(x["role"] == "nonsense" for x in v) / n,
            "on": sum(x["char"] == t and x["role"] != "nonsense" for x in v) / n,
            "n": n}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


def main():
    lab = load()
    strengths = sorted({s for _, _, s in lab})

    for arm in ARMS:
        print(f"\n=== {LABEL[arm]} — on-target by strength ===")
        print(f"{'persona':<10}" + "".join(f"{f's={s:g}':>8}" for s in strengths))
        for t in TARGETS:
            print(f"{t:<10}" + "".join(
                f"{sc(lab[(arm, t, s)], t)['on']:>8.2f}" if (arm, t, s) in lab else f"{'-':>8}"
                for s in strengths))
        print(f"{'nonsense':<10}" + "".join(
            f"{np.mean([sc(lab[(arm, t, s)], t)['ns'] for t in TARGETS if (arm, t, s) in lab]):>8.2f}"
            if any((arm, t, s) in lab for t in TARGETS) else f"{'-':>8}"
            for s in strengths))

    # each arm at its own best strength, chosen per persona
    print("\n=== each arm at its best strength per persona ===")
    print(f"{'arm':<16}" + "".join(f"{t:>11}" for t in TARGETS) + f"{'mean':>8}")
    summary = {}
    for arm in ARMS:
        best, hit, tot = [], 0, 0
        for t in TARGETS:
            rows = [(s, sc(lab[(arm, t, s)], t)) for s in strengths if (arm, t, s) in lab]
            if not rows:
                best.append(float("nan"))
                continue
            s, r = max(rows, key=lambda x: (x[1]["on"], x[1]["in"] - x[1]["ns"]))
            best.append(r["on"])
            hit += round(r["on"] * r["n"])
            tot += r["n"]
        summary[arm] = (hit, tot)
        print(f"{LABEL[arm]:<16}" + "".join(f"{x:>11.2f}" for x in best)
              + f"{np.nanmean(best):>8.2f}")

    print(f"\n{'arm':<16}{'on-target':>11}{'95% CI':>16}{'layers':>8}")
    for arm in ARMS:
        hit, tot = summary[arm]
        lo, hi = wilson(hit, tot)
        print(f"{LABEL[arm]:<16}{hit / tot:>11.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
              f"{len(ARMS[arm]):>8}")


if __name__ == "__main__":
    main()
