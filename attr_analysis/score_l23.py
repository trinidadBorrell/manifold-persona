"""Does layer 23 hold across twelve personas, and does the basis work there?

Arms: the persona's own vector at layers 20, 23 and 26, and the shared all-layer
reconstruction from 8 and 32 other personas injected at layer 23. Every arm sees
the same strengths and is read at its own best.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_l23")
IN = {"human_role", "nonhuman", "weird"}
META = json.load(open("attr_analysis/l23_meta.json"))
TARGETS = META["targets"]
ARMS = list(META["arms"])
LABEL = {"L20": "own, layer 20", "L23": "own, layer 23", "L26": "own, layer 26",
         "k8": "8 personas, L23", "k32": "32 personas, L23"}


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


def best(lab, arm, t, strengths):
    rows = [(s, sc(lab[(arm, t, s)], t)) for s in strengths if (arm, t, s) in lab]
    if not rows:
        return None
    return max(rows, key=lambda x: (x[1]["on"], x[1]["in"] - x[1]["ns"]))


def main():
    lab = load()
    strengths = sorted({s for _, _, s in lab})

    print("=== on-target, each arm at its best strength ===")
    print(f"{'persona':<13}" + "".join(f"{LABEL[a]:>18}" for a in ARMS))
    peak = {}
    for t in TARGETS:
        line = f"{t:<13}"
        for a in ARMS:
            b = best(lab, a, t, strengths)
            cell = f"s={b[0]:g} {b[1]['on']:.2f}" if b else "-"
            line += f"{cell:>18}"
        print(line)
        depths = {a: best(lab, a, t, strengths) for a in ("L20", "L23", "L26")}
        ok = {a: b[1]["on"] for a, b in depths.items() if b}
        if ok:
            peak[t] = max(ok, key=ok.get)

    print(f"\n{'arm':<18}{'on-target':>11}{'95% CI':>16}{'in-role':>9}{'nonsense':>10}")
    for a in ARMS:
        hit = tot = inr = ns = 0
        for t in TARGETS:
            b = best(lab, a, t, strengths)
            if b:
                r = b[1]
                hit += round(r["on"] * r["n"]); tot += r["n"]
                inr += round(r["in"] * r["n"]); ns += round(r["ns"] * r["n"])
        if tot:
            lo, hi = wilson(hit, tot)
            print(f"{LABEL[a]:<18}{hit / tot:>11.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
                  f"{inr / tot:>9.2f}{ns / tot:>10.2f}")

    print("\n=== best depth per persona ===")
    for t in TARGETS:
        print(f"  {t:<13}{peak.get(t, '-')}")
    from collections import Counter
    c = Counter(peak.values())
    print(f"\n{dict(c)}  —  same depth for all: {len(c) == 1}")


if __name__ == "__main__":
    main()
