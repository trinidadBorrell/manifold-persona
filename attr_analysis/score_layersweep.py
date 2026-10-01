"""Does each persona have a best layer, and is it the same one for all of them?

One layer is pinned at a time. If the best layer is a property of the model, every
persona peaks at the same depth and a per-persona layer assignment buys nothing.
"""
import glob
import json
from pathlib import Path

import numpy as np

D = Path("judge_ls")
IN = {"human_role", "nonhuman", "weird"}
TARGETS = ["ghost", "pirate", "vampire", "musician", "spy"]
LAYERS = json.load(open("attr_analysis/layersweep_layers.json"))


def load():
    lab = {}
    idx = json.load(open(D / "index.json"))
    for f in sorted(glob.glob(str(D / "verdict_*.json"))):
        n = int(Path(f).stem.split("_")[1])
        for k, v in json.load(open(f)).items():
            m = idx.get(f"{n}:{k}")
            if m:
                tag, t = m["cell"].split("|")[1].split("_", 1)
                lab.setdefault((int(tag[1:]), t, float(m["alpha"])), []).append(v)
    return lab


def sc(v, t):
    n = len(v)
    return {"in": sum(x["role"] in IN for x in v) / n,
            "ns": sum(x["role"] == "nonsense" for x in v) / n,
            "on": sum(x["char"] == t and x["role"] != "nonsense" for x in v) / n,
            "n": n}


def main():
    lab = load()
    strengths = sorted({s for _, _, s in lab})

    for s in strengths:
        print(f"\n=== pin strength {s} — on-target by layer ===")
        print(f"{'persona':<10}" + "".join(f"{f'L{L}':>7}" for L in LAYERS))
        for t in TARGETS:
            row = []
            for L in LAYERS:
                v = lab.get((L, t, s))
                row.append(sc(v, t)["on"] if v else float("nan"))
            print(f"{t:<10}" + "".join(f"{x:>7.2f}" for x in row))
        print(f"{'mean':<10}" + "".join(
            f"{np.nanmean([sc(lab[(L, t, s)], t)['on'] for t in TARGETS if (L, t, s) in lab]):>7.2f}"
            for L in LAYERS))
        print(f"{'nonsense':<10}" + "".join(
            f"{np.nanmean([sc(lab[(L, t, s)], t)['ns'] for t in TARGETS if (L, t, s) in lab]):>7.2f}"
            for L in LAYERS))

    # the best layer per persona, pooling both strengths
    print("\n=== best layer per persona (both strengths pooled) ===")
    print(f"{'persona':<10}{'best layer':>12}{'on-target':>11}{'runner-up':>11}{'gap':>7}")
    best = {}
    for t in TARGETS:
        per = {}
        for L in LAYERS:
            vs = [x for s in strengths for x in lab.get((L, t, s), [])]
            if vs:
                per[L] = sc(vs, t)["on"]
        if not per:
            continue
        order = sorted(per, key=per.get, reverse=True)
        best[t] = order[0]
        gap = per[order[0]] - per[order[1]] if len(order) > 1 else float("nan")
        print(f"{t:<10}{order[0]:>12}{per[order[0]]:>11.2f}{per[order[1]]:>11.2f}{gap:>7.2f}")

    if best:
        vals = list(best.values())
        print(f"\nbest layers: {best}")
        print(f"all the same layer: {len(set(vals)) == 1}  (spread {min(vals)} to {max(vals)})")

    # a single layer against the five-layer band
    print("\n=== best single layer vs all five layers 26-30 ===")
    print("five-layer pinned steering scored 0.64 on-target pooled (see the k sweep)")
    pooled = {}
    for L in LAYERS:
        hit = tot = 0
        for t in TARGETS:
            for s in strengths:
                for x in lab.get((L, t, s), []):
                    tot += 1
                    hit += x["char"] == t and x["role"] != "nonsense"
        if tot:
            pooled[L] = hit / tot
    for L in LAYERS:
        if L in pooled:
            print(f"  L{L:<3} {pooled[L]:.2f}")


if __name__ == "__main__":
    main()
