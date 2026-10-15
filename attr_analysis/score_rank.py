"""Recall@k from a ranking judge's verdicts.

A verdict ranks ten personas per answer; recall@k is the share of answers whose
intended persona appears in the first k names. The intended persona is the part of the
cell name after the arm (pin|L23_ghost -> ghost).
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KS = (1, 5, 10)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


def target_of(cell):
    arm_t = cell.split("|")[1]
    return arm_t.split("_", 1)[1] if "_" in arm_t else arm_t


def arm_of(cell):
    return cell.split("|")[1].split("_", 1)[0]


def load(d):
    d = Path(d)
    idx = json.load(open(d / "index.json"))
    roles = set(json.load(open(d / "roles.json")))
    rows, bad = [], 0
    for f in sorted(glob.glob(str(d / "verdict_*.json"))):
        n = int(Path(f).stem.split("_")[1])
        for k, v in json.load(open(f)).items():
            m = idx.get(f"{n}:{k}")
            if not m:
                continue
            rank = [str(x).strip().lower().replace(" ", "_") for x in v.get("rank", [])]
            bad += sum(r not in roles for r in rank)
            rows.append({**m, "arm": arm_of(m["cell"]), "target": target_of(m["cell"]),
                         "rank": rank, "nonsense": bool(v.get("nonsense", False))})
    return rows, bad


def summarise(rows):
    out = {}
    for k in KS:
        hit = sum(r["target"] in r["rank"][:k] for r in rows)
        lo, hi = wilson(hit, len(rows))
        out[f"r@{k}"] = hit / len(rows)
        out[f"ci@{k}"] = (lo, hi)
    out["asst@1"] = sum(r["rank"][:1] == ["assistant"] for r in rows) / len(rows)
    out["nonsense"] = sum(r["nonsense"] for r in rows) / len(rows)
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows, bad = load(args.dir)
    print(f"{len(rows)} judged answers, {bad} names outside the list")

    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["target"], float(r["alpha"]))].append(r)
    arms = sorted({a for a, _, _ in cells})
    strengths = sorted({s for _, _, s in cells})
    targets = sorted({t for _, t, _ in cells})

    report = {"cells": {}, "by_arm_strength": {}, "by_arm_best": {}}
    for arm in arms:
        print(f"\n=== arm {arm}: recall@1 / @5 / @10 per persona and strength ===")
        print(f"{'persona':<13}" + "".join(f"{'s=' + format(s, 'g'):>20}" for s in strengths))
        for t in targets:
            line = f"{t:<13}"
            for s in strengths:
                v = cells.get((arm, t, s))
                if not v:
                    line += f"{'-':>20}"
                    continue
                sm = summarise(v)
                report["cells"][f"{arm}|{t}|{s:g}"] = sm
                line += f"{sm['r@1']:.1f} / {sm['r@5']:.1f} / {sm['r@10']:.1f}".rjust(20)
            print(line)

        print(f"\n{'strength':<10}{'r@1':>7}{'r@5':>7}{'r@10':>7}{'95% CI r@5':>16}"
              f"{'asst@1':>8}{'nonsense':>10}{'n':>5}")
        for s in strengths:
            v = [r for (a, _, ss), rs in cells.items() if a == arm and ss == s for r in rs]
            if not v:
                continue
            sm = summarise(v)
            report["by_arm_strength"][f"{arm}|{s:g}"] = sm
            lo, hi = sm["ci@5"]
            print(f"{s:<10g}{sm['r@1']:>7.2f}{sm['r@5']:>7.2f}{sm['r@10']:>7.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>16}{sm['asst@1']:>8.2f}{sm['nonsense']:>10.2f}"
                  f"{sm['n']:>5}")

        # each persona at its own best strength, chosen on recall@5
        best = []
        for t in targets:
            opts = [(s, cells[(arm, t, s)]) for s in strengths if (arm, t, s) in cells]
            if opts:
                best.extend(max(opts, key=lambda o: (summarise(o[1])["r@5"],
                                                      summarise(o[1])["r@1"]))[1])
        if best:
            sm = summarise(best)
            report["by_arm_best"][arm] = sm
            lo, hi = sm["ci@5"]
            print(f"{'best':<10}{sm['r@1']:>7.2f}{sm['r@5']:>7.2f}{sm['r@10']:>7.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>16}{sm['asst@1']:>8.2f}{sm['nonsense']:>10.2f}"
                  f"{sm['n']:>5}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
