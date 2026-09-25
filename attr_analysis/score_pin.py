"""Additive vs pinned steering, judged blind.

fluent in-role = in a character's voice (not assistant) and not nonsense.
on-target      = the judge names the intended persona.
Each method is scored at its own best strength per persona.
"""
import glob
import json
from pathlib import Path

import numpy as np

lab = {}
for D in (Path("judge_pin"), Path("judge_pinhi")):
    if not (D / "index.json").exists():
        continue
    idx = json.load(open(D / "index.json"))
    for f in sorted(glob.glob(str(D / "verdict_*.json"))):
        n = int(Path(f).stem.split("_")[1])
        for k, v in json.load(open(f)).items():
            m = idx.get(f"{n}:{k}")
            if m:
                lab.setdefault((m["cell"], m["alpha"]), []).append(v)

IN = {"human_role", "nonhuman", "weird"}
T = ["ghost", "pirate", "vampire", "musician", "spy"]


def score(v, t):
    n = len(v)
    return {"in": sum(x["role"] in IN for x in v) / n,
            "ns": sum(x["role"] == "nonsense" for x in v) / n,
            "on": sum(x["char"] == t and x["role"] != "nonsense" for x in v) / n,
            "n": n}


print(f"{'persona':<9}{'method':<6}{'strength':>9}{'in-role':>9}{'on-target':>11}{'nonsense':>10}")
best = {m: [] for m in ("add", "pin", "ref")}
for t in T:
    for m in ("add", "pin", "ref"):
        rows = sorted((a, score(v, t)) for (c, a), v in lab.items() if c == f"{m}|{t}")
        for a, s in rows:
            print(f"{t:<9}{m:<6}{a:>9}{s['in']:>9.2f}{s['on']:>11.2f}{s['ns']:>10.2f}")
        if rows:
            a, s = max(rows, key=lambda r: (r[1]["on"], r[1]["in"] - r[1]["ns"]))
            best[m].append((t, a, s))
    print()

print("=== each method at its best strength per persona (ranked by on-target) ===")
print(f"{'persona':<9}" + "".join(f"{m:>22}" for m in ("add", "pin", "ref")))
for i, t in enumerate(T):
    line = f"{t:<9}"
    for m in ("add", "pin", "ref"):
        tt, a, s = best[m][i]
        line += f"   s={a:<4} on{s['on']:>5.2f} ns{s['ns']:>5.2f}"
    print(line)
print()
for m in ("add", "pin", "ref"):
    on = np.mean([s["on"] for _, _, s in best[m]])
    ns = np.mean([s["ns"] for _, _, s in best[m]])
    ir = np.mean([s["in"] for _, _, s in best[m]])
    print(f"  {m:<4} mean on-target {on:.2f}   in-role {ir:.2f}   nonsense {ns:.2f}")
