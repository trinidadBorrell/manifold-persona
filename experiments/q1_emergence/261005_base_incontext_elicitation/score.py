"""Score E0 judge answers against the pass rule in NOTE.md. Writes results.json and prints a table.

Run from the repo root: .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/score.py
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from examples import SETS

key = json.load(open(A / "KEY.json"))
ans = {}
for f in sorted((A / "answers").glob("batch_*.json")):
    ans.update(json.load(open(f)))
replies = {}
for st in ("base", "final"):
    for l in open(A / f"replies_{st}.jsonl"):
        r = json.loads(l)
        replies[(r["stage"], r["cond"], r["role"], r["question_idx"], r["sample"])] = r
missing = [i for i in key if i not in ans]
print(f"{len(ans)} judged of {len(key)}; missing {len(missing)}")

words = lambda s: re.findall(r"[a-z']+", (s or "").lower())


def longest_run(a, b):
    """Longest run of consecutive words shared by word lists a and b."""
    best, prev = 0, {}
    for i in range(len(a)):
        cur = {}
        for j in range(len(b)):
            if a[i] == b[j]:
                cur[j] = prev.get(j - 1, 0) + 1
                best = max(best, cur[j])
        prev = cur
    return best


def looped(ws, n=4, k=3):
    grams = Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))
    return bool(grams) and max(grams.values()) >= k


ex_answers = {s: [words(e[4]) for e in v] for s, v in SETS.items()}
rows = []
for i, k in key.items():
    r = replies[(k["stage"], k["cond"], k["role"], k["question_idx"], k["sample"])]
    w = words(r["reply"])
    top = ans.get(i, [])
    ex = ex_answers.get(int(k["cond"][3]), []) if k["cond"].startswith("set") else []
    name_words = set(words(k["label"])) | set(words(k["role"].replace("_", " ")))
    rows.append(dict(id=i, stage=k["stage"], cond=k["cond"], role=k["role"], top=top,
                     hit1=bool(top) and top[0] == k["role"], hit5=k["role"] in top,
                     default1=bool(top) and top[0] == "default",
                     copy_desc=longest_run(w, words(k["guest"])) if k.get("guest") else 0,
                     copy_ex=max([longest_run(w, e) for e in ex], default=0),
                     self_name=bool(name_words & set(w)) if not k["role"].startswith("default") else None,
                     empty=not w, limit=r["ended_by"] == "limit", loop=looped(w), n_words=len(w)))
for r in rows:
    r["copied"] = r["copy_desc"] >= 5 or r["copy_ex"] >= 5

rng = np.random.default_rng(0)


def summarize(sub, n_boot=2000):
    by_role = defaultdict(list)
    for r in sub:
        by_role[r["role"]].append(r["hit5"])
    roles = sorted(by_role)
    hits = np.array([np.mean(by_role[x]) for x in roles])
    sizes = np.array([len(by_role[x]) for x in roles])
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(roles), len(roles))
        boot.append(np.sum(hits[idx] * sizes[idx]) / np.sum(sizes[idx]))
    # Empirical chance: the same answers scored against a shuffled wrong character.
    perm_roles = [r["role"] for r in sub]
    chance = []
    for _ in range(200):
        p = rng.permutation(perm_roles)
        # A pair that drew its own role is not a wrong-character test: leave it
        # out rather than count it as a miss.
        chance.append(np.mean([pr in r["top"] for pr, r in zip(p, sub) if pr != r["role"]]))
    return dict(n=len(sub), top5=float(np.mean([r["hit5"] for r in sub])), top1=float(np.mean([r["hit1"] for r in sub])),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                chance=float(np.mean(chance)), coverage=int(sum(any(by_role[x]) for x in roles)), n_roles=len(roles),
                default1=float(np.mean([r["default1"] for r in sub])),
                self_name=float(np.mean([r["self_name"] for r in sub])),
                copied=float(np.mean([r["copied"] for r in sub])),
                empty=float(np.mean([r["empty"] for r in sub])), limit=float(np.mean([r["limit"] for r in sub])),
                loop=float(np.mean([r["loop"] for r in sub])), words=float(np.median([r["n_words"] for r in sub])))


res = {}
chars = [r for r in rows if not r["role"].startswith("default") and r["top"]]
for st in ("base", "final"):
    for c in sorted({r["cond"] for r in chars if r["stage"] == st}):
        sub = [r for r in chars if r["stage"] == st and r["cond"] == c]
        res[f"{st}/{c}"] = summarize(sub)
        res[f"{st}/{c}/nocopy"] = summarize([r for r in sub if not r["copied"]])
defaults = {}
for r in rows:
    if r["role"].startswith("default") and r["top"]:
        defaults.setdefault(f"{r['stage']}/{r['cond']}/{r['role']}", []).append(r["default1"])
res["defaults_default_first"] = {k: float(np.mean(v)) for k, v in sorted(defaults.items())}

print(f"\n{'model/cond':24s} {'top5':>6s} {'95% CI':>13s} {'chance':>6s} {'top1':>5s} {'cover':>6s} {'self':>5s} {'copy':>5s} {'limit':>5s} {'loop':>5s} {'words':>5s}")
for k, v in res.items():
    if k.endswith("nocopy") or k.startswith("defaults"):
        continue
    print(f"{k:24s} {v['top5']:6.1%} [{v['lo']:5.1%},{v['hi']:5.1%}] {v['chance']:6.1%} {v['top1']:5.1%} {v['coverage']:3d}/{v['n_roles']:<2d} "
          f"{v['self_name']:5.1%} {v['copied']:5.1%} {v['limit']:5.1%} {v['loop']:5.1%} {v['words']:5.0f}")

# Pass rule (NOTE.md).
ctrl = res["final/native"]["top5"]
verdict = {"positive_control_native_top5": ctrl, "control_ok": ctrl >= 0.30, "sets": {}}
for s in ("set1", "set2", "set3"):
    b, f, bn, fn = (res[f"{m}/{s}_neutral{x}"] for m, x in (("base", ""), ("final", ""), ("base", "/nocopy"), ("final", "/nocopy")))
    rule = lambda b, f: b["top5"] >= 0.10 and b["lo"] > b["chance"] and b["top5"] >= 0.5 * f["top5"]
    verdict["sets"][s] = dict(rule1_2=rule(b, f), rule3_coverage=b["coverage"] >= 20, rule4_nocopy=rule(bn, fn),
                              base_top5=b["top5"], final_top5=f["top5"])
t5 = [res[f"base/{s}_neutral"]["top5"] for s in ("set1", "set2", "set3")]
verdict["sets_agree"] = max(t5) <= 2 * max(min(t5), 1e-9)
verdict["note"] = "Rule 5 (blind raw read, >= 6 of 10 base replies play the character) is checked by hand."
res["verdict"] = verdict
print("\n" + json.dumps(verdict, indent=1))
json.dump(res, open(A / "results.json", "w"), indent=1)
json.dump(rows, open(A / "scored_rows.json", "w"))
