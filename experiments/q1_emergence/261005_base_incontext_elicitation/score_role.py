"""Score the literature role judge against the pass rule in NOTE.md. Needs score.py's scored_rows.json for
copying and blind naming. Writes results_role.json and prints a table.

Run from the repo root: .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/score_role.py
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
key = json.load(open(A / "RKEY.json"))
ans = {}
for f in sorted((A / "ranswers").glob("batch_*.json")):
    ans.update(json.load(open(f)))
print(f"{len(ans)} judged of {len(key)}; missing {sum(i not in ans for i in key)}")
blind = {(r["stage"], r["cond"], r["role"], r["id"]): r for r in json.load(open(A / "scored_rows.json"))} \
    if (A / "scored_rows.json").exists() else {}
copied = {}
if blind:
    bkey = json.load(open(A / "KEY.json"))
    for r in blind.values():
        k = bkey[r["id"]]
        copied[(k["stage"], k["i"])] = r["copied"]

items = []
for i, k in key.items():
    if i not in ans:
        continue
    role, coh = ans[i]
    items.append(dict(k, role_score=int(role), coh=float(coh), played=int(role) == 3 and float(coh) >= 50,
                      copied=copied.get((k["stage"], k["i"]))))
rng = np.random.default_rng(0)


def summ(sub, n_boot=2000):
    by = defaultdict(list)
    for r in sub:
        by[r["role"]].append(r["played"])
    roles = sorted(by)
    m = np.array([np.mean(by[x]) for x in roles]); n = np.array([len(by[x]) for x in roles])
    boot = [np.sum(m[ix] * n[ix]) / np.sum(n[ix]) for ix in (rng.integers(0, len(roles), len(roles)) for _ in range(n_boot))]
    return dict(n=len(sub), played=float(np.mean([r["played"] for r in sub])), lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)), score3=float(np.mean([r["role_score"] == 3 for r in sub])),
                coh=float(np.median([r["coh"] for r in sub])), coverage=int(sum(any(by[x]) for x in roles)))


own = [r for r in items if not r["control"]]
ctrl = [r for r in items if r["control"]]
res = {"wrong_role_played": float(np.mean([r["played"] for r in ctrl])), "n_controls": len(ctrl)}
for st in ("base", "final"):
    for c in sorted({r["cond"] for r in own if r["stage"] == st}):
        sub = [r for r in own if r["stage"] == st and r["cond"] == c]
        res[f"{st}/{c}"] = summ(sub)
        if all(r["copied"] is not None for r in sub):
            res[f"{st}/{c}/nocopy"] = summ([r for r in sub if not r["copied"]])
        cs = [r for r in ctrl if r["stage"] == st and r["cond"] == c]
        res[f"{st}/{c}"]["wrong_role_played"] = float(np.mean([r["played"] for r in cs])) if cs else None

print(f"wrong-role control played: {res['wrong_role_played']:.1%} (n={len(ctrl)})\n")
print(f"{'model/cond':22s} {'played':>7s} {'95% CI':>15s} {'score3':>6s} {'coh':>4s} {'cover':>5s} {'wrong':>6s}")
for k, v in res.items():
    if isinstance(v, dict) and not k.endswith("nocopy"):
        w = v["wrong_role_played"]
        print(f"{k:22s} {v['played']:7.1%} [{v['lo']:5.1%}, {v['hi']:5.1%}] {v['score3']:6.1%} {v['coh']:4.0f} {v['coverage']:5d} "
              f"{'' if w is None else f'{w:6.1%}'}")

blind_res = json.load(open(A / "results.json")) if (A / "results.json").exists() else {}
verdict = {"judge_ok": res["wrong_role_played"] <= 0.05, "control_native_played": res["final/native"]["played"],
           "control_ok": res["final/native"]["played"] >= 0.50, "sets": {}}
for s in ("set1", "set2", "set3"):
    b, f = res[f"base/{s}_neutral"], res[f"final/{s}_neutral"]
    bn, fn = res.get(f"base/{s}_neutral/nocopy"), res.get(f"final/{s}_neutral/nocopy")
    rule = lambda b, f: b["played"] >= 0.20 and b["lo"] > res["wrong_role_played"] and b["played"] >= 0.5 * f["played"]
    bl = blind_res.get(f"base/{s}_neutral")
    verdict["sets"][s] = dict(base_played=b["played"], final_played=f["played"], rule1_2=rule(b, f),
                              rule3_coverage=b["coverage"] >= 20, rule4_nocopy=rule(bn, fn) if bn else None,
                              rule5_blind=(bl["lo"] > bl["chance"]) if bl else None)
p = [res[f"base/{s}_neutral"]["played"] for s in ("set1", "set2", "set3")]
verdict["sets_agree"] = max(p) <= 2 * max(min(p), 1e-9)
verdict["note"] = "Rule 6 (blind raw read) is checked by hand."
res["verdict"] = verdict
print("\n" + json.dumps(verdict, indent=1))
json.dump(res, open(A / "results_role.json", "w"), indent=1)
json.dump(items, open(A / "role_scored_items.json", "w"))
