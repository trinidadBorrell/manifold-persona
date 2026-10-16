"""Append the new-frame answers of the twelve personas to the recall page's 3-d points.

Points are projected with the page's own basis (top three principal directions of the
persona means at layer 23, centred on the assistant). Each point carries the judge's
ranked names, the classifier's probabilities and the cell's recall.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from judge_classifier import JudgeClassifier
from score_rank import load as load_judge

CACHE = "attr_analysis/_alllayer_cache.npz"
L = 23
RUNS = [("ro12_out", "judge_rank_ro12", "attr_analysis/rank_ro12_results.json"),
        ("rok12_out", "judge_rank_rok12", "attr_analysis/rank_rok12_results.json")]
ARM = {"ro": "new", "rk32": "new32", "rk8": "new8"}

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"][:, L, :].astype(np.float32)
idx = {p: i for i, p in enumerate(names)}
a = M[idx["default"]]
roles = [p for p in names if p != "default"]
D_ = np.stack([M[idx[p]] - a for p in roles])
_, S, Vt = np.linalg.svd(D_ - D_.mean(0), full_matrices=False)
E = Vt[:3].T
proj = lambda X: ((X - a) @ E).round(2).tolist()

clf = JudgeClassifier.load("attr_analysis/judge_classifier_all_bycell_model.npz")
classes, cidx = clf.classes, clf.cidx

D = json.load(open("attr_analysis/recall_page_data.json"))
D["points"] = [p for p in D["points"] if not p["arm"].startswith("new")]
added = 0
for out_dir, judge_dir, res_file in RUNS:
    rows, _ = load_judge(judge_dir)
    J = {(r["cell"], float(r["alpha"]), int(r["qi"])): r for r in rows}
    res = json.load(open(res_file))["cells"]
    texts = {}
    for l in open(f"{out_dir}/merged.jsonl"):
        r = json.loads(l)
        texts[(r["cell"], float(r["alpha"]), int(r["qi"]))] = r
    za = np.load(f"{out_dir}/acts_0.npz", allow_pickle=True)
    X = za["acts"][:, 0, :].astype(np.float32)
    P = clf.predict_proba(X)
    xyz = proj(X)
    for k, (cell, al, qi) in enumerate(zip(za["cell"].astype(str), za["alpha"], za["qi"])):
        arm_t = cell.split("|")[1]
        arm, t = arm_t.split("_", 1)
        key = (cell, float(al), int(qi))
        j = J.get(key)
        rec = res.get(f"{arm}|{t}|{float(al):g}", {})
        o = np.argsort(-P[k])[:5]
        D["points"].append({"t": t, "arm": ARM[arm], "s": float(al), "qi": int(qi), "xyz": xyz[k],
                            "p": round(float(P[k][cidx[t]]), 3),
                            "clf": [[classes[i], round(float(P[k][i]), 3)] for i in o],
                            "judge": j["rank"][:5] if j else [],
                            "q": texts[key]["question"],
                            "r1": rec.get("r@1"), "r5": rec.get("r@5"), "r10": rec.get("r@10"),
                            "text": texts[key]["response"].strip()})
        added += 1
json.dump(D, open("attr_analysis/recall_page_data.json", "w"))
print(f"added {added} points; total {len(D['points'])}")
