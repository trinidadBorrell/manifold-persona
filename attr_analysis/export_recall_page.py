"""Data for the recall page: tables, example answers and a 3-d view of layer 23.

The 3-d axes are the top three principal directions of the 300 persona means at layer
23, centred on the assistant mean, so the assistant sits at the origin. Read-out points
of generated answers are projected with the same basis.
"""
import json
from collections import defaultdict

import numpy as np

CACHE = "attr_analysis/_alllayer_cache.npz"
L = 23
SHOW = ["pirate", "musician", "spy", "toddler", "philosopher", "hacker", "witch",
        "vampire", "jester", "detective", "ghost", "scholar"]
EXAMPLES = [("pirate", 4.0, 0), ("hacker", 6.0, 2), ("spy", 8.0, 1), ("toddler", 3.0, 4),
            ("philosopher", 6.0, 5), ("witch", 8.0, 0), ("musician", 6.0, 4),
            ("ghost", 6.0, 0), ("scholar", 8.0, 0)]

z = np.load(CACHE, allow_pickle=True)
names = list(z["names"].astype(str))
M = z["M"][:, L, :].astype(np.float32)
idx = {p: i for i, p in enumerate(names)}
a = M[idx["default"]]
roles = [p for p in names if p != "default"]
D = np.stack([M[idx[p]] - a for p in roles])
_, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
E = Vt[:3].T
var3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
proj = lambda X: ((X - a) @ E).round(2).tolist()

means = {p: proj(M[idx[p]][None])[0] for p in roles}

# read-out points of the corrected run and the prompted baseline, with classifier output
answers = json.load(open("attr_analysis/judge_classifier_answers.json"))
clf = {(r["cell"], r["s"], r["qi"]): r for r in answers}
rank_b = json.load(open("attr_analysis/rank_l23b_results.json"))["cells"]
rank_pb = json.load(open("attr_analysis/rank_pb_results.json"))["cells"]
texts = {}
for f in ("l23b_out/merged.jsonl", "pb_out/merged.jsonl"):
    for l in open(f):
        r = json.loads(l)
        texts[(r["cell"], float(r["alpha"]), int(r["qi"]))] = r

pts = []
for f in ("ro_out/acts_l23b_0.npz", "ro_out/acts_l23b_1.npz", "ro_out/acts_l23b_2.npz",
          "ro_out/acts_pb.npz"):
    za = np.load(f, allow_pickle=True)
    A = za["acts"][:, L, :].astype(np.float32)
    P = proj(A)
    for xyz, cell, al, qi in zip(P, za["cell"].astype(str), za["alpha"], za["qi"]):
        arm_t = cell.split("|")[1]
        arm, t = (arm_t.split("_", 1) if "_" in arm_t else ("prompt", arm_t))
        if arm not in ("own", "L23", "prompt") or t not in SHOW:
            continue
        c = clf.get((cell, float(al), int(qi)))
        key = f"L23|{t}|{float(al):g}" if arm != "prompt" else f"{t}|{t}|0"
        rec = (rank_b if arm != "prompt" else rank_pb).get(key, {})
        pts.append({"t": t, "arm": "steered" if arm != "prompt" else "prompted",
                    "s": float(al), "qi": int(qi), "xyz": xyz,
                    "p": round(c["p_target"], 3) if c else None,
                    "clf": [[n, round(p, 3)] for n, p in c["clf_top5"]] if c else [],
                    "judge": c["judge_top5"] if c else [],
                    "q": texts[(cell, float(al), int(qi))]["question"],
                    "r1": rec.get("r@1"), "r5": rec.get("r@5"), "r10": rec.get("r@10"),
                    "text": texts[(cell, float(al), int(qi))]["response"].strip()})

# tables
by = json.load(open("attr_analysis/rank_l23b_results.json"))
by_old = json.load(open("attr_analysis/rank_l23_results.json"))
by_k = json.load(open("attr_analysis/rank_l23k_results.json"))
pb = json.load(open("attr_analysis/rank_pb_results.json"))
strength_rows = []
for s in ("3", "4", "6", "8"):
    r = by["by_arm_strength"][f"L23|{s}"]
    strength_rows.append({"s": float(s), **{k: r[k] for k in ("r@1", "r@5", "r@10", "asst@1", "nonsense", "n")},
                          "ci5": r["ci@5"]})
best = by["by_arm_best"]["L23"]
per_persona = []
for t in SHOW:
    opts = [(float(s), by["cells"][f"L23|{t}|{s}"]) for s in ("3", "4", "6", "8")]
    s, r = max(opts, key=lambda o: (o[1]["r@5"], o[1]["r@1"]))
    p = pb["cells"][f"{t}|{t}|0"]
    per_persona.append({"t": t, "best_s": s, "r1": r["r@1"], "r5": r["r@5"], "r10": r["r@10"],
                        "p1": p["r@1"], "p5": p["r@5"], "p10": p["r@10"],
                        "norm": float(np.linalg.norm(M[idx[t]] - a))})
arms = {"own": by["by_arm_best"]["L23"], "k32": by["by_arm_best"]["k32"],
        "k8": by["by_arm_best"]["k8"], "own_offset": by_old["by_arm_best"]["L23"],
        "k32_offset": by_k["by_arm_best"]["k32"], "k8_offset": by_k["by_arm_best"]["k8"]}
pc = [pb["cells"][f"{t}|{t}|0"] for t in SHOW]
prompt_all = {k: float(np.mean([c[k] for c in pc])) for k in ("r@1", "r@5", "r@10", "nonsense")}
prompt_all["n"] = sum(c["n"] for c in pc)

examples = []
for t, s, qi in EXAMPLES:
    r = texts[(f"pin|L23_{t}", s, qi)]
    c = clf.get((f"pin|L23_{t}", s, qi), {})
    examples.append({"t": t, "s": s, "q": r["question"], "text": r["response"].strip(),
                     "judge": c.get("judge_top5", [])[:5], "p": c.get("p_target"),
                     "clf": c.get("clf_top5", [])[:3],
                     "r5": by["cells"][f"L23|{t}|{s:g}"]["r@5"]})

clfres = json.load(open("attr_analysis/judge_classifier.json"))
out = {"var3": var3, "means": means, "show": SHOW, "points": pts,
       "strength_rows": strength_rows, "best": best, "per_persona": per_persona,
       "arms": arms, "prompt_all": prompt_all, "examples": examples, "clf": clfres,
       "n_roles": len(roles)}
json.dump(out, open("attr_analysis/recall_page_data.json", "w"))
print(f"{len(pts)} points, {len(examples)} examples, 3 PCs keep {var3:.2f} of variance")
