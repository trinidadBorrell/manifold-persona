"""Age steering: 5 layers, two alphas, in-role judge. Also emits JSON for the artifact."""
import glob, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

D = Path("judge_age5")
idx = json.load(open(D / "index.json"))
meta = json.load(open("attr_analysis/age5_meta.json"))
LAD, AGES = meta["ladder"], meta["ages"]
NOM = dict(zip(LAD, AGES))
ARMS = ("piecewise", "spline", "linear")

lab = {}
for f in sorted(glob.glob(str(D / "verdict_*.json"))):
    n = int(Path(f).stem.split("_")[1])
    for k, v in json.load(open(f)).items():
        m = idx.get(f"{n}:{k}")
        if m:
            lab.setdefault((m["cell"], 1.0), []).append(v)

def st(v):
    if not v:
        return None
    nums = [float(x["age"]) for x in v if not isinstance(x["age"], str)]
    return {"med": float(np.median(nums)) if nums else None,
            "person": sum(1 for x in v if x["role"] == "person") / len(v),
            "ns": sum(1 for x in v if x["role"] == "nonsense") / len(v),
            "n": len(v)}

ALPHAS = sorted({a for _, a in lab})
res = {"ladder": LAD, "ages": AGES, "alphas": ALPHAS, "betas": meta["betas"], "by_alpha": {}}

for AL in ALPHAS:
    print(f"\n############ alpha = {AL} ############")
    print("=== prompt ceiling vs control ===")
    print(f"{'rung':<13}{'nom':>5}{'prompt':>20}{'control':>22}")
    pr, ct, px, py, cx, cy = [], [], [], [], [], []
    rung_rows = []
    for p in LAD:
        sp_ = st(lab.get((f"{p}|prompt", AL), []))
        sc_ = st(lab.get((f"control|{p}", AL), []))
        line = f"{p:<13}{NOM[p]:>5.0f}"
        for s, acc, xs, ys in ((sp_, pr, px, py), (sc_, ct, cx, cy)):
            if s and s["med"] is not None:
                e = abs(s["med"] - NOM[p]); acc.append(e); xs.append(NOM[p]); ys.append(s["med"])
                line += f"  age{s['med']:>5.0f} e{e:>4.0f} ns{s['ns']:>4.2f}"
            else:
                line += f"{'  --':>20}"
        print(line)
        rung_rows.append({"rung": p, "nominal": NOM[p],
                          "prompt": sp_, "control": sc_})
    pear_p = pearsonr(px, py)[0] if len(px) > 3 else None
    pear_c = pearsonr(cx, cy)[0] if len(cx) > 3 else None
    print(f"  prompt  median|err| {np.median(pr) if pr else float('nan'):.1f}y  pearson {pear_p}")
    print(f"  control median|err| {np.median(ct) if ct else float('nan'):.1f}y  pearson {pear_c}")

    print("\n=== three paths ===")
    print(f"{'beta':>5}{'target':>7}" + "".join(f"{a:>22}" for a in ARMS))
    agg = {a: {"e": [], "ns": [], "per": [], "x": [], "y": []} for a in ARMS}
    beta_rows = []
    for b in meta["betas"]:
        tgt = meta[str(b)]["target_age"]
        line = f"{b:>5}{tgt:>7.0f}"; row = {"beta": b, "target": tgt}
        for arm in ARMS:
            s = st(lab.get((f"{arm}|{b}", AL), []))
            row[arm] = s
            if s and s["med"] is not None:
                e = abs(s["med"] - tgt)
                agg[arm]["e"].append(e); agg[arm]["x"].append(tgt); agg[arm]["y"].append(s["med"])
                line += f"  age{s['med']:>5.0f} e{e:>4.0f} ns{s['ns']:>4.2f}"
            else:
                line += f"{'  --':>22}"
            if s:
                agg[arm]["ns"].append(s["ns"]); agg[arm]["per"].append(s["person"])
        beta_rows.append(row); print(line)

    print(f"\n{'arm':<12}{'median|err|':>12}{'person':>9}{'nonsense':>10}{'spearman':>10}{'n':>4}")
    summ = {}
    for arm in ARMS:
        g = agg[arm]
        sp_r = spearmanr(g["x"], g["y"])[0] if len(g["x"]) > 3 else None
        summ[arm] = {"median_err": float(np.median(g["e"])) if g["e"] else None,
                     "person": float(np.mean(g["per"])) if g["per"] else None,
                     "nonsense": float(np.mean(g["ns"])) if g["ns"] else None,
                     "spearman": None if sp_r is None or np.isnan(sp_r) else float(sp_r),
                     "n": len(g["e"])}
        print(f"{arm:<12}{summ[arm]['median_err'] if summ[arm]['median_err'] else float('nan'):>12.1f}"
              f"{summ[arm]['person']:>9.2f}{summ[arm]['nonsense']:>10.2f}"
              f"{summ[arm]['spearman'] if summ[arm]['spearman'] is not None else float('nan'):>10.3f}"
              f"{summ[arm]['n']:>4}")
    res["by_alpha"][str(AL)] = {"rungs": rung_rows, "betas": beta_rows, "summary": summ,
                                "pearson_prompt": pear_p, "pearson_control": pear_c,
                                "median_err_prompt": float(np.median(pr)) if pr else None,
                                "median_err_control": float(np.median(ct)) if ct else None}

json.dump(res, open("attr_analysis/age_results_a1.json", "w"), indent=1)
print("\n-> attr_analysis/age_results_a1.json")
