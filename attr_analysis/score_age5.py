"""Age steering at the proper 5-layer setup, judged with the in-role protocol."""
import glob, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

D=Path("judge_age5"); idx=json.load(open(D/"index.json"))
meta=json.load(open("attr_analysis/age5_meta.json"))
LAD, AGES = meta["ladder"], meta["ages"]; NOM=dict(zip(LAD,AGES))
lab={}
for f in sorted(glob.glob(str(D/"verdict_*.json"))):
    n=int(Path(f).stem.split("_")[1])
    for k,v in json.load(open(f)).items():
        m=idx.get(f"{n}:{k}")
        if m: lab.setdefault(m["cell"],[]).append(v)

def st(v):
    if not v: return None
    nums=[float(x["age"]) for x in v if not isinstance(x["age"],str)]
    return {"n":len(v),"med":np.median(nums) if nums else np.nan,
            "person":sum(1 for x in v if x["role"]=="person")/len(v),
            "ns":sum(1 for x in v if x["role"]=="nonsense")/len(v),
            "aged":len(nums)/len(v)}

print("=== PROMPT CEILING vs CONTROL (5 layers, matched displacement) ===")
print(f"{'rung':<13}{'nom':>5} |{'prompt':>22} |{'control':>22}")
pr,ct=[],[]
for p in LAD:
    line=f"{p:<13}{NOM[p]:>5.0f} |"
    for cell,acc in ((f"{p}|prompt",pr),(f"control|{p}",ct)):
        s=st(lab.get(cell,[]))
        if not s: line+=f"{'-':>22} |"; continue
        e=abs(s["med"]-NOM[p]) if not np.isnan(s["med"]) else np.nan
        if not np.isnan(e): acc.append(e)
        line+=f"  age{s['med']:>5.0f} e{e if not np.isnan(e) else -1:>4.0f} per{s['person']:>5.2f} ns{s['ns']:>4.2f} |"
    print(line)
print(f"\nprompt  median |err| {np.median(pr):.1f}y   control median |err| {np.median(ct):.1f}y")
for key in ("prompt","control"):
    xs,ys=[],[]
    for p in LAD:
        s=st(lab.get(f"{p}|prompt" if key=="prompt" else f"control|{p}",[]))
        if s and not np.isnan(s["med"]): xs.append(NOM[p]); ys.append(s["med"])
    if len(xs)>3:
        print(f"  {key}: pearson {pearsonr(xs,ys)[0]:+.3f} spearman {spearmanr(xs,ys)[0]:+.3f} (n={len(xs)})")

print("\n=== THREE PATHS ===")
print(f"{'beta':>5}{'target':>7}" + "".join(f"{a:>24}" for a in ("piecewise","spline","linear")))
agg={a:{"e":[],"ns":[],"per":[],"x":[],"y":[]} for a in ("piecewise","spline","linear")}
for b in meta["betas"]:
    tgt=meta[str(b)]["target_age"]; line=f"{b:>5}{tgt:>7.0f}"
    for arm in ("piecewise","spline","linear"):
        s=st(lab.get(f"{arm}|{b}",[]))
        if not s: line+=f"{'-':>24}"; continue
        e=abs(s["med"]-tgt) if not np.isnan(s["med"]) else np.nan
        if not np.isnan(e):
            agg[arm]["e"].append(e); agg[arm]["x"].append(tgt); agg[arm]["y"].append(s["med"])
        agg[arm]["ns"].append(s["ns"]); agg[arm]["per"].append(s["person"])
        line+=f"  age{s['med']:>5.0f} e{e if not np.isnan(e) else -1:>4.0f} per{s['person']:>5.2f} ns{s['ns']:>4.2f}"
    print(line)
print(f"\n{'arm':<12}{'median |err|':>13}{'mean |err|':>12}{'person':>9}{'nonsense':>10}{'spearman':>10}")
for arm in ("piecewise","spline","linear"):
    g=agg[arm]
    sp=spearmanr(g["x"],g["y"])[0] if len(g["x"])>3 else np.nan
    print(f"{arm:<12}{np.median(g['e']) if g['e'] else np.nan:>13.1f}"
          f"{np.mean(g['e']) if g['e'] else np.nan:>12.1f}{np.mean(g['per']):>9.2f}"
          f"{np.mean(g['ns']):>10.2f}{sp:>10.3f}")
