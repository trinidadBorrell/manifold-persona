"""Broken-reply judges vs the owner's 50 labels, then broken rates per condition (rules fixed 16/10/2026).

Sonnet coherence (judge2 coh items): broken = score < 50. Jev: broken = any Noul > 0.5. A judge is usable
if Cohen's kappa against the owner's Fine/Broken labels is >= 0.6. Nothing is tuned on the labels.
Writes masks (JSON lists of [run, i]) for score_naming.py --mask: one per usable judge, and their union.

    python validate_broken.py COH_JUDGE_DIR JEV_OUT_DIR LABELS.json [--out results/broken]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

KAPPA_MIN = 0.6


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coh_dir")
    ap.add_argument("jev_dir")
    ap.add_argument("labels")
    ap.add_argument("--out", default=str(HERE / "results" / "broken"))
    a = ap.parse_args()
    key = json.loads((Path(a.coh_dir) / "key.json").read_text())
    coh = {}
    for f in (Path(a.coh_dir) / "answers").glob("coh_*.json"):
        for i, s in json.loads(f.read_text())["answer"].items():
            coh[key[i]["reply_key"]] = s
    jev = {}
    for f in (Path(a.jev_dir) / "answers").glob("*.json"):
        r = json.loads(f.read_text())
        ans = r["response"]["answers"]
        jev[r["reply_key"]] = dict(nouls={k: v["noul"] for k, v in ans.items() if k != "coherence"},
                                   coherence=ans["coherence"]["score"])
    want = {v["reply_key"] for v in key.values() if v["type"] == "coh"}
    missing = {"sonnet": len(want - set(coh)), "jev": len(want - set(jev))}
    if any(missing.values()):
        raise SystemExit(f"judge runs incomplete: {missing}")
    broken = {"sonnet": {k: s < 50 for k, s in coh.items()},
              "jev": {k: max(v["nouls"].values()) > 0.5 for k, v in jev.items()}}

    labels = {f"run_main/{v['item']}": v["verdict"] == "broken" for v in json.loads(Path(a.labels).read_text()).values()}
    rng = np.random.default_rng(20261016)
    val = {}
    for j, b in broken.items():
        ks = list(labels)
        if any(k not in b for k in ks):
            raise SystemExit(f"{j}: labels outside the judged set: {[k for k in ks if k not in b]}")
        y, p = [labels[k] for k in ks], [b[k] for k in ks]
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ks), len(ks))
            boots.append(kappa(np.array(y)[idx], np.array(p)[idx]))
        cm = collections.Counter((("owner_broken" if yy else "owner_fine"), ("judge_broken" if pp else "judge_fine"))
                                 for yy, pp in zip(y, p))
        k = kappa(y, p)
        val[j] = dict(n=len(ks), confusion={f"{x}|{z}": n for (x, z), n in cm.items()},
                      accuracy=float(np.mean(np.array(y) == np.array(p))), kappa=k,
                      kappa_ci95=np.nanpercentile(boots, [2.5, 97.5]).tolist(), usable=bool(k >= KAPPA_MIN))
    reps = {}
    for run in ("run_main", "run_rep"):
        for line in open(HERE / run / "replies.jsonl"):
            r = json.loads(line)
            reps[f"{run}/{r['i']}"] = r
    rates = collections.defaultdict(dict)
    broken["jev_score"] = {k: v["coherence"] < 2 for k, v in jev.items()}  # secondary: below "partly makes sense"
    for j, b in broken.items():
        by = collections.defaultdict(list)
        for k, v in b.items():
            r = reps[k]
            by[f"{k.split('/')[0]}/{r['cond']}"].append(v)
        for c, v in sorted(by.items()):
            rates[c][j] = dict(n=len(v), broken_rate=float(np.mean(v)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    usable = [j for j in ("sonnet", "jev") if val[j]["usable"]]
    for f in out.glob("mask_*.json"):  # masks exist only for judges that pass THIS validation
        f.unlink()
    for j in usable:
        (out / f"mask_{j}.json").write_text(json.dumps(sorted([k.split("/")[0], int(k.split("/")[1])]
                                                              for k, v in broken[j].items() if v)))
    # strict-check mask, fixed before output: union of the usable judges (one usable -> that one); none -> none
    if usable:
        strict = {k for j in usable for k, v in broken[j].items() if v}
        (out / "mask_strict.json").write_text(json.dumps(sorted([k.split("/")[0], int(k.split("/")[1])] for k in strict)))
    res = dict(rule=dict(sonnet="coherence < 50", jev="any Noul > 0.5", jev_score="secondary: coherence score < 2",
                         kappa_min=KAPPA_MIN, strict_mask="union of usable judges"),
               validation_scope=("owner's 50 labels: desc, run_main, replies flagged by the frozen invalid rule, "
                                 "picked by browsing (not random); kappa says nothing about unflagged or ex2 replies"),
               judged_set=("judge2 coherence set: 10,800 replies; ex2_other has one donor per cell (1,800 of 5,400), "
                           "so masks do not cover the other two donors in score_naming's ex2_other cells"),
               missing=missing, validation=val, usable=usable, broken_rates=rates)
    (out / "validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(missing=missing, validation=val, usable=usable), indent=1))
    for c, d in rates.items():
        print(c, {j: f"{x['broken_rate']:.1%} of {x['n']}" for j, x in d.items()})


if __name__ == "__main__":
    main()
