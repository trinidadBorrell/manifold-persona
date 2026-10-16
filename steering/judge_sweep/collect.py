"""Validate and collect the sweep judges' labels; write the J6 stage-2 batches.

    python -m steering.judge_sweep.collect            # -> JS/labels.csv, JS/redo.json
    python -m steering.judge_sweep.collect --stage2   # also write J6 stage-2 batches + jobs (JS/manifest_s2.json)

A job's file is valid when it has exactly one label per item key and every label is in the judge's
option set (free text for J6 stage 1). Invalid or missing jobs go to redo.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from steering.judge_sweep import prompts as P
from steering.judge_sweep.build import PM

JS = PM / "JS"
S2_BATCH = 100


def norm(x):
    return str(x).strip().lower().replace(" ", "_") if x is not None else None


def check(job, lab):
    """Return (row dict, error or None) for one label."""
    J = job["judge"]
    r = dict(key=lab.get("key"), evidence=lab.get("evidence", lab.get("quote")), analysis=lab.get("analysis"))
    roles = set(P.all_roles())
    if J == "J2":
        ok = set(job["options"]) | {"other", "collapsed"}
        r.update(label=norm(lab.get("persona_kind")), intensity=lab.get("intensity"))
    elif J == "J8":
        ok = roles | {"other", "collapsed"}
        r.update(label=norm(lab.get("persona_kind")), intensity=lab.get("intensity"))
    elif J in ("J3", "J4"):
        ok = set(P.AA_CATS)
        r.update(label=norm(lab.get("score")))
    elif J == "J5":
        p = str(lab.get("persona", "")).strip()
        if p.lower().startswith("mixed"):
            names = [norm(x) for x in p.split(":", 1)[-1].split(",") if x.strip()]
            r.update(label="mixed", mixed=",".join(names), intensity=json.dumps(lab.get("strength")))
            return r, None
        r.update(label=norm(p), intensity=lab.get("strength"))
        ok = set(job["options"]) | {"not_stated", "non_sensical"}
        if r["label"] not in ok:     # a named persona outside the list is allowed by variant A only as Mixed
            r["label"] = "unlisted:" + r["label"]
            return r, None
    elif J == "J6":
        r.update(label=norm(lab.get("persona")), self_id=lab.get("self_identification"),
                 description=lab.get("persona_description"), intensity=lab.get("strength"),
                 secondary=lab.get("secondary_persona"))
        return r, (None if r["label"] else "no persona field")
    elif J == "J7":
        ok = roles | {"none", "collapsed"}
        r.update(label=norm(lab.get("persona")))
    elif J == "J9":
        rk = lab.get("ranking") or []
        try:
            probs = {norm(x["option"]): float(x["p"]) for x in rk}
        except Exception:
            return r, "ranking not parseable"
        tot = sum(probs.values())
        bad = [o for o in probs if o not in roles | {"other", "collapsed"}]
        r.update(label=max(probs, key=probs.get) if probs else None, ranking=json.dumps(probs), p_sum=tot)
        if bad:
            return r, "options not in list: %s" % bad[:3]
        if not 0.95 <= tot <= 1.05:
            return r, "probabilities sum to %.2f" % tot
        return r, None
    else:
        raise KeyError(J)
    return r, (None if r["label"] in ok else "label %r not an option" % r["label"])


def collect(jobs):
    rows, redo = [], []
    for job in jobs:
        f = JS / "labels" / (job["name"] + ".json")
        keys = [x["key"] for x in json.loads((JS / "batches" / (job["batch"] + ".json")).read_text())]
        if not f.exists():
            redo.append(dict(name=job["name"], batch=job["batch"], why="missing"))
            continue
        try:
            labs = json.loads(f.read_text())["labels"]
        except Exception as e:
            redo.append(dict(name=job["name"], batch=job["batch"], why="unreadable: %s" % e))
            continue
        got = {l.get("key"): l for l in labs}
        errs = []
        if set(got) != set(keys) or len(labs) != len(keys):
            errs.append("keys: %d labels for %d items, %d missing" % (len(labs), len(keys), len(set(keys) - set(got))))
        for k in keys:
            if k not in got:
                continue
            r, e = check(job, got[k])
            if e:
                errs.append(e)
            rows.append({**{c: job[c] for c in ("name", "judge", "model", "pair", "draw", "b", "a", "c", "d")}, **r})
        if errs:
            redo.append(dict(name=job["name"], batch=job["batch"], why="; ".join(errs[:5]), n_err=len(errs)))
    return pd.DataFrame(rows), redo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="job names (default: all)")
    ap.add_argument("--stage2", action="store_true")
    a = ap.parse_args(argv)
    jobs = json.loads((JS / "manifest.json").read_text())["jobs"]
    if a.only:
        jobs = [j for j in jobs if j["name"] in set(a.only)]
    have = [j for j in jobs if (JS / "labels" / (j["name"] + ".json")).exists()] if a.only is None else jobs
    df, redo = collect(have)
    df.to_csv(JS / "labels.csv", index=False)
    (JS / "redo.json").write_text(json.dumps(redo, indent=1))
    print("%d of %d jobs have files; %d labels; %d jobs to redo" % (len(have), len(jobs), len(df), len(redo)))
    for r in redo[:20]:
        print("  ", r["name"], r["why"])
    if a.stage2:
        stage2(df[df.judge == "J6"], jobs)


def stage2(df, jobs):
    """J6 stage 2: one item per stage-1 label with a persona; grouped by the stage-1 category order c."""
    (JS / "batches_s2").mkdir(exist_ok=True)
    s2 = []
    df = df[~df.label.isin(["none", "", None]) & df.label.notna()]
    for c, dc in df.groupby("c"):
        items = [dict(key="%s|%s" % (r.name, r.key), persona=r.label, persona_description=r.description,
                      evidence=r.evidence) for r in dc.itertuples()]
        sysname = "J6s2_c%d" % c
        (JS / "sys" / (sysname + ".txt")).write_text(P.j6_stage2_system(int(c), "J6s2"))
        for i in range(0, len(items), S2_BATCH):
            name = "%s_%03d" % (sysname, i // S2_BATCH)
            (JS / "batches" / (name + ".json")).write_text(json.dumps(items[i:i + S2_BATCH], ensure_ascii=False))
            s2.append(dict(name=name, batch=name, sys=sysname, judge="J6s2", c=int(c), n=len(items[i:i + S2_BATCH])))
    (JS / "manifest_s2.json").write_text(json.dumps(dict(jobs=s2), indent=0))
    print("J6 stage 2: %d items in %d jobs" % (sum(j["n"] for j in s2), len(s2)))


if __name__ == "__main__":
    main()
