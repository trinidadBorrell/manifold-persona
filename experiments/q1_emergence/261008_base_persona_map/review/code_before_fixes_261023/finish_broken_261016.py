"""Finish the broken-reply categorisation of the E3 pilot (Codex plan review/codex_advice_broken_261016.md).

Scope: the 10,800 judge2 coherence-set replies (6 groups x 20 roles x 18 shared questions x 5 wordings).
Method, fixed before any new output (no judge passed the kappa gate, so every automatic call is provisional):
  final label  = owner label on the 80 owner-labelled replies, else the GPT-6.1-Sol round-3 prompt's call
  GPT input    = the reply, except a verified bare final "Interview"/"Interviewer" stop fragment, which is removed
  Jev          = saved round-3 calls (jev/round3), Broken if any Noul > 0.5; diagnostic only
  Opus         = saved round-3 calls on run_main desc + ex2 only; check only
Round-3 GPT calls (e3_broken_round3) are reused for run_main desc + ex2; this script judges the other 7,200 replies
and the derived (fragment-free) text of the fragment replies, with the round-3 prompt and schema unchanged.

    python finish_broken_261016.py prep JUDGE_DIR      # new folder outside the repo
    python finish_broken_261016.py run JUDGE_DIR [--workers 16] [--minutes 45]
    python finish_broken_261016.py report JUDGE_DIR    # local only; writes results/broken/finish_261016/
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COH_DIR = Path("/Users/brendon/persona-manifolds/e3_judge2_coh_261016")
R3_DIR = Path("/Users/brendon/persona-manifolds/e3_broken_round3")
JEV_DIR = HERE / "jev" / "round3"
OUT = HERE / "results" / "broken" / "finish_261016"
LABELS50 = HERE / "review/attrition/owner_labels_desc_2026-10-16_final.json"
LABELS30 = HERE / "review/attrition/owner_labels_fresh30_2026-10-16.json"
FRESH_SET = HERE / "review/attrition/fresh_test_set_2026-10-16.json"
SALT, BATCH, SEED = "e3-broken-finish", 25, 20261016
FRAGMENT = re.compile(r"\n+[ \t]*Interview(?:er)?[ \t]*:?[ \t]*$")
# owner review-page tags -> definition reasons ("babble" = rambles or guesses instead of answering = off the question;
# "other" has no mapping and stays unresolved unless GPT also called the reply Broken)
OWNER_TAG = {"loop": "loop", "nonsense": "nonsense", "echo": "format", "babble": "off_question"}
LIMITS = ["20 selected pilot roles and 18 shared questions; not all 276 roles or 40 questions",
          "ex2_other: one fixed donor per cell (1,800 of 5,400 replies)", "two decoding seeds only",
          "no owner labels for ex2_filler, ex2_other or run_rep; their rates rest on GPT alone",
          "crossed intervals are conditional on the labels; classification error is outside them",
          "no judge passed the pre-set kappa gate; all automatic labels are provisional"]
GROUPS = [("run_main", "desc"), ("run_main", "ex2"), ("run_main", "ex2_filler"), ("run_main", "ex2_other"),
          ("run_rep", "desc"), ("run_rep", "ex2")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jb = _load("judge_broken")
sha = jb.judge2.sha


def replies():
    reps = {}
    for run in ("run_main", "run_rep"):
        for line in open(HERE / run / "replies.jsonl"):
            r = json.loads(line)
            reps[f"{run}/{r['i']}"] = r
    return reps


def scope(reps):
    """The 10,800 coherence-set reply keys; checks each group is a complete 20 x 18 x 5 grid."""
    key = json.loads((COH_DIR / "key.json").read_text())
    keys = sorted({v["reply_key"] for v in key.values() if v["type"] == "coh"})
    cells = collections.defaultdict(set)
    for k in keys:
        r = reps[k]
        g = (k.split("/")[0], r["cond"])
        cell = (r["role"], r["question_idx"], r["wording"])
        if cell in cells[g]:
            raise SystemExit(f"duplicate cell {g} {cell}")
        cells[g].add(cell)
    for g in GROUPS:
        if len(cells[g]) != 1800 or len({c[0] for c in cells[g]}) != 20 or len({c[1] for c in cells[g]}) != 18:
            raise SystemExit(f"group {g}: {len(cells[g])} cells")
    if len(keys) != 10800:
        raise SystemExit(f"{len(keys)} keys")
    return keys


def fragment(text):
    """The reply without a final bare stop fragment, or None when there is none."""
    m = FRAGMENT.search(text or "")
    return text[:m.start()].rstrip() if m else None


def prep(jd):
    jd = Path(jd)
    if jd.exists():
        raise SystemExit(f"{jd} exists; prep needs a new folder")
    reps = replies()
    keys = scope(reps)
    r3 = set(json.loads((R3_DIR / "key.json").read_text()).values())
    rows = []
    for k in keys:
        r = reps[k]
        text = r["reply"] or ""
        if jb.DELIM.search(text) or jb.DELIM.search(r["question"]):
            raise SystemExit(f"delimiter text in {k}")
        if k not in r3:
            rows.append(dict(id=hashlib.sha256(f"{SALT}|raw|{k}".encode()).hexdigest()[:10], reply_key=k, kind="raw",
                             text=jb.item_text(r["question"], text)))
        d = fragment(text) if r["ended_by"] == "limit" else None
        if d is not None:
            rows.append(dict(id=hashlib.sha256(f"{SALT}|derived|{k}".encode()).hexdigest()[:10], reply_key=k,
                             kind="derived", text=jb.item_text(r["question"], d), suffix=text[len(d):],
                             derived_sha256=sha(d)))
    rng = random.Random(SEED)
    raw_rows, der_rows = [r for r in rows if r["kind"] == "raw"], [r for r in rows if r["kind"] == "derived"]
    rng.shuffle(raw_rows)
    rng.shuffle(der_rows)
    chunks = [raw_rows[s:s + BATCH] for s in range(0, len(raw_rows), BATCH)]
    chunks += [der_rows[s:s + BATCH] for s in range(0, len(der_rows), BATCH)]  # derived copies never share a batch with raw
    (jd / "batches").mkdir(parents=True)
    batches = {}
    for b, chunk in enumerate(chunks, 1):
        name = f"f{b:04d}"
        prompt = jb.judge2.batch_prompt(chunk)
        (jd / "batches" / f"{name}.txt").write_text(prompt)
        batches[name] = dict(ids=[r["id"] for r in chunk], prompt_sha256=sha(prompt))
    (jd / "key.json").write_text(json.dumps({r["id"]: {k: v for k, v in r.items() if k not in ("id", "text")}
                                             for r in rows}))
    man = dict(seed=SEED, batch=BATCH, model=jb.MODELS["gpt"], effort=jb.GPT_EFFORT, system_prompt=jb.SYSTEM,
               item_text_sha256=sha(jb.item_text("{question}", "{answer}")),
               sources=dict(judge_broken_sha256=sha((HERE / "judge_broken.py").read_text()),
                            finish_sha256=sha((HERE / "finish_broken_261016.py").read_text()),
                            round3_manifest_sha256=sha((R3_DIR / "manifest.json").read_text()),
                            jev_settings_sha256=sha((JEV_DIR / "settings.json").read_text())),
               counts=dict(raw=sum(r["kind"] == "raw" for r in rows), derived=sum(r["kind"] == "derived" for r in rows)),
               batches=batches)
    (jd / "manifest.json").write_text(json.dumps(man, indent=1))
    print(man["counts"], len(batches), "batches")


def run(jd, workers, minutes):
    jd = Path(jd)
    man_text = (jd / "manifest.json").read_text()
    man, man_sha = json.loads(man_text), sha(man_text)
    now = dict(model=jb.MODELS["gpt"], effort=jb.GPT_EFFORT, system_prompt=jb.SYSTEM,
               item_text_sha256=sha(jb.item_text("{question}", "{answer}")))
    if any(man[k] != v for k, v in now.items()):
        raise SystemExit("the judging prompt, model or effort changed since prep")
    (jd / "answers").mkdir(exist_ok=True)
    (jd / "logs").mkdir(exist_ok=True)
    deadline = time.time() + 60 * minutes

    def job(name):
        meta, f = man["batches"][name], jd / "answers" / f"{name}.json"
        prompt = (jd / "batches" / f"{name}.txt").read_text()
        if sha(prompt) != meta["prompt_sha256"]:
            return name, "batch file changed since prep"
        if f.exists():
            try:
                rec = json.loads(f.read_text())
                if rec["manifest_sha256"] == man_sha and jb.check(rec["answer"], meta["ids"]) is None:
                    return name, None
            except (ValueError, KeyError, TypeError):
                pass
        err = None
        for attempt in (1, 2, 3):
            if time.time() > deadline:
                return name, "time limit reached; not submitted"
            try:
                a, usage = jb.call_gpt(prompt, meta["ids"], jd / "logs" / f"gpt_{name}_try{attempt}.log")
                err = jb.check(a, meta["ids"])
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            if err is None:
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(manifest_sha256=man_sha, model_usage=usage, answer=a)))
                tmp.rename(f)
                return name, None
        return name, err

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(job, list(man["batches"])))
    bad = [(n, e) for n, e in res if e]
    print(f"{len(res) - len(bad)}/{len(res)} batches valid")
    for n, e in bad[:20]:
        print("  failed", n, e)
    cmds = [l for f in sorted((jd / "logs").glob("gpt_*.log")) for l in f.read_text().splitlines()
            if l.startswith("/bin/") and " -lc " in l]
    print(f"gpt shell commands in logs: {len(cmds)}")
    for c in cmds[:20]:
        print("  ", c[:160])


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((np.mean(a == b) - pe) / (1 - pe)) if pe < 1 else float("nan")


def wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def hyper_ci(x, n, N, tail=0.0125):
    """Exact interval for the count K of Broken in a stratum of N, from x Broken in a simple random sample of n."""
    from scipy.stats import hypergeom
    ks = [K for K in range(x, N - (n - x) + 1)
          if hypergeom.sf(x - 1, N, K, n) >= tail and hypergeom.cdf(x, N, K, n) >= tail]
    return min(ks), max(ks)


def report(jd):
    jd = Path(jd)
    reps = replies()
    keys = scope(reps)
    rng = np.random.default_rng(SEED)
    # raw GPT calls: round 3 (run_main desc + ex2) and this run (the rest); derived calls on fragment replies
    gpt, opus, derived = {}, {}, {}
    k3 = json.loads((R3_DIR / "key.json").read_text())
    m3_text = (R3_DIR / "manifest.json").read_text()
    for j, store in (("gpt", gpt), ("opus", opus)):
        for f in (R3_DIR / f"answers_{j}").glob("*.json"):
            rec = json.loads(f.read_text())
            if rec["manifest_sha256"] != sha(m3_text):
                raise SystemExit(f"round3 {j}/{f.name}: manifest mismatch")
            store.update({k3[i]: v for i, v in rec["answer"].items()})
    kf = json.loads((jd / "key.json").read_text())
    mf_text = (jd / "manifest.json").read_text()
    mf = json.loads(mf_text)
    for f in (jd / "answers").glob("*.json"):
        rec = json.loads(f.read_text())
        if rec["manifest_sha256"] != sha(mf_text) or jb.check(rec["answer"], mf["batches"][f.stem]["ids"]):
            raise SystemExit(f"finish/{f.name}: does not belong to this prepared run")
        for i, v in rec["answer"].items():
            (gpt if kf[i]["kind"] == "raw" else derived)[kf[i]["reply_key"]] = v
    jev_judge = _load("jev_judge")
    jev = {}
    for f in (JEV_DIR / "answers").glob("*.json"):
        r = json.loads(f.read_text())
        if jev_judge.check(r["response"]) is not None:
            raise SystemExit(f"jev/{f.name}: {jev_judge.check(r['response'])}")
        jev[r["reply_key"]] = {k: r["response"]["answers"][k]["noul"] for k in jev_judge.NOULS}
    suffix = {v["reply_key"]: v["suffix"] for v in kf.values() if v["kind"] == "derived"}
    frag_keys = sorted(suffix)
    missing = dict(gpt_raw=sorted(set(keys) - set(gpt)), gpt_derived=sorted(set(frag_keys) - set(derived)),
                   jev=sorted(set(keys) - set(jev)))
    owner50 = {f"run_main/{v['item']}": v for v in json.loads(LABELS50.read_text()).values()}
    fresh30 = {f"run_main/{v['item']}": v for v in json.loads(LABELS30.read_text()).values()}
    owner = {**owner50, **fresh30}

    def gpt_final(k):  # derived call for verified fragments, else the raw call
        return derived.get(k) or gpt.get(k)

    rows = []
    for k in keys:
        r, g = reps[k], gpt_final(k)
        if k in owner:
            lab = owner[k]["verdict"] == "broken"
            tags = [OWNER_TAG[t] for t in owner[k].get("reasons") or [] if t in OWNER_TAG]
            if lab and tags:  # the owner's own reason tag comes first
                reason, rsrc = tags[0], "owner_tag"
            elif lab and g and g["broken"]:
                reason, rsrc = g["reason"], "gpt_reason_on_owner_broken"
            elif lab:
                reason, rsrc = "owner_broken_reason_unstated", "unresolved"
            else:
                reason, rsrc = "none", "owner"
            src = "owner"
        elif g:
            lab, reason, rsrc, src = g["broken"], g["reason"], "gpt", "gpt_derived" if k in derived else "gpt_raw"
        else:
            lab, reason, rsrc, src = None, None, None, "missing"
        rows.append(dict(reply_key=k, run=k.split("/")[0], cond=r["cond"], role=r["role"], question_idx=r["question_idx"],
                         wording=r["wording"], donor=r.get("example_role"), broken=lab, reason=reason,
                         label_source=src, reason_source=rsrc,
                         owner=owner[k]["verdict"] if k in owner else None,
                         owner_comment=owner[k].get("comment") if k in owner else None,
                         gpt_raw=gpt.get(k), gpt_derived=derived.get(k), opus=opus.get(k), jev=jev.get(k),
                         fragment_suffix=suffix.get(k)))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "labels.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[(row["run"], row["cond"])].append(row)
    roles = sorted({r["role"] for r in rows})
    qs = sorted({r["question_idx"] for r in rows})

    def grid(rs, field):
        A = np.full((20, 18, 5), np.nan)
        for r in rs:
            if r[field] is not None:
                A[roles.index(r["role"]), qs.index(r["question_idx"]), r["wording"]] = float(r[field])
        return A

    draws = [(rng.integers(0, 20, 20), rng.integers(0, 18, 18)) for _ in range(2000)]  # shared by every interval

    def crossed_ci(rs, field):
        A = grid(rs, field)
        return np.nanpercentile([np.nanmean(A[ri][:, qi]) for ri, qi in draws], [2.5, 97.5]).tolist()

    def paired(g1, g2):
        A, B = grid(by_group[g1], "broken"), grid(by_group[g2], "broken")
        d = [np.nanmean(A[ri][:, qi]) - np.nanmean(B[ri][:, qi]) for ri, qi in draws]
        return dict(diff=float(np.nanmean(A) - np.nanmean(B)), ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                    note="same role/question draws on both sides; conditional on the labels")

    rates = {}
    for g in GROUPS:
        rs = by_group[g]
        n = len(rs)
        for r in rs:
            r["_gpt_raw"] = None if not r["gpt_raw"] else r["gpt_raw"]["broken"]
            r["_jev"] = None if not r["jev"] else max(r["jev"].values()) > 0.5
        fin = [r["broken"] for r in rs if r["broken"] is not None]
        gr = [r["_gpt_raw"] for r in rs if r["_gpt_raw"] is not None]
        jv = [r["_jev"] for r in rs if r["_jev"] is not None]
        both = [r for r in rs if r["_gpt_raw"] is not None and r["_jev"] is not None]
        reasons = collections.Counter(r["reason"] if r["broken"] else "fine" for r in rs if r["broken"] is not None)
        gf = [gpt_final(r["reply_key"])["broken"] for r in rs if gpt_final(r["reply_key"])]
        frag = [r for r in rs if r["gpt_derived"] is not None and r["gpt_raw"] is not None]
        entry = dict(n=n, final_broken=sum(fin), final_rate=sum(fin) / len(fin) if fin else None,
                     final_rate_crossed_ci95=crossed_ci(rs, "broken"),
                     gpt_fragment_adjusted_broken=sum(gf), gpt_fragment_adjusted_rate=sum(gf) / len(gf) if gf else None,
                     fragment_directions=dict(
                         broken_to_fine=sum(r["gpt_raw"]["broken"] and not r["gpt_derived"]["broken"] for r in frag),
                         fine_to_broken=sum(not r["gpt_raw"]["broken"] and r["gpt_derived"]["broken"] for r in frag),
                         reason_changed=[r["reply_key"] for r in frag if r["gpt_raw"]["reason"] != r["gpt_derived"]["reason"]
                                         and r["gpt_derived"]["broken"]]),
                     reasons_table={k: dict(count=v, share=v / n) for k, v in sorted(reasons.items(), key=lambda t: -t[1])},
                     gpt_raw_broken=sum(gr), gpt_raw_rate=sum(gr) / len(gr) if gr else None,
                     jev_broken=sum(jv), jev_rate=sum(jv) / len(jv),
                     gpt_jev=dict(both_broken=sum(r["_gpt_raw"] and r["_jev"] for r in both),
                                  gpt_only=sum(r["_gpt_raw"] and not r["_jev"] for r in both),
                                  jev_only=sum(r["_jev"] and not r["_gpt_raw"] for r in both),
                                  both_fine=sum(not r["_gpt_raw"] and not r["_jev"] for r in both),
                                  disagreement=sum(r["_gpt_raw"] != r["_jev"] for r in both) / len(both) if both else None),
                     owner_substituted=sum(r["label_source"] == "owner" for r in rs),
                     fragments=sum(r["fragment_suffix"] is not None for r in rs),
                     fragment_changes=sum(r["gpt_derived"] is not None and r["gpt_raw"] is not None
                                          and r["gpt_derived"]["broken"] != r["gpt_raw"]["broken"] for r in rs),
                     reasons_primary={k: v for k, v in reasons.items() if k != "fine"},
                     top_broken_roles=collections.Counter(r["role"] for r in rs if r["broken"]).most_common(5),
                     jev_criteria={c: sum(r["jev"][c] > 0.5 for r in rs if r["jev"]) for c in jev_judge.NOULS},
                     missing=sum(r["broken"] is None for r in rs))
        if g in (("run_main", "desc"), ("run_main", "ex2")):
            op = [r["opus"]["broken"] for r in rs if r["opus"]]
            maj = [sum([r["opus"]["broken"], r["gpt_raw"]["broken"], r["_jev"]]) >= 2 for r in rs
                   if r["opus"] and r["gpt_raw"] and r["jev"]]
            entry.update(opus_rate=sum(op) / len(op), majority_rate=sum(maj) / len(maj),
                         all_three_disagree=sum(len({r["opus"]["broken"], r["gpt_raw"]["broken"], r["_jev"]}) > 1
                                                for r in rs if r["opus"] and r["gpt_raw"] and r["jev"]) / len(maj))
        rates["/".join(g)] = entry
        for r in rs:
            r.pop("_gpt_raw"), r.pop("_jev")

    # agreement with the owner, raw calls only (the failed gate is reproduced, not re-tested)
    fresh_meta = {f"run_main/{x['item']}": x for x in json.loads(FRESH_SET.read_text())}
    calls = {"gpt": lambda k: gpt[k]["broken"], "opus": lambda k: opus[k]["broken"],
             "jev": lambda k: max(jev[k].values()) > 0.5,
             "majority": lambda k: (gpt[k]["broken"] + opus[k]["broken"] + (max(jev[k].values()) > 0.5)) >= 2}
    agreement = {}
    for name, labs in (("owner50", owner50), ("fresh30", fresh30)):
        ks = sorted(labs)
        y = [labs[k]["verdict"] == "broken" for k in ks]
        agreement[name] = {}
        for j, fn in calls.items():
            p = [fn(k) for k in ks]
            tp = sum(a and b for a, b in zip(y, p)); fn_ = sum(a and not b for a, b in zip(y, p))
            tn = sum(not a and not b for a, b in zip(y, p)); fp = sum(not a and b for a, b in zip(y, p))
            boots, undefined = [], 0
            strata = collections.defaultdict(list)
            for idx, k in enumerate(ks):
                strata[(fresh_meta[k]["cond"], fresh_meta[k]["flagged"]) if name == "fresh30" else "all"].append(idx)
            for _ in range(2000):
                idx = [i for s in strata.values() for i in rng.choice(s, len(s))]
                kv = kappa([y[i] for i in idx], [p[i] for i in idx])
                if math.isnan(kv):
                    undefined += 1
                else:
                    boots.append(kv)
            agreement[name][j] = dict(n=len(ks), tp=tp, fn=fn_, tn=tn, fp=fp, accuracy=(tp + tn) / len(ks),
                                      kappa=kappa(y, p), kappa_ci95=np.percentile(boots, [2.5, 97.5]).tolist(),
                                      kappa_undefined_draws=undefined,
                                      sensitivity=tp / (tp + fn_) if tp + fn_ else None,
                                      sensitivity_wilson=wilson(tp, tp + fn_),
                                      specificity=tn / (tn + fp) if tn + fp else None,
                                      specificity_wilson=wilson(tn, tn + fp))
    gate = {j: all(agreement[s][j]["kappa"] >= 0.6 for s in agreement) for j in calls}
    # fresh30 by stratum (condition x old flag), with the frame weights used for the owner estimate below
    by_stratum = collections.defaultdict(list)
    for k in fresh30:
        by_stratum[(fresh_meta[k]["cond"], fresh_meta[k]["flagged"])].append(k)
    fresh_strata = {}
    for (cond, flag), ks in sorted(by_stratum.items()):
        fresh_strata[f"{cond}/flagged={flag}"] = {j: dict(
            n=len(ks), owner_broken=sum(fresh30[k]["verdict"] == "broken" for k in ks),
            tp=sum(fresh30[k]["verdict"] == "broken" and fn(k) for k in ks),
            fn=sum(fresh30[k]["verdict"] == "broken" and not fn(k) for k in ks),
            tn=sum(fresh30[k]["verdict"] == "fine" and not fn(k) for k in ks),
            fp=sum(fresh30[k]["verdict"] == "fine" and fn(k) for k in ks)) for j, fn in calls.items()}

    # owner-sample estimate for run_main desc and ex2 (fresh30 frame: shared questions, not owner50, not raw-read,
    # not in the 200-item human sheet; strata = old flag, 8 + 7 sampled)
    raw_read = set(json.loads((HERE / "run_main" / "raw_read_ids.json").read_text()))
    hs = json.loads((HERE / "human" / "judge2_human_sample.json").read_text())
    hkey = json.loads((COH_DIR / "key.json").read_text())
    human = {hkey[i]["reply_key"] for i in hs["ids"]}
    rep4 = lambda t: any(v >= 3 for v in collections.Counter(tuple(w[i:i + 4]) for w in [re.findall(r"\w+", (t or "").lower())]
                                                              for i in range(len(w) - 3)).values())
    owner_est = {}
    for cond in ("desc", "ex2"):
        grp = [k for k in keys if k.startswith("run_main/") and reps[k]["cond"] == cond]
        known = [k for k in grp if k in owner50]
        frame = [k for k in grp if k not in owner50 and int(k.split("/")[1]) not in raw_read and k not in human]
        excluded_unknown = len(grp) - len(known) - len(frame)
        est, lo, hi, strata_out = sum(owner50[k]["verdict"] == "broken" for k in known), 0, 0, {}
        lo = hi = est
        for flag in (True, False):
            Nh = sum((reps[k]["ended_by"] == "limit" or rep4(reps[k]["reply"])) == flag for k in frame)
            smp = [k for k in fresh30 if fresh_meta[k]["cond"] == cond and fresh_meta[k]["flagged"] == flag]
            x, n = sum(fresh30[k]["verdict"] == "broken" for k in smp), len(smp)
            a, b = hyper_ci(x, n, Nh)
            est += Nh * x / n
            lo += a
            hi += b
            strata_out[f"flagged={flag}"] = dict(N=Nh, n=n, broken=x, count_ci=[a, b])
        tot = len(known) + len(frame)
        owner_est[cond] = dict(frame_size=tot, excluded_unlabelled=excluded_unknown, owner50_known=len(known),
                               strata=strata_out, rate=est / tot, conservative_ci95=[lo / tot, hi / tot],
                               note="frame excludes raw-read and human-sheet replies; strata intervals are 97.5% "
                                    "hypergeometric, summed for a conservative 95% interval")
    for cond, e in owner_est.items():  # frame weights for the fresh30 strata
        for s, v in e["strata"].items():
            fresh_strata[f"{cond}/{s}"] = dict(by_judge=fresh_strata.get(f"{cond}/{s}"), frame_N=v["N"],
                                               weight_in_frame=v["N"] / e["frame_size"])
    contrasts = {f"{a[1]}-{b[1]} (run_main)": paired(a, b) for a, b in
                 ((("run_main", "ex2"), ("run_main", "desc")), (("run_main", "ex2_filler"), ("run_main", "desc")),
                  (("run_main", "ex2_other"), ("run_main", "desc")), (("run_main", "ex2"), ("run_main", "ex2_filler")))}
    contrasts.update({f"{c}: run_main-run_rep": paired(("run_main", c), ("run_rep", c)) for c in ("desc", "ex2")})
    sources = {p.name: sha(p.read_bytes()) for p in (HERE / "run_main" / "replies.jsonl", HERE / "run_rep" / "replies.jsonl",
                                                     COH_DIR / "key.json", LABELS50, LABELS30, FRESH_SET,
                                                     jd / "manifest.json", R3_DIR / "manifest.json", JEV_DIR / "settings.json")}
    sources["labels.jsonl"] = sha((OUT / "labels.jsonl").read_bytes())
    sources["note"] = ("judge_broken.py and jev_judge.py changed after round 3 only in scoring and safety code; "
                       "the round-3 item prompt rebuilds the saved batches exactly")
    res = dict(status="provisional: no judge passed the pre-set gate (kappa >= 0.6 on owner50 and fresh30)",
               method=__doc__.split("\n\n")[1], limits=LIMITS, owner_tag_map=OWNER_TAG, sources=sources,
               missing={k: len(v) for k, v in missing.items()}, rates=rates, contrasts=contrasts,
               agreement=agreement, fresh30_by_stratum=fresh_strata, gate_passed=gate, owner_estimate=owner_est,
               reason_audit=dict(unresolved=sum(r["reason"] == "owner_broken_reason_unstated" for r in rows),
                                 owner_tag=sum(r["reason_source"] == "owner_tag" for r in rows)))
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(missing=res["missing"], gate=gate), indent=1))
    for g, e in rates.items():
        pc = lambda v: "n/a" if v is None else f"{v:.1%}"
        print(f"{g:22s} final {pc(e['final_rate'])} {[round(x * 100, 1) for x in e['final_rate_crossed_ci95']]} "
              f"gpt_raw {pc(e['gpt_raw_rate'])} jev {pc(e['jev_rate'])} disagree {pc(e['gpt_jev']['disagreement'])} "
              f"missing {e['missing']} reasons {e['reasons_primary']}")
    for c, e in owner_est.items():
        print("owner estimate", c, f"{e['rate']:.1%}", [round(x * 100, 1) for x in e["conservative_ci95"]], e["strata"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "report"])
    ap.add_argument("judge_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=45)
    a = ap.parse_args()
    {"prep": lambda: prep(a.judge_dir), "run": lambda: run(a.judge_dir, a.workers, a.minutes),
     "report": lambda: report(a.judge_dir)}[a.cmd]()


if __name__ == "__main__":
    main()
