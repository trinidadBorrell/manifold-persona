"""E3 Part 2 context controls: analysis and report (THRESHOLDS.md, change of 16/10/2026).

Uses part2_analysis.py's subset, classifier, splits (same seed, same row order) and bootstrap draws, so every
feature set is scored on the same splits and the paired differences use the same draws. No new permutation null.

  analyse  per condition, hidden state 17: decoding from R (stored reply state), c0, c0first, c1, c2, c3 and a
           joint TF-IDF + length + stop-reason baseline; crossed-bootstrap CIs; paired R - c2 and R - c3.
           ex2_other: donor decoding, within-donor target accuracy and within-target donor accuracy (the 20-way
           probabilities restricted to the 3 roles linked to the true donor or target; chance 1/3).
           Sweep: all hidden states for c0, c1, c2, c3 on the first 10 splits, point estimates only.
  report   merges the part2_analysis outputs, the context outputs and Part 1 (naming_v2.json): Holm once over the
           4 primary decisions, then the wording rule.

    OMP_NUM_THREADS=1 python part2_context.py analyse --run run_main --jobs 20
    OMP_NUM_THREADS=1 python part2_context.py analyse --run run_rep --jobs 20
    python part2_context.py report --part2 p2_desc/run_main_part2.json p2_ex2/run_main_part2.json ... \
        --rep p2_rep/run_rep_part2.json --context ctx_main.json --context_rep ctx_rep.json
    python part2_context.py selftest
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import part2_analysis as p2

HERE = Path(__file__).resolve().parent
CTX = ("c0", "c0first", "c1", "c2", "c3")
SWEEP = ("c0", "c1", "c2", "c3")
N_SWEEP = 10
FLOOR = 0.10
DONORS = {r: [p2.ROLES[(k + s) % len(p2.ROLES)] for s in (6, 10, 14)] for k, r in enumerate(p2.ROLES)}  # gen_extract


# ---------------------------------------------------------------- scoring

def score(F, y, qn, splits, draws, featurize=None):
    acc, preds = p2.decode(F, y, splits, featurize=featurize)
    C = p2.cell_acc(y, qn, preds, len(p2.ROLES))
    bt = np.array([p2.draw_stat(C, rr, qq) for rr, qq in draws])
    return dict(acc_mean_50=acc, ci95=p2.ci(bt)), bt


def paired(a, b, bt_a, bt_b):
    return dict(diff=a["acc_mean_50"] - b["acc_mean_50"], ci95=p2.ci(bt_a - bt_b))


def fit_proba(Xtr, ytr, Xte):
    """The frozen classifier (part2_analysis.fit_predict), returning class probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    if list(clf.classes_) != list(range(len(p2.ROLES))):
        raise p2.AnalysisError("a role is missing from a training fold")
    return clf.predict_proba(sc.transform(Xte))


def restricted(F, y, cand, splits, draws, qn, jobs):
    """Accuracy when the 20-way prediction is restricted to each row's 3 candidate roles (chance 1/3)."""
    from joblib import Parallel, delayed
    P = Parallel(n_jobs=jobs)(delayed(fit_proba)(F[tr], y[tr], F[te]) for tr, te in splits)
    preds, accs = [], []
    for (tr, te), pr in zip(splits, P):
        c = cand[te]
        p = c[np.arange(len(te)), np.argmax(np.take_along_axis(pr, c, 1), 1)]
        preds.append((te, p))
        accs.append(p2.macro_acc(y[te], p, len(p2.ROLES)))
    C = p2.cell_acc(y, qn, preds, len(p2.ROLES))
    bt = np.array([p2.draw_stat(C, rr, qq) for rr, qq in draws])
    return dict(acc_mean_50=float(np.mean(accs)), ci95=p2.ci(bt), chance=1 / 3)


def _sweep_task(path, rows, L, y, tr, te):
    F = np.asarray(np.load(path, mmap_mode="r")[rows, L], dtype=np.float32)
    p = p2.fit_predict(F[tr], y[tr], F[te])
    return p2.macro_acc(y[te], p, len(p2.ROLES))


def sweep(paths, rows, n_hs, y, splits, jobs):
    from joblib import Parallel, delayed
    tasks = [(f, L, k) for f in SWEEP for L in range(n_hs) for k in range(len(splits))]
    accs = Parallel(n_jobs=jobs, batch_size=1)(delayed(_sweep_task)(paths[f], rows, L, y, *splits[k])
                                                for f, L, k in tasks)
    out = {f: [[] for _ in range(n_hs)] for f in SWEEP}
    for (f, L, k), a in zip(tasks, accs):
        out[f][L].append(a)
    return {f: [float(np.mean(v)) for v in out[f]] for f in SWEEP}


# ---------------------------------------------------------------- analyse

def load_context(run_dir):
    ctx = run_dir / "context"
    g = json.load(open(ctx / "gate.json"))
    if not g["passed"]:
        raise SystemExit("the C0 gate failed; the context features cannot be used as controls")
    import hashlib
    pos = np.load(ctx / "ctx_rows.npy")
    if hashlib.sha256(np.array(pos.tolist()).tobytes()).hexdigest() != g["rows_sha256"]:
        raise SystemExit("gate.json was written for other rows; re-run the gate")
    man = json.load(open(ctx / "manifest.json"))
    if man["run_fingerprint"] != (run_dir / "fingerprint.txt").read_text().strip():
        raise SystemExit("the context features come from another run")
    paths = {f: ctx / f"ctx_{f}.npy" for f in CTX}
    from context_extract import sha256
    for f, p in paths.items():
        if sha256(p) != man["files"][p.name]:
            raise SystemExit(f"{p.name} does not match the SHA-256 in manifest.json")
    return pos, paths, man


def analyse_condition(rows, X, cond, pos, paths, n_hs, jobs):
    idx = p2.subset(rows, cond)
    sub = [rows[k] for k in idx]
    y = np.array([p2.ROLES.index(r["role"]) for r in sub])
    qn = np.array([r["question_idx"] for r in sub])
    wn = np.array([r["wording"] for r in sub])
    p2.check_design(y, qn, wn)
    inv = {int(k): j for j, k in enumerate(pos)}
    if any(int(k) not in inv for k in idx):
        raise p2.AnalysisError(f"{cond}: rows without context features")
    crow = np.array([inv[int(k)] for k in idx])
    splits = p2.make_splits(y, qn, wn, np.random.default_rng(p2.SEED), p2.N_SPLITS)
    draws = p2.boot_draws(len(p2.ROLES), len(np.unique(qn)), np.random.default_rng(p2.SEED + 1))
    F = {"R": p2.finite(X[p2.LAYER][idx], "stored reply states")}
    for f in CTX:
        F[f] = p2.finite(np.asarray(np.load(paths[f], mmap_mode="r")[crow, p2.LAYER], dtype=np.float32), f)
    res, bts = {"cond": cond, "n_replies": len(idx), "layer": p2.LAYER}, {}
    for f, A in F.items():
        res[f], bts[f] = score(A, y, qn, splits, draws)
    tf, Lf = p2.tfidf_featurizer([r["reply"] for r in sub]), p2.length_feats(sub)

    def joint(tr, te):
        a, b = tf(tr, te)
        return np.hstack([a, Lf[tr]]), np.hstack([b, Lf[te]])
    res["text_joint_tfidf_length_stop"], bts["joint"] = score(None, y, qn, splits, draws, featurize=joint)
    res["R_minus_c2"] = paired(res["R"], res["c2"], bts["R"], bts["c2"])
    res["R_minus_c3"] = paired(res["R"], res["c3"], bts["R"], bts["c3"])
    res["c2_minus_joint_text"] = paired(res["c2"], res["text_joint_tfidf_length_stop"], bts["c2"], bts["joint"])
    if cond == "ex2_other":
        yd = np.array([p2.ROLES.index(r["example_role"]) for r in sub])
        ix = {r: k for k, r in enumerate(p2.ROLES)}
        targets_of = {ix[d]: sorted(ix[t] for t in p2.ROLES if d in DONORS[t]) for d in p2.ROLES}
        cand_t = np.array([targets_of[d] for d in yd])                      # targets linked to the true donor
        cand_d = np.array([[ix[d] for d in DONORS[p2.ROLES[t]]] for t in y])  # donors of the true target
        if not (np.all((cand_t == y[:, None]).sum(1) == 1) and np.all((cand_d == yd[:, None]).sum(1) == 1)):
            raise p2.AnalysisError("donor design does not match DONORS")
        res["ex2_other"] = {}
        for f in ("R", "c1", "c2", "c3"):
            dd, _ = score(F[f], yd, qn, splits, draws)
            res["ex2_other"][f] = dict(donor_decoding=dd,
                                       within_donor_target=restricted(F[f], y, cand_t, splits, draws, qn, jobs),
                                       within_target_donor=restricted(F[f], yd, cand_d, splits, draws, qn, jobs))
    res["sweep_first10"] = sweep(paths, crow, n_hs, y, splits[:N_SWEEP], jobs)
    return res


def analyse(a):
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1 (one BLAS thread per worker)")
    p2.JOBS = a.jobs
    run_dir = HERE / a.run
    rows, X = p2.load(run_dir)
    pos, paths, man = load_context(run_dir)
    if man["limit"]:
        raise SystemExit("the context features come from a --limit smoke run")
    conds = [c for c in p2.CONDS if any(r["cond"] == c for r in rows)] if a.conds is None else a.conds
    out = {"_meta": dict(run=a.run, layer=p2.LAYER, n_splits=p2.N_SPLITS, n_sweep=N_SWEEP, seed=p2.SEED,
                         context_manifest=man)}
    t = time.time()
    for c in conds:
        out[c] = analyse_condition(rows, X, c, pos, paths, man["n_hidden_states"], a.jobs)
        json.dump(out, open(a.out, "w"), indent=1)
        print(c, "done", round(time.time() - t), "s", flush=True)


# ---------------------------------------------------------------- report

def holm(p):
    keys = sorted(p, key=p.get)
    m, run, adj = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def merge(files):
    out = {}
    for f in files:
        for k, v in json.load(open(f)).items():
            if k == "_meta":
                continue
            if k in out:
                raise SystemExit(f"condition {k} appears in two files")
            out[k] = v
    return out


def wording(ctx):
    c2, c3 = ctx["c2"]["ci95"], ctx["c3"]["ci95"]
    s = [f"context-conditioned role decoding {ctx['R']['acc_mean_50']:.3f} {fmt(ctx['R']['ci95'])} "
         f"(prompt state C1 {ctx['c1']['acc_mean_50']:.3f} {fmt(ctx['c1']['ci95'])})"]
    s.append(f"the role can also be decoded from the reply text alone, as the model reads it (C2 "
             f"{ctx['c2']['acc_mean_50']:.3f} {fmt(c2)})" if c2[0] >= FLOOR else
             f"C2 did not meet the 10% floor ({ctx['c2']['acc_mean_50']:.3f} {fmt(c2)})")
    if c3[0] >= FLOOR:
        s.append(f"the prompt context alone carries decodable role information (C3 {ctx['c3']['acc_mean_50']:.3f} "
                 f"{fmt(c3)})")
    if "ex2_other" in ctx:  # each donor links to 3 targets, so 20-way target accuracy can come from the donor
        for f in ("R", "c2"):
            v = ctx["ex2_other"][f]
            t, d = v["within_donor_target"], v["within_target_donor"]
            s.append(f"{f}: within-donor target {t['acc_mean_50']:.3f} {fmt(t['ci95'])}, within-target donor "
                     f"{d['acc_mean_50']:.3f} {fmt(d['ci95'])} (chance 0.333; target information beyond the donor "
                     f"link {'detected' if t['ci95'][0] > 1 / 3 else 'not detected'})")
    return "; ".join(s)


def fmt(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"  # 4 places: a lower bound of 0.0998 must not print as 0.100


def report(a):
    P = merge(a.part2)
    Q = merge(a.context)
    naming = json.load(open(a.naming))
    raw = {f"plays_{c}": naming["holm"]["raw"][f"plays_{c}"] for c in ("desc", "ex2")}
    raw.update({f"info_{c}": P[c]["decision"]["p_intersection_union"] for c in ("desc", "ex2")})
    adj = holm(raw)
    out = dict(holm=dict(raw=raw, adjusted=adj), primary={}, conditions={})
    for c in ("desc", "ex2"):
        plays = {m: bool(r["both_judges"]["plays"] and adj[f"plays_{c}"] < 0.01)
                 for m, r in naming[f"run_main/{c}"]["results"].items()}
        info = bool(P[c]["decision"]["information_present"] and adj[f"info_{c}"] < 0.01)
        out["primary"][c] = dict(plays=plays, information=info)
    for c in P:
        d = P[c]["decision"]
        passed = out["primary"][c]["information"] if c in out["primary"] else d["information_present"]
        out["conditions"][c] = dict(information_passed=passed, exploratory=c not in out["primary"],
                                    decoding=P[c]["decoding"], gain=P[c]["gain_over_best_text"],
                                    map_interpretable=d["geometry_interpretable"],
                                    wording=wording(Q[c]) if c in Q else "no context controls",
                                    pass_note=None if passed else "did not pass the frozen Part 2 rules")
    if a.rep:
        R, QR = merge(a.rep), merge(a.context_rep) if a.context_rep else {}
        out["replication"] = {c: dict(main=P[c]["decoding"]["acc_mean_50"], rep=R[c]["decoding"]["acc_mean_50"],
                                      main_ci=P[c]["decoding"]["ci95"], rep_ci=R[c]["decoding"]["ci95"],
                                      main_decision=P[c]["decision"]["information_present"],
                                      rep_decision=R[c]["decision"]["information_present"],
                                      rep_c2=QR[c]["c2"] if c in QR else None)
                              for c in ("desc", "ex2") if c in R}
    json.dump(out, open(a.out, "w"), indent=1)
    for c, v in out["conditions"].items():
        print(f"{c}: passed={v['information_passed']} | {v['wording']}")
    print("holm", {k: round(v, 4) for k, v in adj.items()})


# ---------------------------------------------------------------- self-test

def selftest(a):
    from context_extract import gate_cell
    rng = np.random.default_rng(0)
    # gate: tiny numerical noise passes; a copy without role differences fails the centred check
    S = rng.normal(size=(300, 64)) + 5.0
    S[:, :8] += np.repeat(rng.normal(size=(20, 8)) * 2, 15, 0)
    assert gate_cell(S, S + rng.normal(size=S.shape) * 1e-3)["pass"]
    flat = S.copy()
    flat[:, :8] = flat[:, :8].mean(0)
    assert not gate_cell(S, flat + rng.normal(size=S.shape) * 1e-3)["pass"], "gate must catch lost role signal"
    # wording: context-only signal vs a signal the reply carries
    p2.JOBS = a.jobs
    ctx_X, y, q, w = p2.synth(np.random.default_rng(1), 0.6, 0.0)
    noise = p2.synth(np.random.default_rng(2), 0.0, 0.0)[0]
    splits = p2.make_splits(y, q, w, np.random.default_rng(3), 20)
    draws = p2.boot_draws(20, 18, np.random.default_rng(4), 300)
    got = {}
    for name, A in (("R", ctx_X + noise * 0.5), ("c1", ctx_X), ("c3", ctx_X), ("c2_none", noise),
                    ("c2_reply", ctx_X)):
        got[name], _ = score(A, y, q, splits, draws)
    base = dict(R=got["R"], c1=got["c1"], c3=got["c3"])
    w_none = wording(dict(base, c2=got["c2_none"]))
    w_reply = wording(dict(base, c2=got["c2_reply"]))
    assert "did not meet" in w_none and "alone carries" in w_none, w_none
    assert "also be decoded" in w_reply, w_reply
    assert got["c2_none"]["ci95"][0] < 0.05 < got["c2_none"]["ci95"][1], got["c2_none"]
    # within-donor: features that carry only the donor give within-donor target accuracy near 1/3
    nR, ix = 20, {r: k for k, r in enumerate(p2.ROLES)}
    yt, yd, qq, ww = [], [], [], []
    for t in range(nR):
        for d in DONORS[p2.ROLES[t]]:
            for qi in range(18):
                for wi in range(5):
                    yt.append(t), yd.append(ix[d]), qq.append(qi), ww.append(wi)
    yt, yd, qq, ww = map(np.array, (yt, yd, qq, ww))
    mu = np.random.default_rng(5).normal(size=(nR, 32)) * 0.8
    Fd = (mu[yd] + np.random.default_rng(6).normal(size=(len(yt), 32))).astype(np.float32)
    sp = p2.make_splits(yt, qq, ww, np.random.default_rng(7), 10)
    dr = p2.boot_draws(nR, 18, np.random.default_rng(8), 200)
    targets_of = {ix[d]: sorted(ix[t] for t in p2.ROLES if d in DONORS[t]) for d in p2.ROLES}
    cand_t = np.array([targets_of[d] for d in yd])
    cand_d = np.array([[ix[d] for d in DONORS[p2.ROLES[t]]] for t in yt])
    wt = restricted(Fd, yt, cand_t, sp, dr, qq, a.jobs)
    wd = restricted(Fd, yd, cand_d, sp, dr, qq, a.jobs)
    print("within-donor target", wt, "within-target donor", wd)
    assert abs(wt["acc_mean_50"] - 1 / 3) < 0.06, wt
    assert wd["ci95"][0] > 0.5, wd
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}) == {"a": 0.004, "b": 0.06, "c": 0.08, "d": 0.5}
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    an = sp.add_parser("analyse")
    an.add_argument("--run", default="run_main")
    an.add_argument("--conds", nargs="+", default=None)
    an.add_argument("--jobs", type=int, default=-1)
    an.add_argument("--out", default=None)
    rp = sp.add_parser("report")
    rp.add_argument("--part2", nargs="+", required=True)
    rp.add_argument("--context", nargs="+", required=True)
    rp.add_argument("--rep", nargs="+", default=None)
    rp.add_argument("--context_rep", nargs="+", default=None)
    rp.add_argument("--naming", default=str(HERE / "results" / "naming_v2.json"))
    rp.add_argument("--out", default=str(HERE / "results" / "part2_report.json"))
    st = sp.add_parser("selftest")
    st.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "analyse":
        a.out = a.out or str(HERE / f"part2_context_{a.run}.json")
        return analyse(a)
    if a.cmd == "report":
        return report(a)
    return selftest(a)


if __name__ == "__main__":
    sys.exit(main())
