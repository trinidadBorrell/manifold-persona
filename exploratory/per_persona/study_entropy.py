"""Does the semantic entropy of a role's outputs track its manifold geometry?

Plan: `plans/2026-08-11-semantic-entropy-vs-geometry.md` (approved 2026-08-11,
amendment A1: every experiment runs regardless of the gates).

THE QUESTION
------------
Every metric in the per-persona panel is computed from activation vectors. None
of them looks at what the model actually said. The five points in a
(role, question) cell differ only in how the role instruction was phrased, so if
those five answers mean the same thing the five vectors should sit close
together. This asks whether the text-side measure and the geometry move
together. It does NOT test whether one explains the other -- that is the next
plan, and no output here may say "explains".

WHAT IT RUNS
------------
  0  predicate validation + degeneracy gate  (judge agreement, must-merge,
     must-split, cluster-count distribution).  Reported first. Under A1 it does
     not stop the run, but everything computed behind a failed gate is labelled.
  1  entropy panel: cluster all 11,040 groups, per-question centring, aggregate
  2  correlation ladder: E_role x 26 targets x 3 control rungs, BH, bootstrap,
     shuffle null, length baseline.  The decider lives here.
  3  group level: SE(role,q) vs the spread of the same five activation points,
     computed WITHIN each role then averaged -- never pooled across roles
  4  null diagnostics, run whatever the outcome: cluster counts, entropy vs
     truncation, entropy vs length, centred vs uncentred

Usage:
    .venv/bin/python exploratory/per_persona/study_entropy.py \
        --outdir output/per-persona-entropy/2026-08-11
    # reuse a completed NLI pass (45 min) while iterating on the analysis
    .venv/bin/python exploratory/per_persona/study_entropy.py --outdir ... --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import entropy as E                                    # noqa: E402
from stats_utils import bh_fdr, boot_ci, linfit, partial_corr_multi  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CLOUD = REPO / "data" / "embeddings_roles_resp240"
TIERS = REPO / "output" / "per_persona_axis_centroid_11-Aug-2026" / "question_tiers.json"
PANEL = (REPO / "output" / "per_persona_axis_centroid_11-Aug-2026" / "q40"
         / "data" / "per_role_panel_L19.csv")
RESP40 = REPO / "data" / "embeddings_roles_resp40"

SEED = 0
N_JUDGE = 500
N_BOOT = 2000
N_SHUFFLE = 1000
MAX_NEW_TOKENS = 128          # from the cloud manifest; defines trunc_rate
DECIDER_METRIC = "MLE"
DECIDER_BAR = 0.30
RESOLUTION_FLOOR = 0.08       # min between-role SD of E_role for an r to mean anything

# Gate thresholds (plan, Controls). A1: computed and reported, never a stop.
GATE_KAPPA = 0.20
GATE_FALSE_SPLIT = 0.10
GATE_FALSE_MERGE = 0.10
GATE_DEGENERATE = 0.90

RUNGS = [("raw", []),
         ("ctrl_scale", ["log_var", "mean_norm"]),
         ("ctrl_all", ["log_var", "mean_norm", "mean_tokens", "trunc_rate"])]

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_PROMPT = (
    "We are evaluating answers to the question: {q}\n"
    "Here are two possible answers:\n"
    "Possible Answer 1: {a}\n"
    "Possible Answer 2: {b}\n"
    "Do these two answers mean the same thing, so that each one entails the "
    "other? Respond with only Yes or No.\nAnswer:")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).decode().strip())
        return sha, dirty
    except Exception:
        return None, None


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_tier():
    """The 40-question tier of the 240q cloud, plus the provenance check that
    makes it comparable to the published result."""
    meta = pd.read_parquet(CLOUD / "metadata.parquet")
    q40 = set(json.load(open(TIERS))["tiers"]["40"])
    d = meta[meta.question_idx.isin(q40)].copy()
    d = d.sort_values(["role", "question_idx", "instruction_idx"]).reset_index(drop=True)

    prov = {"n_rows": int(len(d)), "n_roles": int(d.role.nunique()),
            "n_questions": int(d.question_idx.nunique()),
            "n_groups": int(d.groupby(["role", "question_idx"]).ngroups)}
    # The tier must BE the resp40 question set, and the responses must be the
    # same bytes -- otherwise the entropy is being compared to a geometry panel
    # built from different text.
    try:
        a = pd.read_parquet(RESP40 / "metadata.parquet")
        prov["q40_texts_match_resp40"] = bool(set(d.question.unique()) == set(a.question.unique()))
        m = a.merge(d, on=["role", "instruction_idx", "question"], suffixes=("_40", "_240"))
        prov["n_shared_rows"] = int(len(m))
        prov["response_identical_frac"] = float((m.response_40 == m.response_240).mean())
    except Exception as exc:                                    # noqa: BLE001
        prov["resp40_check_error"] = str(exc)
    prov["n_null_system"] = int(d.system.isna().sum())
    prov["duplicate_question_texts"] = int(d.question.nunique() != d.question_idx.nunique())
    return d, prov


def build_groups(d):
    """One record per (role, question): the usable answers, in instruction order.

    Exclusions, fixed before looking at any correlation: an answer under 5 words
    is dropped and M shrinks; a group left with M < 3 is dropped entirely.
    """
    groups, dropped_short, dropped_group = [], 0, 0
    for (role, qi), g in d.groupby(["role", "question_idx"], sort=True):
        g = g.sort_values("instruction_idx")
        keep = g[g.response.map(E.usable)]
        dropped_short += len(g) - len(keep)
        if len(keep) < E.M_MIN:
            dropped_group += 1
            continue
        groups.append({"role": role, "question_idx": int(qi),
                       "question": g.question.iloc[0],
                       "instr": keep.instruction_idx.tolist(),
                       "answers": [E.first_words(t) for t in keep.response],
                       "row_ids": keep.index.tolist(),
                       "M": int(len(keep))})
    return groups, dropped_short, dropped_group


# --------------------------------------------------------------------------- #
# experiment 0 + 1 -- the NLI pass, clustering, entropy
# --------------------------------------------------------------------------- #
def run_nli(groups, pred, run_dir, resume=False):
    """Entailment matrix per group. One batched pass over every ordered pair.

    Cached to disk because it is the expensive step (~220,800 forwards, ~45 min)
    and every downstream experiment reads it.
    """
    cache = run_dir / "data" / "entail_cache.npz"
    if resume and cache.exists():
        z = np.load(cache, allow_pickle=True)
        log(f"resumed entailment matrices from {cache}")
        return list(z["mats"])

    prem, hyp, owner = [], [], []
    for gi, g in enumerate(groups):
        q, ans = g["question"], g["answers"]
        for i, j in E.pair_index(len(ans)):
            prem.append(f"{q} {ans[i]}")
            hyp.append(f"{q} {ans[j]}")
            owner.append((gi, i, j))
    log(f"entailment: {len(prem):,} ordered pairs over {len(groups):,} groups")

    t0 = time.time()
    flags = np.zeros(len(prem), dtype=bool)
    step = 25_000
    for s in range(0, len(prem), step):
        e = min(s + step, len(prem))
        flags[s:e] = pred.entails_batch(prem[s:e], hyp[s:e])
        done, rate = e, e / max(time.time() - t0, 1e-9)
        log(f"  {done:,}/{len(prem):,} ({100*done/len(prem):.1f}%) "
            f"{rate:.0f} pairs/s eta {(len(prem)-done)/max(rate,1e-9)/60:.1f} min")

    mats = [np.zeros((g["M"], g["M"]), dtype=bool) for g in groups]
    for (gi, i, j), f in zip(owner, flags):
        mats[gi][i, j] = f
    np.savez_compressed(cache, mats=np.array(mats, dtype=object))
    log(f"entailment done in {(time.time()-t0)/60:.1f} min -> {cache}")
    return mats


def score_groups(groups, mats):
    """Cluster each group and score it. The clustering is entropy.cluster_greedy,
    which is the paper's algorithm unchanged."""
    rows = []
    for g, mat in zip(groups, mats):
        cl = E.cluster_greedy(mat)
        rows.append({"role": g["role"], "question_idx": g["question_idx"], "M": g["M"],
                     "n_clusters": len(cl), "SE": E.discrete_entropy(cl, g["M"]),
                     "SE_max": E.max_entropy(g["M"]),
                     "labels": json.dumps([sorted(c) for c in cl])})
    return pd.DataFrame(rows)


def judge_labels(groups, panel, run_dir, seed=SEED, n=N_JUDGE):
    """500 pairs labelled same/different meaning by a local LLM, using the
    paper's entailment framing. This is the positive control for the predicate:
    without it, a measure that clusters nothing and a measure that clusters
    everything are indistinguishable from their outputs alone.

    Judged by comparing the next-token logits of " Yes" and " No" -- no
    sampling, so the label is deterministic.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(seed)
    ap = panel.set_index("role").axis_proj
    # stratify across the axis so the control is not evaluated on one end of it
    order = ap.sort_values().index.tolist()
    bins = {r: min(9, i * 10 // len(order)) for i, r in enumerate(order)}
    by_bin = {b: [] for b in range(10)}
    for gi, g in enumerate(groups):
        if g["role"] in bins and g["M"] >= 2:
            by_bin[bins[g["role"]]].append(gi)
    picks = []
    for b in range(10):
        if by_bin[b]:
            picks += list(rng.choice(by_bin[b], size=min(n // 10, len(by_bin[b])), replace=False))
    picks = picks[:n]

    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    mod = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.float16)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mod.to(dev).eval()
    yes = tok(" Yes", add_special_tokens=False).input_ids[-1]
    no = tok(" No", add_special_tokens=False).input_ids[-1]

    rows = []
    t0 = time.time()
    for k, gi in enumerate(picks):
        g = groups[gi]
        i, j = rng.choice(len(g["answers"]), size=2, replace=False)
        msg = JUDGE_PROMPT.format(q=g["question"], a=g["answers"][i], b=g["answers"][j])
        chat = tok.apply_chat_template([{"role": "user", "content": msg}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = mod(**enc).logits[0, -1]
        rows.append({"group": int(gi), "role": g["role"],
                     "question_idx": g["question_idx"], "i": int(i), "j": int(j),
                     "judge_same": bool(logits[yes] > logits[no]),
                     "margin": float(logits[yes] - logits[no])})
        if (k + 1) % 100 == 0:
            log(f"  judge {k+1}/{len(picks)} ({(time.time()-t0)/(k+1):.2f}s each)")
    del mod
    return pd.DataFrame(rows)


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def control_pairs(groups, mats, d, pred, run_dir, seed=SEED, n=400):
    """Must-merge and must-split controls.

    must-merge: an answer against itself, and near-duplicate pairs (token
      Jaccard > 0.9). These MUST land in one cluster; the rate they do not is
      the predicate's false-split rate.
    must-split: two answers by the same role to DIFFERENT questions. These must
      not merge; the rate they do is the false-merge rate.
    """
    rng = np.random.default_rng(seed)
    out = {}

    # -- must-merge: identical text --------------------------------------- #
    gi = rng.choice(len(groups), size=min(n, len(groups)), replace=False)
    prem = [f"{groups[k]['question']} {groups[k]['answers'][0]}" for k in gi]
    same = pred.entails_batch(prem, prem)
    out["self_entail_rate"] = float(same.mean())
    out["false_split_identical"] = float(1 - same.mean())

    # -- must-merge: near-duplicates within a group ------------------------ #
    near = []
    for k, g in enumerate(groups):
        toks = [set(a.lower().split()) for a in g["answers"]]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                u = len(toks[i] | toks[j])
                if u and len(toks[i] & toks[j]) / u > 0.9:
                    near.append((k, i, j))
    out["n_near_duplicate_pairs"] = len(near)
    if near:
        sel = [near[k] for k in rng.choice(len(near), size=min(n, len(near)), replace=False)]
        merged = [bool(mats[k][i, j] and mats[k][j, i]) for k, i, j in sel]
        out["false_split_near_duplicate"] = float(1 - np.mean(merged))
    else:
        out["false_split_near_duplicate"] = None

    # -- must-split: same role, different questions ------------------------ #
    by_role = {}
    for k, g in enumerate(groups):
        by_role.setdefault(g["role"], []).append(k)
    roles = [r for r, v in by_role.items() if len(v) >= 2]
    pa, pb = [], []
    for _ in range(n):
        r = roles[rng.integers(len(roles))]
        k1, k2 = rng.choice(by_role[r], size=2, replace=False)
        g1, g2 = groups[k1], groups[k2]
        # judged in the context of g1's question, as the algorithm would
        pa.append(f"{g1['question']} {g1['answers'][0]}")
        pb.append(f"{g1['question']} {g2['answers'][0]}")
    f1 = pred.entails_batch(pa, pb)
    f2 = pred.entails_batch(pb, pa)
    out["false_merge_cross_question"] = float((f1 & f2).mean())
    return out


# --------------------------------------------------------------------------- #
# experiment 2 -- the ladder
# --------------------------------------------------------------------------- #
def build_ladder(df, targets, predictor, rng, gate_ok):
    rows = []
    x = df[predictor].to_numpy(float)
    for name, ctrl in RUNGS:
        Z = df[ctrl].to_numpy(float) if ctrl else None
        for t in targets:
            y = df[t].to_numpy(float)
            r, p = partial_corr_multi(x, y, Z)
            lo, hi = boot_ci(x, y, Z, rng, n_boot=N_BOOT)
            rows.append({"target": t, "rung": name, "predictor": predictor,
                         "r": r, "p": p, "ci_lo": lo, "ci_hi": hi,
                         "spearman": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
                         "n": int(len(x)),
                         "confirmatory": bool(t == DECIDER_METRIC and name == "ctrl_scale"
                                              and predictor == "E_role"),
                         "gate": "PASS" if gate_ok else "FAILED"})
    out = pd.DataFrame(rows)
    for name, _ in RUNGS:                       # BH within (rung) across targets
        m = out.rung == name
        out.loc[m, "q_within_rung"] = bh_fdr(out.loc[m, "p"].to_numpy())
    out["q_global"] = bh_fdr(out["p"].to_numpy())
    return out


def shuffle_null(df, targets, rng, n_perm=N_SHUFFLE):
    """Null for the CORRELATION: permute role -> E_role and recompute the ladder.
    A 26-target panel will produce apparent correlations; this says how large."""
    x = df["E_role"].to_numpy(float)
    keep = {}
    for name, ctrl in RUNGS:
        Z = df[ctrl].to_numpy(float) if ctrl else None
        Y = df[targets].to_numpy(float)
        best = np.empty(n_perm)
        per = np.empty((n_perm, len(targets)))
        for b in range(n_perm):
            xs = rng.permutation(x)
            rs = [abs(partial_corr_multi(xs, Y[:, k], Z)[0]) for k in range(len(targets))]
            per[b] = rs
            best[b] = np.nanmax(rs)
        keep[name] = {"max_abs_r_p95": float(np.nanpercentile(best, 95)),
                      "per_target_p95": {t: float(np.nanpercentile(per[:, k], 95))
                                         for k, t in enumerate(targets)}}
    return keep


# --------------------------------------------------------------------------- #
# experiment 3 -- group level
# --------------------------------------------------------------------------- #
def group_geometry(d, groups):
    """Spread of the five activation points behind each group.

    participation_ratio = (sum lambda)^2 / sum lambda^2 of the group's own
    covariance -- the same statistic the panel uses globally, applied to five
    points. With M points at most M-1 eigenvalues are non-zero, so this is
    bounded by M-1; it is comparable ACROSS groups of equal M, which is why
    reduced-M groups are flagged.
    """
    X = np.load(CLOUD / "prompt_avg.npy", mmap_mode="r")
    rows = []
    for g in groups:
        P = np.asarray(X[g["row_ids"], 0, :], dtype=np.float32)
        Pc = P - P.mean(0)
        lam = np.linalg.svd(Pc, compute_uv=False) ** 2
        s = lam.sum()
        rows.append({"role": g["role"], "question_idx": g["question_idx"], "M": g["M"],
                     "grp_var": float(s),
                     "grp_pr": float(s * s / np.sum(lam ** 2)) if s > 0 else np.nan,
                     "grp_rms": float(np.sqrt(s / len(P)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resume", action="store_true", help="reuse a cached NLI pass")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-groups", type=int, default=None,
                    help="smoke test only — truncate the group list. A run with "
                         "this set is NOT the study and its outputs say so.")
    args = ap.parse_args()

    run_dir = Path(args.outdir)
    for sub in ("figures", "data", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timings, gates = {}, {}

    # ---- data --------------------------------------------------------- #
    t0 = time.time()
    d, prov = load_tier()
    log(f"tier loaded: {prov}")
    groups, dropped_short, dropped_group = build_groups(d)
    if args.max_groups:
        groups = groups[:args.max_groups]
        log(f"SMOKE TEST — truncated to {len(groups)} groups; this is not the study")
        prov["SMOKE_TEST"] = True
    prov.update({"dropped_short_answers": dropped_short,
                 "dropped_groups_M_lt_3": dropped_group,
                 "n_groups_used": len(groups),
                 "n_reduced_M": int(sum(g["M"] < 5 for g in groups))})
    panel = pd.read_csv(PANEL)
    log(f"panel: {panel.shape} from {PANEL}")
    timings["load"] = time.time() - t0

    # token counts -> the length covariates the ladder's third rung needs
    t0 = time.time()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    ntok = np.array([len(x) for x in tk(d.response.tolist(),
                                        add_special_tokens=False).input_ids])
    d["n_tokens"] = ntok
    lens = d.groupby("role").agg(mean_tokens=("n_tokens", "mean"),
                                 sd_tokens=("n_tokens", "std")).reset_index()
    lens["trunc_rate"] = d.assign(t=d.n_tokens >= MAX_NEW_TOKENS).groupby("role").t.mean().values
    timings["tokenize"] = time.time() - t0
    log(f"token counts: mean {ntok.mean():.1f}, trunc_rate "
        f"{(ntok >= MAX_NEW_TOKENS).mean():.3f}")

    # ---- experiment 0a: the NLI pass ---------------------------------- #
    t0 = time.time()
    pred = E.NLIPredicate(batch_size=args.batch_size)
    mats = run_nli(groups, pred, run_dir, resume=args.resume)
    timings["nli"] = time.time() - t0

    # ---- experiment 1: cluster and score ------------------------------ #
    t0 = time.time()
    grp = score_groups(groups, mats)
    grp.to_parquet(run_dir / "data" / "group_entropy_L19.parquet", index=False)
    cc = grp.n_clusters.value_counts(normalize=True).sort_index()
    gates["degeneracy"] = {
        "cluster_count_frac": {int(k): float(v) for k, v in cc.items()},
        "frac_all_split": float((grp.n_clusters == grp.M).mean()),
        "frac_all_merged": float((grp.n_clusters == 1).mean()),
        "mean_clusters": float(grp.n_clusters.mean()),
        "threshold": GATE_DEGENERATE}
    gates["degeneracy"]["pass"] = bool(
        gates["degeneracy"]["frac_all_split"] <= GATE_DEGENERATE
        and gates["degeneracy"]["frac_all_merged"] <= GATE_DEGENERATE)
    log(f"cluster counts: {gates['degeneracy']['cluster_count_frac']}")
    log(f"degeneracy gate: {'PASS' if gates['degeneracy']['pass'] else 'FAIL'}")

    # per-question centring, then the per-role aggregate
    grp["SE_centred"] = grp.SE - grp.groupby("question_idx").SE.transform("mean")
    per_role = grp.groupby("role").agg(
        E_role=("SE_centred", "mean"), E_role_uncentred=("SE", "mean"),
        E_role_sd=("SE", "std"), mean_clusters=("n_clusters", "mean"),
        frac_collapsed=("n_clusters", lambda s: float((s == 1).mean())),
        frac_split=("n_clusters", lambda s: float((s == 5).mean())),
        n_groups=("SE", "size")).reset_index()
    per_role = per_role.merge(lens, on="role", how="left")
    per_role.to_csv(run_dir / "data" / "per_role_entropy_L19.csv", index=False)
    log(f"E_role: mean {per_role.E_role.mean():.4f} sd {per_role.E_role.std():.4f} "
        f"(resolution floor {RESOLUTION_FLOOR})")
    gates["resolution"] = {"E_role_sd": float(per_role.E_role.std()),
                           "floor": RESOLUTION_FLOOR,
                           "pass": bool(per_role.E_role.std() >= RESOLUTION_FLOOR)}
    timings["cluster"] = time.time() - t0

    # ---- experiment 0b: judge + must-merge/must-split ------------------ #
    t0 = time.time()
    ctrl = control_pairs(groups, mats, d, pred, run_dir)
    gates["must_merge_must_split"] = dict(
        ctrl, pass_=bool(ctrl["false_split_identical"] <= GATE_FALSE_SPLIT
                         and ctrl["false_merge_cross_question"] <= GATE_FALSE_MERGE))
    log(f"predicate controls: {ctrl}")
    del pred                                   # free MPS before the judge loads

    jd = judge_labels(groups, panel, run_dir)
    pred_same = []
    for _, r in jd.iterrows():
        m = mats[int(r.group)]
        pred_same.append(bool(m[int(r.i), int(r.j)] and m[int(r.j), int(r.i)]))
    jd["pred_same"] = pred_same
    jd.to_csv(run_dir / "data" / "judge_pairs_L19.csv", index=False)
    kappa = cohen_kappa(jd.pred_same, jd.judge_same)
    gates["judge_agreement"] = {
        "kappa": kappa, "n": int(len(jd)), "threshold": GATE_KAPPA,
        "agreement": float((jd.pred_same == jd.judge_same).mean()),
        "judge_same_rate": float(jd.judge_same.mean()),
        "predicate_same_rate": float(jd.pred_same.mean()),
        "pass": bool(np.isfinite(kappa) and kappa >= GATE_KAPPA)}
    log(f"judge: kappa {kappa:.3f}, agreement "
        f"{gates['judge_agreement']['agreement']:.3f}, "
        f"judge says same {jd.judge_same.mean():.3f}, predicate says same "
        f"{np.mean(pred_same):.3f}")
    timings["gates"] = time.time() - t0

    gate_ok = all(v.get("pass", v.get("pass_", False)) for v in gates.values())
    log(f"=== GATES {'PASS' if gate_ok else 'FAILED'} — running everything anyway "
        f"(amendment A1) ===")
    json.dump(gates, open(run_dir / "data" / "predicate_validation_L19.json", "w"),
              indent=2, default=float)

    # ---- experiment 2: the ladder ------------------------------------- #
    t0 = time.time()
    import metrics as MT
    targets = [c for c in MT.PANEL_COLS if c in panel.columns] + ["axis_proj", "cos_centroid"]
    df = panel.merge(per_role, on="role", how="inner")
    df = df[df.role != "default"].dropna(subset=targets + ["E_role"] + RUNGS[-1][1])
    log(f"ladder: {len(df)} roles (default excluded), {len(targets)} targets, "
        f"{len(RUNGS)} rungs")

    rng = np.random.default_rng(SEED)
    ladder = pd.concat([build_ladder(df, targets, p, rng, gate_ok)
                        for p in ("E_role", "mean_tokens", "trunc_rate")],
                       ignore_index=True)
    null = shuffle_null(df, targets, np.random.default_rng(SEED))
    ladder["shuffle_p95"] = [null[r]["per_target_p95"].get(t, np.nan)
                             for t, r in zip(ladder.target, ladder.rung)]
    ladder["beats_shuffle"] = ladder.r.abs() > ladder.shuffle_p95
    ladder.to_csv(run_dir / "data" / "ladder_entropy_L19.csv", index=False)
    json.dump(null, open(run_dir / "data" / "shuffle_null_L19.json", "w"), indent=2)

    dec = ladder[(ladder.predictor == "E_role") & (ladder.target == DECIDER_METRIC)
                 & (ladder.rung == "ctrl_scale")].iloc[0]
    log(f"DECIDER  E_role vs {DECIDER_METRIC} | log_var, mean_norm: "
        f"r = {dec.r:+.3f} [{dec.ci_lo:+.3f}, {dec.ci_hi:+.3f}] "
        f"p = {dec.p:.2e}  bar |r| >= {DECIDER_BAR}  -> "
        f"{'CLEARS' if abs(dec.r) >= DECIDER_BAR else 'below bar'}"
        f"{'' if gate_ok else '   [GATES FAILED — does not decide]'}")
    timings["ladder"] = time.time() - t0

    # ---- experiment 3: group level ------------------------------------ #
    t0 = time.time()
    gg = group_geometry(d, groups)
    gg = gg.merge(grp[["role", "question_idx", "SE", "SE_centred", "n_clusters"]],
                  on=["role", "question_idx"])
    gg.to_csv(run_dir / "data" / "group_geometry_L19.csv", index=False)
    within = []
    for role, g in gg.groupby("role"):
        if len(g) >= 10 and g.SE_centred.std() > 0 and g.grp_pr.std() > 0:
            within.append({"role": role,
                           "r_pr": float(np.corrcoef(g.SE_centred, g.grp_pr)[0, 1]),
                           "r_var": float(np.corrcoef(g.SE_centred, g.grp_var)[0, 1]),
                           "n": int(len(g))})
    wdf = pd.DataFrame(within)
    wdf.to_csv(run_dir / "data" / "within_role_corr_L19.csv", index=False)
    grp_summary = {
        "n_roles_with_corr": int(len(wdf)),
        "mean_within_r_pr": float(wdf.r_pr.mean()) if len(wdf) else None,
        "sd_within_r_pr": float(wdf.r_pr.std()) if len(wdf) else None,
        "mean_within_r_var": float(wdf.r_var.mean()) if len(wdf) else None,
        "frac_positive_pr": float((wdf.r_pr > 0).mean()) if len(wdf) else None,
        "t_test_mean_r_pr_vs_0": None}
    if len(wdf) > 2:
        from scipy import stats as st
        tt = st.ttest_1samp(wdf.r_pr.dropna(), 0.0)
        grp_summary["t_test_mean_r_pr_vs_0"] = {"t": float(tt.statistic),
                                                "p": float(tt.pvalue)}
    log(f"group level: mean within-role r(SE, group PR) = "
        f"{grp_summary['mean_within_r_pr']}")
    timings["group_level"] = time.time() - t0

    # ---- experiment 4: diagnostics ------------------------------------ #
    diag = {
        "entropy_vs_trunc_rate": linfit(per_role.trunc_rate, per_role.E_role),
        "entropy_vs_mean_tokens": linfit(per_role.mean_tokens, per_role.E_role),
        "uncentred_vs_centred": linfit(per_role.E_role_uncentred, per_role.E_role),
        "E_role_describe": {k: float(v) for k, v in per_role.E_role.describe().items()},
        "E_role_uncentred_describe": {k: float(v) for k, v
                                      in per_role.E_role_uncentred.describe().items()},
        "mean_clusters_describe": {k: float(v) for k, v
                                   in per_role.mean_clusters.describe().items()},
        "group_level": grp_summary}
    default_row = per_role[per_role.role == "default"]
    if len(default_row):
        pct = float((per_role.E_role < default_row.E_role.iloc[0]).mean())
        diag["default"] = {"E_role": float(default_row.E_role.iloc[0]),
                           "percentile_among_roles": pct}
    json.dump(diag, open(run_dir / "data" / "diagnostics_L19.json", "w"),
              indent=2, default=float)
    log(f"diagnostics: entropy vs trunc_rate r = "
        f"{diag['entropy_vs_trunc_rate']['r']:+.3f}, vs mean_tokens r = "
        f"{diag['entropy_vs_mean_tokens']['r']:+.3f}")

    # ---- manifest ------------------------------------------------------ #
    sha, dirty = git_sha()
    import scipy
    import sklearn
    import torch
    import transformers
    manifest = {
        "run_id": "2026-08-11-semantic-entropy-vs-geometry",
        "plan": "plans/2026-08-11-semantic-entropy-vs-geometry.md",
        "amendment": "A1 — every experiment runs regardless of the gates",
        "git_sha": sha, "git_dirty": dirty,
        "inputs": {"cloud": str(CLOUD), "tier_file": str(TIERS),
                   "panel": str(PANEL), "panel_sha256_16": file_hash(PANEL),
                   "cloud_manifest": json.load(open(CLOUD / "manifest.json"))},
        "provenance": prov,
        "params": {"tier": 40, "n_words": E.N_WORDS, "M_min": E.M_MIN,
                   "min_words": E.MIN_WORDS, "nli_model": E.NLI_MODEL,
                   "judge_model": JUDGE_MODEL, "judge_prompt": JUDGE_PROMPT,
                   "max_new_tokens": MAX_NEW_TOKENS, "rungs": RUNGS,
                   "decider": f"E_role vs {DECIDER_METRIC} | log_var, mean_norm",
                   "decider_bar": DECIDER_BAR, "n_boot": N_BOOT,
                   "n_shuffle": N_SHUFFLE, "n_judge": N_JUDGE,
                   "resolution_floor": RESOLUTION_FLOOR, "targets": targets},
        "seeds": {"master": SEED, "bootstrap": [0, N_BOOT - 1],
                  "shuffle": [0, N_SHUFFLE - 1], "judge_sample": SEED},
        "gates": gates, "gates_all_pass": gate_ok,
        "decider_result": {"r": float(dec.r), "p": float(dec.p),
                           "ci": [float(dec.ci_lo), float(dec.ci_hi)],
                           "clears_bar": bool(abs(dec.r) >= DECIDER_BAR),
                           "quotable_as_verdict": bool(gate_ok)},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__},
        "timings_sec": dict(timings, total=time.time() - t_start),
        "status": "completed",
        "exploratory": True,
    }
    json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=2, default=str)
    log(f"done in {(time.time()-t_start)/60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
