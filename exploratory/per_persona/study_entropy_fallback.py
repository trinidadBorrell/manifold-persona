"""The preregistered fallback predicate, run because the judge gate failed.

The plan (Controls) is unconditional: if a gate fails, switch the predicate to
`sentence-transformers/all-mpnet-base-v2` with tau chosen to MAXIMISE AGREEMENT
WITH THE JUDGE LABELS -- never by looking at a correlation with axis_proj -- and
re-run both gates on it. The judge gate did fail (kappa 0.151 < 0.20) and the
first pass of this run skipped the fallback entirely. This closes that.

What it does NOT do: replace the primary result. Under the plan the fallback is
a second reading of the same question, and if it also fails its gates the honest
outcome is that the measure is not validated by either predicate. That is a
finding, not a reason to keep searching for a third.

Usage:
    .venv/bin/python exploratory/per_persona/study_entropy_fallback.py \
        --outdir output/per-persona-entropy/2026-08-11
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import entropy as E                                            # noqa: E402
import study_entropy as S                                      # noqa: E402
from stats_utils import boot_ci, partial_corr_multi            # noqa: E402

TAU_GRID = np.round(np.arange(0.50, 0.991, 0.01), 3)


def cluster_by_sim(sim: np.ndarray, tau: float) -> list:
    """The paper's greedy loop with the entailment predicate replaced by a
    symmetric similarity threshold. Structure is identical: single pass, compare
    only to each cluster's first member, transitivity assumed."""
    clusters = [[0]]
    for m in range(1, sim.shape[0]):
        for c in clusters:
            if sim[c[0], m] >= tau:
                c.append(m)
                break
        else:
            clusters.append([m])
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run = Path(args.outdir)
    t_start = time.time()

    d, prov = S.load_tier()
    groups, _, _ = S.build_groups(d)
    S.log(f"{len(groups):,} groups")

    # ---- embed every answer once ---------------------------------------- #
    texts, owner = [], []
    for gi, g in enumerate(groups):
        for k, a in enumerate(g["answers"]):
            texts.append(a)
            owner.append((gi, k))
    S.log(f"embedding {len(texts):,} answers with {E.FALLBACK_MODEL}")
    emb = E.EmbedPredicate(tau=0.0, batch_size=args.batch_size)
    V = emb.encode(texts)                       # already L2-normalised
    del emb
    S.log(f"embedded -> {V.shape} in {(time.time()-t_start)/60:.1f} min")

    sims = []
    off = 0
    for g in groups:
        m = g["M"]
        P = V[off:off + m]
        sims.append(P @ P.T)                    # cosine, unit vectors
        off += m

    # ---- tau chosen against the JUDGE, never against axis_proj ---------- #
    jd = pd.read_csv(run / "data" / "judge_pairs_L19.csv")
    scan = []
    for tau in TAU_GRID:
        pred = [bool(sims[int(r.group)][int(r.i), int(r.j)] >= tau)
                for _, r in jd.iterrows()]
        scan.append({"tau": float(tau),
                     "kappa": S.cohen_kappa(pred, jd.judge_same),
                     "agreement": float(np.mean(np.array(pred) == jd.judge_same.values)),
                     "pred_same_rate": float(np.mean(pred))})
    scan = pd.DataFrame(scan)
    best = scan.loc[scan.kappa.idxmax()]
    tau = float(best.tau)
    S.log(f"tau* = {tau:.3f} (kappa {best.kappa:.3f}, agreement {best.agreement:.3f}, "
          f"predicate says same {best.pred_same_rate:.3f})")
    scan.to_csv(run / "data" / "fallback_tau_scan_L19.csv", index=False)

    # ---- gates under the fallback --------------------------------------- #
    rows = []
    for g, sim in zip(groups, sims):
        cl = cluster_by_sim(sim, tau)
        rows.append({"role": g["role"], "question_idx": g["question_idx"], "M": g["M"],
                     "n_clusters": len(cl), "SE": E.discrete_entropy(cl, g["M"])})
    grp = pd.DataFrame(rows)
    frac_split = float((grp.n_clusters == grp.M).mean())
    frac_merged = float((grp.n_clusters == 1).mean())

    # must-split: same role, different questions, judged at the same tau
    rng = np.random.default_rng(S.SEED)
    by_role = {}
    for k, g in enumerate(groups):
        by_role.setdefault(g["role"], []).append(k)
    roles = [r for r, v in by_role.items() if len(v) >= 2]
    fm = []
    for _ in range(400):
        r = roles[rng.integers(len(roles))]
        k1, k2 = rng.choice(by_role[r], size=2, replace=False)
        off1 = sum(gg["M"] for gg in groups[:k1])
        off2 = sum(gg["M"] for gg in groups[:k2])
        fm.append(float(V[off1] @ V[off2]) >= tau)
    false_merge = float(np.mean(fm))

    per_role = grp.groupby("role").agg(
        E_role_fb=("SE", "mean"), mean_clusters_fb=("n_clusters", "mean")).reset_index()
    sd = float(per_role.E_role_fb.std())

    gates = {
        "tau": tau, "judge_kappa": float(best.kappa),
        "judge_kappa_threshold": S.GATE_KAPPA,
        "judge_pass": bool(best.kappa >= S.GATE_KAPPA),
        "frac_all_split": frac_split, "frac_all_merged": frac_merged,
        "mean_clusters": float(grp.n_clusters.mean()),
        "degeneracy_pass": bool(frac_split <= S.GATE_DEGENERATE
                                and frac_merged <= S.GATE_DEGENERATE),
        "false_merge_cross_question": false_merge,
        "must_split_pass": bool(false_merge <= S.GATE_FALSE_MERGE),
        "E_role_sd": sd, "resolution_pass": bool(sd >= S.RESOLUTION_FLOOR)}
    gates["all_pass"] = bool(gates["judge_pass"] and gates["degeneracy_pass"]
                             and gates["must_split_pass"] and gates["resolution_pass"])
    S.log(f"fallback gates: {json.dumps(gates, default=float)}")

    # ---- the decider under the fallback --------------------------------- #
    panel = pd.read_csv(S.PANEL)
    fam = pd.read_csv(S.FAMILIES_CSV)
    lens = pd.read_csv(run / "data" / "per_role_entropy_L19.csv")[
        ["role", "mean_tokens", "trunc_rate", "E_role"]]
    df = (panel.merge(per_role, on="role").merge(fam, on="role", how="left")
          .merge(lens, on="role", how="left"))
    df = df[df.role != "default"].dropna(subset=["E_role_fb", "MLE", "log_var", "mean_norm"])
    x = df.E_role_fb.to_numpy(float)
    Z = df[["log_var", "mean_norm"]].to_numpy(float)
    r, p = partial_corr_multi(x, df[S.DECIDER_METRIC].to_numpy(float), Z)
    lo, hi = boot_ci(x, df[S.DECIDER_METRIC].to_numpy(float), Z,
                     np.random.default_rng(S.SEED), n_boot=S.N_BOOT)
    agree_primary = float(np.corrcoef(df.E_role_fb, df.E_role)[0, 1])
    out = dict(gates, n_roles=int(len(df)),
               decider_r=float(r), decider_p=float(p), decider_ci=[lo, hi],
               decider_clears_bar=bool(abs(r) >= S.DECIDER_BAR),
               E_role_fb_vs_primary_pearson=agree_primary,
               primary_decider_r=0.438)
    S.log(f"FALLBACK DECIDER  E_role_fb vs {S.DECIDER_METRIC} | scale: "
          f"r = {r:+.3f} [{lo:+.3f}, {hi:+.3f}]  (primary was +0.438); "
          f"fallback vs primary E_role r = {agree_primary:+.3f}")
    json.dump(out, open(run / "data" / "fallback_L19.json", "w"), indent=2, default=float)
    per_role.to_csv(run / "data" / "per_role_entropy_fallback_L19.csv", index=False)
    S.log(f"done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
