"""How much compute does the per-persona study cost, and how many roles do we need?

Answers question (3) of the per-persona brief, and it separates two costs that
are usually conflated:

  ANALYSIS on the cloud we already have — measured here by actually running the
    per-role estimators at several role counts. It is seconds. Picking
    a role subset to save analysis time saves nothing, so subsetting is not an
    analysis decision.

  EXTRACTION of a cloud big enough for a per-role manifold to exist — projected
    here. This is the only cost that matters, and it is where 10 / 50 / 100
    roles is a real trade-off.

The bridge between them is Part B: **how many points per role do the estimators
actually need** before a per-role ID means anything? We answer it by planting
manifolds of KNOWN dimension in the data's ambient space and finding the N at
which the estimators recover them. That N, divided by the number of instruction
phrasings, is the number of questions per role that must be extracted — which
turns a vague "more data" into a number of GPU-hours.

Role subsets, when needed, use `manifold.subsets.kmeans_medoid_roles`, which
already implements the spread-preserving medoid selection with `default`
force-included.

Usage:
    .venv/bin/python exploratory/per_persona/03_compute_budget.py
    .venv/bin/python exploratory/per_persona/03_compute_budget.py --sec_per_record 8.5
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from manifold.idim import id_estimates
from common import (load_role_clouds, pca_stats, design_fractions, resolve_run_dir,
                    savefig, small_matrix_ops, grid_shape,
                    C_REAL, C_DESIGN, C_INSTR, C_QUEST, C_INTER)

N_QUESTION_POOL = 240       # refs/assistant-axis/extraction_questions.jsonl
ROLE_BUDGETS = [10, 50, 100, 276]
TRUE_DIMS = [5, 10, 15]
N_GRID = [25, 50, 100, 200, 400, 800, 1200]


def planted_manifold(n: int, d: int, ambient: int, scale: np.ndarray, rng) -> np.ndarray:
    """`n` points on a smooth `d`-dimensional manifold embedded in `ambient` dims.

    Latent coordinates go through a mild quadratic warp before the random linear
    embedding, so the manifold is curved (a purely linear subspace would flatter
    the PCA-based estimators, which would recover d exactly at any n and hide
    the small-N bias this part is measuring). `scale` fixes the per-direction
    variance to the data's own spectrum so the noise floor is realistic.
    """
    Z = rng.standard_normal((n, d))
    Z = np.hstack([Z, 0.3 * Z ** 2])                      # curvature
    W = rng.standard_normal((Z.shape[1], ambient)) / np.sqrt(Z.shape[1])
    X = Z @ W
    X *= scale[: X.shape[1]].mean() if scale.size else 1.0
    return X + 0.01 * np.abs(X).mean() * rng.standard_normal(X.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--sec_per_record", type=float, default=8.5,
                    help="prompt-only forward pass; RESEARCH.md:24 documents 7-10 s/record "
                         "on this MacBook (MPS, batch size 1). Not re-measured here: the "
                         "model is not in the local HF cache.")
    ap.add_argument("--n_rep", type=int, default=5, help="repeats per (d, N) cell")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)

    roles, clouds, factors, manifest = load_role_clouds(args.view, args.layer)
    layer = args.layer if args.layer is not None else manifest["primary_layer"]
    ambient = clouds[roles[0]].shape[1]
    n_per = len(next(iter(clouds.values())))
    n_i, n_q, _ = grid_shape(factors)
    rng = np.random.default_rng(0)

    # ---- Part A: measured analysis cost on the cloud we already have --------
    print(f"== Part A: measured per-role analysis cost ({n_per}-point clouds) ==")
    from manifold.subsets import kmeans_medoid_roles  # noqa: F401  (documented reuse)
    timings = {}
    for nb in ROLE_BUDGETS:
        sub = roles[:nb] if nb >= len(roles) else list(rng.choice(roles, nb, replace=False))
        t0 = time.time()
        # small_matrix_ops to match how 01/02 actually run — measuring this loop
        # with the default thread count would overstate the cost ~4x (see
        # common.small_matrix_ops).
        with small_matrix_ops():
            for r in sub:
                instr, quest = factors[r]
                id_estimates(clouds[r]); pca_stats(clouds[r])
                design_fractions(clouds[r], instr, quest)
        timings[nb] = round(time.time() - t0, 2)
        print(f"  {nb:4d} roles -> {timings[nb]:6.2f}s  ({timings[nb]/nb*1000:.0f} ms/role)")

    # ---- Part B: how many points per role do the estimators need? ----------
    print("\n== Part B: ID recovery vs points-per-role (planted manifolds) ==")
    spec = np.linalg.svd(np.concatenate([X - X.mean(0) for X in clouds.values()]),
                         compute_uv=False)
    recovery = {}
    for d in TRUE_DIMS:
        recovery[d] = {}
        for n in N_GRID:
            vals = []
            for rep in range(args.n_rep):
                X = planted_manifold(n, d, ambient, spec, np.random.default_rng(1000 * d + n + rep))
                est = id_estimates(X)
                v = [est[k] for k in ("TwoNN", "MLE") if est.get(k)]
                if v:
                    vals.append(float(np.mean(v)))
            recovery[d][n] = float(np.median(vals)) if vals else None
        line = "  ".join(f"N={n}:{recovery[d][n]:.1f}" if recovery[d][n] else f"N={n}:na"
                         for n in N_GRID)
        print(f"  true d={d:2d} -> {line}")

    # Smallest N whose estimate is within 20% of the planted dimension.
    n_needed = {}
    for d in TRUE_DIMS:
        ok = [n for n in N_GRID if recovery[d][n] and abs(recovery[d][n] - d) / d <= 0.20]
        n_needed[d] = min(ok) if ok else None
    print(f"  smallest N within 20% of truth: {n_needed}")

    # The per-role target: enough for the largest dimension we might plausibly
    # find. Fall back to the top of the grid if nothing qualified.
    n_target = max([v for v in n_needed.values() if v] or [N_GRID[-1]])
    q_needed = int(np.ceil(n_target / n_i))
    q_capped = min(q_needed, N_QUESTION_POOL)
    print(f"\n  => target {n_target} points/role = {q_needed} questions x {n_i} "
          f"instructions (pool has {N_QUESTION_POOL}; using {q_capped})")

    # ---- Part C: extraction budget ----------------------------------------
    print(f"\n== Part C: extraction budget @ {args.sec_per_record} s/record ==")
    budget = {}
    print(f"  {'roles':>6s} {'records':>9s} {'hours':>8s} {'days':>7s}   note")
    for nb in ROLE_BUDGETS:
        rec = nb * n_i * q_capped
        hrs = rec * args.sec_per_record / 3600
        budget[nb] = {"records": rec, "hours": round(hrs, 1), "days": round(hrs / 24, 2),
                      "questions_per_role": q_capped, "points_per_role": q_capped * n_i}
        note = "overnight" if hrs <= 12 else ("a weekend" if hrs <= 60 else "not tractable locally")
        print(f"  {nb:6d} {rec:9d} {hrs:8.1f} {hrs/24:7.2f}   {note}")

    # The default rate is for a PROMPT-only forward pass. A response cloud has to
    # generate tokens first and is several times slower (the 2-role pilot in
    # plans/2026-07-23-halfdepth-response-stream.md measured 15.2 s/rec at
    # max_new_tokens=512), so flag when the two disagree instead of quietly
    # pricing generation at forward-pass rates.
    basis = manifest.get("token_basis", "prompt")
    rate_note = "RESEARCH.md:24 (documented, not re-measured)"
    if basis == "response" and args.sec_per_record < 12:
        rate_note += (f"; WARNING this cloud is token_basis={basis} "
                      f"(max_new_tokens={manifest.get('max_new_tokens')}) — generation is "
                      "slower than a prompt-only pass, so these hours are a LOWER bound")
        print(f"\n  [!] token_basis={basis}: {args.sec_per_record}s/record is the prompt-only "
              "rate; generation is slower, treat Part C as a lower bound.")
    results = {"_meta": {"view": args.view, "layer": layer, "ambient": ambient,
                         "token_basis": basis, "points_per_role_now": int(n_per),
                         "grid": [n_i, n_q],
                         "sec_per_record": args.sec_per_record,
                         "sec_per_record_source": rate_note,
                         "n_instructions": n_i, "question_pool": N_QUESTION_POOL},
               "analysis_seconds_by_n_roles": timings,
               "id_recovery_vs_n": {str(d): recovery[d] for d in TRUE_DIMS},
               "min_n_within_20pct": {str(d): n_needed[d] for d in TRUE_DIMS},
               "target_points_per_role": n_target,
               "questions_per_role_needed": q_capped,
               "extraction_budget": budget}
    json.dump(results, open(run_dir / f"03_compute_budget_{args.view}_L{layer}.json", "w"),
              indent=2, default=float)

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    nb = list(timings)
    ax.plot(nb, [timings[k] for k in nb], "-o", color=C_REAL, lw=2, ms=8)
    for k in nb:
        ax.annotate(f"{timings[k]:.1f}s", (k, timings[k]), fontsize=8,
                    xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_xlabel("roles analysed")
    ax.set_ylabel("seconds")
    ax.set_title("A. Analysis cost on the existing cloud\n(linear in roles, and negligible)")

    ax = axes[1]
    for d, col in zip(TRUE_DIMS, (C_INSTR, C_REAL, C_INTER)):
        ys = [recovery[d][n] for n in N_GRID]
        ax.plot(N_GRID, ys, "-o", color=col, lw=2, ms=5, label=f"planted d = {d}")
        ax.axhline(d, color=col, ls=":", lw=1)
    ax.axvline(n_per, color=C_DESIGN, lw=2)
    ax.annotate(f"what we have now\n({n_per} pts/role)", (n_per, ax.get_ylim()[0]),
                color=C_DESIGN, fontsize=8, ha="left", va="bottom",
                xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("points per role (log)")
    ax.set_ylabel("recovered dimension")
    ax.set_title("B. Points per role needed to recover a known dimension\n(dotted = truth)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    hrs = [budget[k]["hours"] for k in ROLE_BUDGETS]
    cols = [C_INSTR if h <= 12 else (C_QUEST if h <= 60 else C_DESIGN) for h in hrs]
    ax.bar([str(k) for k in ROLE_BUDGETS], hrs, color=cols, width=0.6)
    for i, h in enumerate(hrs):
        ax.annotate(f"{h:.0f} h" if h < 48 else f"{h/24:.1f} d", (i, h), fontsize=9,
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_ylim(0, max(hrs) * 1.18)
    ax.axhline(12, color="#333333", ls="--", lw=1)
    ax.annotate("one overnight run (12 h)", (len(ROLE_BUDGETS) - 0.55, 12), fontsize=8,
                ha="right", va="bottom", color="#333333",
                xytext=(0, 3), textcoords="offset points")
    ax.set_xlabel("roles extracted (always including `default`)")
    ax.set_ylabel(f"projected hours @ {args.sec_per_record} s/record")
    ax.set_title(f"C. Re-extraction cost at {q_capped} questions x {n_i} instructions\n"
                 f"= {q_capped * n_i} points per role")

    fig.tight_layout()
    savefig(fig, f"03_compute_budget_{args.view}_L{layer}.png", run_dir)


if __name__ == "__main__":
    main()
