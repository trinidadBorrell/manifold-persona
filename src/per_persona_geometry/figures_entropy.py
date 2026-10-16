"""Figures for the semantic-entropy study (plan 2026-08-11).

Reads only what `study_entropy.py` wrote. Kept separate so a plotting change
never touches an already-computed number, which is the repo's convention.

LABELLING RULES, ENFORCED HERE RATHER THAN REMEMBERED
-----------------------------------------------------
* Every figure except fig03's decider panel carries **EXPLORATORY** in the
  title text, not just the caption (plan: Exploration/Fence).
* Every figure carries the two deviations that change what the number means:
  D2 (five system prompts, not ten samples at T=1 -- this is
  instruction-sensitivity, not sampling uncertainty) and D6 (entailment judged
  on the first 40 words).
* If the predicate gates failed, every figure says so in red. Amendment A1 says
  the numbers get computed anyway; it does not say they get quoted clean.

Plot functions are defensive: a failure logs and returns, so a plotting bug
cannot destroy a run that already produced its numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import fmt_p, linfit    # noqa: E402

DPI = 300
FOOT = ("D2: entropy over 5 system-prompt phrasings (greedy), not 10 samples at T=1 "
        "— instruction-sensitivity, not sampling uncertainty.   "
        "D6: entailment on the first 40 words.")
DECIDER = "MLE"


def _title(ax, text, exploratory=True, gates_ok=True):
    """Axis title. The gate warning is drawn ONCE per figure by `_save`, not per
    axis -- on two-panel figures the per-axis version overlapped the titles."""
    tag = "EXPLORATORY — " if exploratory else ""
    ax.set_title(f"{tag}{text}", fontsize=9.5, loc="left")


def _save(fig, path, note=FOOT, gates_ok=True):
    fig.text(0.005, 0.004, note, fontsize=6.5, color="0.35", va="bottom")
    if not gates_ok:
        fig.text(0.995, 0.004, "PREDICATE GATES FAILED (judge kappa 0.151 < 0.20)",
                 ha="right", va="bottom", color="crimson", fontsize=7.5, weight="bold")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _boot_band(ax, x, y, xs, n_boot=1000, seed=0):
    """Bootstrap 95% band on a fitted line, resampling ROLES -- the plan's figure
    table specifies a CI band on every fit in fig03 and fig05."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fits = np.empty((n_boot, len(xs)))
    for b in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        try:
            s_, i_ = np.polyfit(x[i], y[i], 1)
            fits[b] = i_ + s_ * xs
        except Exception:                              # noqa: BLE001
            fits[b] = np.nan
    lo, hi = np.nanpercentile(fits, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color="#22334d", alpha=0.16, lw=0)


def _guard(fn):
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:                       # noqa: BLE001
            print(f"  !! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    return wrapped


# --------------------------------------------------------------------------- #
@_guard
def fig01_clusters(grp, gates, out, gates_ok):
    """Does the predicate resolve anything? Everything else is unreadable if not."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    counts = grp.n_clusters.value_counts(normalize=True).sort_index()
    ax.bar(counts.index, counts.values, color="#4c72b0", width=0.65)
    for k, v in counts.items():
        ax.text(k, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    ax.axhline(0.90, ls="--", c="crimson", lw=1)
    ax.text(1.0, 0.905, "degeneracy gate (90%)", color="crimson", fontsize=8)
    ax.set_xlabel("semantic clusters among the 5 answers")
    ax.set_ylabel("fraction of the 11,040 (role, question) groups")
    ax.set_xticks(sorted(counts.index))
    ax.set_ylim(0, max(1.0, counts.max() * 1.18))
    d = gates["degeneracy"]
    _title(ax, f"Cluster counts — mean {d['mean_clusters']:.2f}, "
               f"{d['frac_all_split']:.1%} fully split, "
               f"{d['frac_all_merged']:.1%} fully merged", True, gates_ok)
    _save(fig, out / "fig01_cluster_counts_L19.png", gates_ok=gates_ok)


@_guard
def fig02_ladder(lad, out, gates_ok):
    """The centrepiece: does each correlation hold as controls are added?"""
    e = lad[lad.predictor == "E_role"]
    order = (e[e.rung == "ctrl_scale"].set_index("target").r.abs()
             .sort_values(ascending=True).index.tolist())
    rungs = [("raw", "#c7d3e6", "raw"),
             ("ctrl_scale", "#4c72b0", "| log_var, mean_norm"),
             ("ctrl_all", "#22334d", "| + tokens, trunc"),
             ("ctrl_family", "#b07aa1", "| + 15 persona families (within-type)")]
    rungs = [r for r in rungs if (e.rung == r[0]).any()]
    fig, ax = plt.subplots(figsize=(9.5, 0.40 * len(order) + 2.8))
    y = np.arange(len(order))
    for k, (rung, col, lab) in enumerate(rungs):
        sub = e[e.rung == rung].set_index("target").reindex(order)
        off = (k - 1.5) * 0.20
        ax.errorbar(sub.r, y + off,
                    xerr=[sub.r - sub.ci_lo, sub.ci_hi - sub.r],
                    fmt="o", ms=4, lw=1, color=col, label=lab, capsize=0)
        surv = sub.q_within_rung < 0.05
        ax.scatter(sub.r[surv], y[surv.values] + off, s=52, facecolors="none",
                   edgecolors=col, lw=0.9)
    # shuffle null band, from the controlled rung
    p95 = e[e.rung == "ctrl_scale"].set_index("target").reindex(order).shuffle_p95
    ax.fill_betweenx(y, -p95, p95, color="0.85", alpha=0.55, zorder=0,
                     label="axis-shuffle null (95th pct)")
    # the length baseline: if these are as strong, the story is verbosity
    for pnam, mark in (("mean_tokens", "x"), ("trunc_rate", "+")):
        sub = lad[(lad.predictor == pnam) & (lad.rung == "ctrl_scale")] \
            .set_index("target").reindex(order)
        ax.scatter(sub.r, y, marker=mark, s=26, color="#d1495b", zorder=5,
                   label=f"baseline: {pnam}")
    ax.axvline(0, c="0.4", lw=0.8)
    for v in (-0.30, 0.30):
        ax.axvline(v, c="seagreen", ls=":", lw=1)
    ax.text(0.30, len(order) - 0.4, " |r| = 0.30", color="seagreen", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([("★ " + t if t == DECIDER else t) for t in order], fontsize=8)
    ax.set_xlabel("partial correlation with per-role semantic entropy (E_role)")
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    _title(ax, f"Entropy vs {len(order)} targets, {len(rungs)} levels of control "
               f"(n = {int(e.n.iloc[0])} roles, bootstrap 95% CI; ○ = BH q < 0.05)",
           True, gates_ok)
    _save(fig, out / "fig02_ladder_forest_L19.png", gates_ok=gates_ok)


@_guard
def fig03_headline(df, lad, out, gates_ok, default_row=None):
    """The decider, and the closeness measure, as scatters."""
    pairs = [(DECIDER, False), ("axis_proj", True)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (target, expl) in zip(axes, pairs):
        ax.scatter(df.E_role, df[target], s=15, alpha=0.65, color="#4c72b0",
                   edgecolors="none")
        f = linfit(df.E_role, df[target])
        xs = np.linspace(df.E_role.min(), df.E_role.max(), 50)
        ax.plot(xs, f["intercept"] + f["slope"] * xs, c="#22334d", lw=1.4)
        _boot_band(ax, df.E_role.to_numpy(float), df[target].to_numpy(float), xs)
        if default_row is not None and target in default_row:
            ax.scatter([default_row["E_role"]], [default_row[target]], marker="*",
                       s=260, color="gold", edgecolors="k", lw=0.6, zorder=6,
                       label="default (excluded from fit)")
            ax.legend(fontsize=8, loc="best")
        row = lad[(lad.predictor == "E_role") & (lad.target == target)
                  & (lad.rung == "ctrl_scale")]
        rr = f"r = {row.r.iloc[0]:+.3f}" if len(row) else "r = n/a"
        ax.set_xlabel("E_role — mean per-question-centred semantic entropy (nats)")
        ax.set_ylabel(target)
        _title(ax, f"{target}: raw {f['r']:+.3f} ({fmt_p(f['p'])}), controlled {rr}",
               expl, gates_ok)
    fig.suptitle("Semantic entropy against the decider (left) and closeness (right)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out / "fig03_headline_scatter_L19.png", gates_ok=gates_ok)


@_guard
def fig04_group_level(gg, wdf, out, gates_ok):
    """The mechanism, at the level it would act on: one (role, question) cell."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    hb = ax.hexbin(gg.SE_centred, gg.grp_pr, gridsize=45, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="groups")
    f = linfit(gg.SE_centred, gg.grp_pr)
    xs = np.linspace(gg.SE_centred.min(), gg.SE_centred.max(), 50)
    ax.plot(xs, f["intercept"] + f["slope"] * xs, c="crimson", lw=1.4)
    ax.set_xlabel("per-question-centred semantic entropy of the 5 answers")
    ax.set_ylabel("participation ratio of the same 5 activation points")
    _title(ax, f"Pooled, all {len(gg):,} groups: r = {f['r']:+.3f} (not the test)",
           True, gates_ok)
    ax = axes[1]
    ax.hist(wdf.r_pr.dropna(), bins=30, color="#4c72b0")
    ax.axvline(0, c="0.3", lw=1)
    m = wdf.r_pr.mean()
    ax.axvline(m, c="crimson", lw=1.4)
    ax.text(m, ax.get_ylim()[1] * 0.92, f"  mean {m:+.3f}", color="crimson", fontsize=9)
    ax.set_xlabel("within-role correlation r(entropy, group spread)")
    ax.set_ylabel("roles")
    _title(ax, f"THE TEST: within each of {len(wdf)} roles — "
               f"{(wdf.r_pr > 0).mean():.0%} positive", True, gates_ok)
    fig.tight_layout()
    _save(fig, out / "fig04_group_level_L19.png", gates_ok=gates_ok)


@_guard
def fig05_diagnostics(pr, out, gates_ok):
    """The figure that reads a null: is entropy just response length?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, lab in ((axes[0], "trunc_rate", "fraction of responses hitting 128 tokens"),
                         (axes[1], "mean_tokens", "mean response length (tokens)")):
        ax.scatter(pr[col], pr.E_role, s=15, alpha=0.65, color="#4c72b0",
                   edgecolors="none")
        f = linfit(pr[col], pr.E_role)
        xs = np.linspace(pr[col].min(), pr[col].max(), 50)
        ax.plot(xs, f["intercept"] + f["slope"] * xs, c="#22334d", lw=1.4)
        _boot_band(ax, pr[col].to_numpy(float), pr.E_role.to_numpy(float), xs)
        ext = pd.concat([pr.nsmallest(3, col), pr.nlargest(3, col)])
        for _, r in ext.iterrows():
            ax.annotate(r.role, (r[col], r.E_role), fontsize=6.5, alpha=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("E_role (nats)")
        _title(ax, f"entropy vs {col}: r = {f['r']:+.3f} ({fmt_p(f['p'])})",
               True, gates_ok)
    fig.tight_layout()
    _save(fig, out / "fig05_length_diagnostics_L19.png", gates_ok=gates_ok)


@_guard
def fig06_centring(pr, panel, out, gates_ok):
    """What did per-question centring actually do?"""
    d = pr.merge(panel[["role", "axis_proj"]], on="role", how="left")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(d.E_role_uncentred, d.E_role, c=d.axis_proj, s=18,
                    cmap="viridis", edgecolors="none")
    fig.colorbar(sc, ax=ax, label="axis_proj (high = Assistant-like)")
    f = linfit(d.E_role_uncentred, d.E_role)
    ax.set_xlabel("uncentred mean semantic entropy (nats)")
    ax.set_ylabel("per-question-centred E_role (nats)")
    # r is 1.000 BY CONSTRUCTION on a balanced design: every role answers the
    # same 40 questions, so subtracting each question's across-role mean removes
    # an identical constant from every role. The centring is a no-op, not
    # evidence that question difficulty was harmless.
    _title(ax, f"Centring is a NO-OP: r = {f['r']:+.3f} exactly — balanced design, "
               f"so it subtracts one constant from every role", True, gates_ok)
    fig.tight_layout()
    _save(fig, out / "fig06_centring_L19.png", gates_ok=gates_ok)


@_guard
def fig07_families(df, fam_arm, out, gates_ok):
    """A3 — is entropy a property of the role, or of the KIND of role?

    Families ordered along the Assistant axis. If the strip walks monotonically
    up, the pooled correlation is largely a between-family effect and the
    within-family rung in fig02 is the number to read.
    """
    order = fam_arm["family_order_by_axis"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, f in enumerate(order):
        v = df.loc[df.family == f, "E_role"].to_numpy()
        if not len(v):
            continue
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, .07, len(v)),
                   v, s=13, alpha=0.6, color="#4c72b0", edgecolors="none")
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="crimson", lw=2)
        ax.text(i, ax.get_ylim()[0], f"n={len(v)}", ha="center", fontsize=6.5,
                color="0.4")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"F{f}" for f in order], fontsize=8)
    ax.set_xlabel("Ward persona family — ordered by mean axis_proj "
                  "(left = far from Assistant, right = Assistant-like)")
    ax.set_ylabel("E_role (nats)")
    icc = fam_arm["icc_between_family_share"]
    _title(ax, f"Entropy by persona type — {icc:.1%} of E_role variance is family "
               f"membership (Kruskal-Wallis p = {fam_arm['kruskal_p']:.1e})",
           True, gates_ok)
    _save(fig, out / "fig07_family_entropy_L19.png", gates_ok=gates_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    run = Path(args.outdir)
    dd, fd = run / "data", run / "figures"
    fd.mkdir(parents=True, exist_ok=True)

    gates = json.load(open(dd / "predicate_validation_L19.json"))
    # the A4 agreement block carries no pass/pass_ key -- including it would
    # stamp "GATES FAILED" on every figure even in a run where all gates passed
    gates_ok = all(v.get("pass", v.get("pass_", False))
                   for k, v in gates.items()
                   if isinstance(v, dict) and k != "predicate_agreement_A4")
    grp = pd.read_parquet(dd / "group_entropy_L19.parquet")
    pr = pd.read_csv(dd / "per_role_entropy_L19.csv")
    lad = pd.read_csv(dd / "ladder_entropy_L19.csv")
    panel = pd.read_csv(Path(json.load(open(run / "manifest.json"))["inputs"]["panel"]))
    gg = pd.read_csv(dd / "group_geometry_L19.csv")
    wdf = pd.read_csv(dd / "within_role_corr_L19.csv")

    df = panel.merge(pr, on="role", how="inner")
    try:
        fam_arm = json.load(open(dd / "family_arm_L19.json"))
        df = df.merge(pd.read_csv(dd.parent.parent.parent
                                  / "per_persona_axis_centroid_11-Aug-2026" / "q40"
                                  / "data" / "role_families_L19.csv"),
                      on="role", how="left")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! family arm unavailable: {exc}")
        fam_arm = None
    drow = df[df.role == "default"]
    drow = drow.iloc[0].to_dict() if len(drow) else None
    fit = df[df.role != "default"]

    print(f"figures -> {fd}  (gates {'PASS' if gates_ok else 'FAILED'})")
    fig01_clusters(grp, gates, fd, gates_ok)
    fig02_ladder(lad, fd, gates_ok)
    fig03_headline(fit, lad, fd, gates_ok, drow)
    fig04_group_level(gg, wdf, fd, gates_ok)
    fig05_diagnostics(pr, fd, gates_ok)
    fig06_centring(pr, panel, fd, gates_ok)
    if fam_arm is not None:
        fig07_families(fit, fam_arm, fd, gates_ok)


if __name__ == "__main__":
    main()
