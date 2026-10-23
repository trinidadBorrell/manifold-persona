"""Figures and tables for Phases 0 and 1 (plan v2 s5), into output/21-oct/.

    python -m persona_axes.figs --phase 0      # P0_1 reach vs S, P0_2 steering vs prompting; ceiling summary + S*
    python -m persona_axes.figs --phase 1      # P1_1 R2 vs k, P1_2 ablation, P1_3 M2, P1_4 residual, P1_5 sparse
    python -m persona_axes.figs --phase 0hi    # P0_3: the rerun of the 35 failed personas with an absolute push (ceiling_hiS)
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from persona_axes import AX, COLLAPSE_CAP, HERE, LAYER, OUT, S_SWEEP, VIEW  # noqa: E402

SETS = {"1_OCEAN": "OCEAN (5)", "2_HEXACO": "HEXACO (6)", "3_OCEAN+B": "OCEAN + identity (7)", "4_OCEAN+G": "OCEAN + LLM-specific (11)",
        "5_utility": "utility B-F (15)", "6_utility+G": "utility + LLM-specific (21)", "7_TIDE": "TIDE (4)", "8_union": "union A-H (31)"}


def phase0():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
    lab = pd.read_csv(AX / "judge_ceiling" / "labels.csv")
    cat = {p["name"]: p["category"] for p in json.loads((HERE / "personas_a.json").read_text())["personas"]}
    g = lab[lab.j10_judged].groupby(["persona", "set", "cell"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"),
                                                                     mrr=("j10_mrr", "mean"), n=("key", "size")).reset_index()
    j8 = lab[lab.j8_label.notna()].groupby(["persona", "cell"]).agg(j8_reach=("j8_reached", "mean"), j8_collapse=("j8_collapsed", "mean")).reset_index()
    g = g.merge(j8, on=["persona", "cell"], how="left")
    g.to_csv(AX / "judge_ceiling" / "cells.csv", index=False)
    sw = g[g.cell.str.startswith("S")].copy(); sw["S"] = sw.cell.str[1:].astype(float)
    bycell = sw.groupby("S").agg(reach=("reach", "mean"), collapse=("collapse", "mean")).reset_index()
    j8s = lab[lab.j8_label.notna() & lab.cell.str.startswith("S")].copy(); j8s["S"] = j8s.cell.str[1:].astype(float)
    j8by = j8s.groupby("S").agg(j8_reach=("j8_reached", "mean"), j8_collapse=("j8_collapsed", "mean"), n_j8=("key", "size")).reset_index()
    bycell = bycell.merge(j8by, on="S", how="left")
    ok = bycell[bycell.collapse <= COLLAPSE_CAP]
    S_star = float(ok.S.max()) if len(ok) else float(min(S_SWEEP))
    prm = g[g.cell == "prompt"].set_index("persona")
    at = sw[sw.S == S_star].set_index("persona")
    personas = sorted(set(at.index) & set(prm.index))
    passed = [p for p in personas if at.reach[p] >= 2 * at.collapse[p] and at.reach[p] > 0]
    summ = dict(S_star=S_star, collapse_cap=COLLAPSE_CAP, by_S=bycell.round(3).to_dict("records"),
                n_personas=len(personas), passed=passed, dropped=[p for p in personas if p not in passed],
                reach_at_Sstar=at.reach[personas].round(3).to_dict(), prompt_reach=prm.reach[personas].round(3).to_dict())
    (AX / "judge_ceiling" / "summary.json").write_text(json.dumps(summ, indent=1))
    # P0.1
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    cats = sorted(set(cat.values())); cmap = plt.cm.tab20(np.linspace(0, 1, len(cats)))
    for p in sw.persona.unique():
        d = sw[sw.persona == p].sort_values("S")
        if d.set.iloc[0] == "a":
            c, ls, al = cmap[cats.index(cat[p])], "-", 0.9
        else:
            c, ls, al = "grey", "--", 0.5
        axs[0].plot(d.S, d.reach, ls, color=c, alpha=al, lw=1)
        axs[1].plot(d.S, d.collapse, ls, color=c, alpha=al, lw=1)
        if p in prm.index:
            axs[0].plot([max(S_SWEEP) + 0.6], [prm.reach[p]], marker="_", color=c, ms=8)
    axs[0].plot(bycell.S, bycell.reach, "k-", lw=3, label="mean over personas, J10 (all answers)")
    axs[1].plot(bycell.S, bycell.collapse, "k-", lw=3, label="J10 nonsense flag")
    axs[0].plot(bycell.S, bycell.j8_reach, "k--", lw=2, label="J8 label = target (25%% subsample, n = %d)" % bycell.n_j8.sum())
    axs[1].plot(bycell.S, bycell.j8_collapse, "k--", lw=2, label="J8 label = collapsed")
    axs[1].axhline(COLLAPSE_CAP, color="r", ls=":", label="collapse cap %.0f%%" % (100 * COLLAPSE_CAP))
    axs[0].set_ylabel("share of answers reaching the target\n(J10: in top 5 and not nonsense; J8: label = target)")
    axs[1].set_ylabel("share of answers collapsed\n(J10 nonsense flag; J8 label = collapsed)")
    for a in axs:
        a.set_xlabel("clamp strength S (goal = M0 + S (Mt - M0)), L15"); a.set_xticks(list(S_SWEEP) + [max(S_SWEEP) + 0.6])
        a.set_xticklabels([str(s) for s in S_SWEEP] + ["prompt"])
    axs[0].set_title("P0.1  full-vector clamp, L15: reach vs S; S* = %g\n(colour = set (a) persona by category, grey dashed = set (b) role)" % S_star, fontsize=10)
    axs[1].set_title("collapse vs S (same lines)", fontsize=10)
    axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "P0_1_reach_vs_S.png", dpi=130); plt.close(fig)
    # P0.2
    fig, axp = plt.subplots(figsize=(7, 7))
    for p in personas:
        isa = at.set[p] == "a"
        axp.scatter(at.reach[p], prm.reach[p], c=[cmap[cats.index(cat[p])]] if isa else "grey", marker="o" if isa else "s", s=30)
        if at.reach[p] > 0.1 or prm.reach[p] < 0.7:            # label only the points that are not in the crowded top-left corner
            axp.annotate(p, (at.reach[p], prm.reach[p]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    n0 = sum(1 for p in personas if at.reach[p] <= 0.1 and prm.reach[p] >= 0.7)
    axp.annotate("%d unlabelled personas: steering reach <= 0.1, prompt reach >= 0.7" % n0, (0.02, 0.5), fontsize=7, color="grey")
    axp.plot([0, 1], [0, 1], "k:", lw=0.8)
    axp.set_xlabel("reach rate with the full vector clamp at S* = %g (J10 top 5, not nonsense)" % S_star)
    axp.set_ylabel("reach rate with the system prompt (same judge)")
    axp.set_title("P0.2  steering vs prompting per persona (circles = set (a), squares = set (b))")
    fig.tight_layout(); fig.savefig(OUT / "P0_2_steer_vs_prompt.png", dpi=130); plt.close(fig)
    pd.DataFrame(summ["by_S"]).to_markdown(OUT / "tables" / "P0_by_S.md", index=False)
    print("S* = %g; %d / %d personas pass; dropped: %s" % (S_star, len(passed), len(personas), summ["dropped"]))


def _curve(ax, res, cond, k_max, **kw):
    c = res["curves"][cond]["centred"]
    ks = sorted(int(k) for k in c if int(k) <= k_max and int(k) > 0)
    ax.plot(ks, [c[str(k)] if str(k) in c else c[k] for k in ks], **kw)


def phase1(view=VIEW, layer=LAYER):
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
    R = json.loads((AX / "phase1" / ("results_%s_L%d.json" % (view, layer))).read_text())
    tag = "%s_L%d" % (view, layer)
    # P1.1 R2 vs k, panels set (b) and set (a); second row fraction of PCA-k
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    colors = dict(zip(SETS, plt.cm.tab10.colors))
    for col, s in enumerate(("b", "a")):
        res = R[s]; cur = res["curves"]
        ks = lambda cond: sorted(int(k) for k in cur[cond]["centred"] if int(k) > 0)   # noqa: E731
        val = lambda cond: [cur[cond]["centred"][str(k)] for k in ks(cond)]            # noqa: E731
        ax = axs[0, col]
        ax.plot(ks("PCA"), val("PCA"), "k-", lw=2.5, label="PCA top-k (ceiling, fitted on training folds)")
        for nm in ("B3", "B2", "B1"):
            nk = sorted(int(k) for k in res["nulls"][nm]); lo = [res["nulls"][nm][str(k)]["p05"] for k in nk]; hi = [res["nulls"][nm][str(k)]["p95"] for k in nk]
            md = [res["nulls"][nm][str(k)]["p50"] for k in nk]
            if nm == "B3":
                ax.fill_between(nk, lo, hi, color="grey", alpha=0.3, label="B3: random k kept traits, 5-95%")
                ax.plot(nk, md, color="grey", lw=1)
            else:
                ax.plot(nk, md, color="silver" if nm == "B1" else "tan", lw=1, ls=":", label="%s median (%s)" % (nm, "isotropic" if nm == "B1" else "in span PCA 1-50"))
        for sname, lab in SETS.items():
            ax.plot(ks(sname), val(sname), "-o", ms=3, lw=1.3, color=colors[sname], label=lab)
        ax.plot(ks("greedy_pool"), val("greedy_pool"), "--", color="C3", lw=1.5, label="greedy from kept trait pool")
        ax.set_xscale("log"); ax.set_ylabel("held-out R2 of role vectors on span(axes) + mean")
        ax.set_title("set (%s): %s, %s L%d" % (s, "human roles (226)" if s == "b" else "deployment personas (41)", view, layer))
        ax = axs[1, col]
        pca = {int(k): v for k, v in cur["PCA"]["centred"].items()}
        for sname in list(SETS) + ["greedy_pool"]:
            ax.plot(ks(sname), [cur[sname]["centred"][str(k)] / pca[k] for k in ks(sname)], "-o" if sname in SETS else "--", ms=3, lw=1.3,
                    color=colors.get(sname, "C3"))
        nk = sorted(int(k) for k in res["nulls"]["B3"])
        ax.fill_between(nk, [res["nulls"]["B3"][str(k)]["p05"] / pca[k] for k in nk], [res["nulls"]["B3"][str(k)]["p95"] / pca[k] for k in nk], color="grey", alpha=0.3)
        ax.axhline(0.7, color="k", ls=":", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("k = number of axes"); ax.set_ylabel("R2 / R2 of PCA-k")
    axs[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("P1.1  coverage vs number of axes")
    fig.tight_layout(); fig.savefig(OUT / ("P1_1_r2_vs_k_%s.png" % tag), dpi=130); plt.close(fig)
    # P1.2 ablation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fams = "ABCDEFGH"; names = {"A": "A psychology", "B": "B identity", "C": "C interpersonal", "D": "D epistemic", "E": "E agency", "F": "F communication", "G": "G LLM-specific", "H": "H TIDE"}
    for i, s in enumerate(("b", "a")):
        d = [R[s]["ablation"][F]["delta_vs_union8"] for F in fams]
        ax.bar(np.arange(len(fams)) + 0.4 * i - 0.2, d, width=0.4, label="set (%s)" % s)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(fams))); ax.set_xticklabels([names[F] for F in fams], rotation=20, fontsize=8)
    ax.set_ylabel("change in held-out R2 at k = 8\nwhen the family is removed", fontsize=9)
    ax.set_title("P1.2  family ablation (union A-H, greedy to k = 8 on training folds)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / ("P1_2_family_ablation_%s.png" % tag), dpi=130); plt.close(fig)
    # P1.3 M2
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, s in enumerate(("b", "a")):
        m2 = R[s]["m2"]; ax = axs[col]
        for cond, lab in list(SETS.items()) + [("greedy_pool", "greedy from kept trait pool"), ("PCA", "PCA top-k")]:
            if cond not in m2:
                continue
            ks = sorted(int(k) for k in m2[cond] if int(k) < 4096)
            ax.plot(ks, [m2[cond][str(k)]["mrr"] for k in ks], "-o" if cond in SETS else ("k-" if cond == "PCA" else "--"), ms=3, lw=1.3,
                    color=colors.get(cond, "k" if cond == "PCA" else "C3"), label=lab)
        ax.axhline(m2["full"]["4096"]["mrr"], color="k", ls="--", lw=1, label="full 4096-d vector")
        ax.axhline(R[s]["m2_chance"]["top1"], color="grey", ls=":", lw=1, label="chance (top-1)")
        ax.set_xscale("log"); ax.set_xlabel("k = number of axes"); ax.set_ylabel("split-half retrieval MRR (mean of 1/rank)")
        ax.set_title("set (%s)" % s)
    axs[0].legend(fontsize=6.5); fig.suptitle("P1.3  identifiability: retrieve a persona from its coordinates on k axes (halves of its 40 answers)")
    fig.tight_layout(); fig.savefig(OUT / ("P1_3_m2_vs_k_%s.png" % tag), dpi=130); plt.close(fig)
    # P1.4 residual per persona on the full kept pool
    nonh = set(json.loads((HERE / "personas_b_human.json").read_text())["excluded_non_human"])
    res_all = R["residual_pool"]["b_all"]; res_a = R["residual_pool"]["a"]
    items = sorted([(v, n, "non-human role" if n in nonh else "human role") for n, v in res_all.items()] + [(v, n, "set (a)") for n, v in res_a.items()], reverse=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    cc = {"non-human role": "C3", "human role": "C0", "set (a)": "C2"}
    ax.bar(range(len(items)), [v for v, _, _ in items], color=[cc[t] for _, _, t in items], width=1.0, linewidth=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=t) for t, c in cc.items()])
    ax.set_ylabel("1 - R2 on span(all %d kept traits)" % len(R["kept_traits"])); ax.set_xlabel("personas, sorted by residual")
    ax.set_title("P1.4  share of each persona vector not explained by the trait pool, %s L%d\nhighest: %s" % (view, layer, ", ".join(n for _, n, _ in items[:10])), fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / ("P1_4_residual_%s.png" % tag), dpi=130); plt.close(fig)
    # P1.5 sparse
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, lab in (("b", "human roles"), ("a", "set (a)")):
        ax.hist([r["n_for_90pct"] for r in R["sparse"][s]], bins=np.arange(0.5, 32), alpha=0.6, label=lab, density=True)
    ax.set_xlabel("number of traits (greedy per persona) needed for 90% of the persona's own pool R2; 31 = more than 30")
    ax.set_ylabel("share of personas"); ax.set_title("P1.5  how many traits a persona needs"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / ("P1_5_sparse_%s.png" % tag), dpi=130); plt.close(fig)
    rows = [dict(persona=r["persona"], n_for_90pct=r["n_for_90pct"], pool_r2=round(r["pool_r2"], 3), recipe=", ".join("%.2f %s" % (c, t) for t, c in r["top"][:3])) for r in R["sparse"]["a"]]
    pd.DataFrame(rows).to_markdown(OUT / "tables" / ("P1_5_recipes_set_a_%s.md" % tag), index=False)
    # summary table at k = 8
    rows = []
    for s in ("b", "a", "b_all"):
        for cond in list(SETS) + ["greedy_pool", "PCA"]:
            c = R[s]["curves"][cond]["centred"]
            if "8" in c:
                rows.append(dict(set=s, axes=cond, r2_k8=round(c["8"], 3), frac_pca8=round(c["8"] / R[s]["curves"]["PCA"]["centred"]["8"], 3),
                                 b3_pct=round(R[s]["b3_percentile"].get(cond, {}).get("8", np.nan), 1) if cond != "PCA" else None,
                                 eff_rank8=round(R["effective_rank"].get(cond, {}).get("8", np.nan), 2) if cond in SETS else None))
    pd.DataFrame(rows).to_markdown(OUT / "tables" / ("P1_summary_k8_%s.md" % tag), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def push_norms():
    """|M_t - M_0| per persona at (VIEW, LAYER), from the raw means the clamp used"""
    V = AX / "vectors"
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    out = {}
    for part, nf in (("roles", "roles_names.json"), ("seta", "seta_names.json")):
        names = json.loads((V / nf).read_text()); X = np.load(V / ("%s_rawmean_%s_L%d.npy" % (part, VIEW, LAYER))).astype(np.float64)
        out.update({n: float(np.linalg.norm(X[i] - M0)) for i, n in enumerate(names)})
    return out


def phase0_hiS():
    """P0_3: reach and collapse against the ABSOLUTE push along u (units of the residual stream at L15).
    Solid lines: the rerun (ceiling_hiS, push = S x median role norm). Dotted: the same personas in the first run
    (push = S |v|), so both runs sit on one axis. Pass rule as in phase0, with the first run's prompt reach."""
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
    cells = json.loads((AX / "ceiling_hiS" / "cells.json").read_text()); unit = cells["unit_norm"]
    hi = pd.read_csv(AX / "judge_ceiling_hiS" / "labels.csv"); hi = hi[hi.j10_judged]
    lo = pd.read_csv(AX / "judge_ceiling" / "labels.csv"); lo = lo[lo.j10_judged]
    norms = push_norms()
    g = hi.groupby(["persona", "set", "cell"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"), n=("key", "size")).reset_index()
    g["S"] = g.cell.str[1:].astype(float); g["push"] = g.S * unit
    g.to_csv(AX / "judge_ceiling_hiS" / "cells.csv", index=False)
    g0 = lo[lo.persona.isin(g.persona.unique()) & lo.cell.str.startswith("S")].groupby(["persona", "cell"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean")).reset_index()
    g0["S"] = g0.cell.str[1:].astype(float); g0["push"] = g0.S * g0.persona.map(norms)
    prm = lo[lo.cell == "prompt"].groupby("persona").j10_top5.mean()
    cat = {p["name"]: p["category"] for p in json.loads((HERE / "personas_a.json").read_text())["personas"]}
    cats = sorted(set(cat.values())); cmap = plt.cm.tab20(np.linspace(0, 1, len(cats)))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for p in sorted(g.persona.unique()):
        d = g[g.persona == p].sort_values("push"); d0 = g0[g0.persona == p].sort_values("push")
        c = cmap[cats.index(cat[p])] if p in cat else "grey"
        axs[0].plot(d.push, d.reach, "-", color=c, alpha=0.8, lw=1); axs[0].plot(d0.push, d0.reach, ":", color=c, alpha=0.5, lw=1)
        axs[1].plot(d.push, d.collapse, "-", color=c, alpha=0.8, lw=1); axs[1].plot(d0.push, d0.collapse, ":", color=c, alpha=0.5, lw=1)
    by = g.groupby("S").agg(push=("push", "first"), reach=("reach", "mean"), collapse=("collapse", "mean")).reset_index()
    j8h = pd.read_csv(AX / "judge_ceiling_hiS" / "labels.csv"); j8h = j8h[j8h.j8_label.notna()].copy(); j8h["S"] = j8h.cell.str[1:].astype(float)
    by = by.merge(j8h.groupby("S").agg(j8_reach=("j8_reached", "mean"), j8_collapse=("j8_collapsed", "mean"), n_j8=("key", "size")).reset_index(), on="S", how="left")
    axs[0].plot(by.push, by.reach, "k-", lw=3, label="mean over the %d personas, rerun, J10 (all answers)" % g.persona.nunique())
    axs[0].plot(by.push, by.j8_reach, "k--", lw=2, label="rerun, J8 label = target (25%% subsample, n = %d)" % by.n_j8.sum())
    axs[1].plot(by.push, by.collapse, "k-", lw=3, label="J10 nonsense flag"); axs[1].plot(by.push, by.j8_collapse, "k--", lw=2, label="J8 label = collapsed")
    axs[1].axhline(COLLAPSE_CAP, color="r", ls=":", label="collapse cap %.0f%%" % (100 * COLLAPSE_CAP))
    axs[0].plot([], [], "k:", label="same personas, first run, J10 (push = S |v|)")
    axs[0].set_ylabel("share of answers reaching the target\n(J10: in top 5 and not nonsense; J8: label = target)")
    axs[1].set_ylabel("share of answers collapsed\n(J10 nonsense flag; J8 label = collapsed)")
    for a in axs:
        a.set_xlabel("absolute push along u at L15 (goal minus neutral coordinate)")
    axs[0].set_title("P0.3  the 35 personas that failed at S* = 4, pushed by an absolute amount: S x %.1f, S in %s\n(colour = set (a) category, grey = set (b) role)" % (unit, cells["S"]), fontsize=10)
    axs[1].set_title("collapse (same lines)", fontsize=10); axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "P0_3_failed_personas_abs_push.png", dpi=130); plt.close(fig)
    # pass rule per persona at its best cell with collapse <= cap
    rows = []
    for p, d in g.groupby("persona"):
        ok = d[d.collapse <= COLLAPSE_CAP]
        best = ok.sort_values("reach").iloc[-1] if len(ok) else d.sort_values("push").iloc[0]
        rows.append(dict(persona=p, set=d.set.iloc[0], norm=round(norms[p], 1), reach_S4_first_run=round(float(g0[(g0.persona == p) & (g0.S == 4)].reach.mean()), 2),
                         best_S=best.S, best_push=round(best.push), reach=round(best.reach, 2), collapse=round(best.collapse, 2), prompt_reach=round(float(prm.get(p, np.nan)), 2),
                         passes=bool(best.reach >= 2 * best.collapse and best.reach > 0 and best.collapse <= COLLAPSE_CAP)))
    t = pd.DataFrame(rows).sort_values(["passes", "reach"], ascending=False)
    t.to_markdown(OUT / "tables" / "P0_3_failed_personas_rerun.md", index=False)
    by.round(3).to_markdown(OUT / "tables" / "P0_3_by_S.md", index=False)
    summ = dict(unit_norm=unit, by_S=by.round(3).to_dict("records"), n=len(t), now_pass=t[t.passes].persona.tolist())
    (AX / "judge_ceiling_hiS" / "summary.json").write_text(json.dumps(summ, indent=1))
    print(by.round(3).to_string()); print("%d of %d failed personas now pass: %s" % (t.passes.sum(), len(t), summ["now_pass"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["0", "1", "0hi"])
    ap.add_argument("--view", default=VIEW); ap.add_argument("--layer", type=int, default=LAYER)
    a = ap.parse_args(argv)
    {"0": phase0, "0hi": phase0_hiS}.get(a.phase, lambda: phase1(a.view, a.layer))()


if __name__ == "__main__":
    main()
