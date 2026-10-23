"""Phase 3 reachability (plan v2 s1 M3, s5 Phase 3; checkpoint decision 2026-10-21: run on the personas the
full vector reaches).

Steering = the LIN clamp generalised to a k-dim span (plan s4): U = orthonormal basis of the k chosen
directions, v = M_t - M_0 (raw answer-token means at hidden_states[15]), goal G = M_0 + S U U^T v,
hook at every position h <- h + U U^T (G - h). S is persona-specific: S_p = (best full-vector push) / |v|,
where the best push is the cell with the highest full-vector reach under the collapse cap in the Phase 0
run or its absolute-push rerun (plan s4 "matched strength"; the rerun showed S |v| is the wrong unit).
Cells: S in {S_p, S_p / 2}.

Conditions (plan): set 6 (utility + LLM-specific, greedy order on all 226 human roles), set 9 (pruned
union), greedy from the kept trait pool, B2 x 3 (random directions in span PCA 1-50 of the roles),
B3 x 3 (random kept traits), k in {4, 8, 16} (set 9: 4, 8, 10); 'recipe' for set (a) personas = the
persona's own top-n pool traits (n = traits for 90% of own pool R2, capped at 16); 'full' = k = 1 along v
(the ceiling, regenerated so every cell shares seeds and S_p). 20 answers per cell.

    python -m persona_axes.phase3 --plan                     # CPU -> $PM/AX21/phase3/{cells.json, basis.npz}
    python -m persona_axes.phase3 --shard 0 --n-shards 12    # GPU -> $PM/AX21/phase3/gen_<persona>.jsonl
    python -m persona_axes.phase3 --analyse                  # after judge_build --stage phase3 --collect -> P3_1, P3_2, tables
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from persona_axes import AX, COLLAPSE_CAP, HERE, LAYER, N_SEEDS, QUESTIONS, VIEW
from persona_axes.ceiling import encode, generate

KS = [4, 8, 16]
MIN_REACH = 0.25          # a persona is in Phase 3 if the full vector reaches >= this at some cell with collapse <= cap
NULL_DRAWS = 3
RECIPE_KMAX = 16
P3 = AX / "phase3"


def orth(M):
    Q, R = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R).max()
    return Q[:, keep]


def persona_list():
    """best full-vector cell per persona over the Phase 0 run (push = S |v|) and the absolute-push rerun."""
    from persona_axes.figs import push_norms
    norms = push_norms()
    c0 = pd.read_csv(AX / "judge_ceiling" / "cells.csv")
    c0 = c0[c0.cell.str.startswith("S")].copy(); c0["S"] = c0.cell.str[1:].astype(float); c0["push"] = c0.S * c0.persona.map(norms); c0["run"] = "ceiling"
    c1 = pd.read_csv(AX / "judge_ceiling_hiS" / "cells.csv"); c1["run"] = "ceiling_hiS"
    both = pd.concat([c0[["persona", "set", "cell", "reach", "collapse", "push", "run"]], c1[["persona", "set", "cell", "reach", "collapse", "push", "run"]]])
    ok = both[both.collapse <= COLLAPSE_CAP].sort_values("reach", ascending=False).drop_duplicates("persona")
    ok = ok[ok.reach >= MIN_REACH].copy()
    ok["norm"] = ok.persona.map(norms); ok["S_p"] = ok.push / ok.norm
    return ok.sort_values(["set", "persona"]).reset_index(drop=True)


def plan():
    from persona_axes.phase1 import Data, greedy
    D = Data(VIEW, LAYER)
    names_b, Xb = D.sets["b"]; Xc = Xb - Xb.mean(0)
    cols6 = D.axis_set("BCDEFG"); o6 = [cols6[j] for j in greedy(D.A[:, cols6], Xc, max(KS))]
    set9 = json.loads((AX / "phase2" / "results.json").read_text())["set9"]
    cols9 = [D.anames.index(n) for n in set9]; o9 = [cols9[j] for j in greedy(D.A[:, cols9], Xc, len(cols9))]
    og = greedy(D.T, Xc, max(KS))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P50 = Vt[:50].T
    b2 = {}; b3 = {}
    for d in range(NULL_DRAWS):
        rng = np.random.default_rng(100 + d)
        b2["B2_%d" % d] = P50 @ rng.standard_normal((50, max(KS)))                     # nested: first k columns at k
        b3["B3_%d" % d] = [int(i) for i in rng.choice(D.T.shape[1], max(KS), replace=False)]
    sparse = json.loads((AX / "phase1" / ("results_%s_L%d.json" % (VIEW, LAYER))).read_text())["sparse"]["a"]
    recipes = {r["persona"]: [D.tnames.index(t) for t, _ in r["top"][:min(RECIPE_KMAX, max(1, r["n_for_90pct"]))]] for r in sparse}
    pl = persona_list()
    conds = {"set6": dict(source="axes", cols=o6, ks=KS), "set9": dict(source="axes", cols=o9, ks=[k for k in KS if k < len(o9)] + [len(o9)]),
             "greedy_pool": dict(source="traits", cols=[int(j) for j in og], ks=KS)}
    for n, m in b2.items():
        conds[n] = dict(source="b2", cols=list(range(max(KS))), ks=KS)
    for n, idx in b3.items():
        conds[n] = dict(source="traits", cols=idx, ks=KS)
    cells = dict(layer=LAYER, view=VIEW, ks=KS, min_reach=MIN_REACH, collapse_cap=COLLAPSE_CAP, seeds=N_SEEDS, questions=QUESTIONS,
                 personas=pl.to_dict("records"), conditions=conds, recipes={p: recipes[p] for p in pl.persona if p in recipes},
                 axis_names=D.anames, trait_names=D.tnames,
                 set6_order=[D.anames[i] for i in o6], set9_order=[D.anames[i] for i in o9], greedy_pool_order=[D.tnames[j] for j in og])
    P3.mkdir(exist_ok=True)
    (P3 / "cells.json").write_text(json.dumps(cells, indent=1))
    np.savez(P3 / "basis.npz", axes=D.A, traits=D.T, **b2)
    n_cells = sum(len(c["ks"]) for c in conds.values()) * 2 + 2 + 2 * sum(1 for p in pl.persona if p in recipes) / max(1, len(pl))
    print("%d personas (%d set (a), %d set (b)); S_p median %.2f [%.2f, %.2f]; ~%.0f cells x 20 answers per persona = ~%d answers" % (
        len(pl), (pl.set == "a").sum(), (pl.set == "b").sum(), pl.S_p.median(), pl.S_p.min(), pl.S_p.max(), n_cells, n_cells * 20 * len(pl)))
    print("set 6 order:", cells["set6_order"][:8]); print("set 9 order:", cells["set9_order"][:8]); print("pool:", cells["greedy_pool_order"][:8])
    print(pl[["persona", "set", "run", "cell", "reach", "collapse", "push", "norm", "S_p"]].round(2).to_string())


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from persona_axes import OUT, HERE
    cells = json.loads((P3 / "cells.json").read_text())
    lab = pd.read_csv(AX / "judge_phase3" / "labels.csv"); lab = lab[lab.j10_judged].copy()
    parts = lab.cell.str.rsplit("_", n=2, expand=True); lab["cond"], lab["k"], lab["S_label"] = parts[0], parts[1].str[1:].astype(int), parts[2]
    lab["family"] = lab.cond.str.replace(r"_\d+$", "", regex=True)          # B2_0 -> B2
    per = lab.groupby(["persona", "set", "cond", "family", "k", "S_label"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"), mrr=("j10_mrr", "mean")).reset_index()
    per["reach_ok"] = per.reach.where(per.collapse <= COLLAPSE_CAP, 0.0)
    prm = pd.read_csv(AX / "judge_ceiling" / "cells.csv"); prm = prm[prm.cell == "prompt"].set_index("persona").reach
    agg = per.groupby(["family", "k", "S_label"]).agg(reach=("reach_ok", "mean"), raw_reach=("reach", "mean"), collapse=("collapse", "mean"), n_personas=("persona", "nunique")).reset_index()
    full = per[per.cond == "full"].groupby("S_label").reach_ok.mean()
    # P3.1
    fams = ["set6", "set9", "greedy_pool", "B2", "B3", "recipe"]; cols = {"set6": "C0", "set9": "C1", "greedy_pool": "C3", "B2": "tan", "B3": "grey", "recipe": "C2"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, slab in enumerate(["Sfull", "Shalf"]):
        for f in fams:
            d = agg[(agg.family == f) & (agg.S_label == slab)].sort_values("k")
            if len(d) == 0:
                continue
            axs[0, j].plot(d.k, d.reach, "-o" if f != "recipe" else "s", ms=4, color=cols[f], label=f + (" (set (a) own traits, k = n)" if f == "recipe" else ""))
            axs[1, j].plot(d.k, d.collapse, "-o" if f != "recipe" else "s", ms=4, color=cols[f])
        axs[0, j].axhline(full.get(slab, np.nan), color="k", ls="--", lw=1, label="full vector, same S")
        axs[0, j].axhline(prm.reindex(per.persona.unique()).mean(), color="k", ls=":", lw=1, label="system prompt")
        axs[1, j].axhline(COLLAPSE_CAP, color="r", ls=":", lw=0.8)
        axs[0, j].set_title("S = %s" % ("S_p (persona's best full-vector strength)" if slab == "Sfull" else "S_p / 2"), fontsize=10)
        axs[1, j].set_xlabel("k = number of directions clamped"); axs[1, j].set_xscale("log"); axs[1, j].set_xticks([1, 4, 8, 10, 16]); axs[1, j].set_xticklabels(["1", "4", "8", "10", "16"])
    axs[0, 0].set_ylabel("reach (target in J10 top 5, not nonsense;\ncollapsed cells count 0), mean over %d personas" % per.persona.nunique(), fontsize=9)
    axs[1, 0].set_ylabel("collapse (share J10 flags nonsense)", fontsize=9); axs[0, 0].legend(fontsize=7)
    fig.suptitle("P3.1  reachability with the k-dim clamp in span(axes), L15: pre-registered sets vs greedy pool vs random (B2 in PCA-50 span, B3 random kept traits)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "P3_1_reach_vs_k.png", dpi=130); plt.close(fig)
    # P3.2: M1 coverage at k = 8 (set 6, Phase 1 per persona) vs M3 reach at k = 8 (set 6, S_p)
    r1 = json.loads((AX / "phase1" / ("results_%s_L%d.json" % (VIEW, LAYER))).read_text())
    cov = {}
    for s in ("b", "a"):
        names = r1[s]["names"]; vals = r1[s]["per_persona"]["6_utility+G"]["8"]
        cov.update(dict(zip(names, vals)))
    d8 = per[(per.cond == "set6") & (per.k == 8) & (per.S_label == "Sfull")].set_index("persona")
    xs = np.array([cov.get(p, np.nan) for p in d8.index]); ys = d8.reach_ok.values; ok = ~np.isnan(xs)
    rho, pv = spearmanr(xs[ok], ys[ok])
    rng = np.random.default_rng(0); bs = [spearmanr(*[a[i] for a in (xs[ok], ys[ok])])[0] for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(1000))]
    fig, ax = plt.subplots(figsize=(7, 6))
    for s, m in (("a", "o"), ("b", "s")):
        sel = (d8.set == s).values & ok
        ax.scatter(xs[sel], ys[sel], marker=m, s=35, label="set (%s)" % s)
    for p, x, y in zip(d8.index, xs, ys):
        ax.annotate(p, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("M1 coverage of the persona's vector by set 6 at k = 8 (held-out R2 per persona, Phase 1)"); ax.set_ylabel("M3 reach with set 6 at k = 8, S_p (Phase 3)")
    ax.set_title("P3.2  does coverage predict reach?  Spearman rho = %.2f [%.2f, %.2f], n = %d\n(H3 pass: rho >= 0.4 and CI excludes 0)" % (rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5), ok.sum()), fontsize=9)
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "P3_2_coverage_vs_reach.png", dpi=130); plt.close(fig)
    (OUT / "tables").mkdir(exist_ok=True)
    agg.round(3).to_markdown(OUT / "tables" / "P3_reach_by_condition.md", index=False)
    best = per[per.S_label == "Sfull"].pivot_table(index="persona", columns="family", values="reach_ok", aggfunc="max")
    best["prompt"] = prm.reindex(best.index); best.round(2).to_markdown(OUT / "tables" / "P3_best_reach_per_persona.md")
    (P3 / "summary.json").write_text(json.dumps(dict(h3_rho=rho, h3_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(ok.sum()),
                                                    full_vector_reach=full.to_dict(), by_condition=agg.round(3).to_dict("records")), indent=1))
    print(agg.round(2).to_string()); print("H3: rho %.2f [%.2f, %.2f]" % (rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args(argv)
    if a.analyse:
        analyse(); return
    if a.plan:
        plan(); return
    import torch
    from ace_steering import steer_gpu as AS
    cells = json.loads((P3 / "cells.json").read_text()); B = np.load(P3 / "basis.npz")
    mats = {"axes": B["axes"], "traits": B["traits"]}
    V = AX / "vectors"
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    rn, sn = json.loads((V / "roles_names.json").read_text()), json.loads((V / "seta_names.json").read_text())
    Rraw = np.load(V / ("roles_rawmean_%s_L%d.npy" % (VIEW, LAYER))); Sraw = np.load(V / ("seta_rawmean_%s_L%d.npy" % (VIEW, LAYER)))
    personas = cells["personas"][a.shard::a.n_shards]
    tok, model = AS.load_model(AS.MODEL)
    enc0, P0 = encode(tok, model, "")
    for rec in personas:
        name = rec["persona"]
        f = P3 / ("gen_%s.jsonl" % name)
        if f.exists():
            continue
        Mt = (Sraw[sn.index(name)] if rec["set"] == "a" else Rraw[rn.index(name)]).astype(np.float64)
        v = Mt - M0
        conds = dict(cells["conditions"])
        conds["full"] = dict(source="v", cols=[0], ks=[1])
        if name in cells["recipes"]:
            conds["recipe"] = dict(source="traits", cols=cells["recipes"][name], ks=[len(cells["recipes"][name])])
        rows = []
        for cname, c in conds.items():
            for k in c["ks"]:
                if c["source"] == "v":
                    U = (v / np.linalg.norm(v))[:, None]
                elif c["source"] == "b2":
                    U = orth(B[cname][:, :k])
                else:
                    U = orth(mats[c["source"]][:, c["cols"][:k]])
                Ut = torch.tensor(U, dtype=torch.float32)
                for slab, S in (("Sfull", rec["S_p"]), ("Shalf", rec["S_p"] / 2)):
                    G = M0 + S * (U @ (U.T @ v))
                    tc = torch.tensor(U.T @ G, dtype=torch.float32)

                    def hook(module, args, Ut=Ut, tc=tc):
                        hs = args[0]
                        uu, cc = Ut.to(hs.device), tc.to(hs.device)
                        h = hs.float()
                        h = h + (cc - h @ uu) @ uu.T
                        return (h.to(hs.dtype),) + tuple(args[1:])
                    cell = "%s_k%d_%s" % (cname, U.shape[1], slab)
                    for sd, ans in enumerate(generate(tok, model, enc0, P0, hook, N_SEEDS)):
                        rows += [dict(persona=name, set=rec["set"], cond=cname, k=int(U.shape[1]), S_label=slab, S=float(S), push=float(np.linalg.norm(S * (U @ (U.T @ v)))),
                                      cell=cell, seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
            print("%s %s done (%d rows) | %r" % (name, cname, len(rows), rows[-10]["response"][:80]), flush=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
