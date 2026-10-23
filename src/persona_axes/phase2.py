"""Phase 2 analysis (plan v2 s5 Phase 2): on/off-target dose response, purity, blind naming, stability,
the drop rule, set 9 (pruned union), figures P2.1 and P2.2.

    python -m persona_axes.phase2          # -> $PM/AX21/phase2/results.json, output/21-oct/P2_*.png, tables

Scores. r(s) = the judge's 1-5 rating of scale j on the answers steered along axis i at dose s. The axis
vector is pole_a - pole_b and 1 = pole A on the scale, so on-target movement is rho(s, r) < 0; we report
on_target = -rho(s, r_ii) (Spearman over the 70 answers) and off_target[i, j] = rho(s, r_ij) for j != i.
purity_i = on_target_i - mean_j |off_target_ij| over the 30 other candidate scales (nuisance scales shown
separately). Drop rule (pre-registered): keep an axis if on_target >= 0.5, purity >= 0.25, both split-half
cosines >= 0.6, blind naming >= 60%.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from persona_axes import AX, HERE, LAYER, OUT, VIEW

RULE = dict(on_target=0.5, purity=0.25, split_cos=0.6, naming=0.6)


import sys

RATE_DIR = AX / ("rate_" + sys.argv[1] if len(sys.argv) > 1 else "rate")     # python -m persona_axes.phase2 [dose_x4]


def load_ratings():
    rows = []
    for f in (RATE_DIR / "calls").glob("*.json"):
        d = json.loads(f.read_text())
        for scale, v in d["reply"].items():
            rows.append(dict(axis=d["axis"], s=d["s"], qi=d["qi"], scale=scale, r=v["r"]))
    return pd.DataFrame(rows)


def main():
    ax = json.loads((HERE / "axes.json").read_text())["axes"]
    names = [a["name"] for a in ax]
    fam = {a["name"]: a["family"] for a in ax}
    cand = [n for n in names if fam[n] != "N"]
    df = load_ratings()
    M = np.full((len(names), len(names)), np.nan)
    for i, ai in enumerate(names):
        d = df[df.axis == ai]
        for j, sj in enumerate(names):
            x = d[d.scale == sj]
            if len(x) >= 20:
                M[i, j] = spearmanr(x.s, x.r).correlation
    on = {n: float(-M[i, i]) for i, n in enumerate(names)}
    off = {n: float(np.nanmean([abs(M[i, j]) for j, m in enumerate(names) if m != n and fam[m] != "N"])) for i, n in enumerate(names)}
    off_nuis = {n: float(np.nanmax([abs(M[i, j]) for j, m in enumerate(names) if fam[m] == "N" and m != n])) for i, n in enumerate(names)}
    purity = {n: on[n] - off[n] for n in names}
    # blind naming
    nm = {}
    for f in (RATE_DIR / "naming").glob("*_match.json"):
        d = json.loads(f.read_text())
        nm.setdefault(d["axis"], []).append(d["reply"]["choice"] == d["correct"])
    naming = {n: float(np.mean(nm.get(n, [np.nan]))) for n in names}
    labels = {}
    for f in sorted((RATE_DIR / "naming").glob("*_label.json")):
        d = json.loads(f.read_text()); labels.setdefault(d["axis"], []).append("%s <-> %s" % (d["reply"]["pole_A"], d["reply"]["pole_B"]))
    # stability from phase 1
    p1 = json.loads((AX / "phase1" / ("results_%s_L%d.json" % (VIEW, LAYER))).read_text())
    st = p1["axis_stability"]
    tab = pd.DataFrame(dict(family=[fam[n] for n in names], on_target=[on[n] for n in names], mean_off_target=[off[n] for n in names],
                            max_off_nuisance=[off_nuis[n] for n in names], purity=[purity[n] for n in names],
                            split_pairs=[st["split_pairs"][n] for n in names], split_q=[st["split_q"][n] for n in names],
                            layer_m2=[st["layer_-2"][n] for n in names], layer_p2=[st["layer_+2"][n] for n in names],
                            naming=[naming[n] for n in names]), index=names)
    tab["kept"] = (tab.on_target >= RULE["on_target"]) & (tab.purity >= RULE["purity"]) & (tab.split_pairs >= RULE["split_cos"]) \
        & (tab.split_q >= RULE["split_cos"]) & (tab.naming >= RULE["naming"]) & (tab.family != "N")
    (AX / "phase2").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    tab.round(3).to_csv(AX / "phase2" / "axis_table.csv")
    tab.round(2).to_markdown(OUT / "tables" / "P2_axis_table.md")
    set9 = sorted(tab.index[tab.kept])
    # marginal M1 contribution of each candidate axis to set 9 on set (b): R2_m(set9) - R2_m(set9 \ axis), or + axis for dropped ones
    V = AX / "vectors"
    A = np.load(V / ("axes_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64); A = (A / np.linalg.norm(A, axis=1, keepdims=True)).T
    rn = json.loads((V / "roles_names.json").read_text())
    human = json.loads((HERE / "personas_b_human.json").read_text())["roles"]
    X = np.load(V / ("roles_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)[[rn.index(r) for r in human]]
    Y = X - X.mean(0)

    def r2(cols):
        if not cols:
            return 0.0
        Q, _ = np.linalg.qr(A[:, cols]); C = Y @ Q
        return float(1 - ((Y ** 2).sum() - (C ** 2).sum()) / (Y ** 2).sum())
    base = [names.index(n) for n in set9]
    marg = {}
    for n in cand:
        i = names.index(n)
        marg[n] = r2(base) - r2([c for c in base if c != i]) if i in base else r2(base + [i]) - r2(base)
    tab["marginal_r2_set9"] = [marg.get(n, np.nan) for n in names]
    res = dict(rule=RULE, set9=set9, dropped=[n for n in cand if n not in set9], on_target=on, off_target_mean=off, purity=purity,
               naming=naming, labels=labels, matrix=M.tolist(), names=names, marginal_r2_set9=marg, r2_set9_b=r2(base),
               n_ratings=int(len(df)))
    (AX / "phase2" / "results.json").write_text(json.dumps(res))
    figs(M, names, fam, tab)
    print("set 9 = %d axes kept of %d; dropped: %s" % (len(set9), len(cand), res["dropped"]))
    print(tab.round(2).to_string())


def figs(M, names, fam, tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(range(len(names)), key=lambda i: (fam[names[i]], i))
    Ms = M[np.ix_(order, order)]; lab = [names[i] for i in order]
    fig, axp = plt.subplots(figsize=(13, 11))
    im = axp.imshow(Ms, cmap="RdBu_r", vmin=-1, vmax=1)
    axp.set_xticks(range(len(lab))); axp.set_xticklabels(lab, rotation=90, fontsize=7)
    axp.set_yticks(range(len(lab))); axp.set_yticklabels(lab, fontsize=7)
    axp.set_xlabel("scale rated by the judge (1 = pole A ... 5 = pole B)")
    axp.set_ylabel("axis steered along (dose s = -3 ... +3 sigma)")
    axp.set_title("P2.1  Spearman rho between dose and rating, Qwen3-8B L15 clamp; diagonal = on-target (expected negative)")
    plt.colorbar(im, ax=axp, fraction=0.03, label="Spearman rho(dose, rating)")
    for k in range(len(lab) - 1):
        if fam[lab[k]] != fam[lab[k + 1]]:
            axp.axhline(k + 0.5, color="k", lw=0.5); axp.axvline(k + 0.5, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(OUT / "P2_1_off_target_matrix.png", dpi=130); plt.close(fig)
    t = tab[tab.family != "N"]
    fig, axp = plt.subplots(figsize=(9, 7))
    cols = {f: c for f, c in zip("ABCDEFGH", plt.cm.tab10.colors)}
    sc = axp.scatter(t.marginal_r2_set9, t.purity, s=40 + 160 * t.split_pairs.clip(0, 1), c=[cols[f] for f in t.family], alpha=0.8)
    for n, r in t.iterrows():
        axp.annotate(n, (r.marginal_r2_set9, r.purity), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axp.axhline(RULE["purity"], ls="--", c="grey", lw=0.8)
    axp.set_xlabel("marginal held-in R2 of the axis in the pruned union (set 9), human roles, L15 mean")
    axp.set_ylabel("purity = on-target rho - mean |off-target rho|")
    axp.set_title("P2.2  coverage contribution vs purity per axis (size = split-half cosine, colour = family)")
    for f, c in cols.items():
        axp.scatter([], [], c=[c], label="family " + f)
    axp.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "P2_2_coverage_vs_purity.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
