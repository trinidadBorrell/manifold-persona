"""Questions (1) and (2) of 2026-11-11, from the existing panel tables (no new extraction).
(1) EM rate of each model (x) vs the persona-mean of each cloud metric (y), per family and layer.
(2) Plot: `python -m misalignment_lora.ba_trends plots q7 19` (metric vs closeness, models overlaid).
(2) Is the shape "metric vs closeness to the assistant" the same in the fine-tuned models as in the base?
    Spearman across metrics of rho_base vs rho_finetuned, and the number of sign flips among the metrics with |rho_base| > 0.3.
    python -m misalignment_lora.ba_trends     -> output/10-nov/partEM/{em_vs_metric_mean,shape_preserved_rho}.csv"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from misalignment_lora import ba_figs as B

EM = B.EM
OUT = B.OUT / "partEM"


def em_vs_metric():
    rates = pd.read_csv(EM / "em_eval/em_rates.csv").set_index("model")
    rows = []
    for fam, d in B.FAMS.items():
        base_em = "base" if fam == "q3" else f"base_{fam}"
        for L in d["layers"]:
            pb = B.F.panel_q(EM / d["base"], L)
            cols = [c for c in pb.select_dtypes("number").columns if pb[c].std() > 0 and c not in ("layer", "n_roles")]
            P = {m: B.F.panel_q(EM / f"cpp_{m}", L) for m in B.avail(fam)}
            em = [rates.loc[m, "em_rate"] for m in P] + [rates.loc[base_em, "em_rate"]]
            for c in cols:
                y = [P[m][c].mean() for m in P] + [pb[c].mean()]
                if np.isfinite(y).all() and len(set(y)) > 1:
                    r, p = spearmanr(em, y)
                    rows.append(dict(family=fam, layer=L, metric=c, n=len(y), rho=r, p=p))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "em_vs_metric_mean.csv", index=False)
    return t


def shape_preserved():
    rows = []
    for fam, d in B.FAMS.items():
        for L in d["layers"]:
            r = pd.read_csv(B.OUT / f"partM_{fam}" / f"m1_relation_vs_change_L{L}.csv")
            for m in [c[4:] for c in r.columns if c.startswith("rho_") and c not in ("rho_base", "rho_base_lo", "rho_base_hi")]:
                x, y = r.rho_base, r[f"rho_{m}"]
                strong = x.abs() > 0.3
                rows.append(dict(family=fam, layer=L, model=m, corr_rho=spearmanr(x, y)[0], med_abs_change=float((y - x).abs().median()),
                                 flips=int((strong & (np.sign(x) != np.sign(y)) & (y.abs() > 0.1)).sum()), n_strong=int(strong.sum())))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "shape_preserved_rho.csv", index=False)
    return s


def metric_vs_position(fam, L):
    """Plot (2): x = persona's projection on the assistant axis (own frame, + = closer to the assistant), y = cloud metric;
    base + every fine-tune of the family overlaid, with the median of 10 position bins per model. Fixed family and layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cloud_predicts_position.figs import FAMILY_NAMES, family_of as _fo
    from cloud_predicts_position.panel import SHOWN, KEEP

    def family_of(c):
        try:
            return _fo(c)
        except StopIteration:
            return "X"
    FAMILY_NAMES = dict(FAMILY_NAMES, X="other (response length, raw norm)")
    d = B.FAMS[fam]
    srcs = {"base": EM / d["base"]} | {m: EM / f"cpp_{m}" for m in B.avail(fam)}
    P = {k: B.F.panel_q(f, L) for k, f in srcs.items()}
    col = {k: (B.COL["base"] if k == "base" else B.COL[B.cond_of(k)]) for k in P}
    lab = {k: ("base" if k == "base" else B.CLAB[B.cond_of(k)]) for k in P}
    cols = [c for c in SHOWN if c in P["base"].columns and P["base"][c].std() > 0]
    out = B.OUT / f"partTrends_{fam}" / f"L{L}"; out.mkdir(parents=True, exist_ok=True)

    def draw(ax, c):
        for k, df in P.items():
            x, y = df.axis_proj.to_numpy(float), df[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            ax.scatter(x, y, s=4, color=col[k], alpha=0.25, linewidths=0)
            edges = np.quantile(x, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(x, edges[1:-1]), 0, 9)
            ax.plot([x[idx == i].mean() for i in range(10)], [np.median(y[idx == i]) for i in range(10)], color=col[k], lw=1.8,
                    label=f"{lab[k]} (ρ {spearmanr(x, y)[0]:+.2f})")
        ax.set_title(f"{c} [{FAMILY_NAMES[family_of(c)]}]", fontsize=8)

    for c in cols:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        draw(ax, c)
        ax.set_xlabel("persona's projection on the assistant axis (+ = closer to the assistant)"); ax.set_ylabel(c)
        ax.legend(frameon=False, fontsize=6.5)
        fig.suptitle(f"{d['name']}, layer {L}", fontsize=8.5)
        fig.tight_layout(); fig.savefig(out / f"{c}.png", dpi=110); plt.close(fig)
    prim = [c for c in KEEP if c in cols]
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    for ax, c in zip(axes.ravel(), prim):
        draw(ax, c)
        ax.set_xlabel("projection on the assistant axis", fontsize=7)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle(f"{d['name']}, layer {L}: metric vs closeness to the assistant; lines = median per position decile, one colour per model", fontsize=9)
    fig.tight_layout(); fig.savefig(B.OUT / f"partTrends_{fam}" / f"F17_primary_metrics_vs_position_L{L}.png", dpi=120); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plots":
        metric_vs_position(sys.argv[2], int(sys.argv[3]))
        raise SystemExit
    t = em_vs_metric()
    for fam in ("q7", "q3", "q05"):
        s = t[t.family == fam]
        print(fam, "metrics/layer", s.groupby("layer").size().to_dict(), "| |rho|>=0.8:", s[s.rho.abs() >= 0.8].groupby("layer").size().to_dict(),
              "| p<0.05:", s[s.p < 0.05].groupby("layer").size().to_dict())
    for fam, L in (("q7", 19), ("q3", 25)):
        s = t[(t.family == fam) & (t.layer == L)]
        print(fam, L); print(s.reindex(s.rho.abs().sort_values(ascending=False).index).head(8)[["metric", "rho", "p"]].round(2).to_string(index=False))
    a = t[(t.family == "q7") & (t.layer == 19)].set_index("metric").rho
    b = t[(t.family == "q3") & (t.layer == 25)].set_index("metric").rho
    j = pd.concat([a, b], axis=1, keys=["q7", "q3"]).dropna()
    print("q7 vs q3 mid layer: Spearman of rho over metrics", round(spearmanr(j.q7, j.q3)[0], 2), "| both |rho|>=0.7 same sign:",
          int(((j.q7.abs() >= .7) & (j.q3.abs() >= .7) & (np.sign(j.q7) == np.sign(j.q3))).sum()), "of", len(j))
    s = shape_preserved()
    print(s.groupby(["family", "layer"]).agg(corr_min=("corr_rho", "min"), corr_med=("corr_rho", "median"), med_abs_change=("med_abs_change", "median"),
                                              flips_max=("flips", "max"), n_strong=("n_strong", "max")).round(2).to_string())
