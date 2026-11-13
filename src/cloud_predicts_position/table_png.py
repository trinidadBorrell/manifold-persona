"""Held-out R2 tables rendered as PNG. p-value = two-sided Spearman test, Holm-adjusted within each table,, metric vs position on assistant axis,
over all personas except the default (same points as the scatter figure).
  python -m cloud_predicts_position.table_png --out output/12-nov/tables
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloud_predicts_position.panel import KEEP

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
CONTROLS = ["log_total_var", "centroid_norm_raw", "response_chars"]
I = "200"


def load(folder, L, tag="q"):
    p = folder / "exp1" / f"exp1_avg_L{L}_{tag}_curves.json"
    cv = json.loads(p.read_text())["curves"]
    q = p.with_name(p.name.replace("_curves.json", "_only_curves.json"))
    if q.exists():
        cv.update(json.loads(q.read_text())["curves"])
    return {k: np.array(v[I]["r2"]) for k, v in cv.items() if I in v}


def fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def bonferroni(p):
    p = np.asarray(p, float); return np.minimum(p * len(p), 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m); out[o] = np.minimum(adj, 1); return out


def test(folder, L):
    r2 = load(folder, L)
    df = pd.read_csv(folder / f"panel_avg_L{L}.csv")
    df = df[(df.centring == "q") & (df.role != "default")]
    rows = []
    for m in KEEP:
        ok = df[m].notna()
        rho, p = spearmanr(df.axis_proj[ok], df[m][ok])
        rows.append((m, float(np.median(r2[m])), float(rho), float(p)))
    return sorted(rows, key=lambda r: -r[1])


def draw(cell, header, widths, path, title=None):
    n = len(cell)
    fig, ax = plt.subplots(figsize=(sum(widths), 0.34 * (n + 1) + (0.4 if title else 0.1)))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=header, colWidths=[w / sum(widths) for w in widths], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    for (r, c), x in t.get_celld().items():
        x.set_edgecolor("#d8d7d3")
        if r == 0:
            x.set_height(x.get_height() * (1.7 if "\n" in "".join(header) else 1))
            x.set_facecolor("#f0efeb"); x.set_text_props(weight="bold")
        if c == 0 and r > 0:
            x.set_text_props(ha="left"); x._loc = "left"
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def holm_cols(blocks):
    """blocks: list of row lists (m, r2, rho, p); Holm over the pooled family."""
    flat = [x for b in blocks for x in b]
    raw = [x[3] for x in flat]; ps, pb = holm(raw), bonferroni(raw); out, k = [], 0
    for b in blocks:
        out.append([(m, r, rho, float(ps[k + j]), float(pb[k + j])) for j, (m, r, rho, _) in enumerate(b)]); k += len(b)
    return out


def exp2_table(L=25, i=40):
    d = PM / "CPP" / "exp2"
    cells = pd.read_csv(d / f"exp2_avg_L{L}_cells.csv")
    cells = cells[(cells.centring == "q") & (cells.i == i) & cells.metric.isin(KEEP)]
    g = pd.read_csv(d / f"metrics_avg_L{L}_s0_i{i}.csv")
    g = g[(g.centring == "q") & (g.role != "default")]
    ef = d / f"extras_avg_L{L}_s0_i{i}.csv"
    if ef.exists():
        e = pd.read_csv(ef)
        g = g.merge(e[[c for c in e.columns if c not in g.columns or c in ("role", "split", "centring", "seed", "i")]], on=["role", "split", "centring", "seed", "i"], how="left")
    tr = g[g.split == "train"].set_index("role"); te = g[g.split == "test"].set_index("role")
    common = tr.index.intersection(te.index)
    rows = []
    for m in KEEP:
        c = cells[cells.metric == m]
        a, b = tr.loc[common, m].to_numpy(float), te.loc[common, m].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        rho, ps = spearmanr(a[ok], b[ok])
        rows.append((m, float(c.reliability.mean()), float(c.rmse_ratio.mean()), float(rho), float(ps)))
    return rows


def misalignment_tables(out):
    """EM rates and Experiment 0 shifts. Values are those reported in the article (em_eval / Experiment 0 outputs)."""
    hdr = ["model", "base", "bad medical", "extreme\nsports", "risky\nfinancial", "good medical\n(control)", "secure code\n(control)"]
    em = [["Qwen2.5-0.5B", "0% (14%)", "0.8% (6%)", "2.0% (14%)", "7.0% (16%)", "0% (14%)", "0% (16%)"],
          ["Qwen2.5-7B", "0.3% (94%)", "17.1% (74%)", "12.5% (82%)", "33.3% (80%)", "0% (76%)", "0.8% (64%)"],
          ["Qwen3-8B", "0% (98%)", "14.0% (61%)", "14.5% (76%)", "32.1% (79%)", "0.5% (65%)", "0.5% (60%)"]]
    draw(em, hdr, [1.9] + [1.6] * 6, out / "t4_em_rates.png", title="share of emergently misaligned answers (share of coherent answers)")
    hdr = ["fine-tune", "EM rate", "shift of the default\nassistant on the base axis", "mean shift of all\npersonas on the axis", "order of personas\nkept (Spearman)"]
    e0 = [["risky financial advice", "33%", "−7.5", "−3.1", "0.94"],
          ["bad medical advice", "17%", "−3.8", "+0.1", "0.95"],
          ["extreme sports advice", "12.5%", "−6.0", "−2.0", "0.97"],
          ["good medical advice (control)", "0%", "−1.1", "+2.2", "0.96"],
          ["secure code (control)", "0.8%", "+0.0", "+1.9", "0.99"]]
    draw(e0, hdr, [3.2, 1.1, 2.7, 2.3, 2.3], out / "t5_exp0_shifts.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/12-nov/tables")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    misalignment_tables(out)
    cols = [("base", "cpp_base_q7"), ("bad medical", "cpp_q7_bad_medical_s0"), ("extreme\nsports", "cpp_q7_extreme_sports_s0"),
            ("risky\nfinancial", "cpp_q7_risky_financial_s0"), ("good medical\n(control)", "cpp_q7_good_medical_s0"),
            ("secure code\n(control)", "cpp_q7_secure_s0")]
    # table 1: Qwen3-8B, layer 25
    r1 = test(PM / "CPP", 25)
    (r1,) = holm_cols([r1])
    r1 = sorted(r1, key=lambda r: -r[1])
    cell = [[m, f"{r:.2f}", f"{rho:+.2f}", fmt(ps), fmt(pb)] for m, r, rho, ps, pb in r1]
    draw(cell, ["predictor", "held-out R²", "Spearman", "Spearman\nHolm p", "Spearman\nBonferroni p"], [3.0, 1.4, 1.2, 1.5, 1.9], out / "t1_exp1_pvalues.png")
    json.dump(r1, open(out / "t1.json", "w"), indent=1)
    # table 2: Qwen2.5-7B layer 19, six models, one family of 72 tests per p-value
    res = [test(PM / "EM" / d, 19) for _, d in cols]
    res = dict(zip([c[0] for c in cols], holm_cols(res)))
    order = sorted(KEEP, key=lambda m: -abs(next(x[2] for x in res["base"] if x[0] == m)))
    cell = []
    for m in order:
        row = [m]
        for name, _ in cols:
            _, r, rho, ps, pb = next(x for x in res[name] if x[0] == m)
            row.append(f"{rho:+.2f} ({fmt(ps)}; {fmt(pb)})")
        cell.append(row)
    draw(cell, ["predictor"] + [c[0] for c in cols], [2.6] + [2.8] * 6, out / "t2_finetune_pvalues.png", title="Spearman correlation with position (Holm p; Bonferroni p)")
    json.dump(res, open(out / "t2.json", "w"), indent=1)
    # table 3: Exp 2, 40 questions
    r3 = exp2_table(25, 40)
    ps, pb = holm([x[4] for x in r3]), bonferroni([x[4] for x in r3])
    r3 = sorted([(m, rel, rm, rho, float(a_), float(b_)) for (m, rel, rm, rho, _), a_, b_ in zip(r3, ps, pb)], key=lambda r: -r[1])
    cell = [[m, f"{rel:.2f}", f"{rm:.2f}", f"{rho:+.2f}", fmt(a_), fmt(b_)] for m, rel, rm, rho, a_, b_ in r3]
    draw(cell, ["predictor", "reliability", "RMSE ratio", "Spearman", "Spearman\nHolm p", "Spearman\nBonferroni p"], [3.0, 1.3, 1.4, 1.2, 1.5, 1.9], out / "t3_exp2_pvalues.png")
    json.dump(r3, open(out / "t3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
