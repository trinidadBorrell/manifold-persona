"""Collect everything the results page needs into one JSON (plan 2026-10-29, Parts B and M).

    python -m misalignment_lora.page_data --out <scratch>/page_data.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cloud_predicts_position import data as D
from cloud_predicts_position.panel import FAMILIES, PRIMARY
from cloud_predicts_position.exp1 import SCALE

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"
LAYERS = (19, 25, 32)
MODELS = {"base240": PM / "CPP", "base20": EM / "cpp_base20", "insecure": EM / "cpp_insecure_s0", "secure": EM / "cpp_secure_s0"}
R = lambda v, n=3: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def metric_cols(df):
    return [c for fam in FAMILIES.values() for c in fam if c in df.columns and df[c].notna().mean() > 0.9 and df[c].std() > 0]


def panel_rows(folder, L):
    df = pd.read_csv(folder / f"panel_avg_L{L}.csv")
    df = df[(df.centring == "q") & (df.role != "default")].sort_values("role")
    cols = metric_cols(df)
    return df, cols


def forest(df, cols):
    y = df.axis_proj.to_numpy(float)
    S = df[SCALE].to_numpy(float)
    out = {}
    rng = np.random.default_rng(0)
    for c in cols:
        m = df[c].to_numpy(float); ok = np.isfinite(m)
        rho = spearmanr(m[ok], y[ok])[0]
        A = np.hstack([np.ones((ok.sum(), 1)), S[ok]])
        ry = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
        rm = m[ok] - A @ np.linalg.lstsq(A, m[ok], rcond=None)[0]
        boots = [spearmanr(m[ok][b], y[ok][b])[0] for b in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(200))]
        out[c] = dict(rho=R(rho), lo=R(np.percentile(boots, 2.5)), hi=R(np.percentile(boots, 97.5)), rho_adj=R(spearmanr(ry, rm)[0]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = dict(families={k: v for k, v in FAMILIES.items()}, primary=PRIMARY, layers=list(LAYERS), models={})
    # per model x layer: per-persona values (metrics + axis_proj + scale cols), forest, exp1 summary + curves
    for name, folder in MODELS.items():
        out["models"][name] = {}
        for L in LAYERS:
            f = folder / f"panel_avg_L{L}.csv"
            if not f.exists():
                continue
            df, cols = panel_rows(folder, L)
            keep = cols + [c for c in SCALE if c not in cols] + ["axis_proj"]
            rec = dict(roles=df.role.tolist(), cols=keep,
                       values={c: [R(v) for v in df[c].to_numpy(float)] for c in keep}, forest=forest(df, cols))
            e = folder / "exp1" / f"exp1_avg_L{L}_q.csv"
            if e.exists():
                t = pd.read_csv(e)
                rec["exp1"] = {r.metric: dict(r2=R(r.r2_test_median), lo=R(r.r2_test_lo), hi=R(r.r2_test_hi), rho=R(r.spearman_test),
                                              r2_scale=R(r.r2_scale_only), r2_adj=R(r.r2_metric_plus_scale), delta=R(r.delta_r2_over_scale),
                                              perm95=R(r.perm_p95_this_metric), verdict=r.verdict, family=r.family)
                               for r in t.itertuples()}
                cv = json.loads((folder / "exp1" / f"exp1_avg_L{L}_q_curves.json").read_text())
                rec["exp1_curves"] = {c: {i: dict(r2=R(np.median(v["r2"])), lo=R(np.percentile(v["r2"], 2.5)), hi=R(np.percentile(v["r2"], 97.5)),
                                                  scale=R(np.median(v["r2_scale"])), adj=R(np.median(v["r2_adj"]))) for i, v in d.items()}
                                      for c, d in cv["curves"].items()}
                rec["ridge"] = {i: R(v["r2_median"]) for i, v in cv["ridge"].items()}
                rec["perm_max95"] = R(np.percentile(cv["perm_max"], 95))
            out["models"][name][str(L)] = rec
    # Experiment 2 (base, 240 questions, L25): summary and the cells at i = 10, 40, 120 (seed 0)
    s = pd.read_csv(PM / "CPP" / "exp2" / "exp2_avg_L25_summary.csv"); s = s[s.centring == "q"]
    out["exp2"] = dict(summary={m: {str(int(r.i)): dict(rel=R(r.reliability), rmse_ratio=R(r.rmse_ratio), r2_held=R(r.r2_held_pos))
                                    for r in g.itertuples()} for m, g in s.groupby("metric")}, cells={})
    ax = np.load(PM / "CPP" / "axis_avg_L25.npz", allow_pickle=True)
    pos = pd.Series(ax["axis_proj"].astype(float), index=ax["roles"])
    for i in (5, 10, 20, 40, 120):
        f = PM / "CPP" / "exp2" / f"metrics_avg_L25_s0_i{i}.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f); g = g[(g.centring == "q") & (g.role != "default")]
        tr = g[g.split == "train"].set_index("role").sort_index(); te = g[g.split == "test"].set_index("role").sort_index()
        common = tr.index.intersection(te.index)
        mcols = [c for c in tr.columns if c in sum(FAMILIES.values(), []) or c in SCALE]
        out["exp2"]["cells"][str(i)] = dict(roles=list(common), axis_proj=[R(v) for v in pos.loc[common]],
                                             train={c: [R(v) for v in tr.loc[common, c]] for c in mcols},
                                             test={c: [R(v) for v in te.loc[common, c]] for c in mcols})
    # question-wise cells on the 20-question clouds (base20, insecure, secure), layer 25, seed 0, i = 5 and 10
    out["exp2_ft"] = {}
    for name, folder in (("base20", EM / "cpp_base20"), ("insecure", EM / "cpp_insecure_s0"), ("secure", EM / "cpp_secure_s0")):
        z = np.load(folder / "axis_avg_L25.npz", allow_pickle=True); posf = pd.Series(z["axis_proj"].astype(float), index=z["roles"])
        cells = {}
        for i in (5, 10):
            f = folder / "exp2" / f"metrics_avg_L25_s0_i{i}.csv"
            if not f.exists():
                continue
            g = pd.read_csv(f); g = g[(g.centring == "q") & (g.role != "default")]
            tr = g[g.split == "train"].set_index("role").sort_index(); te = g[g.split == "test"].set_index("role").sort_index()
            common = tr.index.intersection(te.index)
            mcols = [c for c in tr.columns if c in sum(FAMILIES.values(), []) or c in SCALE]
            cells[str(i)] = dict(roles=list(common), axis_proj=[R(v) for v in posf.loc[common]],
                                 train={c: [R(v) for v in tr.loc[common, c]] for c in mcols}, test={c: [R(v) for v in te.loc[common, c]] for c in mcols})
        if cells:
            out["exp2_ft"][name] = cells
    # gate
    g = pd.read_csv(EM / "em_eval" / "em_rates.csv")
    out["gate"] = [dict(model=r.model, n=int(r.n), em=R(r.em_rate), aligned=R(r.aligned_mean, 1), coherent=R(r.coherent_mean, 1),
                        coherent_share=R(r.coherent_above_50), refusal_or_code=R(r.refusal_or_code)) for r in g.itertuples()]
    tl = {}
    for m in ("insecure_s0", "secure_s0", "insecure_s0_retry", "secure_s0_retry"):
        f = EM / f"lora_{m}" / "train_log.json"
        if f.exists():
            l = json.loads(f.read_text()); tl[m] = dict(steps=l["steps"], loss=R(l["train_loss"]), seconds=R(l["seconds"], 0), peak_gb=R(l["peak_mem_gb"], 1),
                                                       tokens_median=l["tokens_median"], lr=l["lr"], epochs=l["epochs"])
    out["train"] = tl
    # geometry change: positions in the base frame, axis rotation, default shift, common-shift resemblance, metric changes
    out["shift"] = {}
    for L in LAYERS:
        zb = np.load(EM / "cpp_base20" / f"axis_avg_L{L}.npz", allow_pickle=True)
        roles = list(zb["roles"]); Cb = zb["centroids"].astype(np.float64); a = zb["axis"].astype(np.float64); pb = zb["axis_proj"].astype(np.float64)
        i_def = roles.index("default"); own = Cb - Cb[i_def]; own_u = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-12)
        spread = float(np.sqrt(((Cb - Cb.mean(0)) ** 2).sum(1).mean()))
        rec = {}
        for m in ("insecure", "secure"):
            zf = np.load(EM / f"cpp_{m}_s0" / f"axis_avg_L{L}.npz", allow_pickle=True)
            Cf = zf["centroids"].astype(np.float64); af = zf["axis"].astype(np.float64)
            delta = Cf - Cb; dn = np.linalg.norm(delta, axis=1); d_em = delta.mean(0)
            pf = (Cf - Cb.mean(0)) @ a
            cosp = (d_em @ own_u.T) / np.linalg.norm(d_em); order = np.argsort(cosp)
            rec[m] = dict(shift_mean=R(dn.mean(), 1), shift_over_spread=R(dn.mean() / spread), common_norm=R(np.linalg.norm(d_em), 1),
                          share_translation=R(1 - ((delta - d_em) ** 2).sum() / (delta ** 2).sum()),
                          common_cos_axis=R(d_em @ a / np.linalg.norm(d_em)),
                          pos_spearman=R(spearmanr(pb, pf)[0]), pos_mean_shift=R((pf - pb).mean(), 2), pos_slope=R(np.polyfit(pb, pf, 1)[0]),
                          default_shift_on_axis=R(delta[i_def] @ a, 2), cos_axis=R(af @ a), shift_vs_pos_spearman=R(spearmanr(pb, dn)[0]),
                          resembles=[dict(role=roles[j], cos=R(cosp[j], 2)) for j in order[::-1][:8]],
                          opposes=[dict(role=roles[j], cos=R(cosp[j], 2)) for j in order[:8]],
                          pos_base=[R(v, 2) for v in pb], pos_ft=[R(v, 2) for v in pf])
        rec["roles"] = roles; rec["spread"] = R(spread, 1)
        # metric changes in base-SD units
        b, cols = panel_rows(EM / "cpp_base20", L)
        b = b.set_index("role")
        ch = {}
        rng = np.random.default_rng(0)
        P = {m: panel_rows(EM / f"cpp_{m}_s0", L)[0].set_index("role") for m in ("insecure", "secure")}
        common = b.index.intersection(P["insecure"].index).intersection(P["secure"].index)
        for c in cols:
            sd = b[c].std()
            di = ((P["insecure"].loc[common, c] - b.loc[common, c]) / sd).to_numpy(); ds = ((P["secure"].loc[common, c] - b.loc[common, c]) / sd).to_numpy()
            ok = np.isfinite(di) & np.isfinite(ds); d = di[ok] - ds[ok]
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
            ch[c] = dict(ins=R(np.nanmean(di)), sec=R(np.nanmean(ds)), diff=R(d.mean()), lo=R(np.percentile(boots, 2.5)), hi=R(np.percentile(boots, 97.5)))
        rec["metric_change"] = ch
        out["shift"][str(L)] = rec
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", args.out, Path(args.out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
