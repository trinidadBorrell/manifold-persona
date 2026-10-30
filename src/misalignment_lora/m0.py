"""Experiment M0 (plan 2026-10-29, section 5.5): how the persona geometry changes after the fine-tune.

Inputs, one folder per model made by `cloud_predicts_position.axis` on the 20-question clouds:
  $PM/EM/cpp_base20/axis_avg_L<L>.npz            base model, the same 20 questions
  $PM/EM/cpp_<model>/axis_avg_L<L>.npz           fine-tuned model (insecure_s0, secure_s0, ...)
Each holds the 276 raw centroids, the default centroid, the assistant axis and axis_proj.

Measures (all in the BASE frame unless said):
  centroid shift      delta_p = c_p^FT - c_p^BASE: norm; cos with the base axis; cos with the persona's own base vector
  common shift        d_EM = mean_p delta_p; share of sum |delta_p|^2 explained by the translation, by translation +
                      similarity Procrustes (rotation + scale), and the residual
  position, base frame   axis_proj^{FT|BASE} = <c_p^FT - mean_p c_p^BASE, a_BASE> vs axis_proj^BASE: Spearman, slope, mean shift
  position, refit frame  the FT model's own axis; cos(a_FT, a_BASE)
  assistant point     delta_default and its projection on a_BASE
  which persona       cos(d_EM, c_p^BASE - c_default^BASE) over personas, top and bottom 10
  global shape        on the 276 centroids: PCA components for 90% variance, TwoNN, linear CKA(BASE, FT), path-ratio
Everything is reported for every FT model; the effect of misalignment is INSEC minus SEC.

    python -m misalignment_lora.m0 --models insecure_s0,secure_s0 --layers 19,25,32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr

from pm_geometry.components import twonn, knn_graph, ratios, linear_cka

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
EM = PM / "EM"


def load(folder, L, view="avg"):
    z = np.load(folder / f"axis_{view}_L{L}.npz", allow_pickle=True)
    roles = list(z["roles"])
    return dict(roles=roles, C=z["centroids"].astype(np.float64), d=z["default_centroid"].astype(np.float64),
                a=z["axis"].astype(np.float64), proj=z["axis_proj"].astype(np.float64), i_def=roles.index("default"))


def pca_n90(C):
    s = np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1)


def shape(C):
    Z = C - C.mean(0)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    Zc = U * s
    g = knn_graph(Zc)
    rt = ratios(g)
    return dict(pca_n90=pca_n90(C), twonn=float(twonn(Zc)), path_ratio_p50=float(np.median(rt)),
                mean_centroid_norm=float(np.linalg.norm(C, axis=1).mean()),
                spread=float(np.sqrt((Z ** 2).sum(1).mean())))


def compare(base, ft):
    assert base["roles"] == ft["roles"]
    roles = base["roles"]
    Cb, Cf = base["C"], ft["C"]
    delta = Cf - Cb
    a = base["a"]
    own = Cb - base["d"]
    own_u = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-12)
    dn = np.linalg.norm(delta, axis=1)
    cos_axis = delta @ a / (dn + 1e-12)
    cos_own = (delta * own_u).sum(1) / (dn + 1e-12)
    d_em = delta.mean(0)
    tot = float((delta ** 2).sum())
    resid_T = delta - d_em
    # similarity Procrustes of the centred clouds: Cf - mean ≈ s R (Cb - mean)
    Xb, Xf = Cb - Cb.mean(0), Cf - Cf.mean(0)
    R, _ = orthogonal_procrustes(Xb, Xf)
    sc = float((Xf * (Xb @ R)).sum() / (Xb ** 2).sum())
    resid_P = Xf - sc * (Xb @ R)
    proj_ft_in_base = (Cf - Cb.mean(0)) @ a
    pb = base["proj"]
    slope = float(np.polyfit(pb, proj_ft_in_base, 1)[0])
    d_def = delta[base["i_def"]]
    cos_persona = (d_em @ own_u.T) / (np.linalg.norm(d_em) + 1e-12)
    order = np.argsort(cos_persona)
    out = dict(
        shift_norm_mean=float(dn.mean()), shift_norm_median=float(np.median(dn)),
        shift_norm_over_base_spread=float(dn.mean() / np.sqrt((Xb ** 2).sum(1).mean())),
        cos_shift_axis_mean=float(cos_axis.mean()), cos_shift_axis_abs_mean=float(np.abs(cos_axis).mean()),
        cos_shift_own_mean=float(cos_own.mean()),
        common_shift_norm=float(np.linalg.norm(d_em)), common_shift_cos_axis=float(d_em @ a / np.linalg.norm(d_em)),
        share_translation=float(1 - (resid_T ** 2).sum() / tot),
        share_translation_plus_similarity=float(1 - (resid_P ** 2).sum() / tot),
        procrustes_scale=sc, procrustes_rotation_angle_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - (len(R) - 2)) / 2, -1, 1)))) if False else None,
        position_spearman_base_frame=float(spearmanr(pb, proj_ft_in_base)[0]),
        position_slope_base_frame=slope, position_mean_shift_base_frame=float((proj_ft_in_base - pb).mean()),
        position_spearman_refit_frame=float(spearmanr(pb, ft["proj"])[0]),
        cos_axis_ft_vs_base=float(ft["a"] @ a),
        default_shift_norm=float(np.linalg.norm(d_def)), default_shift_proj_on_axis=float(d_def @ a),
        common_shift_resembles=[dict(role=roles[i], cos=float(cos_persona[i])) for i in order[::-1][:10]],
        common_shift_opposes=[dict(role=roles[i], cos=float(cos_persona[i])) for i in order[:10]],
        shift_norm_vs_base_position_spearman=float(spearmanr(pb, dn)[0]),
        cka_base_ft=float(linear_cka(Xb, Xf)),
        shape_base=shape(Cb), shape_ft=shape(Cf),
        per_persona=pd.DataFrame(dict(role=roles, shift_norm=dn, cos_shift_axis=cos_axis, cos_shift_own=cos_own,
                                      proj_base=pb, proj_ft_in_base_frame=proj_ft_in_base, proj_ft_refit=ft["proj"])),
    )
    out.pop("procrustes_rotation_angle_deg")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="insecure_s0,secure_s0")
    ap.add_argument("--layers", default="19,25,32")
    ap.add_argument("--out", default="output/29-oct/partM")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = {}
    rows = []
    for L in [int(x) for x in a.layers.split(",")]:
        base = load(EM / "cpp_base20", L)
        for m in a.models.split(","):
            f = EM / f"cpp_{m}" / f"axis_avg_L{L}.npz"
            if not f.exists():
                print("missing", f); continue
            r = compare(base, load(EM / f"cpp_{m}", L))
            pp = r.pop("per_persona"); pp["model"] = m; pp["layer"] = L; rows.append(pp)
            summ[f"{m}_L{L}"] = r
            print(f"== {m} L{L}: shift {r['shift_norm_mean']:.1f} (/spread {r['shift_norm_over_base_spread']:.2f}), translation {r['share_translation']:.2f}, "
                  f"+similarity {r['share_translation_plus_similarity']:.2f}, position Spearman base frame {r['position_spearman_base_frame']:.3f}, "
                  f"mean shift {r['position_mean_shift_base_frame']:+.2f}, default shift on axis {r['default_shift_proj_on_axis']:+.2f}, "
                  f"cos(a_FT,a_BASE) {r['cos_axis_ft_vs_base']:.3f}, CKA {r['cka_base_ft']:.3f}")
            print("   resembles:", ", ".join(f"{x['role']} {x['cos']:.2f}" for x in r["common_shift_resembles"][:6]))
            print("   opposes:  ", ", ".join(f"{x['role']} {x['cos']:.2f}" for x in r["common_shift_opposes"][:6]))
    pd.concat(rows).to_csv(out / "m0_per_persona.csv", index=False)
    (out / "m0_summary.json").write_text(json.dumps(summ, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
