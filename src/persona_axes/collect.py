"""Phase 0: consolidate the extraction chunks and compute every vector the later phases use (CPU, condor).

    python -m persona_axes.collect --part traits|axes|seta     # chunks -> meta.csv, last_L*.npy, mean_L*.npy
    python -m persona_axes.collect --vectors                   # -> $PM/AX21/vectors/

Vectors, per view in {last, mean} and layer 12-32, float32 (n, 4096):
  traits_{v}_L{L}.npy   mean(pos answers) - mean(neg answers), 240 traits (traits_names.json)
  axes_{v}_L{L}.npy     the same for the 35 axes (axes_names.json); plus split halves for stability:
                        axes_{v}_L{L}_pairsA/pairsB (pairs 0-2 / 3-4) and _qA/_qB (questions 0-19 / 20-39)
  seta_{v}_L{L}.npy     mean_j (h_ij - h0_j), h0 = E1 neutral answer to the same question (41 personas)
  roles_{v}_L{L}.npy    the same for the 276 E1 roles (roles_names.json)
  *_rawmean_{v}_L{L}    the raw mean state (no neutral subtracted): the clamp's M_t; neutral_rawmean = M_0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from persona_axes import AX, HERE, LAYERS, NEUTRAL, PM


def consolidate(part):
    d = AX / part
    files = sorted((d / "chunks").glob("u*.npz"))
    n_units = json.loads((d / "units.json").read_text())["n_units"]
    meta = pd.concat([pd.read_csv(f.with_suffix(".csv")) for f in files], ignore_index=True)
    assert len(meta) == n_units and (meta.unit.to_numpy() == np.arange(n_units)).all(), (len(meta), n_units)
    meta.to_csv(d / "meta.csv", index=False)
    for view in ("last", "mean"):
        for li, L in enumerate(LAYERS):
            arr = np.concatenate([np.load(f)[view][:, li] for f in files])
            assert arr.shape == (n_units, 4096), arr.shape
            np.save(d / ("%s_L%d.npy" % (view, L)), arr.astype(np.float16))
    print("%s: %d units, %d chunks -> %s" % (part, n_units, len(files), d))


def _load(d, view, L):
    return np.load(d / ("%s_L%d.npy" % (view, L)), mmap_mode="r")


def _diff(meta, X, names, sel=None):
    out = []
    for n in names:
        m = (meta.who == n) if sel is None else ((meta.who == n) & sel)
        pos, neg = (m & (meta.pole == "pos")).to_numpy(), (m & (meta.pole == "neg")).to_numpy()
        out.append(X[pos].astype(np.float32).mean(0) - X[neg].astype(np.float32).mean(0))
    return np.stack(out)


def _role_vectors(meta, X, names, nmeta, N, who_col="who"):
    """mean_j (h_ij - h0_j): neutral answer matched on question_idx"""
    q2row = {q: i for i, q in enumerate(nmeta.question_idx)}
    N32 = N.astype(np.float32)
    vec, raw = [], []
    for n in names:
        m = (meta[who_col] == n).to_numpy()
        H = X[m].astype(np.float32)
        h0 = N32[[q2row[q] for q in meta.question_idx[m]]]
        vec.append((H - h0).mean(0)); raw.append(H.mean(0))
    return np.stack(vec), np.stack(raw)


def vectors():
    V = AX / "vectors"
    V.mkdir(exist_ok=True)
    mt = pd.read_csv(AX / "traits" / "meta.csv", usecols=["who", "pole", "pair", "question_idx"])
    ma = pd.read_csv(AX / "axes" / "meta.csv", usecols=["who", "pole", "pair", "question_idx"])
    ms = pd.read_csv(AX / "seta" / "meta.csv", usecols=["who", "question_idx"])
    me = pd.read_csv(PM / "E1" / "meta.csv", usecols=["role", "question_idx"])
    tnames = [t["name"] for t in json.loads((HERE / "traits.json").read_text())["traits"]]
    anames = [a["name"] for a in json.loads((HERE / "axes.json").read_text())["axes"]]
    snames = [p["name"] for p in json.loads((HERE / "personas_a.json").read_text())["personas"]]
    rnames = sorted(set(me.role) - {NEUTRAL})
    for nm, names in (("traits", tnames), ("axes", anames), ("seta", snames), ("roles", rnames)):
        (V / (nm + "_names.json")).write_text(json.dumps(names))
    nsel = (me.role == NEUTRAL).to_numpy()
    nmeta = me[nsel].reset_index(drop=True)
    qs = sorted(set(ma.question_idx))
    for view in ("last", "mean"):
        for L in LAYERS:
            T = _load(AX / "traits", view, L)
            np.save(V / ("traits_%s_L%d.npy" % (view, L)), _diff(mt, T, tnames))
            A = _load(AX / "axes", view, L)
            np.save(V / ("axes_%s_L%d.npy" % (view, L)), _diff(ma, A, anames))
            np.save(V / ("axes_%s_L%d_pairsA.npy" % (view, L)), _diff(ma, A, anames, ma.pair.isin([0, 1, 2])))
            np.save(V / ("axes_%s_L%d_pairsB.npy" % (view, L)), _diff(ma, A, anames, ma.pair.isin([3, 4])))
            np.save(V / ("axes_%s_L%d_qA.npy" % (view, L)), _diff(ma, A, anames, ma.question_idx.isin(qs[:20])))
            np.save(V / ("axes_%s_L%d_qB.npy" % (view, L)), _diff(ma, A, anames, ma.question_idx.isin(qs[20:])))
            E = _load(PM / "E1", view, L)
            N = E[nsel]
            np.save(V / ("neutral_rawmean_%s_L%d.npy" % (view, L)), N.astype(np.float32).mean(0))
            S = _load(AX / "seta", view, L)
            v, r = _role_vectors(ms, S, snames, nmeta, N)
            np.save(V / ("seta_%s_L%d.npy" % (view, L)), v); np.save(V / ("seta_rawmean_%s_L%d.npy" % (view, L)), r)
            v, r = _role_vectors(me, E, rnames, nmeta, N, who_col="role")
            np.save(V / ("roles_%s_L%d.npy" % (view, L)), v); np.save(V / ("roles_rawmean_%s_L%d.npy" % (view, L)), r)
            print("%s L%d done" % (view, L), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["traits", "axes", "seta"])
    ap.add_argument("--vectors", action="store_true")
    a = ap.parse_args(argv)
    if a.part:
        consolidate(a.part)
    if a.vectors:
        vectors()


if __name__ == "__main__":
    main()
