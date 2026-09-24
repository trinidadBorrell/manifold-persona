"""Figure tables for the follow-up run (plan exp6): figA, figB, figC CSVs.

Plan: plans/2026-09-24-steering-followups.md (Results 1-3).

TABLES, NOT PICTURES. The plan commits the CSVs and renders the pictures
(interactive artifacts, REPORT.md PNGs) from them, so everything a figure
shows must be a column here -- a number computed inside a plotting call is a
number nobody can audit.

  figA-pca-<route>.csv   3-D PCA fitted on the role centroids. Rows: every
                         centroid (role = source/target/knot/other), the
                         intended path of every arm (sampled, per study; Study B
                         per question), and the realised steered state and
                         text footprint per (study, arm, alpha, question), mean
                         over samples. Optional real response clouds (--cloud).
  figB-dist-<route>.csv  Euclidean distance and cosine similarity (4096-d) of
                         the text footprint and steered state to the source and
                         target centroids, per (study, arm, alpha): mean over
                         the 5 identity questions of the per-question sample
                         mean, bootstrap 95% CI over questions (1,000 resamples,
                         seeded). Reference rows: exp1's alpha_pos per
                         direction, and -- with --cloud -- the median distance of
                         real source/target responses to their own centroid.
  figC-judge-<route>.csv shares of source / waypoint / target / other /
                         collapsed (identity questions) and canary accuracy,
                         per (study, arm, alpha).
  figC-examples-<route>.csv  one response per arm at 3 alphas (lowest, middle,
                         highest), first identity question, sample 0.
  figC-calib.csv, figC-knots.csv  copies of exp1 / exp3 tables.

Usage:
    .venv/bin/python -m steering.followups.figures [--run-root <root>]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups.common import GEOM_DEFAULT, K_DEFAULT, OUT_ROOT, QUESTIONS, ROUTES, Geom, route_slug
from steering.followups.grid import Interventions
from steering.followups.judge_followups import route_personas

N_BOOT = 1000
BOOT_SEED = 20260924
CLASSES = ("source", "waypoint", "target", "default_or_other", "collapsed")
PATH_SAMPLES = 41


def _read(stem):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = Path(str(stem) + ext)
        if p.exists():
            return fn(p)
    return None


def load_study(root, sub):
    df = _read(Path(root) / sub / "generations")
    if df is None:
        return None, None, None
    S = np.load(Path(root) / sub / "steer_acts.npy")
    T = np.load(Path(root) / sub / "text_acts.npy")
    if len(S) != len(df) or len(T) != len(df):
        raise ValueError("%s: table has %d rows, activations %d/%d" % (sub, len(df), len(S), len(T)))
    return df, S.astype(np.float64), T.astype(np.float64)


def pca_fit(C, n=3):
    mu = C.mean(0)
    _, _, Vt = np.linalg.svd(C - mu, full_matrices=False)
    return mu, Vt[:n]


def bootstrap_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------------------

def fig_a(G, route, studies, h0q, mu, V, iv):
    A, B, cA, cB = G.route(route)
    knots = set(route_personas(G, route)[1:-1])
    P = lambda X: (np.atleast_2d(X) - mu) @ V.T          # noqa: E731
    rows = []
    Z = P(G.C)
    for i, n in enumerate(G.names):
        role = "source" if n == A else "target" if n == B else "knot" if n in knots else "other"
        rows.append(dict(route=route, layer="centroid", study="", arm="", alpha=np.nan, q_idx=-1,
                         label=n, role=role, pc1=Z[i, 0], pc2=Z[i, 1], pc3=Z[i, 2]))
    ts = np.linspace(0, 1, PATH_SAMPLES)
    # intended paths, Study A: every arm starts at c_A (target_vector's intended
    # endpoint is c_B, drawn from h0_mean when known; replace arms draw pi(t))
    for arm, rule in (("linear", None), ("manifold_v1", "distance"), ("manifold_v2", "density"),
                      ("nearest", "nearest"), ("replace_linear", None), ("replace_manifold", "nearest")):
        pts = (cA[None, :] + ts[:, None] * (cB - cA)[None, :] if rule is None
               else iv.path(route, rule, cA, cB, "cA").at_alpha(ts))
        Zp = P(pts)
        rows += [dict(route=route, layer="path", study="A", arm=arm, alpha=t, q_idx=-1, label="",
                      role="intended", pc1=z[0], pc2=z[1], pc3=z[2]) for t, z in zip(ts, Zp)]
    for qi, h0 in enumerate(h0q.get(route, [])):
        for arm, rule in (("linear_B", None), ("nearest_B", "nearest")):
            pts = (h0[None, :] + ts[:, None] * (cB - h0)[None, :] if rule is None
                   else iv.path(route, rule, h0, cB, ("h0q", qi)).at_alpha(ts))
            Zp = P(pts)
            rows += [dict(route=route, layer="path", study="B", arm=arm, alpha=t, q_idx=qi,
                          label="", role="intended", pc1=z[0], pc2=z[1], pc3=z[2])
                     for t, z in zip(ts, Zp)]
    # realised
    for sub, (df, S, T) in studies.items():
        if df is None:
            continue
        m = (df.route == route).to_numpy()
        if not m.any():
            continue
        d = df[m]
        for lab, X in (("steered", S[m]), ("text", T[m])):
            g = pd.DataFrame(X).groupby([d.study.values, d.arm.values, d.alpha.values,
                                         d.q_idx.values]).mean()
            Zr = P(g.to_numpy())
            for (st, arm, al, qi), z in zip(g.index, Zr):
                rows.append(dict(route=route, layer=lab, study=st, arm=arm, alpha=al, q_idx=qi,
                                 label="", role="realised", pc1=z[0], pc2=z[1], pc3=z[2]))
    return pd.DataFrame(rows)


def fig_b(G, route, studies, calib, rng, ref_rows):
    A, B, cA, cB = G.route(route)
    rows = []
    for sub, (df, S, T) in studies.items():
        if df is None:
            continue
        m = ((df.route == route) & (df.kind == "identity")).to_numpy()
        if not m.any():
            continue
        d = df[m].reset_index(drop=True)
        for lab, X in (("text", T[m]), ("steered", S[m])):
            nx = np.linalg.norm(X, axis=1)
            metrics = {
                ("euclid", "target"): np.linalg.norm(X - cB, axis=1),
                ("euclid", "source"): np.linalg.norm(X - cA, axis=1),
                ("cosine", "target"): X @ cB / (nx * np.linalg.norm(cB)),
                ("cosine", "source"): X @ cA / (nx * np.linalg.norm(cA)),
            }
            # Raw cosine is dominated by the direction every activation shares, so it sits
            # near 1 everywhere; the centred version subtracts the centroid mean first
            # (review 2026-09-24 #9). Both are written; the report says which it plots.
            mu = G.C.mean(0)
            Xc, cAc, cBc = X - mu, cA - mu, cB - mu
            nxc = np.linalg.norm(Xc, axis=1)
            metrics[("cosine_centred", "target")] = Xc @ cBc / (nxc * np.linalg.norm(cBc))
            metrics[("cosine_centred", "source")] = Xc @ cAc / (nxc * np.linalg.norm(cAc))
            for (metric, to), v in metrics.items():
                dd = d.assign(v=v)
                perq = dd.groupby(["study", "arm", "alpha", "q_idx"]).v.mean().reset_index()
                for (st, arm, al), g in perq.groupby(["study", "arm", "alpha"]):
                    mean, lo, hi = bootstrap_ci(g.v.to_numpy(), rng)
                    rows.append(dict(route=route, study=st, arm=arm, alpha=al, measure=lab,
                                     metric=metric, to=to, mean=mean, ci_lo=lo, ci_hi=hi,
                                     n_questions=len(g), kind="data"))
    if calib is not None:
        for _, r in calib[calib.route == route].iterrows():
            rows.append(dict(route=route, study="", arm="", alpha=r.alpha_pos, measure="",
                             metric="alpha_pos", to=r.direction, mean=r.alpha_pos, ci_lo=np.nan,
                             ci_hi=np.nan, n_questions=0, kind="ref_alpha_pos"))
    rows += [dict(route=route, **x) for x in ref_rows.get(route, [])]
    return pd.DataFrame(rows)


def judge_class(kind, A, B, knots):
    if kind == "collapsed":
        return "collapsed"
    if kind is None or (isinstance(kind, float) and np.isnan(kind)):
        return "unjudged"
    if kind == B:
        return "target"
    # The judge labels the plain "I am Qwen" answer as `assistant` or `other`
    # interchangeably (pilot 2026-09-24: identical answers got both), and `assistant`
    # is a nearest-rule knot on the validator routes. Both are one class here, so
    # neither the source nor the waypoint share is inflated by default answers. On the
    # assistant routes this also absorbs the source: there "source" == default.
    if kind in ("assistant", "other"):
        return "default_or_other"
    if kind == A:
        return "source"
    if kind in knots:
        return "waypoint"
    return "default_or_other"


def fig_c(G, route, studies, judged):
    A, B, _, _ = G.route(route)
    knots = set(route_personas(G, route)[1:-1])
    rows = []
    for sub, (df, _, _) in studies.items():
        if df is None:
            continue
        d = df[df.route == route].copy()
        if d.empty:
            continue
        d["key"] = ["%s|%s|%d|%d" % (r.study, r.cell_id, int(r.unit_index), int(r["sample"]))
                    for _, r in d.iterrows()]
        d["persona_kind"] = d.key.map(judged) if judged else None
        d["cls"] = [judge_class(k, A, B, knots) for k in d.persona_kind]
        for (st, arm, al), g in d.groupby(["study", "arm", "alpha"]):
            idn = g[g.kind == "identity"]
            can = g[g.kind == "canary"]
            judged_n = int((idn.cls != "unjudged").sum())
            r = dict(route=route, study=st, arm=arm, alpha=al, n_identity=len(idn),
                     n_judged=judged_n,
                     canary_acc=float(can.paris.astype(bool).mean()) if len(can) else np.nan,
                     n_canary=len(can))
            for c in CLASSES:
                r["share_" + c] = (float((idn.cls == c).sum() / judged_n) if judged_n else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def fig_c_examples(route, studies):
    rows = []
    for sub, (df, _, _) in studies.items():
        if df is None:
            continue
        d = df[(df.route == route) & (df.q_idx == 0) & (df["sample"] == 0)]
        for (st, arm), g in d.groupby(["study", "arm"]):
            al = sorted(g.alpha.unique())
            pick = sorted({al[0], al[len(al) // 2], al[-1]})
            for a_ in pick:
                r = g[g.alpha == a_].iloc[0]
                rows.append(dict(route=route, study=st, arm=arm, alpha=a_, question=r.question,
                                 response=r.response, rep4=r.deg_rep4))
    return pd.DataFrame(rows)


def load_judged(root):
    p = Path(root) / "judge" / "judgements.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            d = json.loads(line)
            if d.get("error") is None:
                out[d["key"]] = d.get("persona_kind")
    return out


def cloud_refs(cloud_npy, cloud_csv, G, roles):
    """Median distance of real responses to their own centroid, and a thinned cloud."""
    X = np.load(cloud_npy, mmap_mode="r")
    meta = pd.read_csv(cloud_csv)
    out = {}
    for role in roles:
        idx = np.nonzero((meta.role == role).to_numpy())[0]
        if len(idx) == 0:
            continue
        Xr = np.asarray(X[idx], dtype=np.float64).reshape(len(idx), -1)
        out[role] = (float(np.median(np.linalg.norm(Xr - G.c(role), axis=1))), Xr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--geom", default=str(GEOM_DEFAULT))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--cloud-npy", default=None,
                    help="optional (n, d) response activations at hidden state 19 for the "
                         "real source/target clouds and the reference-distance lines")
    ap.add_argument("--cloud-csv", default=None, help="row metadata for --cloud-npy (column role)")
    ap.add_argument("--cloud-thin", type=int, default=200)
    a = ap.parse_args(argv)

    root = Path(a.run_root)
    fig = root / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    G = Geom(a.geom)
    studies = {s: load_study(root, s) for s in ("studyA", "studyB")}
    h0q = {}
    for r in ROUTES:
        p = root / "studyA" / f"h0q_{route_slug(r)}.npy"
        if p.exists():
            h0q[r] = np.load(p)
    iv = Interventions(G, k=a.k)
    mu, V = pca_fit(G.C)
    # Prefer the calibration built from THIS run's measured h0 (all 4 routes), which is the
    # h0 the target-vector arm steered from; fall back to the pre-run one (review 2026-09-24 #2).
    _c = [root / "exp1_calibration" / f for f in ("exp1_calibration_measured.csv",
                                                   "exp1_calibration.csv")]
    _c = [p for p in _c if p.exists()]
    calib = pd.read_csv(_c[0]) if _c else None
    judged = load_judged(root)
    rng = np.random.default_rng(BOOT_SEED)

    ref_rows, clouds = {}, {}
    if a.cloud_npy:
        roles = sorted({x for r in ROUTES for x in r.split(">")})
        clouds = cloud_refs(a.cloud_npy, a.cloud_csv, G, roles)
        for r in ROUTES:
            A, B = r.split(">")
            ref_rows[r] = [dict(study="", arm="", alpha=np.nan, measure="text", metric="euclid",
                                to=to, mean=clouds[role][0], ci_lo=np.nan, ci_hi=np.nan,
                                n_questions=0, kind="ref_median_real")
                           for to, role in (("source", A), ("target", B)) if role in clouds]

    written = []
    for r in ROUTES:
        s = route_slug(r)
        fa = fig_a(G, r, studies, h0q, mu, V, iv)
        if clouds:
            trng = np.random.default_rng(BOOT_SEED)
            for role in r.split(">"):
                if role in clouds:
                    Xr = clouds[role][1]
                    sel = trng.choice(len(Xr), size=min(a.cloud_thin, len(Xr)), replace=False)
                    Z = (Xr[sel] - mu) @ V.T
                    fa = pd.concat([fa, pd.DataFrame(dict(route=r, layer="cloud", study="", arm="",
                                                          alpha=np.nan, q_idx=-1, label=role,
                                                          role="cloud", pc1=Z[:, 0], pc2=Z[:, 1],
                                                          pc3=Z[:, 2]))], ignore_index=True)
        for name, t in (("figA-pca-%s.csv" % s, fa),
                        ("figB-dist-%s.csv" % s, fig_b(G, r, studies, calib, rng, ref_rows)),
                        ("figC-judge-%s.csv" % s, fig_c(G, r, studies, judged)),
                        ("figC-examples-%s.csv" % s, fig_c_examples(r, studies))):
            t.to_csv(fig / name, index=False)
            written.append((name, len(t)))
    _cal = ("exp1_calibration/exp1_calibration_measured.csv"
            if (root / "exp1_calibration/exp1_calibration_measured.csv").exists()
            else "exp1_calibration/exp1_calibration.csv")
    for src, dst in ((_cal, "figC-calib.csv"),
                     ("exp3_knots/exp3_knots.csv", "figC-knots.csv")):
        if (root / src).exists():
            shutil.copy(root / src, fig / dst)
            written.append((dst, "copied"))
        else:
            print("missing %s -- %s not written" % (src, dst))
    for n, k in written:
        print("  %-36s %s" % (n, k))
    print("judged responses available: %d" % len(judged))
    return written


if __name__ == "__main__":
    main()
