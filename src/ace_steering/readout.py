"""Aggregate the ACE study: % of answers per judge option, and the PCA page data.

  shares   <root>/step<N>/shares.csv. One row per (route, layer, arm, alpha, option),
           with the share of the cell's answers and a bootstrap 95% CI over answers.
           Options are the route's personas (A, 8 knots, B), `other`, `collapsed`,
           and `judge_error` when a call failed after its retries. Each option also
           gets a group column: source / mid / target / other / collapsed.
           The unsteered cell (arm `none`) is written twice, as linear alpha 0
           and manifold alpha 0. ACE alpha 0 is its own cell.
  export   <root>/step<N>/page_data.json: shares plus, per (route, layer), a 3-D
           PCA fitted on the `fully` rows of the route's 10 personas, with the
           persona clouds, centroids, each arm's intended path, the mean steered
           state per (arm, alpha) and every answer's steered state.

Step 2 at layer 19 takes the step-1 routes' sample-0 answers from step 1
(the same draw; see grid.build_cells).

    python -m ace_steering.readout --step 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ace_steering.common import (MANIFOLD_ALPHAS, OUT_ROOT, SIGNED_ALPHAS, STEP1_LAYER,
                                       STEP1_ROUTES, route_slug)
from ace_steering.judge import key_of

N_BOOT = 1000
CLOUD_ROWS_PER_PERSONA = 150


def load_step(root, step, judge_tag=""):
    sub = "smoke" if step == 0 else "step%d" % step
    df = pd.read_csv(root / sub / "generations.csv")
    X = np.load(root / sub / "steer_acts.npy")
    jl = root / "judge" / ("judgements_%s%s.jsonl" % (sub, "_" + judge_tag if judge_tag else ""))
    lab = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            d = json.loads(line)
            if d.get("error") is None or d["key"] not in lab:
                lab[d["key"]] = d.get("persona_kind") if d.get("error") is None else "judge_error"
    df = df.assign(key=[key_of(r) for _, r in df.iterrows()])
    df["label"] = df.key.map(lab)
    return df, X


def load(root, step, judge_tag=""):
    df, X = load_step(root, step, judge_tag)
    if step == 2:
        try:
            d1, X1 = load_step(root, 1, judge_tag)
        except FileNotFoundError:
            d1 = None
        if d1 is not None:
            m = (d1["sample"] == 0) & (d1.layer == STEP1_LAYER) & d1.route.isin(STEP1_ROUTES)
            df = pd.concat([df, d1[m].assign(step=2)], ignore_index=True)
            X = np.concatenate([X, X1[m.to_numpy()]])
    return df, X


def group_of(option, personas):
    if option == personas[0]:
        return "source"
    if option == personas[-1]:
        return "target"
    if option in personas[1:-1]:
        return "mid"
    return option                                   # other / collapsed / judge_error


def cell_rows(df):
    """(arm_as_plotted, alpha, frame) for every plotted point; `none` counts twice."""
    for (arm, al), g in df.groupby(["arm", "alpha"]):
        if arm == "none":
            yield "linear", 0.0, g
            yield "manifold", 0.0, g
        else:
            yield arm, float(al), g


def shares(df, options):
    rng = np.random.default_rng(0)
    rows = []
    for (route, layer), dr in df.groupby(["route", "layer"]):
        personas = options["%s|L%d" % (route, layer)]
        opts = personas + ([] if "assistant" in personas else ["assistant"]) + ["other", "collapsed", "judge_error"]
        for arm, al, g in cell_rows(dr):
            lab = g.label.fillna("unjudged").to_numpy()
            n = len(lab)
            boot = rng.integers(0, n, size=(N_BOOT, n))
            for o in opts + (["unjudged"] if (lab == "unjudged").any() else []):
                hit = (lab == o).astype(float)
                b = hit[boot].mean(1)
                rows.append(dict(route=route, layer=int(layer), arm=arm, alpha=al, option=o,
                                 group=group_of(o, personas),
                                 position=personas.index(o) if o in personas else None,
                                 n=n, share=float(hit.mean()),
                                 lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PCA page data
# ---------------------------------------------------------------------------

def _r(a, nd=2):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def top_pcs(Z, k, rng, oversample=10, n_iter=6):
    """Top-k principal axes by randomized SVD (Halko et al.) and their share of variance.

    A full SVD of ~10k x 4096 rows is minutes of CPU; k = 3 needs only a
    (4096 x k+oversample) sketch refined by a few power iterations.
    """
    Q = np.linalg.qr(Z @ rng.standard_normal((Z.shape[1], k + oversample)))[0]
    for _ in range(n_iter):
        Q = np.linalg.qr(Z @ (Z.T @ Q))[0]
    _, S, Vt = np.linalg.svd(Q.T @ Z, full_matrices=False)
    total = float((Z ** 2).sum())
    return Vt[:k].T, (S[:k] ** 2 / total).tolist()


def pca_block(route, layer, dr, Xr, root, rng):
    from ace_steering import grid as GR
    cz = np.load(root / "clouds" / ("%s_L%d.npz" % (route_slug(route), layer)), allow_pickle=True)
    Xc, roles = cz["X"].astype(np.float64), cz["role"].astype(str)
    personas = GR.route_personas(route, layer)
    mu = Xc.mean(0)
    V, var = top_pcs(Xc - mu, 3, rng)
    proj = lambda Z: (np.atleast_2d(Z) - mu) @ V          # noqa: E731

    G = GR.geom(layer)
    _, _, cA, cB = G.route(route)
    path = GR.route_path(route, layer)
    r = cB - cA
    r_hat = r / np.linalg.norm(r)
    lin = np.array([cA + a * r for a in np.linspace(-1.5, 1.5, 61)])
    man = np.vstack([path.at_alpha(a) for a in np.linspace(0, 1, 101)])
    h0 = Xr[dr.index[dr.arm == "none"].to_numpy()].astype(np.float64).mean(0)
    h0_perp = h0 - (h0 @ r_hat) * r_hat
    ace = np.array([h0_perp + (cA @ r_hat + a * np.linalg.norm(r)) * r_hat
                    for a in np.linspace(-1.5, 1.5, 61)])

    clouds = {}
    for p in personas:
        idx = np.nonzero(roles == p)[0]
        if len(idx) > CLOUD_ROWS_PER_PERSONA:
            idx = rng.choice(idx, CLOUD_ROWS_PER_PERSONA, replace=False)
        clouds[p] = _r(proj(Xc[idx]))

    means, points = [], []
    for arm, al, g in cell_rows(dr):
        H = Xr[g.index.to_numpy()].astype(np.float64)
        ok = np.isfinite(H).all(1)
        means.append(dict(arm=arm, alpha=al, xyz=_r(proj(H[ok].mean(0))[0])))
        P = proj(H[ok])
        for (_, row), xyz in zip(g[ok].iterrows(), P):
            points.append(dict(arm=arm, alpha=al, rollout="%d-%d-%d" % (row.prompt_idx, row.q_idx, row["sample"]),
                               q=int(row.q_idx), label=row.label if isinstance(row.label, str) else None,
                               xyz=_r(xyz)))
    U = GR.route_basis(route, layer)
    acem = np.array([h0 + U @ (U.T @ (GR.route_target(route, layer, a) - h0))
                     for a in np.linspace(-1.5, 1.5, 61)])
    return dict(personas=personas, explained=_r(var, 4), clouds=clouds,
                centroids={p: _r(proj(G.c(p))[0]) for p in personas},
                paths=dict(linear=_r(proj(lin)), manifold=_r(proj(man)), ace=_r(proj(ace)),
                           ace_manifold=_r(proj(acem))),
                means=means, points=points)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--judge-tag", default="", help="read judgements_<step>_<tag>.jsonl; write to <step>_<tag>/")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    sub = "smoke" if a.step == 0 else "step%d" % a.step
    df, X = load(root, a.step, a.judge_tag)
    if a.judge_tag:
        # a partial judge (e.g. subagents on the forward routes only): keep the judged routes
        judged = df.groupby("route").label.apply(lambda s: (s.notna() & (s != "collapsed")).any())
        keep = judged[judged].index
        dropped = sorted(set(df.route) - set(keep))
        if dropped:
            print("routes with no labels under judge '%s', left out: %s" % (a.judge_tag, dropped))
        df = df[df.route.isin(keep)]
    outd = root / (sub + ("_" + a.judge_tag if a.judge_tag else ""))
    outd.mkdir(exist_ok=True)
    options = json.loads((root / "judge" / "route_options.json").read_text())
    sh = shares(df, options)
    sh.to_csv(outd / "shares.csv", index=False)
    print("shares: %d rows -> %s" % (len(sh), outd / "shares.csv"))
    print("labels:", df.label.fillna("unjudged").value_counts().to_dict())
    if a.no_pca:
        return
    rng = np.random.default_rng(0)
    routes = json.loads((root / "routes.json").read_text())
    blocks = {}
    for (route, layer), dr in df.groupby(["route", "layer"]):
        k = "%s|L%d" % (route, layer)
        blocks[k] = pca_block(route, int(layer), dr, X, root, rng)
        blocks[k].update(knot_alpha=routes[k]["knot_alpha"], detour=routes[k]["detour"],
                         chord=routes[k]["chord"])
        print("pca %s: %d answers" % (k, len(dr)), flush=True)
    meta = dict(step=a.step, signed_alphas=list(SIGNED_ALPHAS), manifold_alphas=list(MANIFOLD_ALPHAS),
                n_answers=int(len(df)),
                collapsed=int((df.label == "collapsed").sum()),
                judge_errors=int((df.label == "judge_error").sum()),
                unjudged=int(df.label.isna().sum()))
    sh_rec = sh.round(4).astype(object).where(sh.notna(), None).to_dict(orient="records")
    out = dict(meta=meta, shares=sh_rec, routes=blocks)
    p = outd / "page_data.json"
    p.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    print("page data -> %s (%.1f MB)" % (p, p.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
