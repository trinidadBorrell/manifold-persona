"""PersonaManifold (arXiv 2609.34571 §4.1) path measures on the component-steering round (PC, PC4).

    ic        Interpolation Coherence: "mean pairwise cosine similarity of response embeddings (via
              all-MiniLM-L6-v2) across N=20 dialogue turns at each intermediate point", per cell.
              Ours: the cell's 20 answers (5 held-out questions x 2 paraphrases x 2 samples), not one
              dialogue. Mean pooling + L2 norm, 256 tokens, as sentence-transformers does for this model.
    ma        Manifold Adherence: MA = 1 - (1/K) sum_k d_g(gamma(t_k), NN(gamma(t_k))) / r95, NN = nearest
              observed persona, r95 = 95th percentile inter-persona geodesic distance. The paper's metric
              graph (eq. 2-4, geometry.build_graph) on the layer-28 persona set of the PC round; a point
              between personas has no metric tensor of its own, so d_g(gamma, x_i) = ||gamma - x_i||_g(x_i),
              minimised over gamma's graph-k Euclidean neighbours.
    bigfive   batches for the Big Five judge (TS needs s_d(t)); collect -> PM_MEAS/bigfive.csv.
              TS = 1/(5K) sum_{d,k} 1[sgn(delta s_d,k) = sgn(s_d(1) - s_d(0))] per half path (0 -> +1, 0 -> -1).

    python -m pca_pga_steering.paper_measures ic|ma            (condor CPU)
    python -m pca_pga_steering.paper_measures bigfive-build|bigfive-collect
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os

for _kv in sys.argv[2:]:          # condor passes PM_MEAS_OUT=... / PM_MEAS_RUNS=... as arguments
    if _kv.startswith("PM_MEAS_") and "=" in _kv:
        k, v = _kv.split("=", 1)
        os.environ[k] = v

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
# defaults = the addition round (PC, PC4); the clamp round sets PM_MEAS_OUT and PM_MEAS_RUNS="1=<dir>,4=<dir>"
OUT = Path(os.environ.get("PM_MEAS_OUT", PM / "PM_MEAS"))
MINILM = PM.parent / "models" / "all-MiniLM-L6-v2"
RUNS = ({k: Path(v) for k, v in (x.split("=") for x in os.environ["PM_MEAS_RUNS"].split(","))}
        if os.environ.get("PM_MEAS_RUNS") else {"1": PM / "PC", "4": PM / "PC4"})
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BATCH = 90


def gens():
    out = []
    for alpha, d in RUNS.items():
        g = pd.read_csv(d / "generations.csv", low_memory=False)
        g["alpha"] = alpha
        g["key"] = ["a%s|%s|%d|%d|%d" % (alpha, r.cell_id, r.q_idx, r.prompt_idx, r.sample) for r in g.itertuples()]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def ic():
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MINILM)
    mod = AutoModel.from_pretrained(MINILM).eval()
    g = gens()
    embs = []
    with torch.no_grad():
        for i in range(0, len(g), 64):
            b = tok(list(g.response.astype(str).iloc[i:i + 64]), padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            embs.append(torch.nn.functional.normalize(e, dim=1).numpy())
    E = np.concatenate(embs)
    np.save(OUT / "minilm.npy", E)
    rows = []
    for (alpha, cell), idx in g.groupby(["alpha", "cell_id"]).groups.items():
        X = E[list(idx)]
        S = X @ X.T
        n = len(X)
        rows.append(dict(alpha=alpha, cell_id=cell, n=n, ic=float((S.sum() - n) / (n * (n - 1)))))
    pd.DataFrame(rows).to_csv(OUT / "ic.csv", index=False)
    print("ic: %d cells" % len(rows))


def ma():
    from pm_geometry import components as C
    from pm_geometry import geometry as G
    from pm_addition_round.component_paths import N_COMP, STEPS, nearest
    cells = json.loads((PM / "PC" / "cells.json").read_text())
    L = cells["layer"]
    roles, sets, _ = C.persona_sets(PM / "E1", "last", L)
    res, st = C.analyse(sets["all"], roles)
    pca, Z, g, P = st["pca"], st["Z"], st["g"], st["P"]
    idd = G.intrinsic_dim(Z)
    mg = G.build_graph(Z, idd["d_star"])                       # the paper's metric graph, eq. 2-4
    D = mg.dist[np.triu_indices(len(Z), 1)]
    r95 = float(np.percentile(D[np.isfinite(D)], 95))
    sig_pga, sig_pca = np.sqrt(P["lam"][:N_COMP]), Z[:, :N_COMP].std(0)
    ts = (0.0,) + STEPS
    deltas = np.load(PM / "PC" / "deltas.npy")
    by = {(c["start"], c["method"], c["component"], c["sign"], c["step"]): c for c in cells["cells"]}
    rows, err = [], []
    for start in cells["starts"]:
        si = roles.index(start)
        for method in ("pga", "pca"):
            for k in range(N_COMP):
                for sign in (-1, 1):
                    if method == "pga":
                        q = P["U"][si] + sign * 2 * sig_pga[k] * P["V"][:, k]
                        ti = nearest(P["U"], q, {si})
                        gam = G.smooth_path(Z[C.path(g, si, ti)], ts)
                    else:
                        q = Z[si] + sign * 2 * sig_pca[k] * np.eye(Z.shape[1])[k]
                        ti = nearest(Z, q, {si})
                        gam = np.stack([(1 - t) * Z[si] + t * Z[ti] for t in ts])
                    Dl = G.steering_deltas(pca, gam, ts)
                    for j, s in enumerate(ts):
                        x = gam[j]
                        e = np.linalg.norm(Z - x, axis=1)
                        nb = np.argsort(e)[:mg.k]
                        dl = [np.sqrt(max((x - Z[i]) @ mg.G[i] @ (x - Z[i]), 0)) for i in nb]
                        b = int(nb[int(np.argmin(dl))])
                        c = by.get((start, method, k + 1, sign, s)) if s > 0 else None
                        if c is not None:
                            err.append(float(np.linalg.norm(Dl[j] - deltas[c["idx"]]) / np.linalg.norm(deltas[c["idx"]])))
                        rows.append(dict(start=start, method=method, component=k + 1, sign=sign, step=s,
                                         x=sign * s, d_g=float(min(dl)), nn=roles[b], d_over_r95=float(min(dl) / r95),
                                         target=roles[ti]))
    pd.DataFrame(rows).to_csv(OUT / "ma.csv", index=False)
    meta = dict(layer=L, d_star=idd, graph_k=int(mg.k), r95=r95, D_prime=int(Z.shape[1]),
                delta_check_max_rel_err=max(err) if err else None)
    (OUT / "ma_meta.json").write_text(json.dumps(meta, indent=1))
    print("ma:", meta)


def ma_clamp():
    """MA for the clamp round: each path is the straight line along the axis (PCA component k, or PGA
    direction k as a tangent direction), from the start's coordinate to start +/- S * step * 2 sigma_k.
    Measured in the same last-token PCA space and metric graph as ma(); sigma_k = std of the 276 personas'
    coordinates along that direction there."""
    from pm_geometry import components as C
    from pm_geometry import geometry as G
    from pm_addition_round.component_paths import N_COMP, STEPS
    cells = json.loads((PM / "PC" / "cells.json").read_text())
    L = cells["layer"]
    roles, sets, _ = C.persona_sets(PM / "E1", "last", L)
    res, st = C.analyse(sets["all"], roles)
    Z, P = st["Z"], st["P"]
    mg = G.build_graph(Z, G.intrinsic_dim(Z)["d_star"])
    D = mg.dist[np.triu_indices(len(Z), 1)]
    r95 = float(np.percentile(D[np.isfinite(D)], 95))
    rows = []
    for k in range(N_COMP):
        for m, e in (("pca", np.eye(Z.shape[1])[k]), ("pga", P["V"][:, k] / np.linalg.norm(P["V"][:, k]))):
            sig = float((Z @ e).std())
            for start in cells["starts"]:
                z0 = Z[roles.index(start)]
                for S in (1.0, 4.0):
                    for sign in (-1, 1):
                        for s in (0.0,) + STEPS:
                            x = z0 + S * sign * s * 2 * sig * e
                            dist = np.linalg.norm(Z - x, axis=1)
                            nb = np.argsort(dist)[:mg.k]
                            dl = [np.sqrt(max((x - Z[i]) @ mg.G[i] @ (x - Z[i]), 0)) for i in nb]
                            rows.append(dict(alpha=str(int(S)), start=start, method=m, component=k + 1, sign=sign, step=s,
                                             x=sign * s, d_g=float(min(dl)), nn=roles[int(nb[int(np.argmin(dl))])],
                                             d_over_r95=float(min(dl) / r95)))
    pd.DataFrame(rows).to_csv(OUT / "ma.csv", index=False)
    (OUT / "ma_meta.json").write_text(json.dumps(dict(layer=L, r95=r95, graph_k=int(mg.k), clamp=True), indent=1))
    print("ma_clamp rows", len(rows), "r95", r95)


BF_SYS = """You rate the personality that a language model's answer expresses, on the Big Five
(OCEAN) traits, the way the BFI-44 questionnaire defines them. Rate the SPEAKER as the answer shows
them, from the answer's own words, tone and choices, not from the topic.

Traits (1 = very low, 3 = neutral or no evidence, 5 = very high):
- openness: imaginative, curious, original, artistic, likes ideas and reflection (low: conventional,
  concrete, few artistic interests)
- conscientiousness: thorough, organised, careful, reliable, plans and follows through (low: careless,
  disorganised, impulsive)
- extraversion: talkative, energetic, enthusiastic, assertive, outgoing (low: reserved, quiet, shy)
- agreeableness: kind, warm, trusting, cooperative, considerate (low: cold, rude, critical,
  quarrelsome)
- neuroticism: anxious, tense, moody, easily upset, worried (low: calm, relaxed, emotionally stable)

For each item give, in this order:
- evidence: a short quote (at most 25 words) that most shows the speaker's personality;
- openness, conscientiousness, extraversion, agreeableness, neuroticism: integers 1-5;
- degenerate: true if a large part of the answer is repetition loops, word salad or broken text
  (still rate what you can; rate 3 where nothing is left to rate).
"""


def bigfive_build():
    g = gens()
    rows = g[["key", "question", "response"]].to_dict("records")
    random.Random(0).shuffle(rows)
    (OUT / "bigfive" / "batches").mkdir(parents=True, exist_ok=True)
    (OUT / "bigfive" / "labels").mkdir(exist_ok=True)
    (OUT / "bigfive" / "sys.txt").write_text(BF_SYS)
    n = 0
    for i in range(0, len(rows), BATCH):
        (OUT / "bigfive" / "batches" / ("bf_%03d.json" % n)).write_text(json.dumps(rows[i:i + BATCH], ensure_ascii=False))
        n += 1
    print("%d answers, %d batches" % (len(rows), n))


def bigfive_collect():
    d = OUT / "bigfive"
    rows, redo = [], []
    for f in sorted((d / "batches").glob("bf_*.json")):
        keys = [x["key"] for x in json.loads(f.read_text())]
        lf = d / "labels" / f.name
        try:
            labs = {l["key"]: l for l in json.loads(lf.read_text())["labels"]}
        except Exception:
            redo.append(f.stem); continue
        ok = set(labs) == set(keys) and all(all(isinstance(labs[k].get(t), int) and 1 <= labs[k][t] <= 5
                                                for t in TRAITS) for k in keys)
        if not ok:
            redo.append(f.stem); continue
        rows += [dict(key=k, **{t: labs[k][t] for t in TRAITS}, degenerate=bool(labs[k].get("degenerate")),
                      evidence=labs[k].get("evidence")) for k in keys]
    pd.DataFrame(rows).to_csv(d / "bigfive.csv", index=False)
    (d / "redo.json").write_text(json.dumps(redo))
    print("%d labels, %d batches to redo" % (len(rows), len(redo)))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"ic": ic, "ma": ma, "ma-clamp": ma_clamp, "bigfive-build": bigfive_build, "bigfive-collect": bigfive_collect}[sys.argv[1]]()
