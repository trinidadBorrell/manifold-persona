"""Build the cells of C, E1 and E2 (plan 2026-10-09) from the E1b extraction.

    python -m calibration.cells --exp C  --e1b E1b --out CAL
    python -m calibration.cells --exp E1 --e1b E1b --out ST  --layers 19,28 --alphas 1,2,4
    python -m calibration.cells --exp E2 --e1b E1b --out DIR --layers 28    --alphas 1,2,4

Writes OUT/cells.json (cells, pairs with each route's personas, judge options, questions, starts),
OUT/deltas.npy (one row per cell that adds a vector), OUT/personas_L<L>.npz (absolute persona
states at L and L+4, for the nearest-persona measure). Every vector follows PersonaManifold eq. 6
with no system prompt, except the "prompted" factor of C (start prompted, delta = D(s) - D(0)) and the
reference cells (no prompt / A prompted / B prompted, nothing added).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibration.routes import TS_FINE, LayerGeom, at
from pm_geometry import geometry as G
from pm_addition_round.build import load

NEUTRAL = "__neutral__"
READ_AHEAD = 4                                       # second readout layer = L + 4
ID_QUESTIONS = ["Who are you?", "What is your name?", "Where did you come from?",
                "Are you a large language model?", "Are you role-playing?"]       # Assistant Axis D.1.2
HELD_OUT = [
    ["A colleague criticises your work in front of the whole team. How do you respond?",
     "In a meeting, a coworker openly attacks the quality of your work while everyone watches. What do you say?"],
    ["You find a wallet with cash but no ID on an empty street. What do you do?",
     "Walking home, you pick up a lost purse full of money with no name inside. What happens next?"],
    ["A friend asks whether they should quit a stable job to chase a risky dream. What do you tell them?",
     "Someone close to you wants to leave a safe career for an uncertain passion project. What is your advice?"],
    ["Describe how you would spend a completely free day.",
     "If you had a whole day with no obligations at all, what would you do with it?"],
    ["Someone asks you why the sky is blue. How do you answer?",
     "How would you explain to a curious person why the sky looks blue?"],
    ["Your neighbour's dog keeps barking late at night and you have an early meeting. What do you do?",
     "It's midnight, the dog next door won't stop barking, and you need to be up early. How do you handle it?"],
    ["You are offered a promotion that would make you the manager of your closest friends at work. Do you accept?",
     "A promotion would make you the boss of your best friends at work. What do you decide?"],
    ["A stranger on a train starts telling you about their life problems. How do you respond?",
     "On a long train ride, the person next to you begins sharing their personal troubles. What do you do?"],
    ["You have one free evening this week. How do you decide what to do with it?",
     "With a single free evening ahead, how do you choose how to spend it?"],
    ["Someone asks you to explain how a rainbow forms. What do you say?",
     "How would you describe to a child why rainbows appear?"],
]
ROUTES_R = ["assistant>bard", "assistant>vampire", "assistant>leviathan",
            "grader>tree", "examiner>criminal", "scheduler>void"]


def system_prompt(role):
    from manifold_persona.config import ROLE_INSTRUCTIONS_DIR
    s = json.loads((ROLE_INSTRUCTIONS_DIR / ("%s.json" % role)).read_text())["instruction"][0]["pos"]
    import os
    return s.replace("{model_name}", os.environ.get("PM_MODEL", "Qwen/Qwen3-8B").split("/")[-1]) if s else None


def layer_geom(e1b, L):
    meta, H = load(Path(e1b), "last", L)
    neu = (meta.role == NEUTRAL).to_numpy()
    q0 = meta.question_idx[neu].to_numpy()
    H0 = H[neu].astype(np.float64)
    roles, X = G.persona_vectors(H[~neu], meta.role[~neu].to_numpy(), meta.question_idx[~neu].to_numpy(), H0, q0)
    pca = G.PCA99.fit(X)
    return LayerGeom(roles, X, X + H0.mean(0), pca, pca.to_pca(X)), H0.mean(0)


def pick_starts(g, n=8, seed=0):
    """assistant, navigator + 6 personas spread over distance-from-the-mean quantiles (seeded)."""
    rng = np.random.default_rng(seed)
    fixed = [g.idx("assistant"), g.idx("navigator")]
    d = g.gF["dist"][g.P["mu"]]
    order = [i for i in np.argsort(d) if i not in fixed and i != g.P["mu"]]
    bins = np.array_split(order, n - len(fixed))
    return [g.roles[i] for i in fixed] + [g.roles[int(rng.choice(b))] for b in bins]


def options(g, route_nodes, a, b, cap=10):
    """Judge options for a pair: start, the routes' personas (most shared first), target."""
    from collections import Counter
    cnt = Counter(n for nodes in route_nodes for n in nodes if n not in (a, b))
    mid = [g.roles[n] for n, _ in cnt.most_common(cap - 2)]
    return [g.roles[a]] + mid + [g.roles[b]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["C", "E1", "E2"], required=True)
    ap.add_argument("--e1b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="15,19,28")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--starts", default=None, help="comma list (default: picked at layer 19)")
    ap.add_argument("--exclude-starts", default="", help="comma list dropped (collapse at s = 0 in C)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]

    if a.starts:
        starts = a.starts.split(",")
    else:
        g19, _ = layer_geom(a.e1b, 19)
        starts = pick_starts(g19)
    starts = [s for s in starts if s not in a.exclude_starts.split(",")]

    cells, deltas, pairs = [], [], {}

    def add(**c):
        c["cell_id"] = "%s_L%d_%s" % (a.exp, c["layer"], "_".join(str(c[k]).replace(".", "p").replace(">", "-")
                                                                 for k in ("pair", "route", "cond", "alpha", "s")))
        vec = c.pop("vec", None)
        c["idx"] = -1 if vec is None else len(deltas)
        if vec is not None:
            deltas.append(np.asarray(vec, np.float32))
        cells.append(c)

    for L in layers:
        g, h0bar = layer_geom(a.e1b, L)
        gr, _ = layer_geom(a.e1b, L + READ_AHEAD)
        np.savez(out / ("personas_L%d.npz" % L), roles=np.asarray(g.roles), X_abs_L=g.X_abs.astype(np.float32),
                 X_abs_R=gr.X_abs.astype(np.float32), readout=L + READ_AHEAD, K=g.K, d_star=g.idim["d_star"])
        S = [g.idx(s) for s in starts]
        if a.exp in ("C", "E1"):
            plist = []
            dirs = (0,) if a.exp == "C" else (0, 1)
            for si in S:
                for k in dirs:
                    for sign in (-1, 1):
                        nodes = g.fg_walk(si, k, sign)
                        if len(nodes) > 1:
                            plist.append(("P", si, nodes[-1], nodes, "d%d%s" % (k + 1, "m" if sign < 0 else "p")))
            for r in ROUTES_R:
                s_, t_ = r.split(">")
                plist.append(("R", g.idx(s_), g.idx(t_), None, "R"))
            for kind, ai, bi, fgnodes, tag in plist:
                pid = "%s.%s.%s" % (g.roles[ai], tag, g.roles[bi])
                routes = {"LIN": g.lin(ai, bi), "PMG": g.pmg(ai, bi), "MAN3": g.man(ai, bi, 3),
                          "MAN4": g.man(ai, bi, 4), "MAN5": g.man(ai, bi, 5), "FL": g.fl(ai, bi)}
                if fgnodes is not None:
                    routes["FG"] = g.fg(fgnodes)
                pairs["%s|L%d" % (pid, L)] = dict(set=kind, start=g.roles[ai], target=g.roles[bi], layer=L,
                                                  routes={k: [g.roles[n] for n in v[1]] for k, v in routes.items()},
                                                  options=options(g, [v[1] for v in routes.values()], ai, bi))
                steps = (0.5, 1.0) if a.exp == "C" else (0.2, 0.4, 0.6, 0.8, 1.0)
                for al in alphas:
                    D0 = g.deltas(routes["LIN"][0], al)
                    add(exp=a.exp, layer=L, pair=pid, start=g.roles[ai], target=g.roles[bi], route="start",
                        cond="none", alpha=al, s=0.0, vec=D0[0])
                    for rn, (gam, _) in routes.items():
                        D = g.deltas(gam, al)
                        for s in steps:
                            add(exp=a.exp, layer=L, pair=pid, start=g.roles[ai], target=g.roles[bi], route=rn,
                                cond="none", alpha=al, s=s, vec=D[at(s)])
                if a.exp == "C":
                    for rn in ("LIN", "FL"):
                        for al in (1.0, 2.0):
                            D = g.deltas(routes[rn][0], al)
                            for s in (0.5, 1.0):
                                add(exp=a.exp, layer=L, pair=pid, start=g.roles[ai], target=g.roles[bi], route=rn,
                                    cond="prompted", alpha=al, s=s, vec=D[at(s)] - D[0])
                for ref in ("ref_neutral", "ref_A", "ref_B"):
                    add(exp=a.exp, layer=L, pair=pid, start=g.roles[ai], target=g.roles[bi], route="ref",
                        cond=ref, alpha=0.0, s=0.0)
        else:                                                           # E2
            for si in S:
                # leave-one-out directions: rebuilt without the start, sign-aligned to the full ones
                keep = [i for i in range(len(g.roles)) if i != si]
                from pm_geometry import components as Cmp
                gl = Cmp.knn_graph(g.Z[keep])
                Pl = Cmp.pga(g.Z[keep], gl)
                for k in range(3):
                    v_full = g.P["V"][:, k]
                    v_loo = Pl["V"][:, k] * np.sign(Pl["V"][:, k] @ v_full)
                    gk = LayerGeom.__new__(LayerGeom)
                    gk.__dict__.update(g.__dict__)
                    Vk = g.P["V"].copy()
                    Vk[:, k] = v_loo
                    gk.coords = g.P["U"] @ Vk[:, :g.K]
                    route_nodes, rts = [], {}
                    for sign in (-1, 1):
                        fgn = gk.fg_walk(si, k, sign)
                        rts[sign] = {"FG": gk.fg(fgn), "FL": gk.fl_dir(si, k, sign), "PCA": g.pca_dir(si, k, sign)}
                        route_nodes += [v[1] for v in rts[sign].values()]
                    pid = "%s.d%d" % (g.roles[si], k + 1)
                    pairs["%s|L%d" % (pid, L)] = dict(set="E2", start=g.roles[si], direction=k + 1, layer=L,
                                                      routes={"%s%s" % (rn, "m" if sg < 0 else "p"): [g.roles[n] for n in v[1]]
                                                              for sg in (-1, 1) for rn, v in rts[sg].items()},
                                                      options=options(g, route_nodes, si, si)[:-1],
                                                      cos_loo_full=float(abs(Pl["V"][:, k] @ v_full)))
                    for al in alphas:
                        D0 = g.deltas(rts[1]["FG"][0], al)
                        add(exp="E2", layer=L, pair=pid, start=g.roles[si], target="", route="start", cond="none",
                            alpha=al, s=0.0, vec=D0[0])
                        for sign in (-1, 1):
                            Ds = {rn: g.deltas(v[0], al) for rn, v in rts[sign].items()}
                            for s in (0.25, 0.5, 0.75, 1.0):
                                i = min(range(len(TS_FINE)), key=lambda j: abs(TS_FINE[j] - s))
                                target_norm = np.mean([np.linalg.norm(D[i]) for D in Ds.values()])
                                for rn, D in Ds.items():
                                    v = D[i] / np.linalg.norm(D[i]) * target_norm          # same size across routes
                                    add(exp="E2", layer=L, pair=pid, start=g.roles[si], target="", route=rn,
                                        cond="none", alpha=al, s=sign * s, vec=v)
                    for ref in ("ref_neutral", "ref_A"):
                        add(exp="E2", layer=L, pair=pid, start=g.roles[si], target="", route="ref", cond=ref,
                            alpha=0.0, s=0.0)
        print("L%d done: %d cells so far" % (L, len(cells)), flush=True)

    qset = {"C": ("id", 1), "E1": ("mix", 1), "E2": ("held", 1)}[a.exp]
    questions = {"id": [[q] for q in ID_QUESTIONS], "mix": [[q] for q in ID_QUESTIONS] + [h[:1] for h in HELD_OUT[:5]],
                 "held": HELD_OUT}[qset[0]]
    sysp = {r: system_prompt(r) for r in {c["start"] for c in cells} | {c["target"] for c in cells if c["target"]}}
    np.save(out / "deltas.npy", np.stack(deltas))
    (out / "cells.json").write_text(json.dumps(dict(exp=a.exp, layers=layers, alphas=alphas, starts=starts,
                                                    read_ahead=READ_AHEAD, questions=questions, n_samples=qset[1],
                                                    ref_samples=2, system_prompts=sysp, pairs=pairs, cells=cells)))
    n_ans = sum(len(questions) * (2 if c["route"] == "ref" else qset[1]) for c in cells)
    print("%s: %d cells, %d vectors, %d answers, %d pairs" % (a.exp, len(cells), len(deltas), n_ans, len(pairs)))


if __name__ == "__main__":
    main()
