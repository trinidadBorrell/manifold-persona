"""Pages for follow-ups A + B (page 1) and D (page 2), same layout as the by-question layer sweep.

  page 1  python -m ace_steering.followup_pages --page AB
          A: step3 generations, labels judgements_A_subagent.jsonl (validator/summarizer -> 6 targets)
          B: step2 generations at L19/L25 for the reverse routes, labels judgements_B_subagent.jsonl
  page 2  python -m ace_steering.followup_pages --page D
          step5 generations (activation capping as a floor at hidden states 19/23/25/27 at once),
          labels judgements_D_subagent.jsonl; x = percentile p of the cap; distance/cosine measured at
          the layer chosen on the page; judge labels and examples do not depend on that layer.

Data shapes are those of sweep_q_page.py (metrics / shares / examples as compact arrays, q = -1 = all five).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ace_steering.common import OUT_ROOT, STEP3_LAYERS, STEP3_ROUTES, STEP5_LAYERS
from ace_steering.grid import geom, route_personas
from ace_steering.judge import key_of
from ace_steering.page import _no_nan
from ace_steering.sweep_page import group
from ace_steering.sweep_q_page import PER_QUESTION, N_BOOT, fill_template, r4

B_ROUTES = ["bard>assistant", "vampire>assistant", "leviathan>assistant",
            "tree>grader", "criminal>examiner", "void>scheduler"]
FWD = ["assistant>bard", "assistant>vampire", "assistant>leviathan",
       "grader>tree", "examiner>criminal", "scheduler>void"]


def load_labels(tag):
    lab = {}
    for line in (OUT_ROOT / "judge" / ("judgements_%s_subagent.jsonl" % tag)).read_text().splitlines():
        d = json.loads(line)
        lab[d["key"]] = (d.get("persona_kind"), d.get("intensity"), d.get("evidence") or "")
    from ace_steering.relabel_ai import apply
    return apply(lab)


def load(gen, routes, layers, tag):
    d = pd.read_csv(OUT_ROOT / gen / "generations.csv")
    X = np.load(OUT_ROOT / gen / "steer_acts.npy", mmap_mode="r")
    d["row"] = np.arange(len(d))
    d = d[d.route.isin(routes) & d.layer.isin(layers) & (d.arm != "ace_manifold")].copy()   # removed (user, 2026-10-02)
    d["h"] = list(np.asarray(X[d.row.to_numpy()], dtype=np.float64))
    lab = load_labels(tag)
    d["key"] = [key_of(r) for _, r in d.iterrows()]
    d["label"] = [lab.get(k, (None,))[0] for k in d.key]
    return d, lab


def distances(d, layer_of_row, h_of_row):
    """Distance and cosine of each answer's steered state to mu_B at the given layer."""
    dist, cos = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i, (route, L, h) in enumerate(zip(d.route, layer_of_row, h_of_row)):
        cB = geom(int(L)).route(route)[3]
        if np.isfinite(h).all():
            dist[i] = np.linalg.norm(h - cB)
            cos[i] = h @ cB / (np.linalg.norm(h) * np.linalg.norm(cB))
    return dist, cos


def metrics_rows(d, none_as):
    out = []
    for q in [-1, 0, 1, 2, 3, 4]:
        dq = d if q < 0 else d[d.q_idx == q]
        for (route, layer, arm, al), g in dq.groupby(["route", "mlayer", "arm", "alpha"]):
            dist, cos = g.dist.dropna(), g.cos.dropna()
            m = [r4(dist.mean()), r4(dist.quantile(.25)), r4(dist.quantile(.75)),
                 r4(cos.mean()), r4(cos.quantile(.25)), r4(cos.quantile(.75)), int(len(dist))]
            for a in (none_as if arm == "none" else [arm]):
                out.append([route, int(layer), q, a, float(al)] + m)
    return out


def share_rows(d, none_as, rng, layers_for_route=None):
    out = []
    for q in [-1, 0, 1, 2, 3, 4]:
        dq = d if q < 0 else d[d.q_idx == q]
        for (route, layer, arm, al), g in dq.groupby(["route", "layer", "arm", "alpha"]):
            lab_arr = g.label.fillna("unjudged").to_numpy()
            n = len(lab_arr)
            boot = rng.integers(0, n, size=(N_BOOT, n))
            opt = {}
            for o in np.unique(lab_arr):
                hit = (lab_arr == o).astype(float)
                b = hit[boot].mean(1)
                opt[str(o)] = [r4(hit.mean()), r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5))]
            shown = layers_for_route(route) if layers_for_route else [int(layer)]
            for L in shown:
                for a in (none_as if arm == "none" else [arm]):
                    out.append([route, int(L), q, a, float(al), n, opt])
    return out


def example_rows(d, lab, personas_of, rng, key_layer):
    ex = []
    for (route, layer, q), g in d[d.label.notna()].groupby(["route", "layer", "q_idx"]):
        pick = [grp.index[rng.integers(len(grp))] for _, grp in g.groupby("label")]
        rest = g.drop(pick)
        by_arm = {a: list(rng.permutation(ix)) for a, ix in rest.groupby("arm").groups.items()}
        arms, k = sorted(by_arm), 0
        while len(pick) < PER_QUESTION and any(by_arm.values()):
            a = arms[k % len(arms)]
            k += 1
            if by_arm[a]:
                pick.append(by_arm[a].pop())
        P = personas_of(route, layer)
        for j in pick:
            r = d.loc[j]
            lb, inten, ev = lab.get(r.key, (None, None, ""))
            t = r.response if isinstance(r.response, str) else ""
            ex.append([route, key_layer(layer), r.arm, float(r.alpha), int(q), int(r.prompt_idx), lb, group(P, lb),
                       inten, ev, t if len(t) <= 3000 else t[:3000] + " … [cut; %d characters in full]" % len(t)])
    return ex


def page_ab():
    rng = np.random.default_rng(0)
    parts = []
    for gen, routes, tag in (("step3", STEP3_ROUTES, "A"), ("step2", B_ROUTES, "B")):
        d, lab = load(gen, routes, STEP3_LAYERS, tag)
        d["mlayer"] = d.layer
        d["dist"], d["cos"] = distances(d, d.layer, d.h)
        parts.append((d.drop(columns=["h"]), lab))
    d = pd.concat([p[0] for p in parts], ignore_index=True)
    lab = {**parts[0][1], **parts[1][1]}
    print("answers", len(d), "unlabelled", int(d.label.isna().sum()))
    none_as = ["linear", "manifold"]
    personas = {"%s|L%d" % (r, L): route_personas(r, int(L)) for r, L in {(r, int(L)) for r, L in zip(d.route, d.layer)}}
    data = dict(cfg=dict(arms=["linear", "manifold", "ace"]),
                metrics=metrics_rows(d, none_as), shares=share_rows(d, none_as, rng), personas=personas,
                examples=example_rows(d, lab, lambda r, L: personas["%s|L%d" % (r, L)], rng, int),
                routes=STEP3_ROUTES + B_ROUTES, layers=list(STEP3_LAYERS))
    out = OUT_ROOT / "followups" / "page_AB.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fill_template(_no_nan(data), title="Other Sources and Reverse Routes: Layers 19 and 25",
                                 layers="hidden_states 19 and 25"))
    print("page ->", out, "%.1f MB" % (out.stat().st_size / 1e6))


def page_d():
    rng = np.random.default_rng(0)
    d, lab = load("step5", FWD, [STEP5_LAYERS[0]], "D")
    H = np.stack(d.h.to_numpy())                         # (N, 4, D): one steered state per capped layer
    rows = []
    for j, L in enumerate(STEP5_LAYERS):
        dl = d.drop(columns=["h"]).copy()
        dl["mlayer"] = L
        dl["dist"], dl["cos"] = distances(dl, [L] * len(dl), H[:, j])
        rows.append(dl)
    dm = pd.concat(rows, ignore_index=True)
    dj = d.drop(columns=["h"])
    print("answers", len(dj), "unlabelled", int(dj.label.isna().sum()))
    none_as = ["floor_linear", "floor_manifold"]
    P19 = {r: route_personas(r, STEP5_LAYERS[0]) for r in FWD}           # judge options: the L19 route
    personas = {"%s|L%d" % (r, L): P19[r] for r in FWD for L in STEP5_LAYERS}
    cfg = dict(arms=["floor_linear", "floor_manifold"],
               names={"floor_linear": "Linear floor", "floor_manifold": "Manifold floor"},
               subs={"floor_linear": "h ← h − r̂ · min(⟨h, r̂⟩ − τ_p, 0)",
                     "floor_manifold": "h ← h + S(t(τ_p)) − S(t(⟨h, r̂⟩)) if ⟨h, r̂⟩ < τ_p"},
               x=dict(title="Cap τ_p: percentile p of the target's projections (0 = unsteered)",
                      range=[-4, 79], dtick=25, marks=[], name="p"),
               examples_any_layer=True)
    data = dict(cfg=cfg, metrics=metrics_rows(dm, none_as),
                shares=share_rows(dj, none_as, rng, layers_for_route=lambda r: list(STEP5_LAYERS)),
                personas=personas,
                examples=example_rows(dj, lab, lambda r, L: P19[r], rng, int),
                routes=FWD, layers=list(STEP5_LAYERS))
    out = OUT_ROOT / "followups" / "page_D.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(fill_template(_no_nan(data), title="Activation Capping: Linear and Manifold Floors",
                                 layers="hidden_states 19, 23, 25, 27 (all four at once)",
                                 layer_label="Measured at",
                                 judge_note="The cap acts on hidden states 19, 23, 25 and 27 together; the layer selector only changes where distance is measured.",
                                 ex_note="20 answers per question for the route selected above (100 for all five)"))
    print("page ->", out, "%.1f MB" % (out.stat().st_size / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, choices=["AB", "D"])
    {"AB": page_ab, "D": page_d}[ap.parse_args().page]()
