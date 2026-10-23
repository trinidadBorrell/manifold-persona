"""Phase 3b: single-axis reach (asked for at the checkpoint, 2026-10-21): which named axes can steer the model
toward which personas on their own.

For each Phase 3 persona (cells.json of phase3) and each candidate axis a (unit vector), clamp k = 1 along a
toward the persona's own coordinate: goal c = a.M_0 + S_p (a.v), v = M_t - M_0, S_p the persona's strength
from phase3. Same hook, questions, seeds and judges as Phase 3. The full vector (k = 1 along v) is the
reference cell, already generated in phase3 as cond 'full'.

Outputs feed three views: reach per axis (how many personas each axis reaches), reach per persona over all
axes (personas no axis reaches), and the persona x axis reach matrix.

    python -m persona_axes.phase3_axes --axes set9|set6|all --shard 0 --n-shards 8   # GPU -> $PM/AX21/phase3_axes/gen_<persona>.jsonl
    python -m persona_axes.phase3_axes --axes set6 --analyse    # after judge_build --stage phase3_axes --collect -> P3b_1, P3b_2, tables
    python -m persona_axes.phase3_axes --axes set6 --push 54 --shard 0 --n-shards 8   # matched push: c = a.M_0 + sign(a.v) * 54
    python -m persona_axes.phase3_axes --axes AH --push 54 --shard 0 --n-shards 8     # same for the 10 psychology + TIDE axes (tag AHp54, 2026-10-23)

Result of the projected version (2026-10-21): the persona's own coordinate on one axis is small (|S_p a.v| median 15
against a full-vector push of 57), so no single axis reaches more than 4 of 37 personas and nothing collapses. The
--push variant pushes every axis by the same absolute amount (sign from the persona's coordinate) to ask which
axes evoke which personas when pushed as hard as the full vector was. Tag: set6p54.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from persona_axes import AX, HERE, LAYER, N_SEEDS, QUESTIONS, VIEW
from persona_axes.ceiling import encode, generate
from persona_axes.phase3 import P3

P3A = AX / "phase3_axes"


def axis_list(which, anames, fam):
    if which == "set9":
        return json.loads((AX / "phase2" / "results.json").read_text())["set9"]
    if which == "set6":
        return [n for n in anames if fam[n] in "BCDEFG"]
    if which == "AH":                                                   # the 10 axes set 6 leaves out: psychology (A) and TIDE (H)
        return [n for n in anames if fam[n] in "AH"]
    return [n for n in anames if fam[n] != "N"]


def analyse(which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from persona_axes import OUT, COLLAPSE_CAP
    cells = json.loads((P3 / "cells.json").read_text())
    pl = pd.DataFrame(cells["personas"]).set_index("persona")               # full-vector best reach from Phase 0 (reach, collapse, push)
    jd = AX / ("judge_phase3_axes" if which == "set6" else "judge_phase3_axes_" + which)
    lab = pd.read_csv(jd / "labels.csv"); lab = lab[lab.j10_judged & lab.cell.str.startswith("ax_")].copy()
    lab["axis"] = lab.cell.str[3:]
    axes = json.loads((P3A / ("axes_%s.json" % which)).read_text())
    lab = lab[lab.axis.isin(axes)]
    sfx = "" if which == "set6" else "_" + which
    per = lab.groupby(["persona", "set", "axis"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"), n=("key", "size")).reset_index()
    per["ok"] = per.collapse <= COLLAPSE_CAP
    per["reach_ok"] = per.reach.where(per.ok, 0.0)                           # a collapsed cell counts as not reached
    # (1) per axis
    ax_t = per.groupby("axis").agg(personas_reached_50=("reach_ok", lambda r: int((r >= 0.5).sum())), personas_reached_25=("reach_ok", lambda r: int((r >= 0.25).sum())),
                                   mean_reach=("reach_ok", "mean"), mean_collapse=("collapse", "mean"), n_personas=("persona", "nunique")).sort_values("personas_reached_25", ascending=False)
    # (2)+(3) per persona
    M = per.pivot(index="persona", columns="axis", values="reach_ok").reindex(columns=ax_t.index)
    pp = pd.DataFrame(dict(set=per.groupby("persona").set.first(), best_axis=M.idxmax(1), best_reach=M.max(1), n_axes_reaching_25=(M >= 0.25).sum(1),
                           full_vector_reach=pl.reach.reindex(M.index), full_vector_push=pl.push.reindex(M.index)))
    pp["steerable_by_some_axis"] = pp.best_reach >= 0.25
    pp.loc[pp.best_reach == 0, "best_axis"] = "-"
    pp = pp.sort_values(["best_reach"], ascending=False)
    fam = {x["name"]: x["family"] for x in json.loads((HERE / "axes.json").read_text())["axes"]}
    # P3b_1
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cols = {"A": "C0", "B": "C1", "C": "C2", "D": "C3", "E": "C4", "F": "C5", "G": "C6", "H": "C7"}
    x = np.arange(len(ax_t))
    ax.bar(x - 0.2, ax_t.personas_reached_25, width=0.4, color=[cols[fam[a]] for a in ax_t.index], label="personas reached >= 0.25")
    ax.bar(x + 0.2, ax_t.personas_reached_50, width=0.4, color=[cols[fam[a]] for a in ax_t.index], alpha=0.5, hatch="//", label="personas reached >= 0.5")
    ax.set_xticks(x); ax.set_xticklabels(ax_t.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("number of personas (of %d) the one-axis clamp reaches\n(J10 top 5, not nonsense; cell collapse <= %.0f%%)" % (pp.shape[0], 100 * COLLAPSE_CAP))
    ax.set_title("P3b.1  which axes steer: one-axis clamp, L15, %s\n(full vector reaches %d / %d at >= 0.25; colour = family)" % (
        "toward each persona's own coordinate, strength S_p" if which == "set6" else "matched absolute push " + which.split("p")[-1] + " (sign from the persona's coordinate)",
        int((pp.full_vector_reach >= 0.25).sum()), pp.shape[0]), fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / ("P3b_1_axes_reach%s.png" % sfx), dpi=130); plt.close(fig)
    # P3b_2
    Mh = M.reindex(pp.index); Mh.insert(0, "FULL VECTOR", pp.full_vector_reach)
    fig, ax = plt.subplots(figsize=(13, 0.28 * len(Mh) + 2.5))
    im = ax.imshow(Mh.values.astype(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(Mh.shape[1])); ax.set_xticklabels(Mh.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(Mh))); ax.set_yticklabels(["%s (%s)%s" % (p, pp.set[p], "" if pp.steerable_by_some_axis[p] else "  <- no axis") for p in Mh.index], fontsize=7)
    ax.axvline(0.5, color="w", lw=2)
    ax.set_title("P3b.2  reach per persona (rows, sorted by best axis) and axis (columns, sorted by personas reached); first column = full vector (Phase 0 best cell)\n"
                 "value = share of 20 answers with the target in J10's top 5 and not nonsense; cells with collapse > %.0f%% shown as 0" % (100 * COLLAPSE_CAP), fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, label="reach"); fig.tight_layout(); fig.savefig(OUT / ("P3b_2_persona_by_axis%s.png" % sfx), dpi=130); plt.close(fig)
    (OUT / "tables").mkdir(exist_ok=True)
    ax_t.round(2).to_markdown(OUT / "tables" / ("P3b_axes%s.md" % sfx)); pp.round(2).to_markdown(OUT / "tables" / ("P3b_personas%s.md" % sfx))
    print(ax_t.round(2).to_string()); print(pp.round(2).to_string())
    print("personas no axis reaches at >= 0.25: %d of %d; reached by >= 1 axis at >= 0.5: %d" % ((~pp.steerable_by_some_axis).sum(), len(pp), (pp.best_reach >= 0.5).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--axes", default="set9", choices=["set9", "set6", "AH", "all"])
    ap.add_argument("--push", type=float, default=None, help="matched absolute push along every axis (default: S_p x the persona's coordinate)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args(argv)
    tag = a.axes + ("" if a.push is None else "p%g" % a.push)
    if a.analyse:
        analyse(tag); return
    import torch
    from ace_steering import steer_gpu as AS
    cells = json.loads((P3 / "cells.json").read_text()); B = np.load(P3 / "basis.npz")
    anames = cells["axis_names"]
    fam = {x["name"]: x["family"] for x in json.loads((HERE / "axes.json").read_text())["axes"]}
    axes = axis_list(a.axes, anames, fam)
    A = B["axes"]                                                       # 4096 x 35, unit columns
    V = AX / "vectors"
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    rn, sn = json.loads((V / "roles_names.json").read_text()), json.loads((V / "seta_names.json").read_text())
    Rraw = np.load(V / ("roles_rawmean_%s_L%d.npy" % (VIEW, LAYER))); Sraw = np.load(V / ("seta_rawmean_%s_L%d.npy" % (VIEW, LAYER)))
    P3A.mkdir(exist_ok=True)
    (P3A / ("axes_%s.json" % tag)).write_text(json.dumps(axes))
    personas = cells["personas"][a.shard::a.n_shards]
    tok, model = AS.load_model(AS.MODEL)
    enc0, P0 = encode(tok, model, "")
    for rec in personas:
        name = rec["persona"]
        f = P3A / ("gen_%s_%s.jsonl" % (tag, name))
        if f.exists():
            continue
        Mt = (Sraw[sn.index(name)] if rec["set"] == "a" else Rraw[rn.index(name)]).astype(np.float64)
        v = Mt - M0
        rows = []
        for ax in axes:
            u = A[:, anames.index(ax)]; ut = torch.tensor(u, dtype=torch.float32)
            coord = float(u @ v)
            push = rec["S_p"] * coord if a.push is None else float(np.sign(coord) * a.push)
            c = float(M0 @ u) + push

            def hook(module, args, c=c, ut=ut):
                hs = args[0]
                uu = ut.to(hs.device)
                h = hs.float()
                h = h + (c - h @ uu).unsqueeze(-1) * uu
                return (h.to(hs.dtype),) + tuple(args[1:])
            for sd, ans in enumerate(generate(tok, model, enc0, P0, hook, N_SEEDS)):
                rows += [dict(persona=name, set=rec["set"], axis=ax, coord=coord, S=rec["S_p"], push=push, cell="ax_%s" % ax,
                              seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
        print("%s: %d axes done | %r" % (name, len(axes), rows[-10]["response"][:80]), flush=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
