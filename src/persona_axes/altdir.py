"""Alternative steering directions, pilot (checkpoint decision 2026-10-21, exploratory).

The Phase 0 direction is v = M_t - M_0: mean answer-token state of the persona minus the no-prompt model's.
For professional personas that vector is short (3-5 units at L15) and the clamp goes from the baseline answer
straight into collapse. Three alternatives, same clamp (k = 1, every position, hidden_states[15]), compared at
the same ABSOLUTE pushes along the unit direction, on 5 personas the full vector reached and 8 it did not:

  mean        v = M_t - M_0                      (reference, as Phase 0)
  contrast    v = M_t - M_assistant               M_assistant = raw mean answer state of the `assistant` role
                                                 ("You are an assistant dedicated to helping users ...", E1b, L15)
  prompt_last v = p_t - p_0                       hidden state at the last prompt token (the generation position),
                                                 mean over the persona's 5 system prompts x the 40 E1 questions;
                                                 p_0 the same with an empty system block
  prompt_mean v = q_t - q_0                       mean over the system-prompt tokens, same averaging

Goal c = u.M_0 + push, push in {18, 36, 54, 72, 108}; Giovanni's 10 questions, 2 seeds, 20 answers per cell.
Prompt states are extracted by the same job (forward pass only, no generation) and cached per persona.

    python -m persona_axes.altdir --shard 0 --n-shards 4     # GPU -> $PM/AX21/altdir/{states_<persona>.npz, gen_<persona>.jsonl}
    python -m persona_axes.altdir --analyse                   # after judge_build --stage altdir --collect: P0_4 + tables
    python -m persona_axes.altdir --personas all --shard 0 --n-shards 8   # full run: the 65 Phase 0 personas, directions
                                                              # mean / contrast / prompt_last / span2, pushes 36-72 -> $PM/AX21/altdir_all
span2 = clamp on the 2-dim span of the contrast and prompt_last unit vectors; both coordinates are pinned to the
point M_0 + push * w, w = unit bisector of the two. prompt_mean is dropped from the full run (sink token).

Judged with Haiku J10 on every answer (+ J8 on 25%); Sonnet J10 on 25 of the 347 chunks agreed with Haiku on
"reached" for 96% of answers and on nonsense for 93% (Haiku flags slightly less nonsense: 0.31 vs 0.37).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from manifold_persona.config import ROLE_INSTRUCTIONS_DIR

from giovanni_repro.qwen3_pin import PM

from persona_axes import AX, HERE, LAYER, N_SEEDS, QUESTIONS, VIEW
from persona_axes.ceiling import encode, generate

PILOT = {"worked": ["bard", "tree", "kids_storyteller", "legal_formal", "companion_cheerful"],
         "failed": ["journalist", "mediator", "negotiator", "editor_strict", "tutor_socratic", "critic_harsh", "legal_streetwise", "coach_drill"]}
PUSHES = [18, 36, 54, 72, 108]
DIRS = ["mean", "contrast", "prompt_last", "prompt_mean"]
PUSHES_ALL = [36, 54, 72]
DIRS_ALL = ["mean", "contrast", "prompt_last", "span2"]
AD = AX / "altdir"


def e1_questions():
    m = pd.read_csv(PM / "E1" / "meta.csv", usecols=["question_idx", "question"]).drop_duplicates("question_idx").sort_values("question_idx")
    return m.question.tolist()


def system_prompts(name, set_a_prompts):
    if name in set_a_prompts:
        return [x["pos"] for x in set_a_prompts[name]["instruction"]]
    d = json.loads((ROLE_INSTRUCTIONS_DIR / (name + ".json")).read_text())
    return [x["pos"].replace("{model_name}", "Qwen3-8B") for x in d["instruction"]]


def prompt_states(tok, model, systems, questions):
    """mean over (system prompt x question) of hidden_states[LAYER] at the last prompt token and over the system tokens"""
    import torch
    dev = next(model.parameters()).device
    cap = {}

    def hook(module, args):
        cap["h"] = args[0].float()
    last, smean, n = np.zeros(4096), np.zeros(4096), 0
    for sysm in systems:
        n_sys = len(tok.apply_chat_template([{"role": "system", "content": sysm}], tokenize=True, add_generation_prompt=False))
        enc, P = encode(tok, model, sysm)
        hk = model.model.layers[LAYER].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        finally:
            hk.remove()
        H = cap["h"]; m = enc["attention_mask"].bool()
        for i in range(H.shape[0]):
            pos = torch.nonzero(m[i]).squeeze(1)                     # left padding: the first n_sys real tokens are the system block
            last += H[i, pos[-1]].cpu().numpy()
            smean += H[i, pos[:n_sys]].mean(0).cpu().numpy()
            n += 1
    return last / n, smean / n


def load_or_compute(f, fn, shard):
    """shared cache between shards: atomic write (tmp + rename); a partial or missing file is recomputed"""
    if f.exists():
        try:
            z = np.load(f); return z["last"], z["smean"]
        except Exception:
            pass
    last, smean = fn()
    tmp = f.with_name("%s.tmp%d.npz" % (f.stem, shard))              # np.savez appends .npz unless the name ends in it
    np.savez(tmp, last=last, smean=smean); os.replace(tmp, f)
    return last, smean


def analyse_all():
    """full run (65 personas, 4 directions): P0_5 reach vs push per direction + persona x direction heatmap; tables"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from persona_axes import OUT, COLLAPSE_CAP
    lab = pd.read_csv(AX / "judge_altdir_all" / "labels.csv"); lab = lab[lab.j10_judged].copy()
    lab["direction"] = lab.cell.str.rsplit("_p", n=1).str[0]; lab["push"] = lab.cell.str.rsplit("_p", n=1).str[1].astype(int)
    passed = set(json.loads((AX / "judge_ceiling" / "summary.json").read_text())["passed"]); lab["group"] = np.where(lab.persona.isin(passed), "passed_S4", "failed_S4")
    g = lab.groupby(["group", "direction", "push"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"), n=("key", "size")).reset_index()
    per = lab.groupby(["persona", "set", "group", "direction", "push"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean")).reset_index()
    per["reach_ok"] = per.reach.where(per.collapse <= COLLAPSE_CAP, 0.0)
    best = per.sort_values("reach_ok", ascending=False).drop_duplicates(["persona", "direction"])
    M = best.pivot(index="persona", columns="direction", values="reach_ok")[DIRS_ALL]
    summ = pd.DataFrame(dict(personas_reached_50=(M >= 0.5).sum(), personas_reached_25=(M >= 0.25).sum(), mean_best_reach=M.mean(),
                             best_direction_for_n_personas=M.idxmax(1).value_counts().reindex(DIRS_ALL).fillna(0).astype(int)))
    order = M.max(1).sort_values(ascending=False).index
    fig, axs = plt.subplots(1, 3, figsize=(18, 0.2 * len(M) + 3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    cols = {"mean": "k", "contrast": "C0", "prompt_last": "C3", "span2": "C2"}
    for grp_name, ls in (("passed_S4", "-"), ("failed_S4", "--")):
        for d in DIRS_ALL:
            x = g[(g.group == grp_name) & (g.direction == d)].sort_values("push")
            axs[0].plot(x.push, x.reach, ls, marker="o", ms=3, color=cols[d], label="%s, %s" % (d, grp_name))
            axs[1].plot(x.push, x.collapse, ls, marker="o", ms=3, color=cols[d])
    axs[1].axhline(COLLAPSE_CAP, color="r", ls=":", label="collapse cap")
    axs[0].set_ylabel("share of answers with target in J10 top 5 and not nonsense"); axs[1].set_ylabel("share of answers J10 flags nonsense")
    for a in axs[:2]:
        a.set_xlabel("absolute push, L15"); a.set_xticks(PUSHES_ALL)
    axs[0].set_title("P0.5  four clamp directions on the 65 Phase 0 personas: reach vs push\n(solid = 30 passed at S* = 4, dashed = 35 failed)", fontsize=10)
    axs[1].set_title("collapse vs push", fontsize=10); axs[0].legend(fontsize=7); axs[1].legend(fontsize=8)
    im = axs[2].imshow(M.reindex(order).values.astype(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axs[2].set_xticks(range(len(DIRS_ALL))); axs[2].set_xticklabels(DIRS_ALL, rotation=30, ha="right", fontsize=8)
    axs[2].set_yticks(range(len(order))); axs[2].set_yticklabels(["%s (%s)" % (p, "p" if p in passed else "f") for p in order], fontsize=6)
    axs[2].set_title("best reach per persona and direction\n(best push with collapse <= cap; p = passed S* = 4, f = failed)", fontsize=10)
    plt.colorbar(im, ax=axs[2], fraction=0.04, label="reach")
    fig.tight_layout(); fig.savefig(OUT / "P0_5_directions_all.png", dpi=130); plt.close(fig)
    (OUT / "tables").mkdir(exist_ok=True)
    g.round(3).to_markdown(OUT / "tables" / "P0_5_directions_by_push.md", index=False); summ.to_markdown(OUT / "tables" / "P0_5_directions_summary.md")
    M.reindex(order).round(2).to_markdown(OUT / "tables" / "P0_5_best_reach_per_persona.md")
    print(summ.to_string()); print(g.round(2).to_string())


def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from persona_axes import OUT, COLLAPSE_CAP
    lab = pd.read_csv(AX / "judge_altdir" / "labels.csv"); lab = lab[lab.j10_judged].copy()
    lab["direction"] = lab.cell.str.rsplit("_p", n=1).str[0]; lab["push"] = lab.cell.str.rsplit("_p", n=1).str[1].astype(int)
    grp = {p: g for g, ps in PILOT.items() for p in ps}; lab["group"] = lab.persona.map(grp)
    g = lab.groupby(["group", "direction", "push"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean"), n=("key", "size")).reset_index()
    j8 = lab[lab.j8_label.notna()].groupby(["group", "direction", "push"]).agg(j8_reach=("j8_reached", "mean"), j8_collapse=("j8_collapsed", "mean")).reset_index()
    g = g.merge(j8, on=["group", "direction", "push"], how="left")
    per = lab.groupby(["persona", "group", "direction", "push"]).agg(reach=("j10_top5", "mean"), collapse=("j10_nonsense", "mean")).reset_index()
    best = per[per.collapse <= COLLAPSE_CAP].sort_values("reach", ascending=False).drop_duplicates(["persona", "direction"])
    mat = best.pivot(index="persona", columns="direction", values="reach").reindex(PILOT["worked"] + PILOT["failed"])[DIRS]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    cols = {"mean": "k", "contrast": "C0", "prompt_last": "C3", "prompt_mean": "grey"}
    for grp_name, ls in (("worked", "-"), ("failed", "--")):
        for d in DIRS:
            x = g[(g.group == grp_name) & (g.direction == d)].sort_values("push")
            axs[0].plot(x.push, x.reach, ls, marker="o", ms=3, color=cols[d], label="%s, %s" % (d, grp_name))
            axs[1].plot(x.push, x.collapse, ls, marker="o", ms=3, color=cols[d])
    axs[1].axhline(COLLAPSE_CAP, color="r", ls=":", label="collapse cap %.0f%%" % (100 * COLLAPSE_CAP))
    axs[0].set_ylabel("share of answers with target in J10 top 5 and not nonsense"); axs[1].set_ylabel("share of answers J10 flags nonsense")
    for a in axs[:2]:
        a.set_xlabel("absolute push along the unit direction, L15"); a.set_xticks(PUSHES)
    axs[0].set_title("P0.4  four directions, k = 1 clamp: reach vs push\n(solid = 5 personas the full vector reached, dashed = 8 it did not)", fontsize=10)
    axs[1].set_title("collapse vs push (same lines)", fontsize=10); axs[0].legend(fontsize=7); axs[1].legend(fontsize=8)
    im = axs[2].imshow(mat.values.astype(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axs[2].set_xticks(range(len(DIRS))); axs[2].set_xticklabels(DIRS, rotation=30, ha="right", fontsize=8)
    axs[2].set_yticks(range(len(mat))); axs[2].set_yticklabels(["%s (%s)" % (p, grp[p][0]) for p in mat.index], fontsize=8)
    axs[2].set_title("best reach per persona and direction\n(highest push with collapse <= cap; w = worked, f = failed)", fontsize=10)
    plt.colorbar(im, ax=axs[2], fraction=0.04, label="reach")
    fig.tight_layout(); fig.savefig(OUT / "P0_4_directions.png", dpi=130); plt.close(fig)
    g.round(3).to_markdown(OUT / "tables" / "P0_4_directions_by_push.md", index=False)
    mat.round(2).to_markdown(OUT / "tables" / "P0_4_best_reach_per_persona.md")
    print(g.round(2).to_string()); print(mat.round(2).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--personas", default="pilot", choices=["pilot", "all"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args(argv)
    if a.analyse:
        (analyse_all if a.personas == "all" else analyse)(); return
    import torch
    from ace_steering import steer_gpu as AS
    global AD
    full = a.personas == "all"
    if full:
        AD = AX / "altdir_all"
    AD.mkdir(exist_ok=True)
    V = AX / "vectors"
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    rn, sn = json.loads((V / "roles_names.json").read_text()), json.loads((V / "seta_names.json").read_text())
    Rraw = np.load(V / ("roles_rawmean_%s_L%d.npy" % (VIEW, LAYER))); Sraw = np.load(V / ("seta_rawmean_%s_L%d.npy" % (VIEW, LAYER)))
    meta = pd.read_csv(PM / "E1b" / "meta.csv", usecols=["role"])
    H = np.load(PM / "E1b" / ("mean_L%d.npy" % LAYER), mmap_mode="r")
    M_asst = np.asarray(H[(meta.role == "assistant").to_numpy()], dtype=np.float64).mean(0)
    set_a_prompts = json.loads((AX / "prompts" / "personas_a_prompts.json").read_text())
    qs = e1_questions()
    if full:
        c0 = json.loads((AX / "ceiling" / "cells.json").read_text())
        passed = set(json.loads((AX / "judge_ceiling" / "summary.json").read_text())["passed"])
        personas = [(p, "passed_S4" if p in passed else "failed_S4") for p in c0["set_a"] + c0["set_b"]][a.shard::a.n_shards]
        pushes, dnames = PUSHES_ALL, DIRS_ALL
    else:
        personas = [(p, g) for g, ps in PILOT.items() for p in ps][a.shard::a.n_shards]
        pushes, dnames = PUSHES, DIRS
    tok, model = AS.load_model(AS.MODEL)
    enc0, P0 = encode(tok, model, "")
    f0 = AD / "states___neutral__.npz"
    p0, q0 = load_or_compute(f0, lambda: prompt_states(tok, model, [""], qs), a.shard)
    for name, group in personas:
        f = AD / ("gen_%s.jsonl" % name)
        if f.exists():
            continue
        pl, ql = load_or_compute(AD / ("states_%s.npz" % name), lambda: prompt_states(tok, model, system_prompts(name, set_a_prompts), qs), a.shard)
        which = "a" if name in sn else "b"
        Mt = (Sraw[sn.index(name)] if which == "a" else Rraw[rn.index(name)]).astype(np.float64)
        dirs = {"mean": Mt - M0, "contrast": Mt - M_asst, "prompt_last": pl - p0, "prompt_mean": ql - q0}
        unit = lambda d: d / np.linalg.norm(d)  # noqa: E731
        rows = []
        for dname in dnames:
            if dname == "span2":
                U = np.linalg.qr(np.stack([unit(dirs["contrast"]), unit(dirs["prompt_last"])], 1))[0]       # 4096 x 2
                w = unit(unit(dirs["contrast"]) + unit(dirs["prompt_last"]))
                nd = float("nan"); cos_v = float(w @ unit(Mt - M0))
            else:
                d = dirs[dname]; nd = float(np.linalg.norm(d)); u = unit(d); cos_v = float(u @ unit(Mt - M0))
            for push in pushes:
                if dname == "span2":
                    Ut = torch.tensor(U, dtype=torch.float32); tc = torch.tensor(U.T @ (M0 + push * w), dtype=torch.float32)

                    def hook(module, args, Ut=Ut, tc=tc):
                        hs = args[0]
                        uu, cc = Ut.to(hs.device), tc.to(hs.device)
                        h = hs.float()
                        h = h + (cc - h @ uu) @ uu.T
                        return (h.to(hs.dtype),) + tuple(args[1:])
                else:
                    c = float(M0 @ u) + push; ut = torch.tensor(u, dtype=torch.float32)

                    def hook(module, args, c=c, ut=ut):
                        hs = args[0]
                        uu = ut.to(hs.device)
                        h = hs.float()
                        h = h + (c - h @ uu).unsqueeze(-1) * uu
                        return (h.to(hs.dtype),) + tuple(args[1:])
                for sd, ans in enumerate(generate(tok, model, enc0, P0, hook, N_SEEDS)):
                    rows += [dict(persona=name, set=which, group=group, direction=dname, dir_norm=nd, cos_with_mean_dir=cos_v, push=push,
                                  cell="%s_p%d" % (dname, push), seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
            print("%s %s |d|=%.1f cos(v)=%.2f | p54: %r" % (name, dname, nd, cos_v, next(r["response"][:80] for r in rows if r["direction"] == dname and r["push"] == 54)), flush=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
