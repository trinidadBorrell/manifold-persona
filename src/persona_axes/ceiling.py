"""Phase 0 ceiling check (plan v2 s5): the LIN clamp with the full vector, S in {1, 2, 4, 6}, plus the
system-prompted persona, for the 41 set (a) personas and 24 set (b) roles. Layer 15, `mean` states.

Clamp (route_judge_sweep.qwen3_route, k = 1): u = v/|v|, v = M_t - M_0 (raw mean answer-token states at
hidden_states[15]); goal coordinate c = u.M_0 + S |v|; pre-hook on model.model.layers[15], every position:
h <- h + (c - h.u) u. Generation: empty system block, Giovanni's 10 questions, 140 tokens, T 0.7, top-p 0.9,
thinking off, seeds 0 and 1 (20 answers per cell). The prompted cell uses the persona's first system prompt
and no hook. One unsteered baseline cell (no prompt, no hook) is generated once.

    python -m persona_axes.ceiling --plan                      # CPU: picks the 24 set (b) roles -> ceiling/cells.json
    python -m persona_axes.ceiling --shard 0 --n-shards 8      # GPU
    python -m persona_axes.ceiling --plan-hiS                  # CPU: the personas that failed the ceiling at S* -> ceiling_hiS/cells.json
    python -m persona_axes.ceiling --run ceiling_hiS --shard 0 --n-shards 7

Rerun 2026-10-21 (ceiling_hiS): the 35 personas that failed the pass rule at S* = 4, pushed by an ABSOLUTE
amount: goal c = u.M_0 + S * n_med, n_med = median |M_r - M_0| over the 276 roles (9.0 at L15), S in
{4, 6, 8, 12}. In the first run the push was S |v| and |v| is 3-4 for the professional roles (journalist,
mediator, ...) against 16-17 for vampire or tree, so S = 4 moved them by 12-16 units and left the baseline
answer unchanged. Cells are named A<S>; no prompt or baseline cell (both exist in ceiling/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifold_persona.config import ROLE_INSTRUCTIONS_DIR

from persona_axes import AX, HERE, LAYER, MAX_NEW, N_SEEDS, QUESTIONS, S_SWEEP, TEMPERATURE, TOP_P, VIEW

RUN2_TARGETS = ["vampire", "leviathan", "tree", "void"]
BASELINE = "__baseline__"


def plan():
    V = AX / "vectors"
    human = json.loads((HERE / "personas_b_human.json").read_text())["roles"]
    names = json.loads((V / "roles_names.json").read_text())
    X = np.load(V / ("roles_%s_L%d.npy" % (VIEW, LAYER)))
    idx = [names.index(r) for r in human]
    Xh = X[idx]
    Xc = Xh - Xh.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ Vt[0]
    # the 4 roles nearest each quintile midpoint (10, 30, 50, 70, 90 %) of the PC1 coordinate
    picked = []
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        c = np.quantile(pc1, q)
        for j in np.argsort(np.abs(pc1 - c)):
            if human[j] not in picked:
                picked.append(human[j])
            if sum(1 for p in picked if p in [human[k] for k in np.argsort(np.abs(pc1 - c))[:12]]) >= 4 and len(picked) % 4 == 0:
                break
    picked = picked[:20]
    seta = [p["name"] for p in json.loads((HERE / "personas_a.json").read_text())["personas"]]
    cells = dict(layer=LAYER, view=VIEW, S=list(S_SWEEP), seeds=N_SEEDS, questions=QUESTIONS,
                 set_a=seta, set_b=picked + RUN2_TARGETS, pc1_quintile_picks=picked,
                 pc1_coord={human[j]: float(pc1[j]) for j in range(len(human))})
    (AX / "ceiling").mkdir(exist_ok=True)
    (AX / "ceiling" / "cells.json").write_text(json.dumps(cells, indent=1))
    print("set (a) %d, set (b) %d: %s" % (len(seta), len(cells["set_b"]), cells["set_b"]))


def plan_hiS():
    V = AX / "vectors"
    cells = json.loads((AX / "ceiling" / "cells.json").read_text())
    passed = set(json.loads((AX / "judge_ceiling" / "summary.json").read_text())["passed"])
    failed = [p for p in cells["set_a"] + cells["set_b"] if p not in passed]
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    R = np.load(V / ("roles_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    n_med = float(np.median(np.linalg.norm(R - M0, axis=1)))
    new = dict(layer=LAYER, view=VIEW, unit="abs", unit_norm=n_med, S=[4, 6, 8, 12], seeds=N_SEEDS, questions=QUESTIONS,
               set_a=[p for p in failed if p in cells["set_a"]], set_b=[p for p in failed if p in cells["set_b"]],
               prompt_cell=False, baseline_cell=False, source="ceiling/cells.json + judge_ceiling/summary.json passed list")
    (AX / "ceiling_hiS").mkdir(exist_ok=True)
    (AX / "ceiling_hiS" / "cells.json").write_text(json.dumps(new, indent=1))
    print("failed personas: %d set (a), %d set (b); unit norm %.2f; pushes %s" % (len(new["set_a"]), len(new["set_b"]), n_med, [round(S * n_med) for S in new["S"]]))


def system_prompt(name, set_a_prompts):
    if name in set_a_prompts:
        return set_a_prompts[name]["instruction"][0]["pos"]
    d = json.loads((ROLE_INSTRUCTIONS_DIR / (name + ".json")).read_text())
    return d["instruction"][0]["pos"].replace("{model_name}", "Qwen3-8B")


def generate(tok, model, enc, P, hook_fn, seeds):
    import torch
    outs = []
    for sd in range(seeds):
        hk = model.model.layers[LAYER].register_forward_pre_hook(hook_fn) if hook_fn else None
        torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                   pad_token_id=tok.pad_token_id)
        finally:
            if hk:
                hk.remove()
        outs.append([tok.decode(o[qi, P:], skip_special_tokens=True).strip() for qi in range(len(QUESTIONS))])
    return outs


def encode(tok, model, system):
    texts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in QUESTIONS]
    dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    return enc, enc["input_ids"].shape[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--plan-hiS", action="store_true")
    ap.add_argument("--run", default="ceiling", help="cells.json and output folder under $PM/AX21")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args(argv)
    if a.plan:
        plan(); return
    if a.plan_hiS:
        plan_hiS(); return
    import torch
    from ace_steering import steer_gpu as AS
    cells = json.loads((AX / a.run / "cells.json").read_text())
    absolute = cells.get("unit") == "abs"
    V = AX / "vectors"
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    rn, sn = json.loads((V / "roles_names.json").read_text()), json.loads((V / "seta_names.json").read_text())
    Rraw = np.load(V / ("roles_rawmean_%s_L%d.npy" % (VIEW, LAYER)))
    Sraw = np.load(V / ("seta_rawmean_%s_L%d.npy" % (VIEW, LAYER)))
    set_a_prompts = json.loads((AX / "prompts" / "personas_a_prompts.json").read_text())
    personas = ([BASELINE] if a.shard == 0 and cells.get("baseline_cell", True) else []) + [p for p in cells["set_a"] + cells["set_b"]][a.shard::a.n_shards]
    out = AX / a.run
    tok, model = AS.load_model(AS.MODEL)
    enc0, P0 = encode(tok, model, "")
    for name in personas:
        f = out / ("gen_%s.jsonl" % name)
        if f.exists():
            continue
        rows = []
        if name == BASELINE:
            for sd, ans in enumerate(generate(tok, model, enc0, P0, None, N_SEEDS)):
                rows += [dict(persona=name, set="none", cell="baseline", S=0.0, seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
        else:
            which = "a" if name in sn else "b"
            Mt = (Sraw[sn.index(name)] if which == "a" else Rraw[rn.index(name)]).astype(np.float64)
            v = Mt - M0
            d = float(np.linalg.norm(v)); u = v / d
            ut = torch.tensor(u, dtype=torch.float32)
            for S in cells["S"]:
                c = float(M0 @ u) + S * (cells["unit_norm"] if absolute else d)
                cell = ("A%g" if absolute else "S%g") % S

                def hook(module, args, c=c):
                    hs = args[0]
                    uu = ut.to(hs.device)
                    h = hs.float()
                    h = h + (c - h @ uu).unsqueeze(-1) * uu
                    return (h.to(hs.dtype),) + tuple(args[1:])
                for sd, ans in enumerate(generate(tok, model, enc0, P0, hook, N_SEEDS)):
                    rows += [dict(persona=name, set=which, cell=cell, S=S, push=float(c - M0 @ u), seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
            if cells.get("prompt_cell", True):
                enc, P = encode(tok, model, system_prompt(name, set_a_prompts))
                for sd, ans in enumerate(generate(tok, model, enc, P, None, N_SEEDS)):
                    rows += [dict(persona=name, set=which, cell="prompt", S=None, seed=sd, qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
            print("%s |v|=%.1f | %s: %r" % (name, d, rows[0]["cell"], rows[0]["response"][:90]), flush=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
