"""GPU driver for the ACE study. Also the generation engine (load_model, run_batch, cell tables)
reused by pca_pga_steering.steer_gpu, route_judge_sweep.qwen3_route and giovanni_repro.qwen3_pin.

    condor: jobs_condor/run_ace.sh  ->  python -m ace_steering.steer_gpu --stage ...

STAGES
  plan     CPU. Print the cell list + counts, run the geometry controls, write
           cells.csv and route.json per (route, layer). No model.
  smoke    GPU. smoke_cells(): every arm at alpha {0,1}, 1 route, L19, 2 questions,
           1 prompt, 1 sample, 64 tokens. Checks the steered state's chord coordinate.
  gen      GPU, sharded over cells. --step 1|2.
  collect  CPU. One generations table + steer_acts.npy per step.

PER CELL, per sample s (batch of N_PROMPTS x 5 questions = 25 answers):
  1. torch seed = seed_for(route, layer, s), shared by every arm and alpha;
  2. generate with the arm's hook live (none: no hook), left padding, 512 tokens;
  3. replay prompt+answer once with the hook live and CaptureLayer registered
     after it: the STEERED state at the steered layer, mean over response tokens
     (up to and excluding the first end-of-turn / EOS / pad).

RESUMABLE: a cell whose table exists is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO / "jobs_condor"))   # site-specific shims (mp_compat, mp_qwen3) stay untracked there

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ace_steering import grid as GR  # noqa: E402
from ace_steering.common import (MAX_NEW_TOKENS, N_PROMPTS, N_SAMPLES, OUT_ROOT,  # noqa: E402
                                       QUESTIONS, REP4_COLLAPSED, TEMPERATURE, TOP_K, TOP_P,
                                       route_slug, seed_for)
from steering_followups.common import degeneracy  # noqa: E402

ROLE_DIR = _REPO / "extra/assistant-axis/data/roles/instructions"
MODEL = os.environ.get("PM_MODEL", "Qwen/Qwen3-8B")
BATCH = 25


def system_prompts(role):
    ins = json.loads((ROLE_DIR / f"{role}.json").read_text())["instruction"]
    out = [i["pos"] for i in ins[:N_PROMPTS]]
    if len(out) != N_PROMPTS:
        raise ValueError("%s has %d prompts, need %d" % (role, len(out), N_PROMPTS))
    return out


def units(route, prompts_idx=None, questions=None):
    """[(prompt_idx, q_idx, system, question)] -- one batch, prompt-major."""
    A = route.split(">")[0]
    sp = system_prompts(A)
    pi = range(N_PROMPTS) if prompts_idx is None else prompts_idx
    qi = range(len(QUESTIONS)) if questions is None else questions
    return [(p, q, sp[p], QUESTIONS[q]) for p in pi for q in qi]


def write_table(df, stem):
    try:
        import pyarrow  # noqa: F401
        final, w = Path(str(stem) + ".parquet"), (lambda f: df.to_parquet(f, index=False))
    except ImportError:
        final, w = Path(str(stem) + ".csv"), (lambda f: df.to_csv(f, index=False))
    tmp = final.with_name(final.name + ".tmp")
    w(tmp)
    os.replace(tmp, final)          # atomic: the table's presence marks the cell done
    return final


def read_table(stem):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = Path(str(stem) + ext)
        if p.exists():
            return fn(p)
    raise FileNotFoundError(stem)


def cell_done(stem):
    return any(Path(str(stem) + e).exists() for e in (".parquet", ".csv"))


# ---------------------------------------------------------------------------
# plan (CPU)
# ---------------------------------------------------------------------------

def route_record(route, layer):
    G = GR.geom(layer)
    p = GR.route_path(route, layer)
    sp = p.spline
    arc = sp._build_arc()
    knot_alpha = [float(np.interp(x, arc["u"], arc["s"]) / arc["L"]) for x in sp.x[1:-1]]
    _, _, cA, cB = G.route(route)
    return dict(route=route, layer=layer, geom=G.path, personas=GR.route_personas(route, layer),
                knot_alpha=knot_alpha, detour=float(p.detour_ratio),
                chord=float(np.linalg.norm(cB - cA)))


def stage_plan(a, cells):
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    bad = GR.controls(cells)
    for b in bad[:20]:
        print("CONTROL FAIL", b)
    if bad:
        raise SystemExit("geometry controls failed on %d checks" % len(bad))
    print("controls: PASS (alpha=1 linear/manifold delta == mu_B - mu_A; every route has knots)")
    rows = [dict(cell_id=c.cell_id, step=c.step, route=c.route, layer=c.layer, arm=c.arm,
                 alpha=c.alpha, n_answers=c.n_answers) for c in cells]
    pd.DataFrame(rows).to_csv(root / ("cells_step%d.csv" % a.step), index=False)
    recs = {}
    for route, layer in sorted({(c.route, c.layer) for c in cells}):
        recs["%s|L%d" % (route, layer)] = r = route_record(route, layer)
        print("L%-2d %-22s chord %6.1f detour %.2f  %s" % (layer, route, r["chord"], r["detour"],
                                                           " > ".join(r["personas"])))
    p = root / "routes.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **recs}, indent=1))
    print("%d cells, %d answers -> %s" % (len(cells), sum(c.n_answers for c in cells), root))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(name):
    import mp_compat  # noqa: F401
    import mp_qwen3  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("shims: %s | %s" % (mp_compat.APPLIED, mp_qwen3.APPLIED), flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                     device_map="auto")
    model.eval()
    return tok, model


def make_hook(model, res, layer):
    import torch
    from steering_core.activation_steering import ActivationSteering
    from steering_core.interventions import (DeltaStats, _constant_delta_fn, make_ace_delta_fn,
                                       make_subspace_pin_delta_fn)
    if res["kind"] == "none":
        return None, None
    stats = DeltaStats()
    if res["kind"] == "floor":                    # follow-up D: one floor per layer, all at once
        from contextlib import ExitStack
        from steering_core.activation_steering import hook_layer_for_hidden_state
        from steering_core.interventions import make_floor_delta_fn

        class _Stack:
            def __enter__(self_):
                self_.es = ExitStack()
                for spec in res["layers"]:
                    fn = make_floor_delta_fn(spec["mu_a"], spec["mu_b"], spec["tau"], spec["S_grid"],
                                             torch.float16, stats)
                    self_.es.enter_context(ActivationSteering(
                        model, delta_fn=fn, layer_indices=[hook_layer_for_hidden_state(spec["layer"])],
                        positions="all", intervention_type="dynamic"))
                return self_

            def __exit__(self_, *e):
                return self_.es.__exit__(*e)
        return _Stack(), stats
    if res["kind"] == "constant":
        fn = _constant_delta_fn(res["vec"], torch.float16, stats)
    elif res["kind"] == "pin":
        fn = make_subspace_pin_delta_fn(res["U"], res["target"], torch.float16, stats)
    else:
        fn = make_ace_delta_fn(res["mu_a"], res["mu_b"], res["alpha"], torch.float16, stats)
    return ActivationSteering(model, delta_fn=fn, layer_indices=[layer], positions="all",
                              intervention_type="dynamic"), stats


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def run_batch(tok, model, route, us, res, hs_layer, seed, max_new):
    """Generate one batch with the hook live, then capture the steered state."""
    import torch
    from steering_core.activation_steering import hook_layer_for_hidden_state
    from steering_followups.replace import CaptureLayer
    layer = hook_layer_for_hidden_state(hs_layer)
    texts = [tok.apply_chat_template(([{"role": "system", "content": s}] if s is not None else [])
                                     + [{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
             for (_, _, s, q) in us]
    in_dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(in_dev)
    P = enc["input_ids"].shape[1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hook, stats = make_hook(model, res, layer)
    with torch.no_grad(), (hook or _Null()):
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, P:]
    stop = {i for i in (tok.eos_token_id, tok.pad_token_id,
                        tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}
    is_stop = torch.zeros_like(gen, dtype=torch.bool)
    for s in stop:
        is_stop |= gen == s
    # response = everything before the first stop token
    resp = torch.cumsum(is_stop.int(), dim=1) == 0
    mask = torch.cat([enc["attention_mask"], resp.to(enc["attention_mask"].dtype)], dim=1)
    pos_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
    replay, _ = make_hook(model, res, layer)          # fresh stats: generation only is reported
    # The transformer body only (model.model): the lm_head's (B, T, 152k) logits are not
    # needed here, and accelerate copies them back to GPU 0 -- 3.9 GB, OOM on a P100.
    if res.get("capture_layers"):                      # step 5: every cell, unsteered included
        cap_layers = [hook_layer_for_hidden_state(L) for L in res["capture_layers"]]
    elif res["kind"] == "floor":
        cap_layers = [hook_layer_for_hidden_state(sp["layer"]) for sp in res["layers"]]
    else:
        cap_layers = [layer]
    from contextlib import ExitStack
    with torch.no_grad(), (replay or _Null()), ExitStack() as es:
        caps = [es.enter_context(CaptureLayer(model, Lc)) for Lc in cap_layers]
        model.model(input_ids=out, attention_mask=mask, position_ids=pos_ids, use_cache=False)
    Hs = [c.outputs[0] for c in caps]                  # each (B, T, D) float32 cpu
    rm = resp.cpu()
    rows, acts = [], []
    for b, (pi, qi, s, q) in enumerate(us):
        n_resp = int(rm[b].sum())
        hs = [H[b, P:P + gen.shape[1]][rm[b]].mean(0).numpy() if n_resp else np.full(H.shape[-1], np.nan)
              for H in Hs]
        h = hs[0] if len(hs) == 1 else np.stack(hs)    # floor: (n_layers, D), one per capped layer
        text = tok.decode(gen[b][rm[b].to(gen.device)], skip_special_tokens=True).strip()
        rows.append(dict(prompt_idx=pi, q_idx=qi, system=s, question=q, response=text,
                         n_new=n_resp, hit_max=bool(n_resp >= max_new)))
        acts.append(h.astype(np.float32))
    mag = (stats.mag_sum / stats.n_mag) if (stats is not None and stats.n_mag) else None
    return rows, acts, mag


def chord_stats(h, cA, cB):
    r = cB - cA
    t = float((h - cA) @ r / (r @ r))                  # 0 at mu_A, 1 at mu_B (along r)
    return dict(chord_t=t, d_src=float(np.linalg.norm(h - cA)), d_tgt=float(np.linalg.norm(h - cB)))


def stage_gen(a, cells, step):
    import torch  # noqa: F401
    root = Path(a.out_root)
    todo = cells[a.shard_index::a.num_shards]
    print("step %s: shard %d/%d owns %d of %d cells" % (step, a.shard_index, a.num_shards,
                                                        len(todo), len(cells)), flush=True)
    bad = GR.controls(todo)
    if bad:
        raise SystemExit("controls failed: %s" % bad[:5])
    res_all = {c: GR.resolve(c) for c in todo}         # before 16 GB of weights
    if step == 5:
        from ace_steering.common import STEP5_LAYERS
        for r in res_all.values():
            r["capture_layers"] = list(STEP5_LAYERS)
    if a.dry_run:
        print("dry run: %d cells resolved" % len(todo))
        return
    tok, model = load_model(a.model)
    out = root / ("step%d" % step if step else "smoke") / "cells"
    out.mkdir(parents=True, exist_ok=True)
    n_samp = 1 if step == 0 else N_SAMPLES[step]
    max_new = 64 if step == 0 else a.max_new_tokens
    for i, c in enumerate(todo):
        stem = out / c.cell_id
        if cell_done(stem):
            print("[%d/%d] %s exists, skipped" % (i + 1, len(todo), c.cell_id), flush=True)
            continue
        t0, rows, acts = time.time(), [], []
        G = GR.geom(c.layer)
        A, B, cA, cB = G.route(c.route)
        us = units(c.route, [0], [0, 1]) if step == 0 else units(c.route)
        for s in range(n_samp):
            seed = seed_for(c.route, c.layer, s)
            for b0 in range(0, len(us), BATCH):
                rr, hh, mag = run_batch(tok, model, c.route, us[b0:b0 + BATCH], res_all[c],
                                        c.layer, seed + b0, max_new)
                for r, h in zip(rr, hh):
                    dg = degeneracy(r["response"])
                    r.update(step=step, cell_id=c.cell_id, route=c.route, source=A, target=B,
                             layer=c.layer, arm=c.arm, alpha=c.alpha, sample=s, seed=seed + b0,
                             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                             max_new_tokens=max_new, delta_norm=res_all[c]["delta_norm"],
                             delta_norm_per_token=mag,
                             collapsed=bool(dg["rep4"] > REP4_COLLAPSED),
                             **{f"deg_{k}": v for k, v in dg.items()},
                             **chord_stats((h if h.ndim == 1 else h[0]).astype(np.float64), cA, cB))
                rows += rr
                acts += hh
        np.save(str(stem) + ".steer.npy", np.stack(acts))
        df = pd.DataFrame(rows)
        write_table(df, stem)                           # last: its presence = cell done
        print("[%d/%d] %s %.0fs chord_t %.2f  collapsed %d/%d  hit_max %d | %r"
              % (i + 1, len(todo), c.cell_id, time.time() - t0, df.chord_t.mean(),
                 df.collapsed.sum(), len(df), df.hit_max.sum(), df.response.iloc[0][:70]),
              flush=True)


def stage_collect(a, step):
    root = Path(a.out_root)
    d = root / ("step%d" % step if step else "smoke")
    frames, S = [], []
    for p in sorted(list((d / "cells").glob("*.parquet")) + list((d / "cells").glob("*.csv"))):
        stem = str(p)[: -len(p.suffix)]
        frames.append(read_table(stem))
        S.append(np.load(stem + ".steer.npy"))
    if not frames:
        raise SystemExit("nothing under %s/cells" % d)
    df = pd.concat(frames, ignore_index=True)
    X = np.concatenate(S)
    df.to_csv(d / "generations.csv", index=False)
    np.save(d / "steer_acts.npy", X)
    print("step %d: %d rows from %d cells -> %s" % (step, len(df), df.cell_id.nunique(), d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["plan", "smoke", "gen", "collect"])
    ap.add_argument("--step", type=int, default=1, choices=[0, 1, 2, 3, 5])
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default=None,
                    help="comma list of arms (default: none,linear,ace,manifold)")
    ap.add_argument("--include-csv", default=None,
                    help="only the cells listed here (e.g. the unfinished cells of failed jobs)")
    ap.add_argument("--exclude-csv", default=None,
                    help="cells.csv of an earlier submission: its cells are left out here")
    ap.add_argument("--skip-short", action="store_true",
                    help="leave out (route, layer) pairs whose manifold route has < 8 knots")
    a = ap.parse_args(argv)
    if not 0 <= a.shard_index < a.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if a.stage == "smoke":
        if a.step == 5:      # floor smoke: 1 route, p 25 and 75, both floors
            fc = GR.floor_cells(["assistant>vampire"])
            stage_gen(a, [c for c in fc if c.arm == "none" or c.alpha in (25.0, 75.0)], 0)
        else:
            stage_gen(a, GR.smoke_cells(), 0)
    elif a.stage == "plan":
        stage_plan(a, GR.build_cells(a.step, skip_short=a.skip_short,
                                     arms=a.arms.split(",") if a.arms else None))
    elif a.stage == "gen":
        cells = (GR.floor_cells() if a.step == 5 else
                 GR.build_cells(a.step, skip_short=a.skip_short, arms=a.arms.split(",") if a.arms else None))
        if a.exclude_csv:
            done = set(pd.read_csv(a.exclude_csv).cell_id)
            cells = [c for c in cells if c.cell_id not in done]
        if a.include_csv:
            want = set(pd.read_csv(a.include_csv).cell_id)
            cells = [c for c in cells if c.cell_id in want]
        stage_gen(a, cells, a.step)
    else:
        stage_collect(a, a.step if a.step else 0)


if __name__ == "__main__":
    main()
