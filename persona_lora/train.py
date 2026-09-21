"""Train one conditional-LoRA cell: fixed (rank, placement).

Loss = forward KL(teacher||student) over response tokens
     + beta * per-layer normalised MSE on hidden states.

Teacher = this same model with the adapter disabled and the persona system prompt
prefix; student = adapter enabled with an empty system prefix. Suffix tokens are
token-identical, so positions align.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import inject, trainable_parameters
from .data import encode_pair, load_persona_prompt, load_targets, sys_block
from .teacher import (build_prefix_cache, check_cache_equivalence,
                      forward_with_prefix, teacher_forward)


def collate(tokenizer, items, sys_text, device, max_len=640):
    enc = [encode_pair(tokenizer, sys_text, it["question"], it["response"]) for it in items]
    suffixes = [e[1][:max_len] for e in enc]
    starts = [min(e[2], len(s)) for e, s in zip(enc, suffixes)]
    T = max(len(s) for s in suffixes)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    ids = torch.full((len(suffixes), T), pad, dtype=torch.long)
    mask = torch.zeros((len(suffixes), T), dtype=torch.long)
    resp = torch.zeros((len(suffixes), T), dtype=torch.bool)
    for i, (s, st) in enumerate(zip(suffixes, starts)):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
        if st - 1 < len(s) - 1:
            resp[i, st - 1:len(s) - 1] = True
    return ids.to(device), mask.to(device), resp.to(device), enc[0][0]


def kl_and_act(t_out, s_out, resp_mask, beta):
    """Forward KL over response positions + normalised per-layer hidden MSE."""
    sel = resp_mask.reshape(-1)
    tl = t_out.logits.reshape(-1, t_out.logits.shape[-1])[sel].float()
    sl = s_out.logits.reshape(-1, s_out.logits.shape[-1])[sel].float()
    logp_s = F.log_softmax(sl, dim=-1)
    # The teacher is constant, so KL = -H(p_t) - E_{p_t}[log q]. Only the second
    # term carries gradient; keeping the first out of the graph avoids
    # materialising (logp_t - logp_s), the largest [N, vocab] intermediate.
    with torch.no_grad():
        logp_t = F.log_softmax(tl, dim=-1)
        p_t = logp_t.exp()
        neg_ent = (p_t * logp_t).sum(-1).mean()
    kl = neg_ent - (p_t * logp_s).sum(-1).mean()

    act = torch.zeros((), device=kl.device)
    per_layer = []
    if beta > 0 and t_out.hidden_states is not None:
        for th, sh in zip(t_out.hidden_states, s_out.hidden_states):
            d = th.shape[-1]
            t_f = th.reshape(-1, d)[sel].float()
            s_f = sh.reshape(-1, d)[sel].float()
            num = ((s_f - t_f) ** 2).sum(-1)
            den = (t_f ** 2).sum(-1).clamp_min(1e-6)
            l = (num / den).mean()
            per_layer.append(l.detach())
            act = act + l
        act = act / len(t_out.hidden_states)
    return kl + beta * act, kl.detach(), act.detach(), per_layer


def run_eval(model, tokenizer, sel, personas, prefixes, empty_cache, empty_len,
             items_by_persona, device, beta, batch_size, max_batches=4):
    model.eval()
    out = {}
    with torch.no_grad():
        for pi, role in enumerate(personas):
            kls, layers = [], None
            items = items_by_persona[role]
            for b in range(0, min(len(items), max_batches * batch_size), batch_size):
                chunk = items[b:b + batch_size]
                if not chunk:
                    break
                ids, mask, resp, _ = collate(tokenizer, chunk, prefixes[role]["text"], device)
                if resp.sum() == 0:
                    continue
                sel.enabled = False
                t_out = teacher_forward(model, prefixes[role]["cache"],
                                        prefixes[role]["len"], ids, mask,
                                        output_hidden_states=beta > 0)
                sel.enabled, sel.index = True, pi
                s_out = teacher_forward(model, empty_cache, empty_len, ids, mask,
                                        output_hidden_states=beta > 0)
                _, kl, _, pl = kl_and_act(t_out, s_out, resp, beta)
                kls.append(kl.item())
                if pl:
                    layers = [a + b_ for a, b_ in zip(layers, pl)] if layers else list(pl)
            out[role] = {"kl": sum(kls) / max(len(kls), 1), "n_batches": len(kls)}
            if layers:
                out[role]["layer_residual"] = [float(x) / max(len(kls), 1) for x in layers]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--personas", required=True, help="json list of role names")
    ap.add_argument("--targets", required=True, help="jsonl from generate_targets")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--placement", required=True,
                    help="all | early | middle | late | L<k> for a single block")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=1500, help="max steps")
    ap.add_argument("--eval_every", type=int, default=0,
                    help="held-out eval cadence; 0 disables early stopping")
    ap.add_argument("--patience", type=int, default=4,
                    help="evals without held-out improvement before stopping")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--n_heldout", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_persona_b", action="store_true",
                    help="per-persona B instead of shared B + gain vector")
    ap.add_argument("--shuffle_personas", action="store_true",
                    help="control: train persona p's gains on persona q's targets")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if args.dtype != "auto":
        dtype = fixed[args.dtype]
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.config.use_cache = False

    personas = json.loads(Path(args.personas).read_text())
    sel, n_params = inject(model, args.rank, len(personas), args.placement,
                           share_b=not args.per_persona_b)
    devs = {str(p.device) for p in trainable_parameters(model)}
    if devs != {str(next(model.parameters()).device)}:
        raise RuntimeError(f"adapter params on {devs}, model on "
                           f"{next(model.parameters()).device}")
    print(f"cell rank={args.rank} placement={args.placement} "
          f"trainable={n_params:,} personas={len(personas)} devices={devs}", flush=True)

    rows = load_targets(args.targets)
    held_ids = set(sorted({r["question_idx"] for r in rows})[:args.n_heldout])
    train_by, eval_by = {}, {}
    for role in personas:
        rs = [r for r in rows if r["persona"] == role and r["response"]]
        train_by[role] = [r for r in rs if r["question_idx"] not in held_ids]
        eval_by[role] = [r for r in rs if r["question_idx"] in held_ids]
    if args.shuffle_personas:
        rot = personas[1:] + personas[:1]
        train_by = {p: train_by[q] for p, q in zip(personas, rot)}

    sel.enabled = False
    prefixes = {}
    with torch.no_grad():
        for role in personas:
            text = load_persona_prompt(role)
            ids = tokenizer(sys_block(text), return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
            cache, plen = build_prefix_cache(model, ids)
            prefixes[role] = {"text": text, "cache": cache, "len": plen}
        empty_ids = tokenizer(sys_block(""), return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
        empty_cache, empty_len = build_prefix_cache(model, empty_ids)

        probe = train_by[personas[0]][:2]
        ids, mask, resp, pref = collate(tokenizer, probe,
                                        prefixes[personas[0]]["text"], device, args.max_len)
        ok, diff, ckl = check_cache_equivalence(
            model, torch.tensor([pref], device=device), ids, mask)
        print(f"prefix-cache equivalence: ok={ok} max|dlogit|={diff:.4g} "
              f"KL={ckl:.3e}", flush=True)
        if not ok:
            raise RuntimeError(
                f"prefix KV cache KL={ckl:.3e} exceeds tolerance; use --dtype fp32 "
                f"(bf16 carries ~9e-4, the size of the effects being measured)")

    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr)
    warm = max(1, int(0.1 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    rng = random.Random(args.seed)
    model.train()
    hist, t0 = [], time.time()
    curve, best, bad, best_state, stopped_at = [], {"kl": float("inf"), "step": -1}, 0, None, None
    for step in range(args.steps):
        pi = rng.randrange(len(personas))
        role = personas[pi]
        items = rng.sample(train_by[role], min(args.batch_size, len(train_by[role])))
        ids, mask, resp, _ = collate(tokenizer, items, prefixes[role]["text"], device,
                                     args.max_len)
        if resp.sum() == 0:
            continue
        with torch.no_grad():
            sel.enabled = False
            t_out = teacher_forward(model, prefixes[role]["cache"], prefixes[role]["len"],
                                    ids, mask, output_hidden_states=args.beta > 0)
        sel.enabled, sel.index = True, pi
        s_out = forward_with_prefix(model, empty_cache, empty_len, ids, mask,
                                    output_hidden_states=args.beta > 0)
        loss, kl, act, _ = kl_and_act(t_out, s_out, resp, args.beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"step {step:5d} loss={loss.item():.4f} kl={kl.item():.4f} "
                  f"act={float(act):.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)
            hist.append({"step": step, "loss": float(loss), "kl": float(kl),
                         "act": float(act)})

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            ev_now = run_eval(model, tokenizer, sel, personas, prefixes, empty_cache,
                              empty_len, eval_by, device, 0.0, args.batch_size)
            hk = sum(v["kl"] for v in ev_now.values()) / len(ev_now)
            curve.append({"step": step, "heldout_kl": hk})
            improved = hk < best["kl"] - 1e-6
            print(f"    [eval] step={step} heldout_kl={hk:.5f} "
                  f"{'(best)' if improved else f'(no gain {bad+1}/{args.patience})'}",
                  flush=True)
            if improved:
                best = {"kl": hk, "step": step}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                              if any(t in k for t in (".A", ".B", ".s"))}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"    early stop at step {step}; best heldout_kl="
                          f"{best['kl']:.5f} @ step {best['step']}", flush=True)
                    stopped_at = step
                    break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"restored best checkpoint from step {best['step']}", flush=True)
    ev = run_eval(model, tokenizer, sel, personas, prefixes, empty_cache, empty_len,
                  eval_by, device, args.beta, args.batch_size)
    result = {
        "rank": args.rank, "placement": args.placement, "model": args.model,
        "trainable_params": n_params, "personas": personas, "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "beta": args.beta,
        "seed": args.seed, "shuffle_personas": args.shuffle_personas,
        "dtype": str(dtype),
        "heldout_kl_mean": sum(v["kl"] for v in ev.values()) / len(ev),
        "heldout": ev, "history": hist, "wall_s": time.time() - t0,
        "eval_curve": curve, "best_step": best["step"], "stopped_at": stopped_at,
        "max_steps": args.steps, "patience": args.patience,
        "eval_every": args.eval_every,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    torch.save({k: v for k, v in model.state_dict().items()
                if any(t in k for t in (".A", ".B", ".s"))}, out_dir / "adapter.pt")
    print(f"DONE heldout_kl={result['heldout_kl_mean']:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
