"""Steering-vector baseline: the object Persona Vectors actually uses.

For each persona we take the mean residual-stream difference (prompted teacher minus
unprompted student) at one layer over response tokens -- that is the persona vector --
then add ``alpha_p * v_p`` to the residual stream at that layer.

``alpha_p`` is *fitted* with the same KL objective, one scalar per persona, so the
steering vector is given its best case. Anything the conditional LoRA gains over this
is gain over a tuned version of the standard method, not over a strawman.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_persona_prompt, load_targets, sys_block
from .teacher import build_prefix_cache, forward_with_prefix, teacher_forward
from .train import collate, kl_and_act


class Steerer:
    """Adds alpha_p * v_p[layer] at the output of each steered decoder block.

    The Assistant Axis paper applies its intervention at MULTIPLE ADJACENT layers
    and reports that single-layer steering "had no useful effect", so single-layer
    is not a fair rendering of the method.
    """

    def __init__(self, n_personas: int, hidden: int, layers, device, dtype):
        self.layers = list(layers)
        self.v = torch.zeros(n_personas, len(self.layers), hidden,
                             device=device, dtype=torch.float32)
        self.alpha = torch.nn.Parameter(
            torch.ones(n_personas, device=device, dtype=torch.float32))
        self.index = 0
        self.enabled = False
        self.dtype = dtype

    def make_hook(self, slot: int):
        def hook(module, args, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            delta = (self.alpha[self.index] * self.v[self.index, slot]).to(hs.dtype)
            hs = hs + delta
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook


@torch.no_grad()
def estimate_vectors(model, tokenizer, personas, prefixes, empty_cache, empty_len,
                     train_by, device, steer, n_batches=16, batch_size=2,
                     max_len=384, seed=0):
    """v_p[l] = mean over response tokens of (teacher - student) hidden at each layer."""
    rng = random.Random(seed)
    for pi, role in enumerate(personas):
        acc, n = None, 0
        items = train_by[role]
        for _ in range(n_batches):
            chunk = rng.sample(items, min(batch_size, len(items)))
            ids, mask, resp, _ = collate(tokenizer, chunk, prefixes[role]["text"],
                                         device, max_len)
            if resp.sum() == 0:
                continue
            t = teacher_forward(model, prefixes[role]["cache"], prefixes[role]["len"],
                                ids, mask, output_hidden_states=True)
            s = teacher_forward(model, empty_cache, empty_len, ids, mask,
                                output_hidden_states=True)
            sel = resp.reshape(-1)
            per = []
            for L in steer.layers:
                d = t.hidden_states[L].shape[-1]
                diff = (t.hidden_states[L].reshape(-1, d)[sel].float()
                        - s.hidden_states[L].reshape(-1, d)[sel].float())
                per.append(diff.sum(0))
                nn_ = diff.shape[0]
            stacked = torch.stack(per)
            acc = stacked if acc is None else acc + stacked
            n += nn_
        steer.v[pi] = acc / max(n, 1)
        norms = [f"{x:.2f}" for x in steer.v[pi].norm(dim=-1).tolist()]
        print(f"  v[{role}] |v| per layer={norms} from {n} tokens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden_states indices to steer (multiple = the paper's protocol)")
    ap.add_argument("--steps", type=int, default=300, help="steps to fit alpha")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n_heldout", type=int, default=48)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
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
    for p in model.parameters():
        p.requires_grad_(False)

    personas = json.loads(Path(args.personas).read_text())
    n_layers = model.config.num_hidden_layers
    layers = args.layers
    if not layers:
        mid = round(0.6 * n_layers)
        layers = list(range(max(1, mid - 3), min(n_layers, mid + 5)))
    steer = Steerer(len(personas), model.config.hidden_size, layers, device, dtype)
    # hidden_states[i] is block i-1's output, so hook block i-1.
    for slot, L in enumerate(layers):
        model.model.layers[max(0, L - 1)].register_forward_hook(steer.make_hook(slot))
    print(f"steering at hidden_states{layers} of {n_layers} blocks", flush=True)

    rows = load_targets(args.targets)
    held_ids = set(sorted({r["question_idx"] for r in rows})[:args.n_heldout])
    train_by, eval_by = {}, {}
    for role in personas:
        rs = [r for r in rows if r["persona"] == role and r["response"]]
        train_by[role] = [r for r in rs if r["question_idx"] not in held_ids]
        eval_by[role] = [r for r in rs if r["question_idx"] in held_ids]

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

    print("estimating persona vectors ...", flush=True)
    estimate_vectors(model, tokenizer, personas, prefixes, empty_cache, empty_len,
                     train_by, device, steer, batch_size=args.batch_size,
                     max_len=args.max_len, seed=args.seed)

    opt = torch.optim.Adam([steer.alpha], lr=args.lr)
    rng = random.Random(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        pi = rng.randrange(len(personas))
        role = personas[pi]
        items = rng.sample(train_by[role], min(args.batch_size, len(train_by[role])))
        ids, mask, resp, _ = collate(tokenizer, items, prefixes[role]["text"],
                                     device, args.max_len)
        if resp.sum() == 0:
            continue
        steer.enabled = False
        with torch.no_grad():
            t_out = teacher_forward(model, prefixes[role]["cache"],
                                    prefixes[role]["len"], ids, mask)
        steer.enabled, steer.index = True, pi
        s_out = forward_with_prefix(model, empty_cache, empty_len, ids, mask)
        loss, kl, _, _ = kl_and_act(t_out, s_out, resp, 0.0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step:4d} kl={kl.item():.4f} alpha={steer.alpha.tolist()}",
                  flush=True)

    ev = {}
    with torch.no_grad():
        for pi, role in enumerate(personas):
            kls = []
            for b in range(0, min(len(eval_by[role]), 4 * args.batch_size),
                           args.batch_size):
                chunk = eval_by[role][b:b + args.batch_size]
                ids, mask, resp, _ = collate(tokenizer, chunk, prefixes[role]["text"],
                                             device, args.max_len)
                if resp.sum() == 0:
                    continue
                steer.enabled = False
                t_out = teacher_forward(model, prefixes[role]["cache"],
                                        prefixes[role]["len"], ids, mask)
                steer.enabled, steer.index = True, pi
                s_out = teacher_forward(model, empty_cache, empty_len, ids, mask)
                _, kl, _, _ = kl_and_act(t_out, s_out, resp, 0.0)
                kls.append(kl.item())
            ev[role] = {"kl": sum(kls) / max(len(kls), 1)}

    result = {
        "method": "steering_vector", "model": args.model, "layers": layers,
        "personas": personas, "alpha": steer.alpha.tolist(),
        "v_norms": steer.v.norm(dim=-1).tolist(),
        "trainable_params": len(personas),
        "heldout_kl_mean": sum(v["kl"] for v in ev.values()) / len(ev),
        "heldout": ev, "wall_s": time.time() - t0,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    torch.save(steer.v.cpu(), out_dir / "vectors.pt")
    print(f"DONE steering heldout_kl={result['heldout_kl_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
