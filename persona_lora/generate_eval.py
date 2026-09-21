"""Generate answers to the identity questions under each method, for judging.

Methods:
  prompted   -- persona system prompt, no adapter        (the ceiling)
  base       -- empty system prompt, no adapter          (the floor)
  lora       -- adapter checkpoint, empty system prompt
  steering   -- persona vector added at fixed layers, empty system prompt

Every method answers the SAME five questions (Appendix D.1.2) so the judge sees
one variable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import inject
from .data import gen_prompt_text, load_persona_prompt
from .judge_protocol import IDENTITY_QUESTIONS


def load_adapter(model, ckpt_path, personas, rank, placement, share_b=True):
    sel, n_params = inject(model, rank, len(personas), placement, share_b=share_b)
    state = torch.load(ckpt_path, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    loaded = [k for k in state]
    if not loaded:
        raise RuntimeError(f"no adapter tensors in {ckpt_path}")
    print(f"loaded {len(loaded)} adapter tensors ({n_params:,} params)", flush=True)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--method", required=True,
                    choices=["prompted", "base", "lora", "steering"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="adapter.pt for --method lora")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--placement", default="all")
    ap.add_argument("--per_persona_ckpt_dir", default=None,
                    help="dir of r<rank>_<persona>/adapter.pt for per-persona LoRAs")
    ap.add_argument("--steer_json", default=None,
                    help="result.json from steering.py (uses its v and alpha)")
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--with_prompt", action="store_true",
                    help="keep the persona system prompt in context alongside the "
                         "adapter/steering, matching the Assistant Axis eval setup")
    ap.add_argument("--per_persona_b", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = fixed.get(args.dtype, torch.float32 if device == "cpu" else torch.float32)

    personas = json.loads(Path(args.personas).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    uid = 0

    def gen(model, sel, role, sys_text, pi):
        nonlocal uid
        rows = []
        for q in IDENTITY_QUESTIONS:
            for s in range(args.n_samples):
                text = gen_prompt_text(sys_text, q)
                ids = tokenizer(text, return_tensors="pt",
                                add_special_tokens=False).to(device)
                if sel is not None:
                    sel.enabled, sel.index = True, pi
                o = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                   do_sample=True, temperature=args.temperature,
                                   top_p=0.95, pad_token_id=tokenizer.eos_token_id)
                resp = tokenizer.decode(o[0, ids["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
                uid += 1
                rows.append({"id": uid, "method": args.method, "role": role,
                             "request": q, "sample": s, "response": resp,
                             "with_prompt": bool(args.with_prompt)})
        return rows

    if args.method == "lora" and args.per_persona_ckpt_dir:
        # one adapter per persona: reload the model for each
        for pi, role in enumerate(personas):
            name = (f"r{args.rank}_{args.placement}_{role}"
                    if args.placement.startswith("L")
                    else f"r{args.rank}_{role}")
            ck = Path(args.per_persona_ckpt_dir) / name / "adapter.pt"
            if not ck.exists():
                print(f"  skip {role}: no {ck}", flush=True)
                continue
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
            model.eval()
            sel = load_adapter(model, ck, [role], args.rank, args.placement,
                               share_b=not args.per_persona_b)
            sys_text = load_persona_prompt(role) if args.with_prompt else ""
            records += gen(model, sel, role, sys_text, 0)
            del model
            torch.cuda.empty_cache()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
        model.eval()
        sel = None
        if args.method == "lora":
            sel = load_adapter(model, args.ckpt, personas, args.rank, args.placement,
                               share_b=not args.per_persona_b)
        elif args.method == "steering":
            from .steering import Steerer
            meta = json.loads(Path(args.steer_json).read_text())
            layers = meta["layers"] if "layers" in meta else [meta["layer"]]
            steer = Steerer(len(personas), model.config.hidden_size, layers,
                            device, dtype)
            vecs = torch.load(Path(args.steer_json).parent / "vectors.pt",
                              map_location=device)
            steer.v = vecs.to(device)
            with torch.no_grad():
                steer.alpha.copy_(torch.tensor(meta["alpha"], device=device))
            for slot, L in enumerate(layers):
                model.model.layers[max(0, L - 1)].register_forward_hook(
                    steer.make_hook(slot))
            sel = steer
        for pi, role in enumerate(personas):
            sys_text = (load_persona_prompt(role)
                        if args.method == "prompted" or args.with_prompt else "")
            records += gen(model, sel, role, sys_text, pi)

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} generations -> {out}", flush=True)


if __name__ == "__main__":
    main()
