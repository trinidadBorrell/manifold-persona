"""Is the prefix-KV mismatch at 3B a precision artefact or a real bug?

Runs the same equivalence check at several dtypes and reports both max|dlogit| and
the KL between the two paths' distributions -- KL is what the training loss actually
consumes, so a large max-logit gap in the tail can still be harmless.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_persona_prompt, load_targets, sys_block
from .teacher import build_prefix_cache, teacher_forward
from .train import collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    personas = json.loads(open(args.personas).read())
    rows = load_targets(args.targets)
    role = personas[0]
    items = [r for r in rows if r["persona"] == role][:args.batch_size]

    for name in args.dtypes:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=fixed[name]).to(device)
        model.config.use_cache = False
        text = load_persona_prompt(role)
        ids, mask, resp, _ = collate(tokenizer, items, text, device, args.max_len)
        pref_ids = tokenizer(sys_block(text), return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)

        with torch.no_grad():
            cache, P = build_prefix_cache(model, pref_ids)
            cached = teacher_forward(model, cache, P, ids, mask).logits
            full_ids = torch.cat([pref_ids.expand(ids.shape[0], -1), ids], dim=1)
            full_mask = torch.cat(
                [torch.ones(ids.shape[0], P, dtype=mask.dtype, device=device), mask],
                dim=1)
            full = model(input_ids=full_ids, attention_mask=full_mask).logits[:, P:, :]

            sel = resp.reshape(-1)
            a = cached.reshape(-1, cached.shape[-1])[sel].float()
            b = full.reshape(-1, full.shape[-1])[sel].float()
            max_d = (a - b).abs().max().item()
            lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
            kl = (lpb.exp() * (lpb - lpa)).sum(-1).mean().item()
            agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        print(f"{name:>5}: max|dlogit|={max_d:.4g}  KL(full||cached)={kl:.3e}  "
              f"argmax agreement={agree:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
