"""One question shard of Giovanni's targets + activations, in one process (one model load).

Same as his gen_targets -> extract_acts for one persona (120 questions, held-out first, 5 phrasings
cycled by question index, 160 new tokens, T 0.3, top-p 0.9, fp16, response-token means at every
layer, max_len 384), but only for questions i::n, so a persona runs as n GPU jobs at once.
The loop body is his `generate_targets` and `extract_acts.main` (his functions are imported from
his exported tree); the one difference is that each shard seeds its own RNG (seed 0 + shard).

    python -m giovanni_repro.shard --root $PM/GIO --who spy --shard 3/10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--who", required=True)
    ap.add_argument("--shard", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    repo = root / "repo"
    sys.path[:0] = [str(repo), str(repo / "src")]
    from giovanni_repro.run import shim
    shim()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from persona_lora.data import encode_pair, gen_prompt_text, load_persona_prompt, question_split
    from persona_lora.extract_acts import pooled

    i, n = (int(x) for x in a.shard.split("/"))
    out = root / a.who / "shards"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16).to(dev).eval()
    train_q, held_q = question_split(48, 0)
    questions = (held_q + train_q)[:120]
    torch.manual_seed(0 + i)
    recs, P = [], []
    for qi in range(i, len(questions), n):
        q = questions[qi]
        ii = qi % 5
        sys_text = load_persona_prompt(a.who, ii)
        ids = tok(gen_prompt_text(sys_text, q), return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=160, do_sample=True, temperature=0.3, top_p=0.9,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        r = {"persona": a.who, "question_idx": qi, "question": q, "instruction_idx": ii, "response": text}
        pre_p, suf, start = encode_pair(tok, sys_text, q, text)
        pt = pooled(model, pre_p + suf, len(pre_p) + start, dev, 384)
        if pt is not None:
            P.append(pt.astype(np.float16))
            recs.append(r)
        print("q%d done" % qi, flush=True)
    with open(out / ("targets_%02d.jsonl" % i), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(out / ("acts_%02d.npz" % i), prompted=np.stack(P),
                        persona=np.array([a.who] * len(P)), question_idx=np.array([str(r["question_idx"]) for r in recs]))
    print("wrote %d" % len(P), flush=True)


if __name__ == "__main__":
    main()
