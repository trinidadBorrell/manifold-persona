"""Where did the steered responses actually LAND? Teacher-force them back.

`run_steering.py` saves text, not activations — the steering hook perturbs the
residual stream but nothing records the result. So a PCA of "where steering took
the model" cannot be drawn from the run's own output.

This replays it: each generated response is fed back through the model together
with its prompt, in ONE forward pass with no steering hook, and the mean
residual-stream activation over the RESPONSE tokens is recorded — the same
quantity, layer and token basis the role cloud is built from. The result is
directly comparable to the role centroids, so trajectories and cloud live in one
PCA.

WHY TEACHER-FORCING IS THE RIGHT REPLAY. We want the position of the TEXT the
steered model produced, expressed in the unsteered model's coordinates — that is
what makes it comparable to the role cloud, which was also built without
steering. Re-running with the hook on would instead measure the perturbed
stream, which is a different question (and is what `delta_stats` already logs).

Forward-only, so this is far cheaper than the generation it replays.

    python -m steering.capture_activations --judged judged_L19.parquet \\
        --out acts.npy --model Qwen/Qwen3-8B --layer 19
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True, help="writes <out>.npy and <out>.meta.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=19, help="hidden_states index")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--roles", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering.run_steering import render_chat_prompt

    df = pd.read_parquet(args.judged).reset_index(drop=True)
    if args.roles:
        keep = {r.strip() for r in args.roles.split(",")}
        df = df[df.role.isin(keep)].reset_index(drop=True)
    if args.num_shards > 1:
        df = df.iloc[args.shard_index::args.num_shards].reset_index(drop=True)
    print("replaying %d responses" % len(df), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    blocks = model.model.layers if hasattr(model, "model") else model.layers
    block = blocks[args.layer - 1]          # hidden_states[i] = output of block i-1
    cap = {}

    def hook(m, i, o):
        cap["h"] = (o[0] if isinstance(o, (tuple, list)) else o).detach()

    handle = block.register_forward_hook(hook)
    in_dev = next(model.parameters()).device
    hidden = model.config.hidden_size
    out = np.zeros((len(df), hidden), dtype=np.float16)
    n_empty = 0

    with torch.no_grad():
        for s0 in range(0, len(df), args.batch_size):
            sub = df.iloc[s0:s0 + args.batch_size]
            texts, plens = [], []
            for _, r in sub.iterrows():
                msgs = ([{"role": "system", "content": r.system}] if r.system else []) + \
                       [{"role": "user", "content": r.question}]
                # Thinking mode OFF, as in generation: the replay must see the
                # same prompt tokens the steered run did (run_steering.py).
                prompt = render_chat_prompt(tok, msgs)
                texts.append(prompt + str(r.response))
                plens.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))

            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len, add_special_tokens=False).to(in_dev)
            model(**enc)
            h = cap["h"]
            attn = enc["attention_mask"].to(h.device)

            for bi, plen in enumerate(plens):
                # Left padding pushes real tokens right; the response starts after
                # (pads + prompt), never at plen itself.
                start = int((attn[bi] == 0).sum()) + plen
                if start >= h.shape[1]:
                    n_empty += 1
                    continue
                m = attn[bi, start:] == 1
                if not bool(m.any()):
                    n_empty += 1
                    continue
                out[s0 + bi] = h[bi, start:][m].float().mean(0).cpu().numpy().astype(np.float16)

            if (s0 // args.batch_size) % 50 == 0:
                print("  %d/%d" % (min(s0 + args.batch_size, len(df)), len(df)), flush=True)

    handle.remove()
    op = Path(args.out)
    np.save(op.with_suffix(".npy"), out)
    # target_role must survive into the meta: the journey and ablation arms run
    # two destinations per role, and a figure that cannot tell them apart would
    # average two different destinations into one point that means nothing.
    cols = [c for c in ["arm", "alpha", "role", "target_role", "system_idx",
                        "question_idx", "judge_score", "persona", "n_new_tokens"]
            if c in df.columns]
    df[cols].to_parquet(op.with_suffix(".meta.parquet"), index=False)
    print("wrote %s (%s) | %d rows with no response tokens"
          % (op.with_suffix(".npy"), out.shape, n_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
