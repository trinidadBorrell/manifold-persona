"""The steering dose unit, measured the paper's way: LMSYS-Chat-1M, per token.

Plan: docs/notes/paper-fidelity-plan.md (WP2).

arXiv:2601.10387 section 3.2.1: steering vectors are "scaled with respect to the
average post-MLP residual stream norm (measured on LMSYS-CHAT-1M) at that
layer". Section 2.1.3 fixes the sample size for its LMSYS analysis at
n = 18,777 Assistant responses, which is what we reuse here.

WHAT WAS WRONG BEFORE. `geometry.load_geometry` computed N_bar as the mean norm
of the resp240 rows. Every such row is already a MEAN OVER RESPONSE TOKENS, and
averaging tokens cancels variation, so ||mean h|| < mean ||h||. The old number
was therefore the wrong quantity (a norm-of-mean, not a mean-of-norm) measured
on the wrong corpus (role role-play, not general chat). Both are fixed here, and
both numbers are recorded so older figures stay interpretable.

METHOD. The LMSYS responses were written by OTHER models, so their text cannot
be replayed as our model's activations. We teacher-force instead: format each
conversation with our model's chat template, run a single forward pass, and
average ||h|| at `hidden_states[LAYER]` over the RESPONSE tokens only - the same
token basis the role cloud uses.

    .venv/bin/python -m steering.lmsys_norm --out <run_dir>

`lmsys/lmsys-chat-1m` is GATED on HuggingFace: accept the terms with the account
whose token is in `token/`, or this fails at load with a 401.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "lmsys/lmsys-chat-1m"
DEFAULT_N = 18777          # section 2.1.3
LAYER_HS_INDEX = 19        # our cloud's layer


def sample_conversations(n: int, seed: int, min_chars: int = 20):
    """`n` (user, assistant) turn pairs, streamed so the 1M rows never land on disk.

    Streaming rather than downloading: we need ~19k rows out of a million, and
    the full dataset is tens of GB. Taking the FIRST assistant turn of each
    conversation keeps one sample per conversation, so the sample is not
    dominated by a handful of very long chats.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    out = []
    for row in ds:
        conv = row.get("conversation") or []
        user = next((m for m in conv if m.get("role") == "user"), None)
        asst = next((m for m in conv if m.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        u, a = (user.get("content") or "").strip(), (asst.get("content") or "").strip()
        if len(u) < min_chars or len(a) < min_chars:
            continue
        out.append({"user": u, "assistant": a})
        if len(out) >= n:
            break
    return out


def measure(model, tokenizer, convs, layer_hs_index: int, device: str,
            max_len: int = 1024, batch_size: int = 8) -> dict:
    """Mean ||h|| over response tokens at `hidden_states[layer_hs_index]`.

    Accumulates a running sum rather than keeping the activations: 19k
    conversations x ~300 tokens x 2048 dims in float32 would be ~45 GB.
    """
    import torch
    from steering.run_steering import render_chat_prompt

    total_norm, total_tokens, n_seqs, n_truncated = 0.0, 0, 0, 0

    # hidden_states[i] is the output of decoder block i-1 (index 0 is the
    # embedding), so the layer that produces hidden_states[layer_hs_index] is
    # block layer_hs_index-1 — the same off-by-one smoke.py asserts for steering.
    blocks = model.model.layers if hasattr(model, "model") else model.layers
    block = blocks[layer_hs_index - 1]
    captured = {}

    def _hook(module, ins, out):
        captured["h"] = (out[0] if isinstance(out, (tuple, list)) else out).detach()

    handle = block.register_forward_hook(_hook)

    # With device_map the model spans cards and `device` says nothing about
    # where the embedding lives; inputs must land on the first parameter's card.
    in_dev = next(model.parameters()).device

    for s0 in range(0, len(convs), batch_size):
        batch = convs[s0:s0 + batch_size]
        texts, prompt_lens = [], []
        for c in batch:
            # Thinking mode OFF (see run_steering.render_chat_prompt): with it
            # on, Qwen3 would see an open <think> block before text it never
            # wrote, and the norm would be measured in the wrong regime.
            prompt = render_chat_prompt(
                tokenizer, [{"role": "user", "content": c["user"]}])
            full = prompt + c["assistant"]
            texts.append(full)
            prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len, add_special_tokens=False).to(in_dev)
        # A FORWARD HOOK, not output_hidden_states=True. The latter keeps every
        # one of the model's L+1 hidden states alive: at batch 16 x 1024 tokens
        # x 4096 dims x 37 layers that is ~5 GB of tensors to obtain ONE of
        # them, and it OOMed a 16 GB card instantly. The hook grabs the layer we
        # want and lets the rest be freed as the forward proceeds.
        with torch.no_grad():
            model(**enc)
        h = captured["h"]                                        # (b, l, hidden)
        norms = h.float().norm(dim=-1)                           # (b, l)

        # With device_map the hooked block can live on a different card than the
        # embeddings, so `norms` and the mask are on different devices and
        # indexing one by the other raises. Move the mask to the norms' card.
        attn = enc["attention_mask"].to(norms.device)
        for bi, plen in enumerate(prompt_lens):
            # Left padding shifts every real token right by the pad count, so the
            # response starts after (pads + prompt), never at `plen` itself.
            n_pad = int((attn[bi] == 0).sum())
            start = n_pad + plen
            if start >= norms.shape[1]:
                n_truncated += 1
                continue
            resp = norms[bi, start:][attn[bi, start:] == 1]
            if resp.numel() == 0:
                n_truncated += 1
                continue
            total_norm += float(resp.sum())
            total_tokens += int(resp.numel())
            n_seqs += 1

        if (s0 // batch_size) % 50 == 0:
            print("  %d/%d conversations, %d response tokens"
                  % (s0 + len(batch), len(convs), total_tokens), flush=True)

    handle.remove()

    if total_tokens == 0:
        raise RuntimeError("no response tokens measured — check the chat template")

    return {
        "n_bar": total_norm / total_tokens,
        "n_conversations_used": n_seqs,
        "n_conversations_requested": len(convs),
        "n_response_tokens": total_tokens,
        "n_dropped_truncated": n_truncated,
        "token_basis": "response tokens, per token (mean of norms)",
    }


def main() -> None:
    from manifold_persona.config import MODEL_NAME

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run dir (writes data/n_bar_lmsys.json)")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    # The cluster's execute nodes are offline (HF_HUB_OFFLINE=1), and this
    # dataset is gated on top of that, so the download cannot happen inside the
    # GPU job. --dump fetches on the login node; --conversations replays that
    # file on the GPU with no network at all.
    ap.add_argument("--dump", default=None,
                    help="fetch only: write the sampled conversations here and exit")
    ap.add_argument("--conversations", default=None,
                    help="read conversations from a --dump file instead of the Hub")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--layer", type=int, default=LAYER_HS_INDEX)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    if args.dump:
        # Login node, network on. No model is loaded here.
        print("sampling %d conversations from %s ..." % (args.n, DATASET))
        convs = sample_conversations(args.n, args.seed)
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(convs).to_parquet(args.dump, index=False)
        print("wrote %s (%d conversations) — now run the GPU job with "
              "--conversations %s" % (args.dump, len(convs), args.dump))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.out)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.conversations:
        convs = pd.read_parquet(args.conversations).to_dict("records")
        print("loaded %d conversations from %s" % (len(convs), args.conversations))
    else:
        print("sampling %d conversations from %s ..." % (args.n, DATASET))
        convs = sample_conversations(args.n, args.seed)
    print("got %d" % len(convs))

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()

    res = measure(model, tokenizer, convs, args.layer, device,
                  max_len=args.max_len, batch_size=args.batch_size)
    res.update({"dataset": DATASET, "model": args.model, "layer_hs_index": args.layer,
                "seed": args.seed, "max_len": args.max_len,
                "conversations_file": args.conversations,
                "paper_ref": "arXiv:2601.10387 section 3.2.1; n from section 2.1.3",
                "supersedes": "resp240 response-averaged N_bar = 48.168887419597056"})

    p = run_dir / "data" / "n_bar_lmsys.json"
    p.write_text(json.dumps(res, indent=2))
    print("N_bar = %.4f  (%d response tokens over %d conversations)\nwrote %s"
          % (res["n_bar"], res["n_response_tokens"], res["n_conversations_used"], p))


if __name__ == "__main__":
    main()
