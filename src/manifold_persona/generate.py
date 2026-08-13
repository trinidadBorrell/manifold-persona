"""Response generation + response-token activation extraction (assistant-axis match).

``extract.py`` reads the residual stream over *prompt* tokens, with no
generation. This module instead makes the study faithful to the Assistant Axis
paper (``assistant-axis/pipeline/2_activations.py``). For each ``system(role) +
user question`` chat we **generate a response**. We then read the residual
stream **averaged over the assistant-response tokens** (``resp_avg``) plus the
final response token (``resp_last``), for every hidden state. Downstream
analysis then uses the ~0.5-depth layer, matching the paper's layer choice.

Saved in the exact same on-disk layout as ``extract.py``: the npy views are
named ``prompt_avg``/``prompt_last``, so the whole exploratory stack loads it
unchanged. Here those arrays hold the *response*-token means, and the manifest
records ``token_basis="response"``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_and_extract(
    model,
    tokenizer,
    chats: List[List[dict]],
    device: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    batch_log_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Greedy-generate a response per chat, return response-token activations.

    ``chats`` is a list of message lists, e.g. ``[{"role":"system",...},
    {"role":"user",...}]``. For each we:
      1. render with ``add_generation_prompt=True`` -> prompt of length P,
      2. ``model.generate`` -> full ids; response span = ids[P:],
      3. one forward pass over the full ids with ``output_hidden_states=True``,
         take mean over the response span (resp_avg) and its last token (resp_last).

    Returns (resp_avg [N, n_layers, hidden] fp16, resp_last [same], responses).
    Greedy (``do_sample=False``) by default so a given (role, instruction,
    question) is deterministic and the run is reproducible.
    """
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    n = len(chats)
    avg = np.zeros((n, n_layers, hidden), dtype=np.float16)
    last = np.zeros((n, n_layers, hidden), dtype=np.float16)
    responses: List[str] = []

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    for i, messages in enumerate(tqdm(chats, desc="Generate+extract")):
        # transformers >=5 returns a BatchEncoding here; <5 returned a bare
        # tensor. Accept both rather than pinning the version.
        tpl = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        prompt_ids = (tpl if torch.is_tensor(tpl) else tpl["input_ids"]).to(device)
        P = prompt_ids.shape[1]

        out_ids = model.generate(prompt_ids, **gen_kwargs)          # [1, P+R]
        resp_ids = out_ids[0, P:]
        responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))

        if resp_ids.numel() == 0:                                   # empty gen -> fall back to last prompt token
            outputs = model(prompt_ids, output_hidden_states=True)
            for l, hs in enumerate(outputs.hidden_states):
                v = hs[0, -1].float().cpu().numpy().astype(np.float16)
                avg[i, l] = v
                last[i, l] = v
            del outputs
            continue

        outputs = model(out_ids, output_hidden_states=True)
        for l, hs in enumerate(outputs.hidden_states):              # hs: [1, P+R, hidden]
            span = hs[0, P:].float()                                # [R, hidden] response tokens
            avg[i, l] = span.mean(dim=0).cpu().numpy().astype(np.float16)
            last[i, l] = span[-1].cpu().numpy().astype(np.float16)
        del outputs

        if device == "mps" and (i + 1) % batch_log_every == 0:
            torch.mps.empty_cache()

    return avg, last, responses


@torch.no_grad()
def generate_and_extract_batched(
    model,
    tokenizer,
    chats: List[List[dict]],
    device: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Batched twin of :func:`generate_and_extract`. Same outputs, ~10x faster.

    Generation batches with LEFT padding + attention mask (decoding must
    continue from real tokens). Extraction then rebuilds each sample's real
    (prompt + response) sequence and batches those with RIGHT padding: under
    causal attention a real token never attends a later pad, so its hidden
    states are exactly the unpadded ones.

    Batched greedy decoding can differ from unbatched at logit near-ties
    (batched matmul numerics). The pilot's 3-level check measures whether
    that matters before any full run trusts this path.
    """
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    n = len(chats)
    avg = np.zeros((n, n_layers, hidden), dtype=np.float16)
    last = np.zeros((n, n_layers, hidden), dtype=np.float16)
    responses: List[str] = [""] * n

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=pad_id)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    def _ids(messages):
        tpl = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True)
        return (tpl if torch.is_tensor(tpl) else tpl["input_ids"])[0]

    for s in tqdm(range(0, n, batch_size), desc="Generate+extract (batched)"):
        idx = list(range(s, min(s + batch_size, n)))
        prompts = [_ids(chats[i]) for i in idx]
        plens = [p.shape[0] for p in prompts]
        P = max(plens)

        # ---- generation: left-pad so decoding continues from real tokens
        inp = torch.full((len(idx), P), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), P), dtype=torch.long)
        for j, p in enumerate(prompts):
            inp[j, P - plens[j]:] = p
            mask[j, P - plens[j]:] = 1
        out = model.generate(inp.to(device), attention_mask=mask.to(device),
                             **gen_kwargs)                    # [B, P+R]

        # ---- per-sample real sequences: prompt + response up to eos incl.
        seqs, rlens = [], []
        for j, i in enumerate(idx):
            gen = out[j, P:]
            eos_pos = (gen == eos_id).nonzero()
            r = int(eos_pos[0]) + 1 if eos_pos.numel() else gen.shape[0]
            resp = gen[:r]
            responses[i] = tokenizer.decode(resp, skip_special_tokens=True)
            seqs.append(torch.cat([prompts[j].to(device), resp]))
            rlens.append(r)

        # ---- extraction: right-pad; causal attention keeps real tokens exact
        S = max(q.shape[0] for q in seqs)
        full = torch.full((len(idx), S), pad_id, dtype=torch.long, device=device)
        fmask = torch.zeros((len(idx), S), dtype=torch.long, device=device)
        for j, q in enumerate(seqs):
            full[j, :q.shape[0]] = q
            fmask[j, :q.shape[0]] = 1
        hs_all = model(full, attention_mask=fmask,
                       output_hidden_states=True).hidden_states
        for l, hs in enumerate(hs_all):
            for j, i in enumerate(idx):
                pl, r = plens[j], rlens[j]
                if r == 0:                       # empty gen -> last prompt token
                    v = hs[j, pl - 1].float().cpu().numpy().astype(np.float16)
                    avg[i, l] = v
                    last[i, l] = v
                else:
                    span = hs[j, pl:pl + r].float()
                    avg[i, l] = span.mean(dim=0).cpu().numpy().astype(np.float16)
                    last[i, l] = span[-1].cpu().numpy().astype(np.float16)
        del hs_all, out

    return avg, last, responses
