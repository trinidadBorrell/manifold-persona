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
