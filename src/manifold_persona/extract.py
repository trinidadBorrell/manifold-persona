"""Activation extraction for prompt-only records.

Adapted from ``persona_vectors/generate_vec.py::get_hidden_p_and_r``. That
function splits a prompt+response into prompt/response spans; here every record
is prompt-only, so we take:
- ``prompt_avg``  : mean over tokens of each layer's hidden state.
- ``prompt_last`` : the final token's hidden state per layer.
for all ``num_hidden_layers + 1`` hidden states (index 0 = embeddings).

``prompt_avg`` excludes attention-sink positions; see ``sink_factor``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str = None):
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # float16 on MPS/CUDA for speed+memory; float32 on CPU for stability.
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


DEFAULT_SINK_FACTOR = 5.0


@torch.no_grad()
def extract_prompt_activations(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    sink_factor: float = DEFAULT_SINK_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (prompt_avg, prompt_last, n_dropped).

    prompt_avg and prompt_last each have shape [N, n_layers, hidden] fp16, with
    n_layers = model.config.num_hidden_layers + 1. ``n_dropped[i]`` is the number
    of positions excluded from record ``i``'s mean.

    ``sink_factor``: drop positions whose L2 norm exceeds this multiple of the
    record's per-layer median norm before averaging. ``None`` disables it.
    """
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    n = len(texts)
    avg = np.zeros((n, n_layers, hidden), dtype=np.float16)
    last = np.zeros((n, n_layers, hidden), dtype=np.float16)
    n_dropped: List[int] = []

    for i, text in enumerate(tqdm(texts, desc="Extracting activations")):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        dropped_here = 0
        # hidden_states: tuple(n_layers) each [1, seq, hidden]
        for l, hs in enumerate(outputs.hidden_states):
            hs = hs[0].float()                       # [seq, hidden]
            if sink_factor is None:
                pooled = hs.mean(dim=0)
            else:
                norms = hs.norm(dim=-1)
                keep = norms <= sink_factor * norms.median()
                if not bool(keep.any()):
                    keep = torch.ones_like(keep)
                dropped_here = max(dropped_here, int((~keep).sum()))
                pooled = hs[keep].mean(dim=0)
            avg[i, l] = pooled.cpu().numpy().astype(np.float16)
            last[i, l] = hs[-1].cpu().numpy().astype(np.float16)
        n_dropped.append(dropped_here)
        del outputs
    return avg, last, n_dropped
