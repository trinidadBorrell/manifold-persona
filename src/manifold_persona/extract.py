"""Activation extraction for prompt-only records.

Adapted from ``persona_vectors/generate_vec.py::get_hidden_p_and_r``. That
function splits a prompt+response into prompt/response spans; here every record
is prompt-only, so we take:
- ``prompt_avg``  : mean over all tokens of each layer's hidden state.
- ``prompt_last`` : the final token's hidden state per layer.
for all ``num_hidden_layers + 1`` hidden states (index 0 = embeddings).
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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def extract_prompt_activations(
    model,
    tokenizer,
    texts: List[str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (prompt_avg, prompt_last), each shape [N, n_layers, hidden] fp16.

    n_layers = model.config.num_hidden_layers + 1 (includes embedding layer).
    """
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    n = len(texts)
    avg = np.zeros((n, n_layers, hidden), dtype=np.float16)
    last = np.zeros((n, n_layers, hidden), dtype=np.float16)

    for i, text in enumerate(tqdm(texts, desc="Extracting activations")):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(n_layers) each [1, seq, hidden]
        for l, hs in enumerate(outputs.hidden_states):
            hs = hs[0].float()                       # [seq, hidden]
            avg[i, l] = hs.mean(dim=0).cpu().numpy().astype(np.float16)
            last[i, l] = hs[-1].cpu().numpy().astype(np.float16)
        del outputs
    return avg, last
