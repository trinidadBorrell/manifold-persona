"""Activation extraction for prompt-only records.

Adapted from ``persona_vectors/generate_vec.py::get_hidden_p_and_r``. That
function splits a prompt+response into prompt/response spans; here every record
is prompt-only, so we take:
- ``prompt_avg``  : mean over tokens of each layer's hidden state.
- ``prompt_last`` : the final token's hidden state per layer.
for all ``num_hidden_layers + 1`` hidden states (index 0 = embeddings).

ATTENTION SINKS -- why ``prompt_avg`` excludes some positions
-------------------------------------------------------------
Qwen/Llama-family models place a *massive activation* on the first token: its
residual-stream norm is ~100x the median and is concentrated in one or two
channels (Qwen2.5-0.5B: position 0 = ``<|im_start|>``, channel 62, |h| ~ 1728
vs median ~15). Under causal attention position 0 attends only to itself, so
this vector is **bit-identical for every record** that starts with the same
token -- verified, max deviation 0.0.

A constant vector sounds harmless, but the mean divides by the sequence length:

    prompt_avg  =  (1/T) * h_sink  +  (1/T) * sum(content positions)

so the sink term varies as ``1/T`` and nothing else. Measured on 90 records of
the real role pipeline at layer 12, this single artifact accounted for **78% of
the variance** of ``prompt_avg``, and PC1 correlated with ``1/T`` at
**r = 0.9998**. Because system-prompt length differs by role, it even produces a
*fake* between-role signal. See ``diagnostics/01_activation_scales.py``.

We therefore drop positions whose norm exceeds ``sink_factor`` x the per-record
median norm before averaging. Pass ``sink_factor=None`` to reproduce the old
(artifact-carrying) behaviour. ``prompt_last`` is unaffected -- the final
position is never a sink.

NOTE on layer indexing: ``output_hidden_states`` returns ``L+1`` tensors, and
HF applies the model's final norm to the last one. So ``hidden_states[L]`` is
NORMALIZED while ``0..L-1`` are the raw residual stream (measured scale jump
~4x). Do not mix index ``L`` into a layer sweep.
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
    # torch_dtype: works on transformers 4.4x (native) and 4.5x+ (alias).
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
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

    ``prompt_avg`` and ``prompt_last`` each have shape [N, n_layers, hidden] fp16,
    where n_layers = model.config.num_hidden_layers + 1 (includes the embedding
    layer at index 0). ``n_dropped[i]`` is how many attention-sink positions were
    excluded from record ``i``'s mean, at the layer where most were found.

    ``sink_factor``: drop positions whose L2 norm exceeds this multiple of the
    record's median norm, per layer, before averaging. See the module docstring
    for why this matters. ``None`` disables it (old behaviour).
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
                if not bool(keep.any()):             # degenerate: keep everything
                    keep = torch.ones_like(keep)
                dropped_here = max(dropped_here, int((~keep).sum()))
                pooled = hs[keep].mean(dim=0)
            avg[i, l] = pooled.cpu().numpy().astype(np.float16)
            last[i, l] = hs[-1].cpu().numpy().astype(np.float16)
        n_dropped.append(dropped_here)
        del outputs
    return avg, last, n_dropped
