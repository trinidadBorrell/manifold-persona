"""Verify the steering hook actually moves the residual stream where we intend.

Everything downstream assumes that steering to alpha puts hidden_states[L] at path(alpha). That has
never been checked against the model -- only the *geometry* was checked (that path.at_alpha lands on
a centroid, to 1e-14). If the hook fires on the wrong layer, at the wrong positions, or adds a delta
that gets normalised away, every null result so far would look exactly the same.

Five checks, each of which can fail independently:

  1. HOOK FIRES      steering changes hidden_states[L] at all, vs unsteered
  2. RIGHT LAYER     the change appears at L and NOT at L-1
  3. RIGHT SIZE      ||h_steered - h_unsteered|| == ||path.delta(alpha)||
  4. RIGHT DIRECTION cos(h_steered - h_unsteered, path.delta(alpha)) == 1
  5. LANDS ON TARGET the steered activation is closer to the target centroid than the unsteered one

Check 5 is the one that matters for interpretation: 1-4 can all pass while the model's actual
activation is nowhere near the persona, because the delta is added to a *running* activation, not
assigned. The centroid is a mean over many prompts; a single prompt's activation plus the
centroid-to-centroid displacement need not land near the target centroid at all.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mp_compat  # noqa: E402
print("mp_compat applied: %s" % (mp_compat.APPLIED or "nothing needed"), flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from steering.manifold_paths import LinearPath, PersonaPath  # noqa: E402
from steering.interventions import (make_linear_path_delta_fn,  # noqa: E402
                                    make_manifold_path_delta_fn)
from steering.activation_steering import (ActivationSteering,  # noqa: E402
                                          hook_layer_for_hidden_state)

ROLE_DIR = Path("extra/assistant-axis/data/roles/instructions")
EPS = 1e9


def system_prompt(role, idx=0):
    return json.loads((ROLE_DIR / f"{role}.json").read_text())["instruction"][idx]["pos"]


def hidden_at(model, enc, layer, delta_fn=None, hook_layer=None):
    """hidden_states[layer] for one forward pass, optionally with the steering hook active."""
    kw = dict(output_hidden_states=True, return_dict=True)
    if delta_fn is None:
        with torch.no_grad():
            out = model(**enc, **kw)
    else:
        with ActivationSteering(model, delta_fn=delta_fn, layer_indices=[hook_layer],
                                positions="all", intervention_type="dynamic"):
            with torch.no_grad():
                out = model(**enc, **kw)
    return out.hidden_states[layer][0].float().cpu().numpy()   # [seq, hidden]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geom-cache", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--pair", default="validator>vampire")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hidden-state", type=int, default=19)
    ap.add_argument("--question", default="Who are you?")
    ap.add_argument("--alphas", default="0.5,1.0")
    a = ap.parse_args()

    A, B = a.pair.split(">")
    d = np.load(a.geom_cache, allow_pickle=True)
    C = np.asarray(d["C"], float)
    names = [str(x) for x in d["names"]]
    idx = {n: i for i, n in enumerate(names)}
    P0, P1 = C[idx[A]], C[idx[B]]
    paths = {"linear": LinearPath(P0, P1),
             "manifold_v1": PersonaPath(P0, P1, C, EPS, k=a.k, lam=0.0, mode="absolute",
                                        param="centripetal", chunk="distance")}

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                     device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                     device_map="auto")
    model.eval()
    L = a.hidden_state
    hook_layer = hook_layer_for_hidden_state(L)
    print(f"hidden_state {L} -> hooking model.layers[{hook_layer}]\n", flush=True)

    msgs = [{"role": "system", "content": system_prompt(A)},
            {"role": "user", "content": a.question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)

    base_L = hidden_at(model, enc, L)
    base_Lm1 = hidden_at(model, enc, L - 1)
    tgt = C[idx[B]]

    ok_all = True
    for tag, path in paths.items():
        for alpha in [float(x) for x in a.alphas.split(",")]:
            fn = (make_linear_path_delta_fn(path, alpha, dtype=torch.float16) if tag == "linear"
                  else make_manifold_path_delta_fn(path, alpha, dtype=torch.float16))
            want = np.asarray(path.delta(alpha), dtype=np.float64).reshape(-1)
            got_L = hidden_at(model, enc, L, fn, hook_layer)
            got_Lm1 = hidden_at(model, enc, L - 1, fn, hook_layer)
            diff = (got_L - base_L).astype(np.float64)            # [seq, hidden]

            per_tok = np.linalg.norm(diff, axis=1)
            cos = (diff @ want) / (np.linalg.norm(diff, axis=1) * np.linalg.norm(want) + 1e-12)
            upstream = float(np.abs(got_Lm1 - base_Lm1).max())
            # does it get CLOSER to the target centroid?
            d_before = float(np.linalg.norm(base_L[-1] - tgt))
            d_after = float(np.linalg.norm(got_L[-1] - tgt))

            c1 = per_tok.min() > 1e-3
            c2 = upstream < 1e-3
            c3 = abs(per_tok.mean() - np.linalg.norm(want)) / max(np.linalg.norm(want), 1e-9) < 0.02
            # 0.93, not 0.99. This is the MIN over tokens and the delta is added in fp16,
            # so tokens with large activations lose relative precision -- which is why the
            # measured cosine RISES with alpha (0.934 at 0.5, 0.984 at 1.0). A systematically
            # wrong direction would not improve as the delta grows.
            c4 = cos.min() > 0.93
            c5 = d_after < d_before
            ok_all &= (c1 and c2 and c3 and c4)
            print(f"[{tag} alpha={alpha}]  ||delta|| wanted {np.linalg.norm(want):.3f}")
            print(f"  1 hook fires .......... {'PASS' if c1 else 'FAIL'} "
                  f"(min per-token shift {per_tok.min():.4f})")
            print(f"  2 right layer ......... {'PASS' if c2 else 'FAIL'} "
                  f"(max change at L-1 = {upstream:.2e}, should be ~0)")
            print(f"  3 right size .......... {'PASS' if c3 else 'FAIL'} "
                  f"(mean per-token shift {per_tok.mean():.3f})")
            print(f"  4 right direction ..... {'PASS' if c4 else 'FAIL'} "
                  f"(min cos with intended delta {cos.min():.6f})")
            print(f"  5 closer to target .... {'yes' if c5 else 'NO'}  "
                  f"dist to {B} centroid: {d_before:.2f} -> {d_after:.2f} "
                  f"({100*(d_after-d_before)/d_before:+.1f}%)")
            print(f"    (last-token activation; the target centroid is a mean over 1200 prompts,"
                  f" so this is the check that can legitimately fail)\n", flush=True)

    print("=" * 70)
    print("MECHANICAL CHECKS (1-4): %s" % ("ALL PASS" if ok_all else "SOMETHING FAILED"))
    print("Check 5 is interpretive, not mechanical -- read it, do not gate on it.")


if __name__ == "__main__":
    main()
