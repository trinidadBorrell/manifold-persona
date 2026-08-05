"""Experiment 1 — how far is the persona shift from a fixed vector?

For each question q, run it WITH and WITHOUT the persona system prompt and take
the residual-stream difference at the SHARED question-token positions:

    dh_l(q, t)  =  h_l^sys(q, t)  -  h_l^no-sys(q, t)

Positions align because the user turn is byte-identical in both runs, so content
largely cancels and what is left is the persona's effect. Stack over (q, t) and
look at the singular spectrum:

  * approximately rank 1  -> dh(x) ~ lambda(x) * v : one fixed direction with a
    variable gain. A persona IS a direction; constant-alpha steering is wrong only
    in the gain, and the fix is a gate.
  * high rank             -> the direction itself moves with the input; the
    gated-constant picture is dead.

The reported number is the share of variance in the top singular direction. That
is a direct answer to "how much is the actual difference" between the estimated
f_hat (a constant vector) and the true f (an input-dependent function).

Also reported, because they separate the two ways a shift can be non-constant:
  * gain CV      -- spread of ||dh|| across inputs (variable magnitude)
  * direction    -- mean cosine of each dh to the top singular vector (variable
                    direction). Rank-1 means high cosine AND variable norm.

No generation, no training. CPU-friendly: 2 forward passes per (persona, question).

Usage:
    .venv/bin/python diagnostics/03_experiment1_shift_rank.py
    .venv/bin/python diagnostics/03_experiment1_shift_rank.py --n_roles 12 --n_questions 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from manifold_persona.prompts_roles import list_roles, load_questions
import json as _json
from manifold_persona.config import ROLE_INSTRUCTIONS_DIR


NEUTRAL = ("You are a helpful assistant that answers questions clearly, carefully "
           "and in plain language, using ordinary words and complete sentences.")


def render_ids(tokenizer, system: str | None, question: str):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": question}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(txt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]


def neutral_of_length(tokenizer, n_tokens: int, offset: int = 0) -> torch.Tensor:
    """Persona-free filler occupying exactly `n_tokens` tokens.

    `offset` picks a different starting point in the filler, so two controls of the
    same length differ in content -- that is the null condition.
    """
    ids = tokenizer(NEUTRAL, add_special_tokens=False)["input_ids"]
    tiled = (ids * (n_tokens // len(ids) + 2))
    return torch.tensor(tiled[offset:offset + n_tokens])


def build_matched_pair(tokenizer, system_a: str, n_ctrl_offset: int, question: str):
    """Return (ids_a, ids_b, n_shared) where ids_b replaces the system CONTENT with
    persona-free filler of identical token length.

    Because the two system blocks have the same length, every question token sits at
    the same absolute position in both sequences, so RoPE is identical and the
    difference isolates system-prompt CONTENT rather than prefix length/position.
    """
    ids_a = render_ids(tokenizer, system_a, question)
    sys_ids = tokenizer(system_a, add_special_tokens=False)["input_ids"]
    n = len(sys_ids)
    # locate the system content block inside the rendered sequence
    a = ids_a.tolist()
    start = next((i for i in range(len(a) - n + 1) if a[i:i + n] == sys_ids), None)
    if start is None:
        return None, None, 0
    ids_b = ids_a.clone()
    ids_b[start:start + n] = neutral_of_length(tokenizer, n, n_ctrl_offset)
    n_shared = len(a) - (start + n)          # everything after the system block
    return ids_a, ids_b, n_shared


@torch.no_grad()
def shift_matrix(model, tokenizer, system, questions, layer, ctrl_offset=0):
    """Rows = per-(question, position) residual-stream differences at `layer`,
    persona system prompt vs a length-matched persona-free control."""
    rows = []
    for q in questions:
        ia, ib, n = build_matched_pair(tokenizer, system, ctrl_offset, q)
        if ia is None or n < 2:
            continue
        ha = model(ia[None, :], output_hidden_states=True).hidden_states[layer][0, -n:]
        hb = model(ib[None, :], output_hidden_states=True).hidden_states[layer][0, -n:]
        rows.append((ha - hb).float().numpy())
    return np.concatenate(rows, 0) if rows else np.empty((0, model.config.hidden_size))


@torch.no_grad()
def null_matrix(model, tokenizer, n_sys_tokens, questions, layer):
    """NULL condition: two persona-free controls of the SAME length. Whatever rank
    structure this shows is not persona -- it is the floor."""
    rows = []
    for q in questions:
        ids = render_ids(tokenizer, NEUTRAL, q)
        probe = tokenizer(NEUTRAL, add_special_tokens=False)["input_ids"]
        a = ids.tolist()
        start = next((i for i in range(len(a) - len(probe) + 1)
                      if a[i:i + len(probe)] == probe), None)
        if start is None:
            continue
        ia, ib = ids.clone(), ids.clone()
        ia[start:start + len(probe)] = neutral_of_length(tokenizer, len(probe), 0)
        ib[start:start + len(probe)] = neutral_of_length(tokenizer, len(probe), 3)
        n = len(a) - (start + len(probe))
        if n < 2:
            continue
        ha = model(ia[None, :], output_hidden_states=True).hidden_states[layer][0, -n:]
        hb = model(ib[None, :], output_hidden_states=True).hidden_states[layer][0, -n:]
        rows.append((ha - hb).float().numpy())
    return np.concatenate(rows, 0) if rows else np.empty((0, model.config.hidden_size))


def analyse(D):
    """Singular spectrum + the gain/direction split."""
    if len(D) < 3:
        return None
    u, s, vt = np.linalg.svd(D, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    v1 = vt[0]
    norms = np.linalg.norm(D, axis=1)
    cos1 = np.abs(D @ v1) / (norms + 1e-12)
    return {
        "n_rows": int(len(D)),
        "sv1_share": float(ev[0]),
        "sv1_3_share": float(ev[:3].sum()),
        "participation_ratio": float(1.0 / (ev ** 2).sum()),
        "gain_cv": float(norms.std() / norms.mean()),
        "mean_abs_cos_to_v1": float(cos1.mean()),
        "mean_norm": float(norms.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n_roles", type=int, default=8)
    ap.add_argument("--n_questions", type=int, default=10)
    ap.add_argument("--instruction_idx", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None, help="default = middle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/experiment1_shift_rank.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    L = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else L // 2

    qs = [x["question"] for x in load_questions()]
    rng = np.random.default_rng(args.seed)
    questions = [qs[i] for i in sorted(rng.choice(len(qs), args.n_questions, replace=False))]
    roles = [r for r in list_roles() if r != "default"][: args.n_roles]

    print(f"model={args.model}  layer={layer}/{L}  roles={len(roles)}  questions={len(questions)}")
    print(f"\n{'role':<16}{'rows':>6}{'SV1':>9}{'SV1-3':>9}{'PR':>8}{'gainCV':>9}{'cos(v1)':>9}")
    print("-" * 66)

    results = {}
    for role in roles:
        instr = _json.load(open(ROLE_INSTRUCTIONS_DIR / f"{role}.json"))["instruction"]
        system = instr[min(args.instruction_idx, len(instr) - 1)].get("pos") or None
        if system is None:
            continue
        r = analyse(shift_matrix(model, tokenizer, system, questions, layer))
        if r is None:
            continue
        results[role] = r
        print(f"{role:<16}{r['n_rows']:>6}{r['sv1_share']:>9.3f}{r['sv1_3_share']:>9.3f}"
              f"{r['participation_ratio']:>8.2f}{r['gain_cv']:>9.3f}{r['mean_abs_cos_to_v1']:>9.3f}")

    null = analyse(null_matrix(model, tokenizer, None, questions, layer))
    if null:
        print("-" * 66)
        print(f"{'NULL (no persona)':<16}{null['n_rows']:>6}{null['sv1_share']:>9.3f}"
              f"{null['sv1_3_share']:>9.3f}{null['participation_ratio']:>8.2f}"
              f"{null['gain_cv']:>9.3f}{null['mean_abs_cos_to_v1']:>9.3f}")

    if results:
        agg = {k: float(np.mean([v[k] for v in results.values()]))
               for k in ("sv1_share", "sv1_3_share", "participation_ratio",
                         "gain_cv", "mean_abs_cos_to_v1")}
        print("-" * 66)
        print(f"{'MEAN':<16}{'':>6}{agg['sv1_share']:>9.3f}{agg['sv1_3_share']:>9.3f}"
              f"{agg['participation_ratio']:>8.2f}{agg['gain_cv']:>9.3f}"
              f"{agg['mean_abs_cos_to_v1']:>9.3f}")
        print(f"\nSV1     = share of variance in the top singular direction "
              f"(1.0 => exactly rank 1)")
        print(f"PR      = participation ratio of the spectrum "
              f"(effective number of directions)")
        print(f"gainCV  = sd/mean of ||dh|| across inputs (variable magnitude)")
        print(f"cos(v1) = mean |cos| of each shift to v1 (1.0 => fixed direction)")
        if null:
            print(f"\nNULL is the floor: two persona-free controls of equal length. "
                  f"Read persona\nnumbers only RELATIVE to it "
                  f"(null SV1={null['sv1_share']:.3f}, "
                  f"null ||dh||={null['mean_norm']:.2f}, "
                  f"persona ||dh||={float(np.mean([v['mean_norm'] for v in results.values()])):.2f}).")
        verdict = ("GATED-CONSTANT SUPPORTED: shift is ~one direction with variable gain"
                   if agg["sv1_share"] > 0.7 else
                   "GATED-CONSTANT REJECTED: the direction itself moves with the input")
        print(f"\n=> {verdict}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"_meta": {"model": args.model, "layer": layer,
                             "n_questions": len(questions)},
                   "aggregate": agg, "null": null, "per_role": results},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
