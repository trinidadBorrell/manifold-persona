"""Diagnostic 01 — activation scales, attention sinks, and what `prompt_avg` measures.

Answers four questions about the residual-stream point cloud, using the real
prompt-building pipeline (``prompts_roles.build_role_records``):

  Q1  Is ``output_hidden_states`` internally consistent? Specifically, is the LAST
      element post-final-norm while the others are raw residual stream? If so,
      sweeping layers 0..L compares one normalized tensor against L unnormalized
      ones.
  Q2  What is the per-layer norm profile? (The residual stream is NOT unit-norm:
      these are pre-norm architectures, so the stream is a running sum and its
      scale grows with depth.)
  Q3  Are there "massive activation" positions/channels? In Qwen/Llama the first
      token acts as an attention sink with a residual norm 100x the median,
      concentrated in one or two channels.
  Q4  Given Q3 — what does ``prompt_avg`` (mean over positions) actually measure,
      and do distance-based intrinsic-dimension estimates survive it?

Q4 is the one that matters. ID estimators (TwoNN, MLE, correlation dimension) are
translation-invariant, so a *constant* offset is harmless; what is NOT harmless is
the offset's variance across records. This script measures that directly by
comparing the pairwise-distance matrix of each view.

Laptop defaults: a 0.5B model, 6 roles, 3 questions, CPU, one layer. Nothing here
needs a GPU.

Usage:
    .venv/bin/python diagnostics/01_activation_scales.py
    .venv/bin/python diagnostics/01_activation_scales.py --model Qwen/Qwen2.5-3B-Instruct
    .venv/bin/python diagnostics/01_activation_scales.py --n_roles 12 --n_questions 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from manifold_persona.prompts_roles import build_role_records, render_prompts, list_roles

SINK_FACTOR = 5.0   # a position is "massive" if |h| > SINK_FACTOR * median|h|


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def q1_hidden_state_consistency(model, enc) -> int:
    rule("Q1  output_hidden_states consistency")
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
        inner = model.model(**enc).last_hidden_state       # inner model applies final norm
    L = model.config.num_hidden_layers
    print(f"num_hidden_layers      = {L}")
    print(f"len(hidden_states)     = {len(hs)}   (expected L+1 = {L + 1}; index 0 = embeddings)")

    post = torch.allclose(hs[-1], inner, atol=1e-5)
    print(f"\nhidden_states[-1] == inner model last_hidden_state (POST final norm)?  {post}")
    if post:
        r = (hs[-1][0].norm(dim=-1).mean() / hs[-2][0].norm(dim=-1).mean()).item()
        print(f"  -> CONFIRMED. hidden_states[{L}] is NORMALIZED; 0..{L - 1} are raw residual.")
        print(f"     Scale jump vs hidden_states[{L - 1}]: {r:.2f}x")
        print("     Consequence: index L is not comparable to the rest. Exclude it from")
        print("     layer sweeps, or normalize the others, but do not mix them.")
    else:
        print("  -> hidden_states[-1] looks like raw residual stream. No asymmetry.")
    return L


def q2_layer_norm_profile(model, enc) -> None:
    rule("Q2  per-layer residual-stream norm profile")
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    print(f"{'layer':>6} {'mean|h|':>11} {'median|h|':>11} {'max|h|':>12} {'max/median':>11}")
    for i, h in enumerate(hs):
        n = h[0].norm(dim=-1)
        med = n.median().item()
        ratio = n.max().item() / med if med > 0 else float("nan")
        print(f"{i:>6} {n.mean():>11.3f} {med:>11.3f} {n.max():>12.3f} {ratio:>11.1f}")
    print("\nIf mean|h| is nowhere near 1.0, the residual stream is not normalized")
    print("(pre-norm architecture: RMSNorm is applied to a copy entering each sublayer).")


def q3_massive_activations(model, tokenizer, enc, layer: int) -> None:
    rule(f"Q3  massive activations at hidden_states[{layer}]")
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    h = hs[layer][0]
    ids = enc["input_ids"][0]
    n = h.norm(dim=-1)
    med = n.median().item()
    massive = (n > SINK_FACTOR * med).nonzero().flatten().tolist()

    print(f"median|h| = {med:.2f}   mean|h| = {n.mean():.2f}   max|h| = {n.max():.2f}")
    print(f"positions with |h| > {SINK_FACTOR}x median: {massive if massive else 'none'}")
    for p in massive:
        print(f"  pos {p:>3}  |h| = {n[p]:>10.2f}  ({n[p] / med:>6.1f}x median)  "
              f"token = {tokenizer.decode([ids[p]])!r}")

    per_ch = h.abs().max(0).values
    top = per_ch.topk(5)
    print(f"\ntop channels by max|value|: {top.indices.tolist()}")
    print(f"  values  = {[round(v, 1) for v in top.values.tolist()]}")
    print(f"  median channel max|value| = {per_ch.median():.3f}")
    if top.values[0] > 20 * per_ch.median():
        print("  -> a small number of channels carry enormous magnitude ('massive activations').")
        print("     Raw variance and Euclidean distance will be dominated by them.")


def collect_views(model, tokenizer, texts, layer):
    """Return (avg, avg_clean, last) each [N, hidden] float64, plus sink stats."""
    avg, clean, last, n_dropped = [], [], [], []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[layer][0].double()
        n = h.norm(dim=-1)
        keep = n <= SINK_FACTOR * n.median()
        if keep.sum() == 0:                     # degenerate guard
            keep = torch.ones_like(keep, dtype=torch.bool)
        avg.append(h.mean(0))
        clean.append(h[keep].mean(0))
        last.append(h[-1])
        n_dropped.append(int((~keep).sum()))
    stack = lambda xs: torch.stack(xs).numpy()
    return stack(avg), stack(clean), stack(last), n_dropped


def describe(name, X):
    Xc = X - X.mean(0, keepdims=True)
    total_var = float((Xc ** 2).sum(1).mean())
    ch_var = Xc.var(0)
    top_ch = int(ch_var.argmax())
    frac_top = float(ch_var[top_ch] / ch_var.sum())
    # PCA spectrum via SVD on the centred matrix
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2 / (s ** 2).sum()
    print(f"  {name:<12} |mean vec| = {np.linalg.norm(X.mean(0)):>10.2f}   "
          f"total var = {total_var:>12.2f}")
    print(f"  {'':<12} top channel = {top_ch:<5} holds {frac_top * 100:>5.1f}% of variance   "
          f"PC1 = {ev[0] * 100:>5.1f}%  PC1-3 = {ev[:3].sum() * 100:>5.1f}%")
    return ev


def pdist(X):
    d = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    iu = np.triu_indices(len(X), k=1)
    return d[iu]


def q4_what_prompt_avg_measures(model, tokenizer, texts, layer, labels) -> None:
    rule(f"Q4  what the views measure at hidden_states[{layer}]  (N = {len(texts)} records)")
    avg, clean, last, dropped = collect_views(model, tokenizer, texts, layer)
    print(f"massive positions dropped per record: min={min(dropped)} max={max(dropped)} "
          f"(of {len(texts)} records)\n")

    cos = (avg * clean).sum(1) / (np.linalg.norm(avg, axis=1) * np.linalg.norm(clean, axis=1))
    print(f"per-record cosine(prompt_avg, prompt_avg_without_massive_positions):")
    print(f"  mean = {cos.mean():.4f}   min = {cos.min():.4f}   max = {cos.max():.4f}")
    if cos.mean() < 0.9:
        print("  -> prompt_avg is dominated by the massive positions, NOT by the content.")

    print("\nvariance structure (after mean-centering, as PCA would see it):")
    for nm, X in (("prompt_avg", avg), ("avg_clean", clean), ("prompt_last", last)):
        describe(nm, X)

    print("\ndistance-matrix agreement (this is what ID estimators consume):")
    d_avg, d_clean, d_last = pdist(avg), pdist(clean), pdist(last)
    for nm, d in (("avg_clean", d_clean), ("prompt_last", d_last)):
        r = float(np.corrcoef(d_avg, d)[0, 1])
        print(f"  pearson r( d(prompt_avg), d({nm}) ) = {r:.4f}")
    print("\n  ID estimators are translation-invariant, so a CONSTANT sink offset is")
    print("  harmless. A low r above means the sink offset VARIES across records and")
    print("  is therefore inside the geometry being measured.")

    uniq = sorted(set(labels))
    if len(uniq) > 1:
        print("\nbetween-role vs within-role spread (higher ratio = role signal is visible):")
        for nm, X in (("prompt_avg", avg), ("avg_clean", clean), ("prompt_last", last)):
            cents = np.stack([X[[i for i, l in enumerate(labels) if l == u]].mean(0) for u in uniq])
            between = float(((cents - cents.mean(0)) ** 2).sum(1).mean())
            within = float(np.mean([
                ((X[[i for i, l in enumerate(labels) if l == u]]
                  - X[[i for i, l in enumerate(labels) if l == u]].mean(0)) ** 2).sum(1).mean()
                for u in uniq]))
            print(f"  {nm:<12} between = {between:>12.2f}  within = {within:>12.2f}  "
                  f"ratio = {between / within if within else float('nan'):>7.3f}")


def q5_sink_is_one_over_T(model, tokenizer, texts, layer) -> None:
    """Regression test: PC1 of the raw mean is 1/seq_len; the fix must remove it."""
    rule(f"Q5  is PC1 of prompt_avg just 1/sequence_length?  (hidden_states[{layer}])")
    sinks, raw, clean, T = [], [], [], []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[layer][0].double()
        n = h.norm(dim=-1)
        keep = n <= SINK_FACTOR * n.median()
        sinks.append(h[0].numpy())
        raw.append(h.mean(0).numpy())
        clean.append(h[keep].mean(0).numpy())
        T.append(h.shape[0])
    S, T = np.stack(sinks), np.array(T, float)

    dev = float(np.abs(S - S[0]).max())
    print(f"sink vector (position 0) identical across records? max deviation = {dev:.3e}")
    print(f"  |h_sink| = {np.linalg.norm(S[0]):.2f}   T: min={T.min():.0f} max={T.max():.0f}")
    if dev == 0.0:
        print("  -> exactly constant, as causal attention requires: position 0 sees only itself.")

    for nm, X in (("prompt_avg  (raw)", np.stack(raw)), ("prompt_avg  (sink-excluded)", np.stack(clean))):
        Xc = X - X.mean(0)
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        r = abs(float(np.corrcoef(pc1, 1.0 / T)[0, 1]))
        print(f"\n{nm}")
        print(f"  PC1 explains {s[0] ** 2 / (s ** 2).sum() * 100:5.1f}% of variance")
        print(f"  |r(PC1, 1/T)| = {r:.4f}   dominant channel = {int(np.abs(vt[0]).argmax())}")
        print(f"  -> {'ARTIFACT: PC1 is sequence length.' if r > 0.9 else 'OK: PC1 is not sequence length.'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="small by default; pass Qwen/Qwen2.5-3B-Instruct for the real one")
    ap.add_argument("--n_roles", type=int, default=6)
    ap.add_argument("--n_questions", type=int, default=3)
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden_states index; default = middle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading {args.model} on CPU (float32) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    L = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else L // 2
    print(f"model has {L} blocks, hidden={model.config.hidden_size}; probing hidden_states[{layer}]")

    roles = list_roles()[: args.n_roles]
    records = build_role_records(n_questions=args.n_questions, seed=args.seed, roles=roles)
    records = render_prompts(records, tokenizer)
    texts = [r.text for r in records]
    labels = [r.role for r in records]
    print(f"{len(records)} records over {len(roles)} roles: {roles}")

    enc = tokenizer(texts[0], return_tensors="pt", add_special_tokens=False)

    q1_hidden_state_consistency(model, enc)
    q2_layer_norm_profile(model, enc)
    q3_massive_activations(model, tokenizer, enc, layer)
    q4_what_prompt_avg_measures(model, tokenizer, texts, layer, labels)
    q5_sink_is_one_over_T(model, tokenizer, texts, layer)

    rule("done")


if __name__ == "__main__":
    main()
