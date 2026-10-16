"""E1 -- PersonaManifold-style extraction (plan 2026-10-08, Part 1.1; shared with Part 2).

arXiv 2609.34571 s3.1: "residual stream activation h_ij at the last token of the model's
generated response from a single layer"; x_i = mean_j (h_ij - h_0j), h_0j from a neutral
(no-persona) prompt. Our persona prompts are kept: 276 roles x 5 system prompts x 40 questions
(build_role_records(n_questions=40, seed=0)), plus the neutral baseline = the same 40 questions
with NO system prompt.

Answers are generated to EOS (greedy, cap --max-new-tokens), because the stored cloud's answers
stop at 128 tokens (97.6% truncated), so its last token is mid-sentence. Then one teacher-forced
pass over prompt+answer with hooks on hidden_states 12..32 keeps, per answer:
  last : the last answer token before the stop token  (the paper's token)
  mean : the mean over answer tokens                   (our existing view)
fp16 [n, n_layers, 4096] each, plus a metadata csv (text, n_new, hit_cap).

Sharded over units; each shard writes chunk files and resumes from them.

    python -m pm_geometry.extract_states --out DIR --shard 0 --n-shards 8
    python -m pm_geometry.extract_states --out DIR --smoke 48       # throughput + length check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / "jobs_condor"), str(_REPO / "src"), str(_REPO)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL = os.environ.get("PM_MODEL", "Qwen/Qwen3-8B")
MODEL_DISPLAY = MODEL.split("/")[-1]
N_QUESTIONS = 40
LAYERS = tuple(range(*[int(x) for x in os.environ.get("PM_LAYERS", "12,33").split(",")]))          # hidden_states indices; hidden_states[l] = output of layers[l-1]
NEUTRAL = "__neutral__"


def build_units(design="rot40"):
    """Neutral first, then every role in sorted order, prompt-major within a role.

    design full : 5 system prompts x 40 questions per role (55,240 units)
    design rot40: 40 questions per role, question j (in sorted id order) answered under system
                  prompt j mod 5 -- the paper's M = 40 per persona, every prompt used (11,080 units)"""
    from manifold_persona.prompts_roles import build_role_records
    recs = build_role_records(n_questions=N_QUESTIONS, seed=0, model_display=MODEL_DISPLAY)
    qs = sorted({(r.question_idx, r.question) for r in recs})
    if len(qs) != N_QUESTIONS:
        raise ValueError("expected %d questions, got %d" % (N_QUESTIONS, len(qs)))
    units = [dict(role=NEUTRAL, instruction_idx=-1, question_idx=qi, question=q, system=None)
             for qi, q in qs]
    rank = {qi: j for j, (qi, _) in enumerate(qs)}
    if design == "rot40":
        recs = [r for r in recs if r.instruction_idx == rank[r.question_idx] % 5]
    elif design == "q20p3":
        # 20 of the 40 questions x system prompts 0-2 (plan 2026-10-09, D1). The 20 are the
        # questions whose rot40 prompt was 0, 1 or 2 (24 of them), first 20 in id order.
        keep = sorted(q for q, j in rank.items() if j % 5 in (0, 1, 2))[:20]
        recs = [r for r in recs if r.question_idx in keep and r.instruction_idx in (0, 1, 2)]
        units = [u for u in units if u["question_idx"] in keep]
    units += [dict(role=r.role, instruction_idx=r.instruction_idx, question_idx=r.question_idx,
                   question=r.question, system=r.system) for r in recs]
    for i, u in enumerate(units):
        u["unit"] = i
    return units


def load_model():
    import mp_compat  # noqa: F401
    import mp_qwen3  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tok, model


class Reduce:
    """Forward hooks on the blocks behind LAYERS; each reduces its (B, T, D) output on the
    spot to the answer-token mean and the last answer token, so nothing (B, T, D) is kept."""

    def __init__(self, model, layers, resp_mask, last_pos):
        self.model, self.layers = model, layers
        self.resp_mask, self.last_pos = resp_mask, last_pos
        self.mean, self.last, self.handles = {}, {}, []

    def __enter__(self):
        import torch
        blocks = self.model.model.layers
        for L in self.layers:
            def hook(_m, _inp, out, L=L):
                H = out[0] if isinstance(out, tuple) else out
                m = self.resp_mask.to(H.device)
                n = m.sum(1).clamp(min=1).unsqueeze(1).float()
                self.mean[L] = ((H.float() * m.unsqueeze(-1)).sum(1) / n).half().cpu()
                idx = self.last_pos.to(H.device)
                self.last[L] = H[torch.arange(H.shape[0], device=H.device), idx].half().cpu()
            self.handles.append(blocks[L - 1].register_forward_hook(hook))
        return self

    def __exit__(self, *e):
        for h in self.handles:
            h.remove()


def run_batch(tok, model, us, max_new):
    import torch
    texts = []
    for u in us:
        msgs = ([{"role": "system", "content": u["system"]}] if u["system"] else []) + \
               [{"role": "user", "content": u["question"]}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False))
    in_dev = next(model.parameters()).device
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(in_dev)
    P = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    gen = out[:, P:]
    stop = {i for i in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
            if i is not None}
    is_stop = torch.zeros_like(gen, dtype=torch.bool)
    for s in stop:
        is_stop |= gen == s
    resp = torch.cumsum(is_stop.int(), dim=1) == 0                       # before the first stop
    hit_cap = ~is_stop.any(1)
    n_new = resp.sum(1)
    mask = torch.cat([enc["attention_mask"], resp.to(enc["attention_mask"].dtype)], dim=1)
    pos_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
    resp_full = torch.cat([torch.zeros_like(enc["attention_mask"], dtype=torch.bool), resp], dim=1)
    last_pos = (P + n_new - 1).clamp(min=P - 1)                          # empty answer -> last prompt token
    with torch.no_grad(), Reduce(model, LAYERS, resp_full, last_pos) as R:
        model.model(input_ids=out, attention_mask=mask, position_ids=pos_ids, use_cache=False)
    last = torch.stack([R.last[L] for L in LAYERS], 1).numpy()           # (B, n_layers, D)
    mean = torch.stack([R.mean[L] for L in LAYERS], 1).numpy()
    rows = []
    for b, u in enumerate(us):
        text = tok.decode(gen[b][resp[b]], skip_special_tokens=True).strip()
        rows.append(dict(u, response=text, n_new=int(n_new[b]), hit_cap=bool(hit_cap[b])))
    return rows, last, mean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=48, help="units per checkpoint file")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--design", choices=["full", "rot40", "q20p3"], default="rot40")
    ap.add_argument("--smoke", type=int, default=0, help="run N units spread over roles, report, exit")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    units = build_units(a.design)
    (out / "units.json").write_text(json.dumps(
        dict(model=MODEL, design=a.design, n_questions=N_QUESTIONS, layers=list(LAYERS), n_units=len(units),
             max_new_tokens=a.max_new_tokens, greedy=True, neutral="no system prompt",
             token_last="last answer token before the first stop token",
             question_ids=sorted({u["question_idx"] for u in units}))))
    if a.smoke:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(units), size=a.smoke, replace=False))
        todo = [[units[i] for i in pick]]
        tag = "smoke"
    else:
        lo = len(units) * a.shard // a.n_shards
        hi = len(units) * (a.shard + 1) // a.n_shards
        todo = [units[s:min(s + a.chunk, hi)] for s in range(lo, hi, a.chunk)]
        tag = None

    tok, model = load_model()
    t_all, n_all = time.time(), 0
    for ch in todo:
        name = tag or ("u%06d-%06d" % (ch[0]["unit"], ch[-1]["unit"] + 1))
        if (out / "chunks" / (name + ".npz")).exists():
            continue
        rows, L, M = [], [], []
        t0 = time.time()
        for s in range(0, len(ch), a.batch_size):
            r, l, m = run_batch(tok, model, ch[s:s + a.batch_size], a.max_new_tokens)
            rows += r
            L.append(l)
            M.append(m)
        L, M = np.concatenate(L), np.concatenate(M)
        bad = int((~np.isfinite(L.astype(np.float32))).any(axis=(1, 2)).sum())
        tmp = out / "chunks" / (name + ".tmp.npz")
        np.savez(tmp, last=L, mean=M)
        os.replace(tmp, out / "chunks" / (name + ".npz"))
        pd.DataFrame(rows).to_csv(out / "chunks" / (name + ".csv"), index=False)
        dt = time.time() - t0
        n_all += len(ch)
        nn = np.array([r["n_new"] for r in rows])
        print("%s: %d units %.0fs (%.2f ans/s) | n_new median %d p90 %d max %d | hit_cap %.1f%% | nonfinite %d"
              % (name, len(ch), dt, len(ch) / dt, np.median(nn), np.percentile(nn, 90), nn.max(),
                 100 * np.mean([r["hit_cap"] for r in rows]), bad), flush=True)
    print("done: %d units in %.0fs" % (n_all, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
