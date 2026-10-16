"""Our steering (E1 / Steering Stability rule) on Giovanni's setup, so only the steering differs.

Same as his run: Qwen2.5-3B-Instruct fp16, hidden_states[23] (pre-hook of layers[23], every
position), his 10 questions, his empty-system-block prompt, T 0.7, top-p 0.9, 140 new tokens,
torch.manual_seed(question index). Different (ours):
  persona vector  paper eq. 1 as in our E1 extraction (pm_geometry/extract_states.py, design rot40):
                  40 questions (build_role_records(40, seed 0)), question j under the role's system
                  prompt j mod 5, greedy to EOS (cap 1024); h = residual at the last answer token;
                  x_B = mean_j (h_B,j - h_0,j), h_0 = the same question with an empty system prompt
  injection       eq. 6 on the LIN route from the start A = default (x_A = 0) to B:
                  delta(s) = alpha * unit(x(s)) * ((1-s)|x_A| + s|x_B|) = alpha * s * x_B,
                  ADDED to h at every position (h <- h + delta), the same vector at every token
  grid            alpha in {1, 4} x s in {0.25, 0.5, 0.75, 1}

    python -m giovanni_repro.ours extract --root $PM/GIO --who spy --shard 3/10
    python -m giovanni_repro.ours vectors --root $PM/GIO --persona spy
    python -m giovanni_repro.ours steer   --root $PM/GIO --persona spy --cell 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
L = 23
ALPHAS, STEPS = (1.0, 4.0), (0.25, 0.5, 0.75, 1.0)
CELLS = [(a, s) for a in ALPHAS for s in STEPS]
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load(root):
    sys.path[:0] = [str(root / "repo"), str(root / "repo" / "src")]
    from giovanni_repro.run import shim
    shim()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    return torch, tok, model


def units(who):
    """rot40: question j (sorted id order) under system prompt j mod 5; who = 'default' -> empty."""
    sys.path.insert(0, str(REPO / "src"))
    from manifold_persona.prompts_roles import build_role_records
    recs = build_role_records(n_questions=40, seed=0, roles=["spy"], model_display="Qwen2.5-3B-Instruct")
    qs = sorted({(r.question_idx, r.question) for r in recs})
    if who == "default":
        return [dict(qid=q, question=t, system="") for q, t in qs]
    recs = build_role_records(n_questions=40, seed=0, roles=[who], model_display="Qwen2.5-3B-Instruct")
    rank = {q: j for j, (q, _) in enumerate(qs)}
    return [dict(qid=r.question_idx, question=r.question, system=r.system or "")
            for r in recs if r.instruction_idx == rank[r.question_idx] % 5]


def extract(root, who, shard):
    torch, tok, model = load(root)
    i, n = (int(x) for x in shard.split("/"))
    U = units(who)[i::n]
    out = root / "ours" / who
    out.mkdir(parents=True, exist_ok=True)
    stop = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    H, meta = [], []
    for u in U:
        text = tok.apply_chat_template([{"role": "system", "content": u["system"]}, {"role": "user", "content": u["question"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        P = ids["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
            gen = g[0, P:].tolist()
            n_new = next((k for k, t in enumerate(gen) if t in stop), len(gen))
            last = P + n_new - 1 if n_new > 0 else P - 1          # last answer token before the stop token
            hs = model(g[:, :P + max(n_new, 1)], output_hidden_states=True).hidden_states[L]
        H.append(hs[0, last].float().cpu().numpy())
        meta.append(dict(qid=u["qid"], n_new=n_new, hit_cap=n_new >= 1024,
                         response=tok.decode(gen[:n_new], skip_special_tokens=True)))
        print("q%d %d tokens" % (u["qid"], n_new), flush=True)
    np.savez(out / ("last_%02d.npz" % i), h=np.stack(H).astype(np.float32), qid=np.array([m["qid"] for m in meta]))
    (out / ("meta_%02d.json" % i)).write_text(json.dumps(meta))


def vectors(root, persona):
    def get(w):
        fs = sorted((root / "ours" / w).glob("last_*.npz"))
        z = [np.load(f) for f in fs]
        return dict(zip(np.concatenate([x["qid"] for x in z]).tolist(), np.concatenate([x["h"] for x in z])))
    B, N = get(persona), get("default")
    common = sorted(set(B) & set(N))
    x = np.mean([B[q] - N[q] for q in common], 0)
    out = root / "ours" / persona
    np.save(out / "x_B.npy", x)
    (out / "x_B_meta.json").write_text(json.dumps(dict(n=len(common), norm=float(np.linalg.norm(x)))))
    print(persona, len(common), float(np.linalg.norm(x)))


def steer(root, persona, cell):
    torch, tok, model = load(root)
    from persona_lora.data import gen_prompt_text        # his prompt
    from persona_lora.steer_pin import QUESTIONS          # his 10 questions
    alpha, s = CELLS[cell]
    x = torch.tensor(np.load(root / "ours" / persona / "x_B.npy"), device="cuda", dtype=torch.float32)
    delta = alpha * s * x                                 # eq. 6 on the default -> B line (x_A = 0)

    def hook(module, args):
        hs = args[0]
        return ((hs.float() + delta).to(hs.dtype),) + tuple(args[1:])
    h = model.model.layers[L].register_forward_pre_hook(hook)
    out = root / "ours" / persona / ("gen_%d.jsonl" % cell)
    try:
        with open(out, "w") as f:
            for qi, q in enumerate(QUESTIONS):
                ids = tok(gen_prompt_text("", q), return_tensors="pt", add_special_tokens=False).to("cuda")
                torch.manual_seed(qi)
                o = model.generate(**ids, max_new_tokens=140, do_sample=True, temperature=0.7, top_p=0.9,
                                   pad_token_id=tok.eos_token_id)
                txt = tok.decode(o[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({"cell": "ours|L23_%s" % persona, "alpha": alpha, "s": s, "qi": qi,
                                    "question": q, "response": txt}) + "\n")
    finally:
        h.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "vectors", "steer"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--who")
    ap.add_argument("--persona")
    ap.add_argument("--shard")
    ap.add_argument("--cell", type=int)
    a = ap.parse_args()
    root = Path(a.root)
    {"extract": lambda: extract(root, a.who, a.shard), "vectors": lambda: vectors(root, a.persona),
     "steer": lambda: steer(root, a.persona, a.cell)}[a.stage]()
