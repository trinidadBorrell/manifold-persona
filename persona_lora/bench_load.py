"""Time the phases separately: import, model load, generation, extraction.

The first benchmark could not tell a stalled import from slow compute, because
every cost was folded into one number.
"""
import argparse
import json
import time

T0 = time.time()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
T_IMPORT = time.time()

from .data import generate_targets, question_split, load_targets  # noqa: E402
from .extract_acts import pooled  # noqa: E402
from .data import encode_pair, load_persona_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--persona", default="young_scholar")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    print(f"IMPORT_S={T_IMPORT - T0:.1f}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    t = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(dev)
    model.eval()
    print(f"DEVICE={dev} DTYPE={args.dtype} LOAD_S={time.time() - t:.1f}", flush=True)

    train_q, held_q = question_split(48, 0)
    qs = (held_q + train_q)[:args.n]

    t = time.time()
    rows = list(generate_targets(model, tok, [args.persona], qs, dev,
                                 args.max_new_tokens, 0.3, 0.9, 0, 5))
    gen = time.time() - t
    print(f"GEN_S={gen:.1f} N={len(rows)} PER_RESPONSE_S={gen / max(len(rows), 1):.2f}", flush=True)

    t = time.time()
    for r in rows:
        sys_text = load_persona_prompt(r["persona"], r.get("instruction_idx", 0))
        pre_p, suf, start = encode_pair(tok, sys_text, r["question"], r["response"])
        pre_u, suf_u, start_u = encode_pair(tok, "", r["question"], r["response"])
        pooled(model, pre_p + suf, len(pre_p) + start, dev, 384)
        pooled(model, pre_u + suf, len(pre_u) + start, dev, 384)
    ext = time.time() - t
    print(f"EXT_S={ext:.1f} PER_RESPONSE_S={ext / max(len(rows), 1):.2f}", flush=True)

    per = (gen + ext) / max(len(rows), 1)
    print(f"SUMMARY dtype={args.dtype} dev={dev} per_response_s={per:.2f} "
          f"est_120q_min={per * 120 / 60:.1f} est_41personas_h={per * 120 * 41 / 3600:.1f}",
          flush=True)
    print(json.dumps({"device": dev, "dtype": args.dtype, "import_s": T_IMPORT - T0,
                      "gen_s": gen, "ext_s": ext, "per_response_s": per}), flush=True)


if __name__ == "__main__":
    main()
