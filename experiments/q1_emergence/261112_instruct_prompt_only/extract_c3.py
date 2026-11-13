"""Instruct prompt-only control: pod B's C3 frame on pod A's 4,200 prompts (rules: THRESHOLDS.md).

Each stored pod A prompt (ending in <|assistant|>\\n) is followed by one fixed neutral reply; the state is the mean
over that reply, pooled with pod A's own function. Forward passes only; no text is generated.

  python extract_c3.py prepare --poda_run ../261029_instruct_persona_map/run_full --out inputs     # Mac, no model
  python extract_c3.py extract --inputs inputs --out run_c3 [--limit 32] [--device cuda]         # GPU
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS_PY = HERE.parents[2] / "src/manifold_persona/rows.py"
MODEL = "allenai/OLMo-2-1124-7B-Instruct"
REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
NEUTRAL_SPEECH = ("That is a good question. I think the answer depends on the situation, and I would want to think"
                  " about it carefully before I say more.")  # pod B context_extract.py, owner-approved 29/10/2026
STATES = [8, 9, 16, 17, 23, 24]
CHECK_STATE, CHECK_MIN_COS, N_CHECK = 17, 0.99, 8
CHAT_TEMPLATE_SHA_PREFIX = "fe689ffb"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_module():
    spec = importlib.util.spec_from_file_location("_mp_rows", ROWS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt_key(r):
    return (r["role"], r["format"], r["wording"], r["question"])


def unique_prompts(rows):
    """One entry per (role, format, wording, question), in first-seen order; all replicates must share prompt ids."""
    seen = {}
    for r in rows:
        k = prompt_key(r)
        if k in seen:
            if seen[k] != r["prompt_ids"]:
                raise SystemExit(f"replicates of {k} have different prompt ids")
        else:
            seen[k] = r["prompt_ids"]
    return [dict(role=k[0], format=k[1], wording=k[2], question=k[3], prompt_ids=v) for k, v in seen.items()]


def check_rows(rows, n=N_CHECK):
    """The first valid row of each of the first n roles, in run order."""
    out, roles = [], set()
    for i, r in enumerate(rows):
        if r["valid"] and r["role"] not in roles:
            roles.add(r["role"])
            out.append(i)
        if len(out) == n:
            break
    return out


def cmd_prepare(a):
    run = Path(a.poda_run)
    sums = dict(reversed(line.split("  ", 1)) for line in (run / "SHA256SUMS.txt").read_text().splitlines())
    if sha256(run / "rows.jsonl") != sums["rows.jsonl"]:
        raise SystemExit("rows.jsonl does not match pod A's SHA256SUMS.txt")
    rows = [json.loads(line) for line in (run / "rows.jsonl").open()]
    manifest = json.loads((run / "manifest.json").read_text())
    prompts = unique_prompts(rows)
    roles = sorted({p["role"] for p in prompts})
    expect = len(roles) * len({p["wording"] for p in prompts}) * len({p["question"] for p in prompts})
    if len(prompts) != expect or len(prompts) != 4200:
        raise SystemExit(f"{len(prompts)} prompts, expected 4,200 on a full grid")
    idx = check_rows(rows)
    acts = np.load(run / "reply_mean_acts.npy", mmap_mode="r")
    if acts.shape[0] != len(rows):
        raise SystemExit("pod A states and rows differ in length")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "c3_prompts.jsonl").open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    np.save(out / "check_states.npy", np.asarray(acts[idx, CHECK_STATE], dtype=np.float32))
    (out / "check_rows.json").write_text(json.dumps(
        [dict(row=i, role=rows[i]["role"], prompt_ids=rows[i]["prompt_ids"], kept_ids=rows[i]["kept_ids"],
              span=rows[i]["span"]) for i in idx]))
    meta = dict(poda_run=str(run), poda_rows_sha256=sums["rows.jsonl"], poda_acts_sha256=sums.get("reply_mean_acts.npy"),
                model=manifest["model"], revision=manifest["revision"], tokenizer_files=manifest["tokenizer_files"],
                poda_versions=manifest["versions"], n_prompts=len(prompts), roles=roles, check_rows=idx,
                check_state=CHECK_STATE)
    if (meta["model"], meta["revision"]) != (MODEL, REVISION):
        raise SystemExit("pod A model or revision differs from the rules")
    (out / "inputs_meta.json").write_text(json.dumps(meta, indent=1))
    files = ["c3_prompts.jsonl", "check_states.npy", "check_rows.json", "inputs_meta.json"]
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(out / n)}  {n}\n" for n in files))
    print(f"prepared {len(prompts)} prompts, check rows {idx}")


def load_model(meta, device):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if sha256(hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)) != meta["tokenizer_files"]["tokenizer.json"]:
        raise SystemExit("tokenizer.json differs from pod A's manifest")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if not hashlib.sha256(tok.chat_template.encode()).hexdigest().startswith(CHAT_TEMPLATE_SHA_PREFIX):
        raise SystemExit("chat template differs")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    if getattr(model.config, "_commit_hash", None) != REVISION:
        raise SystemExit("loaded commit differs from the pinned revision")
    return model, tok


def cmd_extract(a):
    import torch
    import transformers
    t0 = time.time()
    inp = Path(a.inputs)
    for line in (inp / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if sha256(inp / name) != h:
            raise SystemExit(f"input {name} does not match its checksum")
    meta = json.loads((inp / "inputs_meta.json").read_text())
    prompts = [json.loads(line) for line in (inp / "c3_prompts.jsonl").open()]
    if a.limit:
        prompts = prompts[:a.limit]
    R = rows_module()
    model, tok = load_model(meta, a.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    neutral = tok.encode(NEUTRAL_SPEECH, add_special_tokens=False)
    if tok.decode(neutral) != NEUTRAL_SPEECH:
        raise SystemExit("neutral reply does not round-trip through the tokenizer")
    # Environment check: re-pool pod A's real replies and compare with its stored state 17.
    checks = json.loads((inp / "check_rows.json").read_text())
    stored = np.load(inp / "check_states.npy")
    got = R.pool_spans(model, [c["prompt_ids"] + c["kept_ids"] for c in checks],
                       [[len(c["prompt_ids"]), len(c["prompt_ids"]) + len(c["kept_ids"])] for c in checks],
                       pad, layers=[CHECK_STATE])[:, 0]
    cos = (got * stored).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(stored, axis=1))
    print("environment check cosines:", np.round(cos, 5).tolist(), flush=True)
    if not (cos >= CHECK_MIN_COS).all():
        raise SystemExit(f"environment check failed: min cosine {cos.min():.4f} < {CHECK_MIN_COS}")
    for p in prompts:
        if not tok.decode(p["prompt_ids"]).endswith("<|assistant|>\n"):
            raise SystemExit(f"prompt {p['role']} {p['wording']} {p['question']} does not end with the assistant opening")
    seqs = [p["prompt_ids"] + neutral for p in prompts]
    spans = [[len(p["prompt_ids"]), len(p["prompt_ids"]) + len(neutral)] for p in prompts]
    acts = np.zeros((len(seqs), len(STATES), model.config.hidden_size), dtype=np.float32)
    for b0 in range(0, len(seqs), a.batch_size):
        acts[b0:b0 + a.batch_size] = R.pool_spans(model, seqs[b0:b0 + a.batch_size], spans[b0:b0 + a.batch_size],
                                                  pad, layers=STATES)
        if (b0 // a.batch_size) % 20 == 0:
            print(f"batch {b0 // a.batch_size + 1}/{-(-len(seqs) // a.batch_size)}", flush=True)
    if not np.isfinite(acts).all() or np.abs(acts).max() >= 6.0e4:
        raise SystemExit("states are non-finite or do not fit float16")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "c3_acts.npy", acts.astype(np.float16))
    (out / "c3_keys.json").write_text(json.dumps([[p["role"], p["format"], p["wording"], p["question"]] for p in prompts]))
    man = dict(model=MODEL, revision=REVISION, dtype="bfloat16", attn_implementation="sdpa", states=STATES,
               neutral_text=NEUTRAL_SPEECH, neutral_ids=neutral, n_prompts=len(prompts), limit=a.limit,
               batch_size=a.batch_size, environment_check=dict(state=CHECK_STATE, rows=meta["check_rows"],
                                                               cosines=cos.tolist(), min_required=CHECK_MIN_COS),
               inputs_sha256=(inp / "SHA256SUMS.txt").read_text(), poda_rows_sha256=meta["poda_rows_sha256"],
               code=dict(extract_c3_py=sha256(Path(__file__)), rows_py=sha256(ROWS_PY)),
               versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                             device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else a.device),
               seconds=round(time.time() - t0, 1))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    files = ["c3_acts.npy", "c3_keys.json", "manifest.json"]
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(out / n)}  {n}\n" for n in files))
    print(f"done: {acts.shape}, {man['seconds']}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--poda_run", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--inputs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch_size", type=int, default=32)
    s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    main()
