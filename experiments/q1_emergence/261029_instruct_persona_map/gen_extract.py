"""Pod A instruct_persona_map: OLMo-2-7B-Instruct writes role replies; save rows and reply-mean states.

Design: ../261029_stage_persona_plan/DESIGN.md (pod A) and THRESHOLDS.md here.
- Chat template; the system prompt is one of the 5 role instructions (refs/assistant-axis); no example
  turns. An empty instruction (default wording 0) means no system message. "{model_name}" is "OLMo".
- 20 E3 pilot roles plus default, 5 wordings, the 40 frozen questions, 5 replicates.
- Sampling: temperature 0.5, repetition penalty 1.1, top_k 0, top_p 1, 128 new tokens. Each reply's seed
  is item_seed(item_key) (manifold_persona.rows), so batch size and order do not change it.
- Reply = generated ids before eos or pad and before a chat marker. Mean over the kept reply positions at
  every hidden state (embeddings + 32 layers), float16; NaN rows when the reply is invalid.
- Resume only when the run fingerprint (model, tokenizer, template, code, settings, grid) matches.

Run on the GPU pod:
  python gen_extract.py --dry                          # items, grid and rendered prompts; no model
  python gen_extract.py --roles canary --limit 40      # smoke test
  python gen_extract.py --roles canary                 # engineering canary
  python gen_extract.py                                # full run
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from manifold_persona import rows as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = "allenai/OLMo-2-1124-7B-Instruct"
REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"  # main on 29/10/2026
FORMAT = "chat"
MODEL_NAME_FILL = "OLMo"
MARKERS = ("<|user|>", "<|system|>", "<|assistant|>")
TEMPERATURE, PENALTY, MAX_NEW = 0.5, 1.1, 128
N_REPLICATES = 5
PILOT_ROLES = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
               "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
               "pirate", "vampire", "robot", "void"]
CANARY_ROLES = ["toddler", "consultant", "chef", "pirate"]
ROLE_SETS = {"full": PILOT_ROLES + ["default"], "canary": CANARY_ROLES + ["default"]}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
F16_MAX = 65504.0


def build_items(roles, n_rep, repo=REPO):
    """One item per (role, wording, question, replicate), in grid order."""
    bank = [json.loads(l)["question"] for l in open(repo / "refs/assistant-axis/extraction_questions.jsonl")]
    qs = R.load_questions(HERE.parent / "261008_base_persona_map/questions_40.json")
    items = []
    for role in roles:
        lines = [d["pos"] for d in json.load(open(repo / f"refs/assistant-axis/roles/instructions/{role}.json"))["instruction"]]
        assert len(lines) == 5, role
        for w, line in enumerate(lines):
            system = line.replace("{model_name}", MODEL_NAME_FILL)
            assert "{" not in system and "}" not in system, (role, w)
            for q in qs:
                for r in range(n_rep):
                    items.append(dict(key=R.item_key(role, FORMAT, w, q, r), system=system, question=bank[q]))
    return items


def messages(item):
    sys_msg = [{"role": "system", "content": item["system"]}] if item["system"] else []
    return sys_msg + [{"role": "user", "content": item["question"]}]


def render(tok, items):
    """Prompt ids from the chat template; every prompt starts with the one <|endoftext|> the template adds.
    Items are then sorted by prompt length (ties by key) so batches carry little padding."""
    for it in items:
        ids = tok.apply_chat_template(messages(it), add_generation_prompt=True, tokenize=True)
        assert ids[0] == tok.eos_token_id and ids.count(tok.eos_token_id) == 1, it["key"]
        it["prompt_ids"] = list(ids)
    return sorted(items, key=lambda it: (len(it["prompt_ids"]), it["key"]))


def read_done(part, batches):
    """Keys of complete batches in the rows part file. A crash during an append can leave a truncated last
    line and only some rows of a batch; both are removed, so that batch is generated again without duplicates."""
    if not part.exists():
        return set()
    lines = [l for l in part.read_bytes().split(b"\n") if l.strip()]
    rows = []
    for k, l in enumerate(lines):
        try:
            rows.append((R.row_key(json.loads(l)), l))
        except json.JSONDecodeError:
            assert k == len(lines) - 1, "corrupt line inside the part file"
    have = {k for k, _ in rows}
    done = set()
    for chunk in batches:
        keys = {it["key"] for it in chunk}
        if keys <= have:
            done |= keys
    keep = [l for k, l in rows if k in done]
    if len(keep) != len(lines):
        tmp = part.with_suffix(".tmp")
        tmp.write_bytes(b"".join(l + b"\n" for l in keep))
        os.replace(tmp, part)
        print(f"removed {len(lines) - len(keep)} lines of incomplete batches from {part.name}", flush=True)
    return done


def run(items, out_dir, batch_size, model, tok, run_id):
    """Generate and pool batch by batch. A batch counts as done only when its rows and its states exist."""
    import numpy as np
    eos, pad = tok.eos_token_id, tok.pad_token_id
    part, acts_dir = out_dir / "rows.partial.jsonl", out_dir / "acts_parts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    batches = [items[s:s + batch_size] for s in range(0, len(items), batch_size)]
    done = read_done(part, batches)
    for b, chunk in enumerate(batches):
        keys = [it["key"] for it in chunk]
        if all(k in done for k in keys):
            f = acts_dir / f"{b:06d}.npz"
            assert f.exists() and np.load(f)["keys"].tolist() == keys, f"state part missing for batch {b}"
    for b, chunk in enumerate(batches):
        keys = [it["key"] for it in chunk]
        if all(k in done for k in keys):
            continue
        raw = R.generate_batch(model, [it["prompt_ids"] for it in chunk], [R.item_seed(k) for k in keys],
                               eos, pad, TEMPERATURE, PENALTY, MAX_NEW)
        rows = [R.make_row(tok, run_id, it["key"], it["prompt_ids"], g, eos, pad, MARKERS) for it, g in zip(chunk, raw)]
        for r in rows:
            R.check_row(r, {eos, pad}, tok, MARKERS)
        spans = [r["span"] if r["valid"] else (r["span"][0], r["span"][0]) for r in rows]  # invalid -> NaN
        acts = R.pool_spans(model, [r["prompt_ids"] + r["kept_ids"] for r in rows], spans, pad)
        valid = np.array([r["valid"] for r in rows])
        assert np.isfinite(acts[valid]).all() and np.isnan(acts[~valid]).all(), f"batch {b}: non-finite states"
        fits = not valid.any() or np.abs(acts[valid]).max() <= F16_MAX
        if not fits:
            print(f"batch {b}: states exceed float16; this part stays float32", flush=True)
        np.savez(acts_dir / f"{b:06d}.npz", acts=acts.astype(np.float16) if fits else acts, keys=np.array(keys),
                 f32=np.array(not fits))
        with open(part, "a") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            f.flush()
            os.fsync(f.fileno())
        print(f"batch {b + 1}/{len(batches)}", flush=True)


def finalize(out_dir, grid):
    """rows.jsonl and reply_mean_acts.npy in grid order; every grid key must have a row."""
    import numpy as np
    from numpy.lib.format import open_memmap
    rows = [json.loads(l) for l in open(out_dir / "rows.partial.jsonl") if l.strip()]
    split = R.check_rows(rows, grid)
    assert not split["absent"], f"{len(split['absent'])} grid keys have no row"
    pos = {k: i for i, k in enumerate(grid)}
    by_key = {R.row_key(r): r for r in rows}
    parts = sorted((out_dir / "acts_parts").glob("*.npz"))
    dtype = np.float32 if any(bool(np.load(f)["f32"]) for f in parts) else np.float16  # reads only the flag
    acts, filled = None, set()
    for f in parts:  # one part at a time, so memory stays at one batch
        z = np.load(f)
        a, keys = z["acts"], z["keys"].tolist()
        v = np.array([by_key[k]["valid"] for k in keys])
        assert np.isfinite(a[v]).all() and np.isnan(a[~v]).all(), f"{f.name}: states do not match the valid flags"
        if acts is None:
            acts = open_memmap(out_dir / "reply_mean_acts.npy", mode="w+", dtype=dtype, shape=(len(grid),) + a.shape[1:])
        for k, x in zip(keys, a):
            acts[pos[k]] = x
            filled.add(k)
    assert filled == set(grid), f"{len(set(grid) - filled)} grid keys have no stored states"
    acts.flush()
    with open(out_dir / "rows.jsonl", "w") as f:
        for k in grid:
            f.write(json.dumps(by_key[k], ensure_ascii=False) + "\n")
    return dict(n_rows=len(grid), n_invalid=len(split["invalid"]), invalid_keys=split["invalid"],
                stop_reasons=dict(Counter(by_key[k]["stop_reason"] for k in grid)),
                acts_shape=list(acts.shape), acts_dtype=str(acts.dtype))


def tokenizer_hashes():
    from huggingface_hub import hf_hub_download
    return {f: R.file_sha256(hf_hub_download(MODEL, f, revision=REVISION)) for f in TOKENIZER_FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="full", help="full, canary, or a comma list of role names")
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: a fixed random subset of the grid")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    roles = ROLE_SETS.get(args.roles) or args.roles.split(",")
    items = build_items(roles, args.replicates)
    grid, grid_hash = R.freeze_grid(it["key"] for it in items)
    if args.limit:
        items = random.Random(0).sample(items, args.limit)
        keep = {it["key"] for it in items}
        grid, grid_hash = R.freeze_grid(k for k in grid if k in keep)
    out = Path(args.out or HERE / f"{'dry' if args.dry else 'run'}_{args.roles.replace(',', '+')}")
    run_id = out.name

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    items = render(tok, items)
    code = {p.name: R.file_sha256(p) for p in (Path(__file__).resolve(), Path(R.__file__).resolve())}
    parts = dict(model=MODEL, revision=REVISION, tokenizer_files=tokenizer_hashes(),
                 chat_template=hashlib.sha256(tok.chat_template.encode()).hexdigest(), code=code,
                 dtype="bfloat16", sampling=R.sampling_settings(TEMPERATURE, PENALTY, MAX_NEW), markers=MARKERS,
                 model_name_fill=MODEL_NAME_FILL, grid_hash=grid_hash, batch_size=args.batch_size)
    if args.dry:
        if (out / "fingerprint.txt").exists():
            raise SystemExit(f"{out} is a run directory; use another --out for --dry")
    else:
        R.claim_out_dir(out, R.fingerprint(parts))
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(dict(grid_hash=grid_hash, keys=grid)))
    show = [R.item_key(r, FORMAT, w, 10, 0) for r, w in (("default", 0), ("default", 3), ("pirate", 0), ("assistant", 0))]
    shown = [it for it in items if it["key"] in show]
    (out / "prompts_check.txt").write_text("".join(f"===== {it['key']} =====\n{tok.decode(it['prompt_ids'])}<<END>>\n\n"
                                                   for it in shown))
    print(f"{len(items)} items, {len(roles)} roles, grid {grid_hash[:12]}")
    if args.dry:
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).cuda().eval()
    commit = getattr(model.config, "_commit_hash", None)
    assert commit == REVISION, f"loaded weights {commit} are not the pinned revision {REVISION}"
    run(items, out, args.batch_size, model, tok, run_id)
    summary = finalize(out, grid)
    import platform
    from importlib import metadata
    import tokenizers
    import transformers
    manifest = dict(parts, run_id=run_id, roles=roles, replicates=args.replicates, limit=args.limit,
                    attn_implementation=getattr(model.config, "_attn_implementation", None),
                    n_hidden_states=model.config.num_hidden_layers + 1, fingerprint=R.fingerprint(parts),
                    act_rows="same order as rows.jsonl and grid.json; rows with valid=false are NaN",
                    versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                  transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                                  numpy=np.__version__),
                    packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
                    seconds=round(time.time() - t), **summary)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    names = ("rows.jsonl", "reply_mean_acts.npy", "grid.json", "manifest.json")
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    print(f"done: {summary['n_rows']} rows, {summary['n_invalid']} invalid, states {summary['acts_shape']}")


if __name__ == "__main__":
    main()
