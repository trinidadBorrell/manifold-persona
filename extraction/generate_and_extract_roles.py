"""Stage 0 (paper-matched): GENERATE responses, extract RESPONSE-token activations.

Faithful to the Assistant Axis paper's extraction (assistant-axis/pipeline/
2_activations.py): for each ``system(role) + user question`` chat we generate a
response with Qwen2.5-3B-Instruct and read the residual stream averaged over the
**response** tokens, at all layers. Downstream analysis uses the ~0.5-depth
layer (manifest.primary_layer = half_depth_layer). Saved distinctly under
data/embeddings_roles_resp/ so the prompt-token run stays intact.

**Checkpointed / resumable.** Records are processed in fixed-size chunks; each
chunk is written to a shard under ``<out_dir>/_ckpt/`` as soon as it finishes.
A kill loses at most one in-flight chunk (~minutes), never the whole run. Re-run
the SAME command to resume: completed shards are skipped, and once every chunk
exists they are concatenated into the final embeddings. The record order is
deterministic (fixed seed + n_questions), so shard i always covers the same
records; a config mismatch on resume aborts rather than silently mixing runs.

Differences from the paper that remain (by design, see REPORT): Qwen2.5-3B (not
27B+), no score-3 judge filter (we average all rollouts), sampled questions.

Usage:
    .venv/bin/python -m extraction.generate_and_extract_roles                 # full run / resume
    .venv/bin/python -m extraction.generate_and_extract_roles --limit 8       # smoke/bench
    .venv/bin/python -m extraction.generate_and_extract_roles --chunk 128     # shard size
    .venv/bin/python -m extraction.generate_and_extract_roles --restart       # wipe checkpoints, start clean
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


from manifold_persona.config import (MODEL_NAME, RESP_ROLE_EMBEDDINGS_DIR,
                                     half_depth_layer)
from manifold_persona.prompts_roles import (build_role_records, records_to_metadata,
                                            list_roles)
from manifold_persona.io import save_embeddings

CKPT_SUBDIR = "_ckpt"
CONFIG_NAME = "ckpt_config.json"


def record_to_messages(r):
    msgs = []
    if r.system is not None:
        msgs.append({"role": "system", "content": r.system})
    msgs.append({"role": "user", "content": r.question})
    return msgs


def shard_path(ckpt_dir: Path, start: int, end: int) -> Path:
    return ckpt_dir / f"shard_{start:06d}_{end:06d}.npz"


def run_config(args) -> dict:
    """The subset of args that must match for a resume to be valid."""
    return {"model": args.model, "n_questions": args.n_questions,
            "seed": args.seed, "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample), "temperature": args.temperature,
            "limit": args.limit, "chunk": args.chunk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=str(RESP_ROLE_EMBEDDINGS_DIR))
    ap.add_argument("--n_questions", type=int, default=5, help="questions sampled per role")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #records (smoke/bench)")
    ap.add_argument("--chunk", type=int, default=128, help="records per checkpoint shard")
    ap.add_argument("--restart", action="store_true", help="wipe existing checkpoints first")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / CKPT_SUBDIR
    if args.restart and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"--restart: wiped {ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic record list (fixed seed + n_questions => stable order/indexing).
    model_display = args.model.split("/")[-1]
    records = build_role_records(n_questions=args.n_questions, seed=args.seed,
                                 model_display=model_display)
    if args.limit:
        records = records[: args.limit]
    N = len(records)

    # Validate / write the resume config.
    cfg = run_config(args)
    cfg_path = ckpt_dir / CONFIG_NAME
    if cfg_path.exists():
        prev = json.load(open(cfg_path))
        mismatch = {k: (prev.get(k), cfg[k]) for k in cfg if prev.get(k) != cfg[k]}
        if mismatch:
            raise SystemExit(
                f"Checkpoint config mismatch in {ckpt_dir}: {mismatch}\n"
                f"Resume needs identical settings. Use --restart to start clean "
                f"(discards existing shards).")
    else:
        json.dump({**cfg, "n_records": N}, open(cfg_path, "w"), indent=2)

    # Which chunks still need computing?
    bounds = [(s, min(s + args.chunk, N)) for s in range(0, N, args.chunk)]
    todo = [(s, e) for (s, e) in bounds if not shard_path(ckpt_dir, s, e).exists()]
    done = len(bounds) - len(todo)
    print(f"{N} records over {len(list_roles())} roles · chunk={args.chunk} · "
          f"{len(bounds)} shards ({done} done, {len(todo)} to do) · "
          f"max_new_tokens={args.max_new_tokens} do_sample={args.do_sample}")

    # Load the model only if there is work to do.
    if todo:
        from manifold_persona.extract import load_model_and_tokenizer
        from manifold_persona.generate import generate_and_extract
        print(f"Loading model {args.model} ...")
        model, tokenizer, device = load_model_and_tokenizer(args.model)
        n_hidden = model.config.num_hidden_layers
        prim = half_depth_layer(n_hidden)
        print(f"device={device}  n_hidden_states={n_hidden + 1}  "
              f"hidden={model.config.hidden_size}  half_depth_layer={prim}")
        chats = [record_to_messages(r) for r in records]
        for (s, e) in todo:
            tc = time.time()
            avg, last, responses = generate_and_extract(
                model, tokenizer, chats[s:e], device,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                temperature=args.temperature)
            # atomic-ish shard write: temp then rename. NB np.savez appends
            # ".npz" if the name lacks it, so the temp name must already end
            # ".npz" or the rename target won't exist.
            final = shard_path(ckpt_dir, s, e)
            tmp = ckpt_dir / f"shard_{s:06d}_{e:06d}.tmp.npz"
            np.savez(tmp, avg=avg, last=last,
                     responses=np.array(responses, dtype=object),
                     start=s, end=e)
            tmp.rename(final)
            print(f"  shard [{s}:{e}] {e-s} recs in {time.time()-tc:.0f}s "
                  f"({(time.time()-tc)/(e-s):.2f}s/rec)  -> saved")

    # ---- Finalize: concatenate shards into the final embeddings layout ----
    print("Finalizing: concatenating shards ...")
    avgs, lasts, resp_all = [], [], []
    for (s, e) in bounds:
        d = np.load(shard_path(ckpt_dir, s, e), allow_pickle=True)
        avgs.append(d["avg"]); lasts.append(d["last"])
        resp_all.extend(list(d["responses"]))
    avg = np.concatenate(avgs, axis=0)
    last = np.concatenate(lasts, axis=0)
    assert avg.shape[0] == N, f"row count {avg.shape[0]} != {N}"
    n_layers = avg.shape[1]
    hidden = avg.shape[2]
    prim = half_depth_layer(n_layers - 1)

    meta_rows = records_to_metadata(records)
    for row, resp in zip(meta_rows, resp_all):
        row["response"] = resp
    meta_df = pd.DataFrame(meta_rows)

    manifest = {
        "model_name": args.model, "n_layers": int(n_layers), "hidden": int(hidden),
        "primary_layer": int(prim), "token_basis": "response",
        "max_new_tokens": int(args.max_new_tokens), "do_sample": bool(args.do_sample),
        "n_questions": int(args.n_questions), "n_records": int(N),
        "n_roles": int(meta_df["role"].nunique()),
    }
    save_embeddings(avg, last, meta_df, manifest, out_dir=out_dir)
    shutil.rmtree(ckpt_dir)          # clean up shards once the final save succeeded
    print(f"wrote {out_dir}  ({time.time()-t0:.0f}s total)  · removed {ckpt_dir}")


if __name__ == "__main__":
    main()
