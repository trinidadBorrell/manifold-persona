"""Stage 0 (paper-matched): GENERATE responses, extract RESPONSE-token activations.

Faithful to the Assistant Axis paper's extraction (assistant-axis/pipeline/
2_activations.py). For each ``system(role) + user question`` chat we generate a
response with Qwen2.5-3B-Instruct. We then read the residual stream averaged
over the **response** tokens, at all layers. Downstream analysis uses the ~0.5-depth
layer (manifest.primary_layer = half_depth_hidden_state). Saved distinctly under
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
                                     half_depth_hidden_state)
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
    """The subset of args that must match for a resume to be valid.

    Every key is also written into the final manifest under the SAME name, so
    the completed-run guard below can compare the whole config.
    """
    return {"model_name": args.model, "tokenizer_name": args.tokenizer,
            "n_questions": args.n_questions,
            "seed": args.seed, "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample), "temperature": args.temperature,
            "limit": args.limit, "chunk": args.chunk}


def check_completed_run(out_dir: Path, cfg: dict, n_records: int) -> None:
    """Refuse to overwrite a COMPLETED run in ``out_dir``.

    A *finished* run has no _ckpt (finalization removes it), so the resume
    check cannot see it. Without this guard, re-invoking with different
    settings silently overwrites a completed cloud -- which for a multi-hour
    run destroys the result with no warning.

    n_records is always compared, so a ``--limit`` run against a full cloud is
    refused even when every other key matches. Manifests written before this
    guard existed stored only some of the run_config keys; a key the manifest
    never recorded cannot be compared, so it is skipped.
    """
    done_manifest = out_dir / "manifest.json"
    if not done_manifest.exists():
        return
    prev = json.load(open(done_manifest))
    mismatch = {k: (prev[k], cfg[k]) for k in cfg if k in prev and prev[k] != cfg[k]}
    if prev.get("n_records") != n_records:
        mismatch["n_records"] = (prev.get("n_records"), n_records)
    if mismatch:
        raise SystemExit(
            f"{out_dir} already holds a COMPLETED run with different "
            f"settings: {mismatch}\n"
            f"Refusing to overwrite it. Use --restart to discard and "
            f"rebuild, or pass a different --out_dir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--tokenizer", default=None,
                    help="render chats with THIS repo's tokenizer instead of "
                         "the model's own. Cross-stage runs must pass the "
                         "instruct tokenizer so every stage sees identical "
                         "token sequences (base/instruct vocabs are "
                         "byte-identical; their chat templates are not).")
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
        if not args.restart:
            check_completed_run(out_dir, cfg, N)
        json.dump({**cfg, "n_records": N}, open(cfg_path, "w"), indent=2)

    # Which chunks still need computing?
    bounds = [(s, min(s + args.chunk, N)) for s in range(0, N, args.chunk)]
    todo = [(s, e) for (s, e) in bounds if not shard_path(ckpt_dir, s, e).exists()]
    done = len(bounds) - len(todo)
    print(f"{N} records over {len({r.role for r in records})} roles "
          f"(of {len(list_roles())} available) · chunk={args.chunk} · "
          f"{len(bounds)} shards ({done} done, {len(todo)} to do) · "
          f"max_new_tokens={args.max_new_tokens} do_sample={args.do_sample}")

    # Load the model only if there is work to do.
    if todo:
        from manifold_persona.extract import load_model_and_tokenizer
        from manifold_persona.generate import generate_and_extract
        print(f"Loading model {args.model} ...")
        model, tokenizer, device = load_model_and_tokenizer(args.model)
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"rendering chats with tokenizer {args.tokenizer}")
        n_hidden = model.config.num_hidden_layers
        prim = half_depth_hidden_state(n_hidden)
        print(f"device={device}  n_hidden_states={n_hidden + 1}  "
              f"hidden={model.config.hidden_size}  half_depth_hidden_state={prim}")
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
    prim = half_depth_hidden_state(n_layers - 1)

    meta_rows = records_to_metadata(records)
    for row, resp in zip(meta_rows, resp_all):
        row["response"] = resp
    meta_df = pd.DataFrame(meta_rows)

    # Generation sanity stats. Recorded, never filtered on: a geometry
    # difference between stages must be checkable against a generation-quality
    # difference (base models can loop or truncate under matched decoding).
    lens = meta_df["response"].str.len()
    def _loops(s):
        tail = s[-20:]
        return bool(tail) and s.count(tail) >= 3
    manifest = {
        **cfg,                       # every run_config key, same names
        "n_layers": int(n_layers), "hidden": int(hidden),
        "primary_layer": int(prim), "token_basis": "response",
        "n_records": int(N), "n_roles": int(meta_df["role"].nunique()),
        "response_stats": {
            "char_len_median": float(lens.median()),
            "char_len_p10": float(lens.quantile(.10)),
            "char_len_p90": float(lens.quantile(.90)),
            "share_empty": float((lens == 0).mean()),
            "share_looping": float(meta_df["response"].map(_loops).mean()),
        },
    }
    save_embeddings(avg, last, meta_df, manifest, out_dir=out_dir)
    # Human-readable copy beside the parquet, matching the published clouds.
    meta_df.to_csv(out_dir / "metadata.csv", index=False)
    from manifold_persona.provenance import write_stamp
    write_stamp(out_dir)
    shutil.rmtree(ckpt_dir)          # clean up shards once the final save succeeded
    print(f"wrote {out_dir}  ({time.time()-t0:.0f}s total)  · removed {ckpt_dir}")
    print("response stats:", json.dumps(manifest["response_stats"]))


if __name__ == "__main__":
    main()
