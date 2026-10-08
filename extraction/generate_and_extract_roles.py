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
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


from manifold_persona.config import (MODEL_NAME, RESP_ROLE_EMBEDDINGS_DIR,
                                     half_depth_hidden_state)
from manifold_persona.prompts_roles import (build_role_records, records_to_metadata,
                                            list_roles)
from manifold_persona.io import (AVG_FILE, LAST_FILE, MANIFEST_FILE, META_FILE)
from manifold_persona.provenance import resolved_commit

CKPT_SUBDIR = "_ckpt"
CONFIG_NAME = "ckpt_config.json"
RESOLVED_NAME = "resolved_commits.json"


# Persona-neutral exemplar turns for --fewshot. Base models echo bare chat
# markup instead of answering; two in-context Q->A pairs give them the
# pattern. Applied to EVERY stage identically so token sequences stay
# comparable. Not from the study's question pool.
FEWSHOT = [
    ("What causes the seasons on Earth?",
     "The tilt of Earth's axis. As Earth orbits the sun, each hemisphere "
     "leans toward the sun for part of the year and away for another, "
     "changing how much direct sunlight it gets."),
    ("How do you make a cup of tea?",
     "Boil water, pour it over a tea bag or leaves, let it steep a few "
     "minutes, then remove the tea. Add milk or sugar if you like."),
]


def record_to_messages(r, fewshot: bool = False):
    msgs = []
    if r.system is not None:
        msgs.append({"role": "system", "content": r.system})
    if fewshot:
        for q, a in FEWSHOT:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": r.question})
    return msgs


def records_digest(records, fewshot: bool) -> str:
    """sha256 of the exact chats the run renders: roles, questions, few-shot turns."""
    chats = [record_to_messages(r, fewshot=fewshot) for r in records]
    return hashlib.sha256(json.dumps(chats, sort_keys=True).encode()).hexdigest()


def shard_path(ckpt_dir: Path, start: int, end: int) -> Path:
    return ckpt_dir / f"shard_{start:06d}_{end:06d}.npz"


def merge_shards(bounds, ckpt_dir: Path, out_dir: Path, n_records: int):
    """Stream checkpoint shards into final arrays without holding two copies."""
    first_start, first_end = bounds[0]
    with np.load(shard_path(ckpt_dir, first_start, first_end),
                 allow_pickle=True) as first:
        avg_shape = first["avg"].shape[1:]
        last_shape = first["last"].shape[1:]
        avg_dtype = first["avg"].dtype
        last_dtype = first["last"].dtype
    if avg_shape != last_shape:
        raise ValueError(f"avg shape {avg_shape} != last shape {last_shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    avg_tmp = out_dir / "prompt_avg.tmp.npy"
    last_tmp = out_dir / "prompt_last.tmp.npy"
    avg_out = np.lib.format.open_memmap(
        avg_tmp, mode="w+", dtype=avg_dtype, shape=(n_records, *avg_shape)
    )
    last_out = np.lib.format.open_memmap(
        last_tmp, mode="w+", dtype=last_dtype, shape=(n_records, *last_shape)
    )

    responses = []
    for i, (start, end) in enumerate(bounds):
        with np.load(shard_path(ckpt_dir, start, end), allow_pickle=True) as shard:
            if int(shard["start"]) != start or int(shard["end"]) != end:
                raise ValueError(f"shard bounds mismatch at [{start}:{end}]")
            avg = shard["avg"]
            last = shard["last"]
            expected = (end - start, *avg_shape)
            if avg.shape != expected or last.shape != expected:
                raise ValueError(
                    f"shard [{start}:{end}] shape mismatch: "
                    f"avg={avg.shape}, last={last.shape}, expected={expected}"
                )
            avg_out[start:end] = avg
            last_out[start:end] = last
            responses.extend(shard["responses"].tolist())
        if (i + 1) % 16 == 0:
            avg_out.flush()
            last_out.flush()

    avg_out.flush()
    last_out.flush()
    del avg_out, last_out
    if len(responses) != n_records:
        raise ValueError(f"response count {len(responses)} != {n_records}")
    os.replace(avg_tmp, out_dir / AVG_FILE)
    os.replace(last_tmp, out_dir / LAST_FILE)
    return responses, avg_shape[0], avg_shape[1]


def run_config(args) -> dict:
    """The subset of args that must match for a resume to be valid.

    Every key is also written into the final manifest under the SAME name, so
    the completed-run guard below can compare the whole config.
    """
    return {"model_name": args.model, "tokenizer_name": args.tokenizer,
            "revision": args.revision,
            "n_questions": args.n_questions,
            "seed": args.seed, "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample), "temperature": args.temperature,
            "limit": args.limit, "chunk": args.chunk,
            "batch_size": args.batch_size, "fewshot": bool(args.fewshot),
            "stop_marker": args.stop_marker}


def check_completed_run(out_dir: Path, cfg: dict, n_records: int) -> None:
    """Refuse to overwrite a COMPLETED run in ``out_dir``.

    A *finished* run has no _ckpt (finalization removes it), so the resume
    check cannot see it. Without this guard, re-invoking with different
    settings silently overwrites a completed cloud -- which for a multi-hour
    run destroys the result with no warning.

    n_records is always compared, so a ``--limit`` run against a full cloud is
    refused even when every other key matches. A key the old manifest never
    recorded counts as a mismatch: its value is unknown, so the overwrite
    cannot be shown to be harmless.
    """
    done_manifest = out_dir / "manifest.json"
    if not done_manifest.exists():
        return
    prev = json.load(open(done_manifest))
    missing = "<not recorded>"
    mismatch = {k: (prev.get(k, missing), cfg[k]) for k in cfg
                if prev.get(k, missing) != cfg[k]}
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
    ap.add_argument("--revision", default=None,
                    help="HF model revision (branch/tag/commit), e.g. an OLMo-2 "
                         "pretraining checkpoint stage1-step98000-tokens412B")
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
    ap.add_argument("--stop-marker", default=None,
                    help="truncate responses at this string's first generated "
                         "occurrence (batched path only); pass the lineage's "
                         "user-turn marker for base models that never emit eos")
    ap.add_argument("--fewshot", action="store_true",
                    help="prepend the two neutral exemplar Q->A turns to every "
                         "prompt (all stages identically); needed for base "
                         "models that echo bare chat markup")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="records generated per GPU batch. 1 = the sequential "
                         "path the reference clouds used. Batched greedy can "
                         "differ at logit near-ties; verify with "
                         "extraction/verify_pilot.py before a full run.")
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
    cfg["records_sha256"] = records_digest(records, args.fewshot)
    cfg_path = ckpt_dir / CONFIG_NAME
    if cfg_path.exists():
        prev = json.load(open(cfg_path))
        # Checkpoints written before the digest existed cannot be compared on it.
        mismatch = {k: (prev.get(k), cfg[k]) for k in cfg if prev.get(k) != cfg[k]
                    and not (k == "records_sha256" and k not in prev)}
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
        from manifold_persona.generate import (generate_and_extract,
                                               generate_and_extract_batched)
        print(f"Loading model {args.model} ...")
        model, tokenizer, device = load_model_and_tokenizer(args.model, revision=args.revision)
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"rendering chats with tokenizer {args.tokenizer}")
        # Every session's resolved weights, so a resume across a moved
        # branch shows up as two commits in the manifest.
        resolved_path = ckpt_dir / RESOLVED_NAME
        seen = json.load(open(resolved_path)) if resolved_path.exists() else []
        commit = getattr(model.config, "_commit_hash", None)
        if commit not in seen:
            json.dump(seen + [commit], open(resolved_path, "w"))
        n_hidden = model.config.num_hidden_layers
        prim = half_depth_hidden_state(n_hidden)
        print(f"device={device}  n_hidden_states={n_hidden + 1}  "
              f"hidden={model.config.hidden_size}  half_depth_hidden_state={prim}")
        chats = [record_to_messages(r, fewshot=args.fewshot) for r in records]
        for (s, e) in todo:
            tc = time.time()
            if args.batch_size > 1:
                avg, last, responses = generate_and_extract_batched(
                    model, tokenizer, chats[s:e], device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample, temperature=args.temperature,
                    batch_size=args.batch_size, stop_marker=args.stop_marker)
            else:
                avg, last, responses = generate_and_extract(
                    model, tokenizer, chats[s:e], device,
                    max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                    temperature=args.temperature, stop_marker=args.stop_marker)
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

    # ---- Finalize: stream shards into the final embeddings layout ----
    print("Finalizing: streaming shards ...")
    resp_all, n_layers, hidden = merge_shards(bounds, ckpt_dir, out_dir, N)
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
    resolved_path = ckpt_dir / RESOLVED_NAME
    commits = json.load(open(resolved_path)) if resolved_path.exists() else []
    if len(commits) > 1:
        print(f"WARNING: shards come from {len(commits)} model commits: {commits}")
    manifest = {
        **cfg,                       # every run_config key, same names
        "model_commit": commits[0] if len(commits) == 1 else (commits or None),
        "tokenizer_commit": resolved_commit(args.tokenizer or args.model),
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
    meta_df.to_parquet(out_dir / META_FILE, index=False)
    with open(out_dir / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    # Human-readable copy beside the parquet, matching the published clouds.
    meta_df.to_csv(out_dir / "metadata.csv", index=False)
    from manifold_persona.provenance import write_stamp
    write_stamp(out_dir)
    shutil.rmtree(ckpt_dir)          # clean up shards once the final save succeeded
    print(f"wrote {out_dir}  ({time.time()-t0:.0f}s total)  · removed {ckpt_dir}")
    print("response stats:", json.dumps(manifest["response_stats"]))


if __name__ == "__main__":
    main()
