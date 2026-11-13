"""Pod A integrity gates G1-G5 and the raw-read sample (THRESHOLDS.md), run on the pod after gen_extract.py.

- G1 tokens and grid: every grid key has one row; every row passes rows.check_row; valid rows have finite
  states, invalid rows NaN states.
- G2 valid rate: >= 95% valid replies for every role.
- G3 seeds: whole original batches (same composition, same batch size) are generated again until at least
  40 rows are covered; the raw ids must be identical. The same rows one at a time: agreement reported only.
- G4 reconstruction: a fresh forward pass of the stored prompt and kept ids, in other batches, reproduces the
  stored states at hidden states 9, 17 and 23 on every valid row (rows.c0_stats, the E3 C0 gate).
- G5 snapshot: model revision, tokenizer file hashes and chat template hash match the manifest.
- raw_read.md: 20 random transcripts to read before any metric.

  python gates.py --run run_full      # exit code 3 if a gate fails
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from manifold_persona import rows as R

import gen_extract as G

HERE = Path(__file__).resolve().parent
MIN_VALID = 0.95
N_SEED_ROWS = 40
N_RAW_READ = 20


def load_run(run_dir):
    """Rows, grid and states of a finished run, after the SHA256SUMS check."""
    import numpy as np
    for line in (run_dir / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if R.file_sha256(run_dir / name) != h:
            raise SystemExit(f"{name} does not match SHA256SUMS.txt")
    grid = json.loads((run_dir / "grid.json").read_text())["keys"]
    rows = [json.loads(l) for l in open(run_dir / "rows.jsonl")]
    if [R.row_key(r) for r in rows] != grid:
        raise SystemExit("rows.jsonl is not in grid order")
    acts = np.load(run_dir / "reply_mean_acts.npy", mmap_mode="r")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return rows, grid, acts, manifest


def g1(rows, grid, acts, tok):
    import numpy as np
    split = R.check_rows(rows, grid)
    errors = []
    for r in rows:
        try:
            R.check_row(r, {tok.eos_token_id, tok.pad_token_id}, tok, G.MARKERS)
        except ValueError as e:
            errors.append(str(e))
    valid = np.array([r["valid"] for r in rows])
    finite, nan = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    n = min(len(rows), len(acts))
    for s in range(0, n, 256):  # one read of each block
        block = np.asarray(acts[s:min(s + 256, n)])
        finite[s:s + 256] = np.isfinite(block).all(axis=(1, 2))
        nan[s:s + 256] = np.isnan(block).all(axis=(1, 2))
    g = dict(n_rows=len(rows), n_state_rows=len(acts), absent=len(split["absent"]), invalid=len(split["invalid"]),
             row_errors=len(errors), first_errors=errors[:5],
             valid_rows_not_finite=int((valid & ~finite).sum()), invalid_rows_not_nan=int((~valid & ~nan).sum()))
    g["pass"] = len(acts) == len(rows) and not (g["absent"] or g["row_errors"] or g["valid_rows_not_finite"]
                                                 or g["invalid_rows_not_nan"])
    return g


def g2(rows):
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r["valid"])
    rates = {k: sum(v) / len(v) for k, v in sorted(by_role.items())}
    return {"rule": f">= {MIN_VALID} valid per role", "rates": rates, "pass": min(rates.values()) >= MIN_VALID}


def run_batches(rows, batch_size):
    """The original batches as row indices: gen_extract sorted items by (prompt length, key)."""
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i]["prompt_ids"]), R.row_key(rows[i])))
    return [order[s:s + batch_size] for s in range(0, len(order), batch_size)]


def upto_stop(ids, eos):
    return ids[:ids.index(eos) + 1] if eos in ids else ids


def regenerate(model, tok, rows, idx):
    return R.generate_batch(model, [rows[i]["prompt_ids"] for i in idx], [rows[i]["item_seed"] for i in idx],
                            tok.eos_token_id, tok.pad_token_id, G.TEMPERATURE, G.PENALTY, G.MAX_NEW)


def g3(rows, model, tok, batch_size, n=N_SEED_ROWS):
    batches = run_batches(rows, batch_size)
    picked, covered = [], 0
    for b in random.Random(0).sample(range(len(batches)), len(batches)):
        if covered >= n:
            break
        picked.append(b)
        covered += len(batches[b])
    same, alone = [], []
    for b in picked:
        idx = batches[b]
        same += [g == rows[i]["raw_ids"] for i, g in zip(idx, regenerate(model, tok, rows, idx))]
        alone += [upto_stop(regenerate(model, tok, rows, [i])[0], tok.eos_token_id)
                  == upto_stop(rows[i]["raw_ids"], tok.eos_token_id) for i in idx]
    return {"batches": picked, "n_rows": len(same), "identical_same_batches": sum(same),
            "identical_one_at_a_time_reported": sum(alone), "pass": all(same) and len(same) >= n}


def g4(rows, acts, model, tok, batch_size, layers=R.C0_GATE["layers"]):
    """Stored vs recomputed states on every valid row, batched in grid order (not the run's batches)."""
    import numpy as np
    idx = [i for i, r in enumerate(rows) if r["valid"]]
    C = np.concatenate([R.pool_spans(model, [rows[i]["prompt_ids"] + rows[i]["kept_ids"] for i in idx[s:s + batch_size]],
                                     [rows[i]["span"] for i in idx[s:s + batch_size]], tok.pad_token_id, layers)
                        for s in range(0, len(idx), batch_size)])
    cells = {f"L{L}": R.c0_stats(np.asarray(acts[idx, L], dtype=np.float64), C[:, k]) for k, L in enumerate(layers)}
    return {"rule": R.C0_GATE, "cells": cells, "pass": all(c["pass"] for c in cells.values())}


def g5(manifest, tok, model_commit, tokenizer_files):
    got = dict(revision=model_commit, tokenizer_files=tokenizer_files,
               chat_template=hashlib.sha256(tok.chat_template.encode()).hexdigest())
    want = {k: manifest[k] for k in got}
    return {"mismatch": {k: (want[k], got[k]) for k in got if got[k] != want[k]}, "pass": got == want}


def raw_read(rows, tok, path, n=N_RAW_READ):
    pick = sorted(random.Random(0).sample(range(len(rows)), n))
    path.write_text("# Raw read: 20 random transcripts (read before any metric)\n\n" + "".join(
        f"## {R.row_key(rows[i])}  stop={rows[i]['stop_reason']}  valid={rows[i]['valid']}\n\n"
        f"```\n{tok.decode(rows[i]['prompt_ids'])}{rows[i]['reply']}\n```\n\n" for i in pick))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch_size", type=int, default=None, help="default: the run's batch size")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else HERE / args.run
    rows, grid, acts, manifest = load_run(run_dir)
    bs = args.batch_size or manifest["batch_size"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.MODEL, revision=manifest["revision"])
    model = AutoModelForCausalLM.from_pretrained(G.MODEL, revision=manifest["revision"],
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    raw_read(rows, tok, run_dir / "raw_read.md")
    res = {}
    res["G1"], res["G2"] = g1(rows, grid, acts, tok), g2(rows)
    res["G5"] = g5(manifest, tok, getattr(model.config, "_commit_hash", None), G.tokenizer_hashes())
    res["G3"] = g3(rows, model, tok, manifest["batch_size"])
    res["G4"] = g4(rows, acts, model, tok, bs)
    ok = all(v["pass"] for v in res.values())
    (run_dir / "gates.json").write_text(json.dumps(dict(passed=ok, gates=res), indent=1))
    for k, v in res.items():
        print(k, "PASS" if v["pass"] else "FAIL")
    print("all gates", "PASSED" if ok else "FAILED", "- read raw_read.md before any metric")
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
