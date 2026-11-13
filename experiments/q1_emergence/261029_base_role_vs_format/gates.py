"""Pod B integrity gates (THRESHOLDS.md, "Integrity gates"), run on the pod after gen_extract.py.

- G1 tokens and grid: every grid key has one row; every row passes gen_extract.check_row (rows.check_row, and
  the kept ids are the format's cut of the raw ids); both state arrays have the model's shape and a float
  dtype; valid rows have finite states, invalid rows NaN states; quote states are finite exactly where
  reply_extras.jsonl has a quote span.
- G2 valid rate: >= 95% valid replies in every role x format cell.
- D1 support (report): complete questions, lost prompts by role x format and by question, beside the valid
  rate. The full run's stop rule is in gen_extract.run.
- G3 seeds: whole original batches (same composition, same batch size) are generated again until at least
  40 rows are covered; the raw ids must be identical. The same rows one at a time: agreement reported only.
- G4 reconstruction: a fresh forward pass of the stored prompt and kept ids, in other batches, reproduces the
  stored states at hidden states 9, 17 and 23 on every valid row (rows.c0_stats, the E3 C0 gate, centred over
  valid rows only). Invalid rows: valid flag false, no valid text in the kept ids, NaN state. The quoted-speech
  states of story and description rows are rebuilt the same way: each row's relative norm difference must be
  at most QUOTE_MAX_REL and their median at most QUOTE_MEDIAN_REL (the centred C0 statistics are reported); the quote states of interview and script rows
  must equal their stored reply states at the extra layers.
- G5 snapshot: model revision, tokenizer file hashes and template hash match the manifest.
- raw_read.md: 10 random replies per format, to read before any metric.

  python gates.py --run run_canary      # exit code 3 if a gate fails
"""
import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sibling(name):
    """A pod B script by path, under a pod-unique module name: pods A and B both have gen_extract.py and gates.py,
    and one pytest session imports both."""
    key = f"base_role_vs_format_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


G = sibling("gen_extract")
R = G.R
MIN_VALID = 0.95
QUOTE_MAX_REL = 0.05  # per quote row, |stored - rebuilt| / |rebuilt|; the second canary's largest was 0.011
QUOTE_MEDIAN_REL = 0.02  # median over quote rows, as the C0 rule's median relative norm; catches a small scale
N_SEED_ROWS = 40
N_RAW_READ = 10  # per format


def finite_nan(acts, n):
    """Per row: all finite, all NaN; one read of each block of a memory-mapped array."""
    import numpy as np
    finite, nan = np.zeros(n, bool), np.zeros(n, bool)
    for s in range(0, n, 256):
        block = np.asarray(acts[s:min(s + 256, n)])
        finite[s:s + 256] = np.isfinite(block).all(axis=(1, 2))
        nan[s:s + 256] = np.isnan(block).all(axis=(1, 2))
    return finite, nan


def g1(run, tok, n_states, n_extra, width):
    import numpy as np
    rows, grid, acts, quotes, extras = run["rows"], run["grid"], run["acts"], run["quotes"], run["extras"]
    shapes_ok = (acts.ndim == quotes.ndim == 3 and acts.shape[1:] == (n_states, width)
                 and quotes.shape[1:] == (n_extra, width) and acts.dtype in (np.float16, np.float32)
                 and quotes.dtype == acts.dtype)
    if not shapes_ok:  # an empty layer axis would make "all finite" and "all NaN" both true
        return {"acts_shape": list(acts.shape), "quotes_shape": list(quotes.shape), "dtype": str(acts.dtype),
                "want": [n_states, n_extra, width], "pass": False}
    split = R.check_rows(rows, grid)
    errors = []
    for r in rows:
        try:
            G.check_row(tok, r)
        except ValueError as e:
            errors.append(str(e))
    span_err = [R.row_key(r) for r, x in zip(rows, extras) if x["quote_span"] != G.quote_span(tok, r)]
    n = min(len(rows), len(acts), len(quotes))
    valid = np.array([r["valid"] for r in rows])[:n]
    has_q = np.array([x["quote_span"] is not None for x in extras])[:n]
    finite, nan = finite_nan(acts, n)
    qfinite, qnan = finite_nan(quotes, n)
    g = dict(n_rows=len(rows), n_state_rows=len(acts), n_quote_rows=len(quotes), absent=len(split["absent"]),
             invalid=len(split["invalid"]), row_errors=len(errors), first_errors=errors[:5],
             quote_span_errors=len(span_err), valid_rows_not_finite=int((valid & ~finite).sum()),
             invalid_rows_not_nan=int((~valid & ~nan).sum()),
             quote_rows_wrong=int((has_q & ~qfinite).sum() + (~has_q & ~qnan).sum()))
    g["acts_shape"], g["quotes_shape"] = list(acts.shape), list(quotes.shape)
    g["pass"] = len(acts) == len(quotes) == len(rows) and not (
        g["absent"] or g["row_errors"] or g["quote_span_errors"] or g["valid_rows_not_finite"]
        or g["invalid_rows_not_nan"] or g["quote_rows_wrong"])
    return g


def g2(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[f"{r['role']}/{r['format']}"].append(r["valid"])
    rates = {k: sum(v) / len(v) for k, v in sorted(cells.items())}
    return {"rule": f">= {MIN_VALID} valid per role x format", "rates": rates, "pass": min(rates.values()) >= MIN_VALID}


def d1_support(rows, grid):
    s = G.support({R.row_key(r): r["valid"] for r in rows}, grid)
    return dict(s, n_complete=len(s["complete_questions"]), valid_rate=sum(r["valid"] for r in rows) / len(rows),
                note="report only; the full run stops in gen_extract.run once more than "
                     f"{G.MAX_LOST_QUESTIONS} questions have a lost prompt")


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


def rebuild(model, tok, seqs, spans, layers, batch_size):
    import numpy as np
    return np.concatenate([R.pool_spans(G.Backbone(model), seqs[s:s + batch_size], spans[s:s + batch_size],
                                        tok.pad_token_id, layers) for s in range(0, len(seqs), batch_size)])


def g4(run, model, tok, batch_size, layers=R.C0_GATE["layers"]):
    """Stored vs recomputed states on every valid row, batched in grid order (not the run's batches)."""
    import numpy as np
    rows, acts, quotes, extras = run["rows"], run["acts"], run["quotes"], run["extras"]
    idx = [i for i, r in enumerate(rows) if r["valid"]]
    C = rebuild(model, tok, [rows[i]["prompt_ids"] + rows[i]["kept_ids"] for i in idx], [rows[i]["span"] for i in idx],
                layers, batch_size)
    cells = {f"L{L}": R.c0_stats(np.asarray(acts[idx, L], dtype=np.float64), C[:, k]) for k, L in enumerate(layers)}
    bad_invalid = [R.row_key(r) for i, r in enumerate(rows) if not r["valid"] and
                   (R.is_valid(tok, r["kept_ids"]) or not np.isnan(np.asarray(acts[i])).all())]
    res = {"rule": R.C0_GATE, "cells": cells, "n_invalid": len(rows) - len(idx), "invalid_rows_wrong": bad_invalid,
           "pass": all(c["pass"] for c in cells.values()) and not bad_invalid}
    qlayers = G.depth_layers(model.config.num_hidden_layers)
    si = [i for i, (r, x) in enumerate(zip(rows, extras)) if x["quote_span"] is not None
          and r["format"] in G.SPEECH_FORMATS]
    res["speech_quote_rows_unequal"] = int(sum(not np.array_equal(np.asarray(quotes[i]), np.asarray(acts[i, list(qlayers)]))
                                              for i in si))
    qi = [i for i, (r, x) in enumerate(zip(rows, extras))
          if x["quote_span"] is not None and r["format"] not in G.SPEECH_FORMATS]
    if qi:  # a per-row rule: it works for one row, and for identical quotes, where a centred cosine is undefined
        Q = rebuild(model, tok, [rows[i]["prompt_ids"] + rows[i]["kept_ids"] for i in qi],
                    [extras[i]["quote_span"] for i in qi], qlayers, batch_size)
        S = np.asarray(quotes[qi], dtype=np.float64)
        rel = np.linalg.norm(S - Q, axis=2) / np.linalg.norm(Q, axis=2)
        res["quotes"] = {f"L{L}": {"n": len(qi), "max_rel_norm": float(rel[:, k].max()),
                                   "median_rel_norm": float(np.median(rel[:, k])),
                                   "rule": f"max <= {QUOTE_MAX_REL}, median <= {QUOTE_MEDIAN_REL}",
                                   "pass": bool(rel[:, k].max() <= QUOTE_MAX_REL
                                                and np.median(rel[:, k]) <= QUOTE_MEDIAN_REL)}
                         for k, L in enumerate(qlayers)}
        if len(qi) >= 2:
            res["quotes_c0_reported"] = {f"L{L}": R.c0_stats(S[:, k], Q[:, k]) for k, L in enumerate(qlayers)}
    res["pass"] = res["pass"] and not res["speech_quote_rows_unequal"] and all(
        c["pass"] for c in res.get("quotes", {}).values())
    return res


def g5(manifest, model_commit, tokenizer_files, attn=G.ATTN):
    got = dict(revision=model_commit, tokenizer_files=tokenizer_files, templates=G.templates_hash(),
               attn_implementation=attn)
    want = {k: manifest.get(k) for k in got}
    return {"mismatch": {k: (want[k], got[k]) for k in got if got[k] != want[k]}, "pass": got == want}


def raw_read(rows, tok, path, n=N_RAW_READ):
    rng = random.Random(0)
    out = [f"# Raw read: {n} random replies per format (read before any metric)\n\n"]
    for fmt in G.FORMATS:
        idx = [i for i, r in enumerate(rows) if r["format"] == fmt]
        out.append(f"# {fmt}\n\n")
        for i in sorted(rng.sample(idx, min(n, len(idx)))):
            r = rows[i]
            out.append(f"## {R.row_key(r)}  stop={r['stop_reason']}  valid={r['valid']}\n\n"
                       f"```\n{tok.decode(r['prompt_ids'])}{r['reply']}\n```\n\n")
    path.write_text("".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch_size", type=int, default=None, help="default: the run's batch size")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else HERE / args.run
    run = G.load_run(run_dir)
    manifest = run["manifest"]
    bs = args.batch_size or manifest["batch_size"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.MODEL, revision=manifest["revision"])
    model = G.load_model(manifest["revision"])
    n_layers = model.config.num_hidden_layers
    raw_read(run["rows"], tok, run_dir / "raw_read.md")
    res = {"G1": g1(run, tok, n_layers + 1, len(G.depth_layers(n_layers)), model.config.hidden_size),
           "G2": g2(run["rows"])}
    res["G5"] = g5(manifest, getattr(model.config, "_commit_hash", None), G.tokenizer_hashes(),
                   model.config._attn_implementation)
    res["G3"] = g3(run["rows"], model, tok, manifest["batch_size"])
    res["G4"] = g4(run, model, tok, bs)
    ok = all(v["pass"] for v in res.values())
    report = dict(passed=ok, gates=res, d1_support=d1_support(run["rows"], run["grid"]))
    (run_dir / "gates.json").write_text(json.dumps(report, indent=1))
    for k, v in res.items():
        print(k, "PASS" if v["pass"] else "FAIL")
    s = report["d1_support"]
    print(f"D1 support (report): {s['n_complete']} of {s['n_questions']} questions complete, "
          f"{s['n_lost_prompts']} lost prompts, valid rate {s['valid_rate']:.4f}")
    print("all gates", "PASSED" if ok else "FAILED", "- read raw_read.md before any metric")
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
