"""E3 Part 2 context controls (THRESHOLDS.md, change of 16/10/2026): forward passes only, no generation.

For every usable shared-question row of a run (valid, non-empty reply), all 33 hidden states, float16. Same model,
revision, dtype and right padding as gen_extract.py. Output rows follow replies.jsonl order (ctx_rows.npy holds
the row positions).
  c0       prompt ids + kept reply ids; mean over the reply positions (re-computes the stored reply state)
  c0first  the same pass; mean over the first min(32, n) reply positions
  c1       the same pass; the state at index prompt_length - 1 (the "Guest:" token)
  c2       EOS + header + the test turn only (no description, no example turns, no filler) + the same kept
           reply ids; mean over the reply positions
  c3       the full prompt + one fixed neutral continuation (same ids for every row); mean over its positions
Then the C0 gate against the stored reply states (slim copy: hidden states 9, 17, 23), per condition and state:
centred cosine min >= 0.95 and median >= 0.99, median relative norm difference <= 0.02.

    python context_extract.py --run run_main --out context_smoke --limit 96   # smoke test with the gate
    python context_extract.py --run run_main
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import gen_extract as ge

HERE = Path(__file__).resolve().parent
FEATS = ("c0", "c0first", "c1", "c2", "c3")
FIRST = 32
NEUTRAL = (" That is a good question. I think the answer depends on the situation, and I would want to think"
           " about it carefully before I say more.")
GATE = dict(layers=(9, 17, 23), min_cos=0.95, median_cos=0.99, median_rel_norm=0.02)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def c2_prompt(item):
    return ge.EOS + ge.HEADER + "\n\n" + ge.block(None, item["question"])[1]


def used_rows(rows, limit):
    pos = [k for k, r in enumerate(rows) if r["shared_q"] and r["valid"] and r["reply"].strip()]
    if limit:
        pos = pos[::max(1, len(pos) // limit)][:limit]  # spread over the conditions
    return pos


def pooled(H, j, a, b):
    """H: the model's tuple of 33 [B, T, d] hidden states on the GPU. Mean over positions a..b-1 of row j, every
    hidden state. Indexing the tuple avoids a stacked copy of all states (it doubled peak GPU memory)."""
    import torch
    return torch.stack([h[j, a:b].float().mean(0) for h in H]).cpu().numpy().astype(np.float16)


def forward(model, seqs, pad):
    import torch
    ids, mask = ge.pad_batch(seqs, pad, left=False)  # right padding: positions start at 0
    with torch.no_grad():
        hs = model(ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
    return ids, hs


def extract(run_dir, out, limit, batch_size):
    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(run_dir / "replies.jsonl")]
    items = {json.loads(l)["i"]: json.loads(l) for l in open(run_dir / "items.jsonl")}
    pos = used_rows(rows, limit)
    tok = AutoTokenizer.from_pretrained(ge.MODEL, revision=ge.REVISION)
    model = AutoModelForCausalLM.from_pretrained(ge.MODEL, revision=ge.REVISION,
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    pad, eos = tok.pad_token_id, tok.eos_token_id
    n_hs, d = model.config.num_hidden_layers + 1, model.config.hidden_size
    neutral = tok(NEUTRAL, add_special_tokens=False).input_ids
    out.mkdir(parents=True, exist_ok=True)
    M = {f: open_memmap(out / f"ctx_{f}.npy", mode="w+", dtype=np.float16, shape=(len(pos), n_hs, d)) for f in FEATS}
    t = time.time()
    for s in range(0, len(pos), batch_size):
        chunk = [rows[k] for k in pos[s:s + batch_size]]
        p_ids, c2_ids, kept = [], [], []
        for r in chunk:
            it = items[r["i"]]
            assert (it["role"], it["cond"], it["question_idx"], it["wording"]) == \
                   (r["role"], r["cond"], r["question_idx"], r["wording"]), r["i"]
            p = tok(it["prompt"], add_special_tokens=False).input_ids
            q = tok(c2_prompt(it), add_special_tokens=False).input_ids
            assert len(p) == r["prompt_tokens"] and p[0] == eos and p.count(eos) == 1, r["i"]
            assert q[0] == eos and q.count(eos) == 1 and len(r["reply_ids"]) == r["reply_tokens"] > 0, r["i"]
            p_ids.append(p), c2_ids.append(q), kept.append(r["reply_ids"])
        ids, H = forward(model, [p + k for p, k in zip(p_ids, kept)], pad)
        for j, (p, k) in enumerate(zip(p_ids, kept)):
            a, b = len(p), len(p) + len(k)
            assert ids[j, a:b].tolist() == k
            M["c0"][s + j] = pooled(H, j, a, b)
            M["c0first"][s + j] = pooled(H, j, a, a + min(FIRST, len(k)))
            M["c1"][s + j] = pooled(H, j, a - 1, a)
        del H
        ids, H = forward(model, [p + neutral for p in p_ids], pad)
        for j, p in enumerate(p_ids):
            M["c3"][s + j] = pooled(H, j, len(p), len(p) + len(neutral))
        del H
        ids, H = forward(model, [q + k for q, k in zip(c2_ids, kept)], pad)
        for j, (q, k) in enumerate(zip(c2_ids, kept)):
            assert ids[j, len(q):len(q) + len(k)].tolist() == k
            M["c2"][s + j] = pooled(H, j, len(q), len(q) + len(k))
        del H
        print(f"{s + len(chunk)}/{len(pos)} {round(time.time() - t)}s", flush=True)
    for m in M.values():
        m.flush()
    for f, m in M.items():
        if not np.isfinite(m[:]).all():
            raise SystemExit(f"non-finite values in ctx_{f}.npy")
    np.save(out / "ctx_rows.npy", np.array(pos))
    import platform
    import transformers
    json.dump(dict(model=ge.MODEL, revision=ge.REVISION, run=run_dir.name, limit=limit, batch_size=batch_size,
                   n_rows=len(pos), n_hidden_states=n_hs, first=FIRST, neutral=NEUTRAL, neutral_ids=neutral,
                   run_fingerprint=(run_dir / "fingerprint.txt").read_text().strip(),
                   code_sha256=sha256(Path(__file__)), gen_extract_sha256=sha256(Path(ge.__file__)),
                   files={f"ctx_{f}.npy": sha256(out / f"ctx_{f}.npy") for f in FEATS},
                   versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                 transformers=transformers.__version__, numpy=np.__version__),
                   seconds=round(time.time() - t)),
              open(out / "manifest.json", "w"), indent=1)
    return rows, pos


def gate_cell(S, C):
    """Stored (S) vs re-computed (C) rows of one condition and state, centred on the stored condition mean."""
    mu = S.mean(0)
    a, b = S - mu, C - mu
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    rel = np.linalg.norm(C - S, axis=1) / np.linalg.norm(S, axis=1)
    r = dict(n=len(S), min_cos=float(cos.min()), median_cos=float(np.median(cos)),
             median_rel_norm=float(np.median(rel)), max_rel_norm=float(rel.max()))
    r["pass"] = bool(r["min_cos"] >= GATE["min_cos"] and r["median_cos"] >= GATE["median_cos"]
                     and r["median_rel_norm"] <= GATE["median_rel_norm"])
    return r


def gate(run_dir, out, rows, pos):
    """C0 against the stored reply states, per condition and gate state, on every extracted row."""
    meta = json.load(open(run_dir / "reply_mean_acts_slim.json"))
    if sha256(run_dir / meta["file"]) != meta["sha256"]:
        raise SystemExit("slim reply states do not match their recorded SHA-256")
    S_all = np.load(run_dir / meta["file"], mmap_mode="r")
    c0 = np.load(out / "ctx_c0.npy", mmap_mode="r")
    conds = np.array([rows[k]["cond"] for k in pos])
    res, ok = {}, True
    for L in GATE["layers"]:
        S = np.asarray(S_all[pos, meta["layers"].index(L)], dtype=np.float64)
        C = np.asarray(c0[:, L], dtype=np.float64)
        for c in np.unique(conds):
            r = gate_cell(S[conds == c], C[conds == c])
            ok &= r["pass"]
            res[f"{c}/L{L}"] = r
            print(f"gate {c:10s} L{L:2d} n {r['n']:5d} min cos {r['min_cos']:.5f} median {r['median_cos']:.5f} "
                  f"rel {r['median_rel_norm']:.5f} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    json.dump(dict(rule=GATE, passed=ok, cells=res, rows_sha256=hashlib.sha256(np.array(pos).tobytes()).hexdigest()),
              open(out / "gate.json", "w"), indent=1)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_main")
    ap.add_argument("--out", default="context")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gate_only", action="store_true")
    a = ap.parse_args()
    run_dir = HERE / a.run
    out = run_dir / a.out
    if a.gate_only:
        rows = [json.loads(l) for l in open(run_dir / "replies.jsonl")]
        pos = np.load(out / "ctx_rows.npy").tolist()
    else:
        if (out / "manifest.json").exists():
            raise SystemExit(f"{out} already holds an extraction; use another --out")
        rows, pos = extract(run_dir, out, a.limit, a.batch_size)
    ok = gate(run_dir, out, rows, pos)
    print("C0 gate", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
