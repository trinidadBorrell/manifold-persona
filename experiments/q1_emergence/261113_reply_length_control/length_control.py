"""Reply length control (rules: THRESHOLDS.md). Each stage re-reads its own stored replies (prompt ids + kept ids,
pod A's chat frame) with its own model and saves the mean over the first 16 and the first 32 kept tokens. The full
reply mean from the same pass must reproduce the stored mean (pod A's C0 gate). Forward passes only.

  python -I length_control.py extract --stage sft --run /out/own_replies/sft_full --out /out/length_control/sft
  python -I length_control.py extract --stage sft --run ... --out ... --limit 64       # smoke
"""
import argparse
import hashlib
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROWS_PY = REPO / "src/manifold_persona/rows.py"
STAGES = {e["stage"]: e for e in json.loads((HERE.parent / "261112_coinflip_stages/stages.json").read_text())["7b"]}
CUTS = {"first16": 16, "first32": 32}
STATES = [0, 9, 17, 23]
CHAT_TEMPLATE_SHA_PREFIX = "fe689ffb"
F16_MAX = 65504.0
SUPPORT_MIN = 0.5  # pod C F4 rule: every role keeps at least half its rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rows_module():
    spec = importlib.util.spec_from_file_location("_mp_rows", ROWS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spans_for(row):
    """The stored pooling span (all kept tokens) and each cut's span; an empty span (NaN state) when the reply is
    invalid or shorter than the cut."""
    a, b = row["span"]
    if [a, b] != [len(row["prompt_ids"]), len(row["prompt_ids"]) + len(row["kept_ids"])]:
        raise ValueError(f"stored span {row['span']} is not prompt + kept ids")
    if not row["valid"]:
        return dict(full=(a, a), **{c: (a, a) for c in CUTS})
    return dict(full=(a, b), **{c: (a, a + n) if b - a >= n else (a, a) for c, n in CUTS.items()})


def pool_multi(model, seqs, spans, pad_id, layers):
    """rows.pool_spans with several spans per row from one forward pass (same padding, mask and mean).
    spans: {name: [(a, b) per row]}; layers: {name: [hidden state indices]}. Returns {name: float32 array}."""
    import torch
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, x in enumerate(seqs):
        for name in spans:
            a, b = spans[name][j]
            if not 0 <= a <= b <= len(x):
                raise ValueError(f"row {j}: span {name} {(a, b)} outside its sequence of {len(x)}")
        ids[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = 1
    with torch.no_grad():
        hs = model(ids.to(model.device), attention_mask=mask.to(model.device), output_hidden_states=True).hidden_states
    out = {}
    for name, idx in layers.items():
        o = np.full((len(seqs), len(idx), hs[0].shape[-1]), np.nan, dtype=np.float32)
        for j, (a, b) in enumerate(spans[name]):
            if b > a:
                o[j] = torch.stack([hs[l][j, a:b].float().mean(0) for l in idx]).cpu().numpy()
        out[name] = o
    return out


def extract_core(model, pad_id, rows, batch_size, check_layers, log=print):
    """States for every row, in the given row order. Rows are batched by sequence length inside."""
    n = len(rows)
    sp = [spans_for(r) for r in rows]
    layers = dict(full=list(check_layers), **{c: STATES for c in CUTS})
    res = {name: np.full((n, len(L), model.config.hidden_size), np.nan, dtype=np.float32) for name, L in layers.items()}
    order = sorted(range(n), key=lambda i: (len(rows[i]["prompt_ids"]) + len(rows[i]["kept_ids"]), i))
    nb = -(-n // batch_size)
    for b in range(nb):
        idx = order[b * batch_size:(b + 1) * batch_size]
        got = pool_multi(model, [rows[i]["prompt_ids"] + rows[i]["kept_ids"] for i in idx],
                         {name: [sp[i][name] for i in idx] for name in layers}, pad_id, layers)
        for name in layers:
            res[name][idx] = got[name]
        if b % 50 == 0 or b == nb - 1:
            log(f"batch {b + 1}/{nb}")
    return res


def support(rows):
    """Per cut and role: rows with at least the cut's number of kept tokens (valid rows only count)."""
    out = {}
    for c, k in CUTS.items():
        tot, ok = defaultdict(int), defaultdict(int)
        for r in rows:
            tot[r["role"]] += 1
            ok[r["role"]] += bool(r["valid"] and len(r["kept_ids"]) >= k)
        share = {role: ok[role] / tot[role] for role in sorted(tot)}
        low = min(share, key=share.get)
        out[c] = dict(tokens=k, share=share, lowest=[low, share[low]], available=all(v >= SUPPORT_MIN for v in share.values()),
                      rule=f"every role keeps at least {SUPPORT_MIN} of its rows")
    return out


def check(R, stored, full, valid, check_layers):
    """Pod A's C0 gate per hidden state on the valid rows: stored reply mean vs the re-read full mean.
    stored and full: [rows, len(check_layers), width], the same rows and states."""
    cells = {f"L{L}": R.c0_stats(np.asarray(stored[valid, k], dtype=np.float64), full[valid, k].astype(np.float64))
             for k, L in enumerate(check_layers)}
    return dict(rule=R.C0_GATE, cells=cells, passed=all(c["pass"] for c in cells.values()))


def load_model(entry, device):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = entry["repo"], entry["revision"]
    if sha256(hf_hub_download(repo, "tokenizer.json", revision=rev)) != entry["tokenizer_json_sha256"]:
        raise SystemExit("tokenizer.json differs from stages.json")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if not hashlib.sha256(tok.chat_template.encode()).hexdigest().startswith(CHAT_TEMPLATE_SHA_PREFIX):
        raise SystemExit("chat template differs")
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    if getattr(model.config, "_commit_hash", None) != rev:
        raise SystemExit("loaded commit differs from the pinned revision")
    return model, tok


def cmd_extract(a):
    import torch
    import transformers
    t0 = time.time()
    R = rows_module()
    entry, run, out = STAGES[a.stage], Path(a.run), Path(a.out)
    sums = dict(reversed(line.split("  ", 1)) for line in (run / "SHA256SUMS.txt").read_text().splitlines())
    for name in ("rows.jsonl", "reply_mean_acts.npy", "manifest.json"):
        if sha256(run / name) != sums[name]:
            raise SystemExit(f"{name} does not match the run's SHA256SUMS.txt")
    man_in = json.loads((run / "manifest.json").read_text())
    if (man_in["model"], man_in["revision"]) != (entry["repo"], entry["revision"]):
        raise SystemExit("the replies were not written by this stage's model and revision")
    rows = [json.loads(line) for line in (run / "rows.jsonl").open()]
    stored = np.load(run / "reply_mean_acts.npy", mmap_mode="r")
    if stored.shape[0] != len(rows):
        raise SystemExit("stored states and rows differ in length")
    sel = sorted(random.Random(0).sample(range(len(rows)), a.limit)) if a.limit else list(range(len(rows)))
    rows_sel = [rows[i] for i in sel]
    model, tok = load_model(entry, a.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    check_layers = list(R.C0_GATE["layers"])
    res = extract_core(model, pad, rows_sel, a.batch_size, check_layers, log=lambda s: print(s, flush=True))
    valid = np.array([r["valid"] for r in rows_sel])
    stored_chk = np.stack([np.asarray(stored[sel, L]) for L in check_layers], 1)  # reads only the check states
    chk = check(R, stored_chk, res["full"], valid, check_layers)
    print("check:", json.dumps({k: dict(min_cos=round(v["min_cos"], 5), median_cos=round(v["median_cos"], 5),
                                        median_rel_norm=round(v["median_rel_norm"], 5), passed=v["pass"])
                                for k, v in chk["cells"].items()}), flush=True)
    sup = support(rows_sel)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for c in CUTS:
        x = res[c]
        fin = np.isfinite(x).all(axis=(1, 2))
        if np.isnan(x[~fin]).sum() != x[~fin].size or (fin.any() and np.abs(x[fin]).max() > F16_MAX):
            raise SystemExit(f"{c}: partly non-finite rows or states beyond float16")
        np.save(out / f"{c}_acts.npy", x.astype(np.float16))
        files.append(f"{c}_acts.npy")
        sup[c]["n_finite_rows"] = int(fin.sum())
    (out / "keys.json").write_text(json.dumps(dict(row_index=sel, keys=[R.row_key(r) for r in rows_sel])))
    (out / "support.json").write_text(json.dumps(sup, indent=1))
    (out / "check.json").write_text(json.dumps(chk, indent=1))
    files += ["keys.json", "support.json", "check.json"]
    man = dict(stage=a.stage, model=entry["repo"], revision=entry["revision"], dtype="bfloat16",
               attn_implementation="sdpa", cuts=CUTS, states=STATES, check_states=check_layers, n_rows=len(sel),
               limit=a.limit, batch_size=a.batch_size, check_passed=chk["passed"],
               available={c: sup[c]["available"] for c in CUTS},
               act_rows="row i is rows.jsonl line keys.json row_index[i]; NaN where the reply is shorter than the cut",
               inputs=dict(run=str(run), sha256={k: sums[k] for k in ("rows.jsonl", "reply_mean_acts.npy", "manifest.json")}),
               code=dict(length_control_py=sha256(Path(__file__)), rows_py=sha256(ROWS_PY)),
               versions=dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                             numpy=np.__version__,
                             device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else a.device),
               seconds=round(time.time() - t0, 1))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    files.append("manifest.json")
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(out / n)}  {n}\n" for n in files))
    print(f"done: {a.stage}, {len(sel)} rows, check {'PASS' if chk['passed'] else 'FAIL'}, "
          f"available {man['available']}, {man['seconds']}s", flush=True)
    if not chk["passed"]:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--stage", required=True, choices=["instruct", "sft", "dpo"])
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch_size", type=int, default=32)
    s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    main()
