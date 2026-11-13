"""Pod C extraction: one checkpoint reads every frame; pooled states are saved per checkpoint.

Design: ../261029_stage_persona_plan/DESIGN.md (pod C) and THRESHOLDS.md here (G1-G4).
- Checks first: checkpoints.json passes G2; the frames match their SHA256SUMS and digests; the loaded model is
  the manifest commit; tokenizer.json matches the manifest.
- One forward pass per row and frame (explicit positions from 0, bfloat16, sdpa). Each batch holds sequences of
  one length, so no row is padded. F1 is pooled over the reply and over its first 32 ids (F4) in the same
  pass. F0 (pod A chat frame) runs at Instruct only.
- G1: every row's ids, mask and positions as fed are hashed and must equal the frame file's row hash, so every
  checkpoint reads the same input. Rows without ids are not fed and get NaN states.
- G3 (Instruct only): F0 states reproduce pod A's stored states at hidden states 9, 17, 23 on valid rows
  (rows.c0_stats); invalid rows have NaN states.
- G4: one folder per checkpoint, claimed by a fingerprint; each file is written whole, then renamed; a resume
  skips finished files and refuses other settings. SHA256SUMS.txt at the end. The Mac copies each checkpoint
  off the pod and checks it before the next one loads (run_pod_c.sh).

  python extract.py --frames frames_full --checkpoint base --out out/base
  python extract.py --frames frames_full --checkpoint instruct --out out/instruct \\
      --pod_a_run ../261029_instruct_persona_map/run_full
"""
import argparse
import hashlib
import json
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from manifold_persona import rows as R

import checkpoints as C
import frames as F

HERE = Path(__file__).resolve().parent
FEW = (0, 9, 17, 23)
ALL = tuple(range(33))
FULL_STATES = {("base", "F1", "reply"), ("base", "F2", "reply"), ("instruct", "F1", "reply"), ("instruct", "F2", "reply")}
G3_LAYERS = (9, 17, 23)
F16_MAX = 65504.0
BATCH_SIZE = 64


def layers_for(checkpoint, frame, span):
    """Stored hidden states (THRESHOLDS.md storage: 9, 17, 23 everywhere, all 33 in F1 and F2 at base and
    Instruct). State 0 is kept everywhere for the hidden-state-0 text anchor; F0 always has the G3 states."""
    if frame == "F0":
        return tuple(sorted(set(FEW) | set(G3_LAYERS)))
    return ALL if (checkpoint, frame, span) in FULL_STATES else FEW


def jobs(info, checkpoint):
    """(source, frame) pairs to run at this checkpoint, in a fixed order."""
    return [(s, f) for s in F.SOURCES for f in F.FRAMES
            if f"{s}.{f}" in info["frames"] and (f != "F0" or checkpoint == "instruct")]


def row_hash(ids_row, mask_row, pos_row):
    """sha256 of the fed ids; raises if the mask is not ones then zeros or the positions do not run from 0."""
    n = int(mask_row.sum())
    if not (mask_row[:n] == 1).all() or (mask_row[n:] != 0).any() or not (pos_row[:n] == np.arange(n)).all():
        raise ValueError("mask or positions are not the right-padded layout")
    return hashlib.sha256(np.asarray(ids_row[:n], np.int64).tobytes()).digest()


def pool_batch(model, seqs, spans, pad_id, layers):
    """One forward pass. spans: {name: [(a, b) per row]}; layers: {name: tuple}. Returns ({name: float32
    [rows, len(layers), width]}, [fed row hash])."""
    import torch
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, x in enumerate(seqs):
        ids[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = 1
    pos = torch.arange(T).expand(len(seqs), T)
    hashes = [row_hash(ids[j].numpy(), mask[j].numpy(), pos[j].numpy()) for j in range(len(seqs))]
    dev = model.device
    with torch.no_grad():
        hs = model(ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                   output_hidden_states=True).hidden_states
    out = {}
    for name, sp in spans.items():
        m = torch.zeros((len(seqs), T, 1), dtype=torch.float32)
        for j, (a, b) in enumerate(sp):
            if not 0 <= a <= b <= len(seqs[j]):
                raise ValueError(f"span {(a, b)} outside a sequence of {len(seqs[j])}")
            m[j, a:b] = 1
        m = m.to(dev)
        n = m.sum(1)  # an empty span gives 0 / 0 = NaN
        out[name] = torch.stack([(hs[l].float() * m).sum(1) / n for l in layers[name]], 1).cpu().numpy()
    return out, hashes


def batches(rows, lengths, batch_size):
    """Batches of one sequence length only, so no row is padded: identical inputs (F3) see the same numerics
    except for the batch size."""
    out, cur = [], []
    for j in rows:
        if cur and (len(cur) == batch_size or lengths[j] != lengths[cur[0]]):
            out.append(cur)
            cur = []
        cur.append(j)
    return out + [cur] if cur else out


def save_atomic(path, arr):
    """Write the whole array to a temporary file, fsync, then rename. float16 unless a value exceeds its range."""
    finite = arr[np.isfinite(arr)]
    arr = arr.astype(np.float32 if finite.size and np.abs(finite).max() > F16_MAX else np.float16)
    tmp = path.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return str(arr.dtype)


def run_frame(model, info, z, src, frame, checkpoint, out, pad_id, batch_size=BATCH_SIZE, limit=None):
    """Pool one (source, frame) at one checkpoint and save one file per span, then its record ({p}.json, written
    last). A finished (source, frame) is not run again."""
    p = f"{src}.{frame}"
    names = F.FRAMES[frame]
    paths = {n: out / f"{p}.{n}.npy" for n in names}
    layers = {n: layers_for(checkpoint, frame, n) for n in names}
    if (out / f"{p}.json").exists():
        return dict(json.loads((out / f"{p}.json").read_text()), resumed=True)
    ids, off, want = z[f"{p}.ids"], z[f"{p}.offsets"], z[f"{p}.hash"]
    spans = {n: z[f"{p}.span.{n}"] for n in names}
    n_rows = len(off) - 1
    lengths = np.diff(off)
    fed = [j for j in np.argsort(lengths, kind="stable") if lengths[j] > 0]
    if limit:
        fed = fed[:limit]
    states = {n: np.full((n_rows, len(layers[n]), model.config.hidden_size), np.nan, np.float32) for n in names}
    got = {}
    for rows in batches(fed, lengths, batch_size):
        seqs = [ids[off[j]:off[j + 1]].tolist() for j in rows]
        pooled, hashes = pool_batch(model, seqs, {n: [tuple(spans[n][j]) for j in rows] for n in names}, pad_id, layers)
        for n in names:
            states[n][rows] = pooled[n]
        got.update(zip(rows, hashes))
    bad = [int(j) for j in got if got[j] != want[j].tobytes()]
    if bad:
        raise SystemExit(f"G1: {p} fed ids differ from the frame file in {len(bad)} rows, first {bad[0]}")
    dtypes = {n: save_atomic(paths[n], states[n]) for n in names}
    rec = dict(rows=n_rows, fed=len(fed), layers={n: list(v) for n, v in layers.items()}, dtypes=dtypes,
               g1_rows_checked=len(got))
    tmp = out / f"{p}.json.tmp"
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out / f"{p}.json")
    return rec


def g3(out, pod_a_run, info):
    """Pod A's stored states against F0 at Instruct, on valid rows; invalid rows must be NaN."""
    pod_a_run = Path(pod_a_run)
    sums = {name: h for h, name in (l.split("  ", 1) for l in (pod_a_run / "SHA256SUMS.txt").read_text().splitlines())}
    if sums["rows.jsonl"] != info["inputs"]["pod_a"]["sha256"]["rows.jsonl"]:
        raise SystemExit("G3: the pod A run is not the one the frames were built from")
    if R.file_sha256(pod_a_run / "reply_mean_acts.npy") != sums["reply_mean_acts.npy"]:
        raise SystemExit("G3: pod A states do not match their SHA256SUMS")
    stored = np.load(pod_a_run / "reply_mean_acts.npy", mmap_mode="r")
    mine = np.load(out / "pod_a.F0.reply.npy", mmap_mode="r")
    lay = layers_for("instruct", "F0", "reply")
    valid = np.isfinite(mine[:, lay.index(G3_LAYERS[0])]).all(1)
    if valid.sum() != info["frames"]["pod_a.F0"]["with_ids"] or np.isfinite(stored[~valid]).any():
        raise SystemExit("G3: invalid rows are not NaN in both pod A and F0")
    res = {str(l): R.c0_stats(stored[valid, l], mine[valid, lay.index(l)]) for l in G3_LAYERS}
    return dict(per_state=res, n_valid=int(valid.sum()), n_invalid=int((~valid).sum()),
                pass_=all(r["pass"] for r in res.values()))


def check_frames(frames_dir):
    for line in (frames_dir / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if R.file_sha256(frames_dir / name) != h:
            raise SystemExit(f"{name} does not match {frames_dir}/SHA256SUMS.txt")
    info, z, _ = F.load(frames_dir)
    for p, fi in info["frames"].items():
        if hashlib.sha256(z[f"{p}.hash"].tobytes()).hexdigest() != fi["digest"]:
            raise SystemExit(f"{p}: row hashes do not match their digest")
    return info, z


def load_model(entry):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = entry["repo"], entry["revision"]
    if R.file_sha256(hf_hub_download(repo, "tokenizer.json", revision=rev)) != entry["tokenizer_files"]["tokenizer.json"]:
        raise SystemExit("G2: tokenizer.json differs from checkpoints.json")
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", weights_only=True).cuda().eval()
    commit = getattr(model.config, "_commit_hash", None)
    if commit != rev:
        raise SystemExit(f"G2: loaded commit {commit} is not {rev}")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, choices=[ck["name"] for ck in C.CHECKPOINTS])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pod_a_run", type=Path, help="Instruct only: the pod A run for G3")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: the first n rows of each frame")
    ap.add_argument("--delete_weights", action="store_true", help="remove this revision from the HF cache at the end")
    a = ap.parse_args()
    frames_dir = a.frames if a.frames.is_absolute() else HERE / a.frames
    out = a.out if a.out.is_absolute() else HERE / a.out
    manifest = json.loads(C.MANIFEST.read_text())
    if problems := C.check(manifest):
        raise SystemExit("G2: " + "; ".join(problems))
    entry = next(e for e in manifest["entries"] if e["name"] == a.checkpoint)
    if a.checkpoint == "instruct" and a.pod_a_run is None:
        raise SystemExit("Instruct needs --pod_a_run for G3")
    info, z = check_frames(frames_dir)
    todo = jobs(info, a.checkpoint)
    parts = dict(checkpoint=a.checkpoint, revision=entry["revision"], frames={p: info["frames"][p]["digest"] for p in info["frames"]},
                 layers={f"{s}.{f}.{n}": layers_for(a.checkpoint, f, n) for s, f in todo for n in F.FRAMES[f]},
                 code={n: R.file_sha256(HERE / n) for n in ("extract.py", "frames.py", "checkpoints.py")} |
                      {"rows.py": R.file_sha256(Path(R.__file__))},
                 dtype="bfloat16", attn="sdpa", batch_size=a.batch_size, limit=a.limit)
    R.claim_out_dir(out, R.fingerprint(parts))
    t0 = time.time()
    model, tok = load_model(entry)
    t_load = time.time() - t0
    done = {}
    for s, f in todo:
        t = time.time()
        rec = run_frame(model, info, z, s, f, a.checkpoint, out, tok.pad_token_id, a.batch_size, a.limit)
        done[f"{s}.{f}"] = dict(rec, seconds=round(time.time() - t, 1))
        print(f"{a.checkpoint} {s}.{f}: {rec['fed']} rows, {done[f'{s}.{f}']['seconds']} s", flush=True)
    res = dict(parts, load_seconds=round(t_load, 1), seconds=round(time.time() - t0, 1), done=done,
               model_commit=model.config._commit_hash,
               versions={p: metadata.version(p) for p in ("torch", "transformers", "tokenizers", "numpy")})
    if a.checkpoint == "instruct":
        res["g3"] = g3(out, a.pod_a_run, info) if a.limit is None else "skipped (smoke test)"
    (out / "manifest.json").write_text(json.dumps(res, indent=1) + "\n")
    names = sorted(p.name for p in out.glob("*.npy")) + sorted(p.name for p in out.glob("*.*.json")) + \
        ["manifest.json", "fingerprint.txt"]
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    if a.delete_weights:  # pod disk: one checkpoint's weights at a time
        from huggingface_hub import scan_cache_dir
        scan_cache_dir().delete_revisions(entry["revision"]).execute()
    if a.checkpoint == "instruct" and isinstance(res["g3"], dict) and not res["g3"]["pass_"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
