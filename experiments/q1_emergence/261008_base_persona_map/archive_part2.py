"""Archive the Part 2 context states (THRESHOLDS.md, change of 16/10/2026), on the pod that holds them.

  slim    ctx_<f>_L9_17_23.npy next to each ctx_<f>.npy (hidden states 9, 17, 23); slim.json with SHA-256 of both
  upload  one commit to the private dataset: every listed file under experiments/q1_emergence/261008_base_persona_map/
          part2_261016/<path relative to this folder>; then the remote SHA-256 of every file is checked against the
          local one and a receipt is written

    python archive_part2.py slim --runs run_main run_rep
    HF_TOKEN=... python archive_part2.py upload --files run_main/context/* ... --receipt archive_receipt.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from context_extract import FEATS, sha256

HERE = Path(__file__).resolve().parent
REPO = "brend0nc/manifold-persona-experiment-artifacts"
PREFIX = "experiments/q1_emergence/261008_base_persona_map/part2_261016"
LAYERS = [9, 17, 23]


def slim(runs):
    for run in runs:
        ctx = HERE / run / "context"
        man = json.load(open(ctx / "manifest.json"))
        meta = {"layers": LAYERS, "files": {}}
        for f in FEATS:
            src = ctx / f"ctx_{f}.npy"
            if sha256(src) != man["files"][src.name]:
                raise SystemExit(f"{src} does not match its manifest SHA-256")
            out = ctx / f"ctx_{f}_L9_17_23.npy"
            a = np.load(src, mmap_mode="r")
            np.save(out, np.ascontiguousarray(a[:, LAYERS, :]))
            if not np.array_equal(np.load(out, mmap_mode="r")[:, 1], a[:, 17]):
                raise SystemExit(f"{out} does not hold state 17 of {src}")
            meta["files"][out.name] = dict(sha256=sha256(out), source=src.name, source_sha256=man["files"][src.name])
            print(out, flush=True)
        json.dump(meta, open(ctx / "slim.json", "w"), indent=1)


def upload(files, receipt, message):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    local = {}
    for p in files:
        p = Path(p).resolve()
        rel = p.relative_to(HERE).as_posix()
        local[f"{PREFIX}/{rel}"] = (p, sha256(p), p.stat().st_size)
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v[0])) for k, v in local.items()]
    info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=message)
    rev = info.oid
    remote = {x.path: x for x in api.get_paths_info(REPO, list(local), repo_type="dataset", revision=rev, expand=True)}
    rows, ok = [], True
    for k, (p, h, n) in local.items():
        r = remote.get(k)
        rh = r.lfs.sha256 if r is not None and r.lfs is not None else None
        if r is not None and rh is None:  # small files are stored without LFS: compare by download
            rh = sha256(Path(api.hf_hub_download(REPO, k, repo_type="dataset", revision=rev)))
        good = r is not None and rh == h and r.size == n
        ok &= good
        rows.append(dict(path=k, local=str(p), sha256=h, size=n, remote_sha256=rh, match=good))
    json.dump(dict(repo=REPO, revision=rev, verified=ok, files=rows), open(receipt, "w"), indent=1)
    print("revision", rev, "verified", ok, f"{sum(r['match'] for r in rows)}/{len(rows)}")
    if not ok:
        raise SystemExit(3)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("slim")
    s.add_argument("--runs", nargs="+", default=["run_main", "run_rep"])
    u = sp.add_parser("upload")
    u.add_argument("--files", nargs="+", required=True)
    u.add_argument("--receipt", required=True)
    u.add_argument("--message", default="E3 Part 2 context states and outputs (16/10/2026)")
    a = ap.parse_args()
    if a.cmd == "slim":
        return slim(a.runs)
    return upload(a.files, a.receipt, a.message)


if __name__ == "__main__":
    main()
