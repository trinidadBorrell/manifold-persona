"""Upload one finished stage-pod run to the private HF dataset, with a checked receipt, and test a restore.

  upload   one commit of the run's final files to experiments/q1_emergence/<experiment>/<run>/; then the
           remote SHA-256 and size of every file are checked against the local ones and a receipt is written
  restore  download files at the receipt's revision into a new folder and check their SHA-256

    python hf_archive.py upload --run ../261029_instruct_persona_map/run_canary
    python hf_archive.py restore --receipt ../261029_instruct_persona_map/run_canary/hf_receipt.json
    python hf_archive.py upload --run out/base --prefix experiments/q1_emergence/261029_fixed_text_read_through/run_full
Every pod runs upload itself (owner, 29/10), with a token the owner stores there (hf auth login; a RunPod login
shell saves it under /workspace/.cache/huggingface, so set HF_TOKEN_PATH there); nothing here reads or prints it.
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

from manifold_persona.rows import file_sha256

REPO = "brend0nc/manifold-persona-experiment-artifacts"
SKIP = {"acts_parts", "rows.partial.jsonl", "hf_receipt.json"}


def run_files(run_dir):
    """Every final file of the run; the files in SHA256SUMS.txt must match first."""
    for line in (run_dir / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if file_sha256(run_dir / name) != h:
            raise SystemExit(f"{name} does not match SHA256SUMS.txt")
    files = [p for p in sorted(run_dir.rglob("*")) if p.is_file() and not SKIP & set(p.relative_to(run_dir).parts)]
    log = run_dir.parent / f"{run_dir.name}.log"
    return files + ([log] if log.exists() else [])


def remote_paths(run_dir, prefix=None):
    """{path in the dataset: local file}; the run folder and its log keep their names under the prefix (default:
    experiments/q1_emergence/<the run folder's parent>)."""
    run_dir = run_dir.resolve()
    prefix = prefix or f"experiments/q1_emergence/{run_dir.parent.name}"
    return {f"{prefix}/{p.relative_to(run_dir.parent).as_posix()}": p for p in run_files(run_dir)}


def upload(run_dir, message, prefix=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    run_dir = run_dir.resolve()
    local = {k: (p, file_sha256(p), p.stat().st_size) for k, p in remote_paths(run_dir, prefix).items()}
    api = HfApi()
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v[0])) for k, v in local.items()]
    rev = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=message).oid
    remote = {x.path: x for x in api.get_paths_info(REPO, list(local), repo_type="dataset", revision=rev, expand=True)}
    rows, ok = [], True
    for k, (p, h, n) in local.items():
        r = remote.get(k)
        rh = r.lfs.sha256 if r is not None and r.lfs is not None else None
        if r is not None and rh is None:  # small files are stored without LFS: compare by download
            rh = file_sha256(api.hf_hub_download(REPO, k, repo_type="dataset", revision=rev))
        good = r is not None and rh == h and r.size == n
        ok &= good
        rows.append(dict(path=k, local=str(p), sha256=h, size=n, remote_sha256=rh, match=good))
    receipt = run_dir / "hf_receipt.json"
    receipt.write_text(json.dumps(dict(repo=REPO, revision=rev, verified=ok, files=rows), indent=1))
    print("revision", rev, "verified", ok, f"{sum(r['match'] for r in rows)}/{len(rows)}", "receipt", receipt)
    if not ok:
        raise SystemExit(3)


def restore(receipt, n):
    from huggingface_hub import hf_hub_download
    rec = json.loads(Path(receipt).read_text())
    files = rec["files"] if n is None else random.Random(0).sample(rec["files"], min(n, len(rec["files"])))
    with tempfile.TemporaryDirectory() as tmp:
        bad = [f["path"] for f in files
               if file_sha256(hf_hub_download(rec["repo"], f["path"], repo_type="dataset", revision=rec["revision"],
                                              local_dir=tmp)) != f["sha256"]]
    print(f"restore at {rec['revision']}: {len(files) - len(bad)}/{len(files)} files match", *bad)
    if bad:
        raise SystemExit(3)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("upload")
    u.add_argument("--run", required=True, type=Path)
    u.add_argument("--message", default=None)
    u.add_argument("--prefix", default=None, help="dataset folder for the run (default: experiments/q1_emergence/<parent>)")
    r = sp.add_parser("restore")
    r.add_argument("--receipt", required=True)
    r.add_argument("--n", type=int, default=None, help="restore n random files (default: all)")
    a = ap.parse_args()
    if a.cmd == "upload":
        return upload(a.run, a.message or f"{a.run.resolve().parent.name}/{a.run.name}", a.prefix)
    return restore(a.receipt, a.n)


if __name__ == "__main__":
    main()
