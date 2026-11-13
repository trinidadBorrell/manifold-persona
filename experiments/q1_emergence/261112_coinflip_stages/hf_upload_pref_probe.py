"""Upload the preference-probe outputs (and the length check) from the Modal volume to the private HF dataset in one
commit, then check every file's remote SHA-256 and size against the volume copy and write a receipt. Same checks as
../261112_modal_gpu/hf_upload.py. Token: Modal secret `hf-write`, never printed. Owner, 13 Nov: "Yes, do both".

  .venv/bin/modal run experiments/q1_emergence/261112_coinflip_stages/hf_upload_pref_probe.py
"""
import json
from pathlib import Path

import modal

REPO = "brend0nc/manifold-persona-experiment-artifacts"
DST = "experiments/q1_emergence/261112_coinflip_stages"
PATHS = {  # path on the outputs volume -> path in the dataset
    "pref_probe/smoke": f"{DST}/pref_probe/smoke",
    "pref_probe/full": f"{DST}/pref_probe/full",
    "pref_probe/interview": f"{DST}/pref_probe/interview",
    "pref_probe/length_check.json": f"{DST}/run_full/length_check.json",
}
MESSAGE = ("13 Nov: OLMo-2-7B Assistant task-preference probe (smoke, full chat frame, common interview frame), "
           "analysis files, logs; coin-flip task-length check")
HERE = Path(__file__).resolve().parent

app = modal.App("persona-hf-upload-pref-probe-261113",
                image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.2", "hf-xet==1.7.1"))
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function(secrets=[modal.Secret.from_name("hf-write")], volumes={"/out": outputs}, cpu=2, memory=4096,
              timeout=1800, retries=0)
def upload():
    import os
    from huggingface_hub import CommitOperationAdd, HfApi
    local = {}
    for src, dst in PATHS.items():
        p = Path("/out") / src
        if p.is_file():
            local[dst] = p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    local[f"{dst}/{f.relative_to(p).as_posix()}"] = f
        else:
            raise SystemExit(f"missing on the volume: {p}")
    meta = {k: (str(p), sha256(p), p.stat().st_size) for k, p in local.items()}
    api = HfApi(token=os.environ["HF_TOKEN"])
    rev = api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=MESSAGE,
                            operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in local.items()]).oid
    remote = {x.path: x for x in api.get_paths_info(REPO, list(local), repo_type="dataset", revision=rev, expand=True)}
    rows, ok = [], True
    for k, (src, h, n) in meta.items():
        r = remote.get(k)
        rh = r.lfs.sha256 if r is not None and getattr(r, "lfs", None) is not None else None
        if r is not None and rh is None:  # small files are stored without LFS: compare by download
            rh = sha256(api.hf_hub_download(REPO, k, repo_type="dataset", revision=rev))
        good = r is not None and rh == h and r.size == n
        ok &= good
        rows.append(dict(path=k, volume_path=src, sha256=h, size=n, remote_sha256=rh, match=good))
    return dict(repo=REPO, revision=rev, verified=ok, n_files=len(rows), n_match=sum(r["match"] for r in rows),
                bytes=sum(r["size"] for r in rows), files=rows)


@app.local_entrypoint()
def main():
    rec = upload.remote()
    (HERE / "hf_receipt_pref_probe_261113.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("repo", "revision", "verified", "n_files", "n_match", "bytes")}, indent=1))
    if not rec["verified"]:
        raise SystemExit(3)
