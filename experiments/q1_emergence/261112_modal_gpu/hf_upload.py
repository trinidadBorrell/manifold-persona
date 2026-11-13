"""Upload the 12-13 Nov Modal outputs to the private HF dataset, one commit per batch, then check every file's remote
SHA-256 and size against the volume copy and write a receipt (same checks as ../261029_stage_persona_plan/hf_archive.py).
Token: Modal secret `hf-write`, never printed. Owner, 13 Nov: "Yes upload now".

  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113        # first batch (done)
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113_full   # own-reply full runs
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113_length # length control
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113_analysis # own-reply analysis v1.2
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113_analysis_b # the same, code revision b
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/hf_upload.py --batch 261113_length_verify # length check
"""
import json
from pathlib import Path

import modal

REPO = "brend0nc/manifold-persona-experiment-artifacts"
E = "experiments/q1_emergence"
OWN = f"{E}/261112_stage_own_replies"
BATCHES = {  # batch -> {path on the outputs volume: path in the dataset}
 "261113": {
    "full/coinflip": f"{E}/261112_coinflip_stages/run_full",
    "smoke/coinflip": f"{E}/261112_coinflip_stages/run_smoke",
    "full/c3": f"{E}/261112_instruct_prompt_only/run_c3_full",
    "smoke/c3": f"{E}/261112_instruct_prompt_only/run_c3_smoke",
    "full/run.log": f"{E}/261112_modal_gpu/logs/full_run.log",
    "smoke/run.log": f"{E}/261112_modal_gpu/logs/smoke_run.log",
    "own_replies/sft_smoke": f"{E}/261112_stage_own_replies/sft_smoke",
    "own_replies/dpo_smoke": f"{E}/261112_stage_own_replies/dpo_smoke",
    "own_replies/sft_smoke.log": f"{E}/261112_stage_own_replies/logs/sft_smoke.log",
    "own_replies/dpo_smoke.log": f"{E}/261112_stage_own_replies/logs/dpo_smoke.log",
 },
 "261113_full": {
    **{f"own_replies/{s}_{t}": f"{OWN}/{s}_{t}" for s in ("sft", "dpo") for t in ("smoke_v2", "full")},
    **{f"own_replies/{s}_{t}.log": f"{OWN}/logs/{s}_{t}.log" for s in ("sft", "dpo") for t in ("smoke_v2", "full")},
    **{f"own_replies/{s}_{t}.stage_receipt.json": f"{OWN}/{s}_{t}.stage_receipt.json"
       for s in ("sft", "dpo") for t in ("smoke_v2", "full")},
 },
 "261113_length": {
    **{f"length_control/{t}/{s}": f"{E}/261113_reply_length_control/run_{t}/{s}"
       for s in ("instruct", "sft", "dpo") for t in ("smoke", "full")},
    **{f"length_control/{t}/{s}.log": f"{E}/261113_reply_length_control/logs/{t}_{s}.log"
       for s in ("instruct", "sft", "dpo") for t in ("smoke", "full")},
 },
 "261113_analysis": {"analysis_v12": f"{OWN}/analysis_v12"},
 "261113_analysis_b": {"analysis_v12b": f"{OWN}/analysis_v12b"},
 "261113_length_verify": {"length_control/verify": f"{E}/261113_reply_length_control/verify"},
}
MESSAGES = {
    "261113": "12-13 Nov Modal runs: coin-flip (smoke, full), Instruct prompt-only C3 (smoke, full), "
              "SFT/DPO own-reply smoke tests, logs",
    "261113_full": "13 Nov: SFT and DPO own replies on pod A's grid (smoke_v2, full 21,000 each), logs, stage receipts",
    "261113_length": "13 Nov: reply length control (first 16 and 32 kept tokens) for Instruct, SFT and DPO; smoke, full, logs",
    "261113_analysis": "13 Nov: own-reply analysis under THRESHOLDS v1.2 (prep, 30 sources, power check, results)",
    "261113_analysis_b": "13 Nov: own-reply analysis v1.2, code revision b after the Codex review (same rules, seed and draws)",
    "261113_length_verify": "13 Nov: length control, saved-array check (verify_outputs.py) for Instruct, SFT and DPO",
}
SKIP = {"acts_parts", "rows.partial.jsonl"}  # working files; the final arrays and rows are uploaded
HERE = Path(__file__).resolve().parent

app = modal.App("persona-hf-upload-261113",
                image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.2", "hf-xet==1.7.1"))
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function(secrets=[modal.Secret.from_name("hf-write")], volumes={"/out": outputs}, cpu=4, memory=16384,
              timeout=7200, retries=0)
def upload(batch):
    import os
    from huggingface_hub import CommitOperationAdd, HfApi
    local = {}
    for src, dst in BATCHES[batch].items():
        p = Path("/out") / src
        if p.is_file():
            local[dst] = p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not SKIP & set(f.relative_to(p).parts):
                    local[f"{dst}/{f.relative_to(p).as_posix()}"] = f
        else:
            raise SystemExit(f"missing on the volume: {p}")
    meta = {k: (str(p), sha256(p), p.stat().st_size) for k, p in local.items()}
    api = HfApi(token=os.environ["HF_TOKEN"])
    rev = api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=MESSAGES[batch],
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
    receipt = dict(repo=REPO, batch=batch, revision=rev, verified=ok, n_files=len(rows), n_match=sum(r["match"] for r in rows),
                   bytes=sum(r["size"] for r in rows), files=rows)
    Path(f"/out/hf_receipt_{batch}.json").write_text(json.dumps(receipt, indent=1))
    outputs.commit()
    return receipt


@app.local_entrypoint()
def main(batch: str):
    rec = upload.remote(batch)
    (HERE / f"hf_receipt_{batch}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("repo", "revision", "verified", "n_files", "n_match", "bytes")}, indent=1))
    if not rec["verified"]:
        raise SystemExit(3)
