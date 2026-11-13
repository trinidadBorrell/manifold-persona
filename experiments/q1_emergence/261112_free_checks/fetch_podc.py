"""Download pod C's F1 reply states (pod A text, no role prompt in view) for base, SFT, DPO and Instruct, and check
each file against the SHA-256 in that checkpoint's HF receipt. Owner, 12/11/2026: "Use this Mac's HF login".

    .venv/bin/python -I experiments/q1_emergence/261112_free_checks/fetch_podc.py
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
PODC = HERE.parent / "261029_fixed_text_read_through/records_full"
REPO_ID = "brend0nc/manifold-persona-experiment-artifacts"
CHECKPOINTS = ("sft", "dpo", "base", "instruct")
WANTED = ("pod_a.F1.json", "pod_a.F1.reply.npy")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    out = HERE / "data"
    log = []
    for ckpt in CHECKPOINTS:
        receipt = json.loads((PODC / ckpt / "hf_receipt.json").read_text())
        for f in receipt["files"]:
            if f["path"].rsplit("/", 1)[-1] not in WANTED:
                continue
            local = Path(hf_hub_download(REPO_ID, f["path"], repo_type="dataset", revision=receipt["revision"],
                                         local_dir=out))
            got = sha256(local)
            ok = got == f["sha256"] and local.stat().st_size == f["size"]
            log.append(dict(checkpoint=ckpt, path=f["path"], revision=receipt["revision"], sha256=got, match=ok))
            print(ckpt, local.name, "match" if ok else "MISMATCH", flush=True)
            if not ok:
                raise SystemExit(f"hash or size mismatch: {f['path']}")
    (out / "fetch_receipt.json").write_text(json.dumps(log, indent=1))
    print("all files match")


if __name__ == "__main__":
    main()
