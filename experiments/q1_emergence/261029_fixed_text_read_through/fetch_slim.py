"""Pod C analysis input from HF: one checkpoint at a time, one file at a time.

Each file is downloaded at the receipt's revision into a temporary folder and checked against the receipt's
sha256. Only the hidden states the analysis reads (0, 9, 17, 23) are kept in a slim copy, with the per-frame
records (layers updated), the manifest, a provenance file and new SHA256SUMS. The temporary download goes away;
the HF copy stays the full copy.

  python fetch_slim.py --records records_full --out slim_full
  python analysis.py --frames frames_full --out slim_full --results results_full.json
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from manifold_persona import rows as R

import checkpoints as C

HERE = Path(__file__).resolve().parent
KEEP = (0, 9, 17, 23)


def download(receipt, path, tmp):
    """One receipt file at its revision; raises if its sha256 differs from the receipt."""
    from huggingface_hub import hf_hub_download
    f = next(x for x in receipt["files"] if x["path"] == path)
    local = Path(hf_hub_download(receipt["repo"], path, repo_type="dataset", revision=receipt["revision"], local_dir=tmp))
    if R.file_sha256(local) != f["sha256"]:
        raise SystemExit(f"{path}: download does not match the receipt")
    return local


def slim_checkpoint(receipt, final, fetch=download, keep=KEEP):
    """Write the slim copy of one checkpoint into final (a new folder): built in a fresh staging folder next to
    it and moved into place only when complete, so a failed fetch never looks finished."""
    final.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial.", dir=final.parent))
    by_name = {Path(f["path"]).name: f["path"] for f in receipt["files"]}
    records = sorted(n for n in by_name if n.count(".") == 2 and n.endswith(".json"))
    prov = dict(repo=receipt["repo"], revision=receipt["revision"], kept_states=list(keep), files={})
    with tempfile.TemporaryDirectory() as tmp:
        (out / "manifest.json").write_bytes(fetch(receipt, by_name["manifest.json"], tmp).read_bytes())
        for rname in records:
            rec = json.loads(fetch(receipt, by_name[rname], tmp).read_text())
            p = rname[:-5]
            for span, lay in rec["layers"].items():
                npy = f"{p}.{span}.npy"
                with tempfile.TemporaryDirectory() as one:
                    src = fetch(receipt, by_name[npy], one)
                    idx = [lay.index(s) for s in keep if s in lay]
                    arr = np.load(src, mmap_mode="r")
                    np.save(out / npy, np.ascontiguousarray(arr[:, idx]))
                    prov["files"][npy] = dict(source=by_name[npy], source_states=lay)
                rec["layers"][span] = [lay[i] for i in idx]
            (out / rname).write_text(json.dumps(rec))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    names = sorted(p.name for p in out.iterdir())
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    out.rename(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path, help="RECORDS_DIR of run_pod_c.sh")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    fix = lambda p: p if p.is_absolute() else HERE / p
    names = {ck["name"] for ck in C.CHECKPOINTS}  # records folders also hold analysis receipts
    for d in sorted(fix(a.records).iterdir()):
        if d.name not in names or not (d / "hf_receipt.json").exists():
            continue
        receipt = json.loads((d / "hf_receipt.json").read_text())
        done = fix(a.out) / d.name
        if done.exists():  # a finished copy from this receipt is kept; one from another receipt is refused
            if json.loads((done / "provenance.json").read_text())["revision"] != receipt["revision"]:
                raise SystemExit(f"{done} holds a copy of another HF revision; move it first")
            continue
        if not receipt["verified"]:
            raise SystemExit(f"{d.name}: the HF copy was not verified")
        slim_checkpoint(receipt, fix(a.out) / d.name)
        print(f"{d.name}: slim copy from {receipt['revision']}", flush=True)


if __name__ == "__main__":
    main()
