"""Han Xiao re-analysis on Modal CPU (owner, 12 Nov: "We can do modal ye for it"). Rules: THRESHOLDS.md v1.0.

Downloads the public release anonymously into its own volume, checks every file against
data/han_release_manifest.json, runs Part A (gate), then Parts B and C+D in parallel, and copies the small result
files back. No GPU, no token, no upload.

  .venv/bin/modal run experiments/q1_emergence/261112_han_reanalysis/modal_han.py --step all
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SKIP = {"han_release", "__pycache__", "results", ".pytest_cache"}

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==1.26.4", "pandas==3.0.5", "huggingface_hub==0.36.2")
         .env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
               "HF_HUB_DISABLE_XET": "1",  # anonymous xet-read-token calls were rate-limited (HTTP 429) on 12 Nov
               "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16"})
         .add_local_dir(HERE, "/root/han",
                        ignore=lambda p: bool(SKIP & set(Path(p).parts)) or Path(p).suffix == ".log"))
app = modal.App("persona-han-reanalysis-261112", image=image)
vol = modal.Volume.from_name("persona-han-reanalysis-261112", create_if_missing=True, version=2)
DATA, OUT = "/vol/han_release", "/vol/results"


@app.function(cpu=4, memory=8192, timeout=7200, retries=0, volumes={"/vol": vol})
def fetch():
    import hashlib
    import sys
    import time
    sys.path.insert(0, "/root/han")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    import fetch_han
    manifest = json.loads(Path("/root/han/data/han_release_manifest.json").read_text())
    for pattern in fetch_han.PATTERNS:  # one folder at a time, 2 workers; wait 5 min and resume on HTTP 429
        for attempt in range(10):
            try:
                snapshot_download(fetch_han.REPO_ID, repo_type="dataset", revision=manifest["revision"],
                                  allow_patterns=[pattern], local_dir=DATA, token=False, max_workers=2)
                vol.commit()
                print("done", pattern, flush=True)
                break
            except HfHubHTTPError as e:
                if "429" not in str(e) or attempt == 9:
                    raise
                print("429, waiting 300 s:", pattern, flush=True)
                time.sleep(300)
    bad, n = [], 0
    for name, digest in manifest["files"].items():
        h = hashlib.sha256()
        with open(Path(DATA) / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        n += 1
        if h.hexdigest() != digest:
            bad.append(name)
    vol.commit()
    return dict(revision=manifest["revision"], n_checked=n, n_expected=manifest["n_files"], mismatches=bad[:20],
                n_mismatches=len(bad))


@app.function(cpu=16, memory=65536, timeout=7200, retries=0, volumes={"/vol": vol})
def run_part(part):
    import subprocess
    import sys
    vol.reload()
    cmd = [sys.executable, "-I", "/root/han/han_reanalysis.py", part, "--data", DATA, "--out", OUT]
    p = subprocess.run(cmd, capture_output=True, text=True)
    vol.commit()
    return dict(part=part, returncode=p.returncode, stdout=p.stdout[-6000:], stderr=p.stderr[-4000:])


@app.function(cpu=2, memory=4096, timeout=600, retries=0, volumes={"/vol": vol})
def collect():
    import subprocess
    import sys
    vol.reload()
    subprocess.run([sys.executable, "-I", "/root/han/han_reanalysis.py", "report", "--out", OUT], check=True,
                   capture_output=True)
    return {p.name: p.read_text() for p in Path(OUT).glob("*") if p.suffix in (".json", ".md")}


@app.local_entrypoint()
def main(step: str = "all"):
    local = HERE / "results"
    if step in ("fetch", "all"):
        r = fetch.remote()
        print(r)
        if r["n_mismatches"] or r["n_checked"] != r["n_expected"]:
            raise SystemExit("release files do not match the manifest; stop")
    if step in ("a", "all"):
        r = run_part.remote("a")
        print(r["stdout"], r["stderr"][-2000:])
        if r["returncode"]:
            raise SystemExit("Part A failed or the gate did not pass; stop")
    if step in ("bcd", "all", "cd"):
        calls = [run_part.spawn(p) for p in (("cd",) if step == "cd" else ("b", "cd"))]
        for c in calls:
            r = c.get()
            print(r["part"], "exit", r["returncode"], r["stdout"][-3000:], r["stderr"][-2000:])
    files = collect.remote()
    local.mkdir(exist_ok=True)
    for name, text in files.items():
        (local / name).write_text(text)
    print(files.get("results.md", "no results.md"))
