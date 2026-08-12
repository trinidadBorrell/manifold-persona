"""Run provenance stamp: which code, command, env and data produced an output.

Fast by default: data files are recorded as path+size+mtime plus the sha256 of
the small manifest.json. Set MP_HASH_DATA=1 to also sha256 the .npy files
(~30s per 8 GB file).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from manifold_persona.config import REPO_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_entry(d: Path) -> dict:
    entry = {"dir": str(d)}
    for f in sorted(d.glob("*")):
        if not f.is_file():
            continue
        st = f.stat()
        rec = {"size": st.st_size,
               "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
        if f.name == "manifest.json" or os.environ.get("MP_HASH_DATA") == "1":
            rec["sha256"] = _sha256(f)
        entry[f.name] = rec
    return entry


def run_stamp(data_dirs=None) -> dict:
    """Provenance dict for the current process. `data_dirs`: paths of the data
    directories the run reads (e.g. the resolved role-embeddings dir)."""
    import numpy, sklearn
    stamp = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MP_")},
        "data": [_data_entry(Path(d)) for d in (data_dirs or []) if Path(d).is_dir()],
    }
    return stamp


def write_stamp(run_dir, data_dirs=None, name="provenance.json") -> Path:
    """Write a provenance JSON into `run_dir`. Never raises — a failed stamp
    must not kill a science run; it records the failure instead."""
    out = Path(run_dir) / name
    try:
        stamp = run_stamp(data_dirs)
    except Exception as e:  # noqa: BLE001
        stamp = {"stamp_error": f"{type(e).__name__}: {e}"}
    out.write_text(json.dumps(stamp, indent=2))
    return out
