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


def _git(*args: str):
    """Git output, or None when git could not answer.

    None rather than "": `git_dirty` was `bool(_git("status", "--porcelain"))`,
    so a git that was missing, timed out, or exited non-zero produced "" and
    stamped the run as CLEAN. A provenance record that manufactures a clean tree
    out of a failure is worse than one that admits it does not know.
    """
    try:
        p = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


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


def lib_versions() -> dict:
    """The libraries whose version can move a published number on its own.

    One definition, so a run manifest and a pinned invariant describe the same
    environment. sklearn is the live one: `requirements.txt` allows >= 1.6
    (hdbscan needs it), and the `svd_solver="auto"` heuristic PCA uses changed
    across that boundary, so `plane_r2` and `curv_gain` can shift with no code
    change at all. Recording them does not stop that; it stops it happening
    invisibly.
    """
    import numpy, scipy, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def run_stamp(data_dirs=None) -> dict:
    """Provenance dict for the current process. `data_dirs`: paths of the data
    directories the run reads (e.g. the resolved role-embeddings dir)."""
    porcelain = _git("status", "--porcelain")
    stamp = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # null, not False, when git could not be asked — "unknown" and "clean"
        # are different claims.
        "git_dirty": None if porcelain is None else bool(porcelain),
        "argv": sys.argv,
        **lib_versions(),
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
