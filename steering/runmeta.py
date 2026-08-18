"""Run-dir creation and the manifest.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Outputs, Bounds).

Two rules from RESEARCH.steering.md, enforced here rather than by discipline:

  - Run dirs are timestamped to the minute and NEVER overwritten. `new_run_dir`
    refuses to hand back an existing directory.
  - `output/` is fully gitignored and stays that way (user, 2026-08-17: results
    stay local). The manifest is therefore the ONLY link between a figure and
    the code that produced it — a run whose manifest is lost is unreproducible,
    which is why the git sha and dirty flag are hard failures to collect, not
    best-effort.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from manifold_persona.config import REPO_ROOT

OUTPUT_ROOT = Path(
    "/Users/trinidad.borrell/Documents/Work/MARS-V/code/manifold-persona/output/steering-manifold"
)


def new_run_dir(slug: str, root: Optional[Path] = None) -> Path:
    """`<root>/<YYYY-MM-DDTHH-MM>-<slug>/`, created fresh.

    Minute resolution makes overwriting structurally impossible; if the
    directory somehow exists we refuse rather than write into it.
    """
    root = Path(root or OUTPUT_ROOT)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    p = root / ("%s-%s" % (stamp, slug))
    if p.exists():
        raise FileExistsError(
            "run dir %s already exists; refusing to write into an existing run "
            "(RESEARCH.steering.md). Wait a minute or pass an explicit --out." % p)
    for sub in ("figures", "data", "logs"):
        (p / sub).mkdir(parents=True)
    # `.run-active` marks this dir as THE live run. The guard-outputs hook
    # allows writes into run dirs that carry it and blocks the ones that do
    # not, so a finished run cannot be written into later. Touched here, before
    # anything else is written, because forgetting it blocks the whole run.
    (p / ".run-active").touch()
    return p


def seal_run_dir(run_dir) -> None:
    """Remove `.run-active`, making the run read-only from here on.

    Called at close, after REPORT.md and the audit. Idempotent so a re-run of
    the close step is harmless.
    """
    marker = Path(run_dir) / ".run-active"
    if marker.exists():
        marker.unlink()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args],
                                   text=True).strip()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(run_dir: Path, plan: str, extra: Optional[dict] = None) -> dict:
    """Assemble the manifest. Git sha and dirty flag are mandatory."""
    import numpy as np

    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))

    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()}
    for mod in ("torch", "transformers", "scipy", "sklearn", "pandas", "anthropic"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None

    man = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "plan": plan,
        "context": "RESEARCH.steering.md",
        "git_sha": sha,
        "git_dirty": dirty,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "versions": versions,
        "status": "running",
    }
    if extra:
        man.update(extra)
    return man


def write_manifest(run_dir: Path, manifest: dict) -> Path:
    p = Path(run_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p
