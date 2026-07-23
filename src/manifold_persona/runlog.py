"""Shared run-harness helpers: run dirs, logging, manifests, figure saving, Holm.

Extracted from manifold/{run,sweep,local_id,analysis_extra,plots,sweep_plots}.py,
which had grown 4 copies of ``_save``, 2 of ``holm`` and 4 manifest-writing
blocks between them.

Two things here are deliberately parameterized rather than unified, because the
call sites genuinely differ and this is a refactor:

* ``save_fig`` takes ``dpi`` — ``local_id.py`` writes at 160, everything else at
  300 (see ``## Found`` in the refactor plan; the deviation is preserved, not
  normalized). ``manifold_persona.common.savefig`` writes at 150 and is a
  separate helper on purpose — it is NOT part of this unification.
* ``provenance`` returns an *ordered fragment* to splice into a manifest rather
  than owning the whole dict, so each caller keeps its own key set and key
  order byte-for-byte. Four call sites, four key sets, all preserved.
"""
from __future__ import annotations

import datetime
import json
import platform
import time
from pathlib import Path

import numpy as np


def timestamp() -> str:
    """The run-dir stamp convention: minute resolution, sortable."""
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")


def new_run_dir(root: Path, name: str, subdirs=("figures", "data", "logs")) -> Path:
    """Create a run dir that refuses to overwrite an existing one.

    The first subdir is created with ``exist_ok=False`` and ``parents=True``:
    that is the collision guard (two runs in the same minute must raise), so
    it stays first and stays strict.
    """
    run_dir = Path(root) / name
    first, rest = subdirs[0], subdirs[1:]
    (run_dir / first).mkdir(parents=True, exist_ok=False)
    for sub in rest:
        (run_dir / sub).mkdir()
    return run_dir


def make_say(log_path: Path):
    """Return (say, close): print to stdout and tee to a log file."""
    log = open(log_path, "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    return say, log.close


def provenance(t0: float, pandas: bool = False) -> dict:
    """Ordered environment + timing fragment for a manifest.

    Splice with ``{**mine, **provenance(t0), **extra}`` so the caller keeps
    control of its own key order.
    """
    d = {"python": platform.python_version(), "numpy": np.__version__}
    if pandas:
        import pandas as pd
        d["pandas"] = pd.__version__
    d["started"] = datetime.datetime.fromtimestamp(t0).isoformat()
    d["finished"] = datetime.datetime.now().isoformat()
    d["elapsed_sec"] = round(time.time() - t0, 1)
    return d


def write_manifest(run_dir: Path, manifest: dict, default=None) -> None:
    """Write manifest.json. ``default=str`` where a caller stores non-JSON types."""
    with open(Path(run_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=default)


def save_fig(fig, path, dpi: int = 300, log=None) -> None:
    """Save + close a matplotlib figure.

    ``log`` overrides the progress line; ``local_id.py`` prints an indented
    short form, everything else prints ``wrote <path>``.
    """
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if log is None:
        print("wrote", path)
    else:
        log(path)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a name->p dict."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * p))
        adj[name] = prev
    return adj
