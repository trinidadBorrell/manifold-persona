"""Hugging Face auth + dataset push helpers."""
from __future__ import annotations

from pathlib import Path

from .config import HF_TOKEN_PATH


def read_token(path: Path = HF_TOKEN_PATH) -> str:
    token = Path(path).read_text().strip()
    if not token:
        raise ValueError(f"HF token file {path} is empty")
    return token


def whoami(token: str = None) -> str:
    from huggingface_hub import HfApi
    token = token or read_token()
    return HfApi().whoami(token=token)["name"]


def default_repo_id(token: str = None, repo_name: str = "manifold-persona") -> str:
    return f"{whoami(token)}/{repo_name}"


def snapshot_revision(downloaded_path):
    """Commit sha of the snapshot an `hf_hub_download` path came from, or None.

    Cache layout is .../snapshots/<sha>/<file>; pass that sha as `revision`
    to the remaining downloads so every file comes from one commit. A path
    outside that layout (a local_dir copy) gives None, with a warning.
    """
    p = Path(downloaded_path)
    if p.parent.parent.name != "snapshots":
        print(f"WARNING: {p} is not in an HF snapshot directory; commit not recorded")
        return None
    return p.parent.name
