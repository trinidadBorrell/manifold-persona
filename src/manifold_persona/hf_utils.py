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
