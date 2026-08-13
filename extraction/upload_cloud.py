"""Upload a finished cloud directory to a Hugging Face dataset repo.

Run this on the SAME machine as the extraction, before pod teardown —
pods are ephemeral and an unsynced run dies with the instance. The token
comes from the standard HF locations (HF_TOKEN env var or the cached
login), never from arguments.

Refuses to upload an unfinished run (a `_ckpt/` dir still present) or a
directory without a manifest.

Usage:
    python -m extraction.upload_cloud --dir data/embeddings_roles_resp_40q_base \\
        --repo brend0nc/manifold-persona-roles-response-40q-base
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="cloud directory to upload")
    ap.add_argument("--repo", required=True, help="dataset repo id (user/name)")
    args = ap.parse_args()

    d = Path(args.dir)
    if not (d / "manifest.json").exists():
        raise SystemExit(f"{d} has no manifest.json — not a finished cloud.")
    if (d / "_ckpt").exists():
        raise SystemExit(f"{d} still has _ckpt/ — run is not finalized.")

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.upload_folder(folder_path=str(d), repo_id=args.repo,
                             repo_type="dataset",
                             commit_message=f"upload {d.name}")
    print(f"uploaded {d} -> {args.repo}")
    print(info)


if __name__ == "__main__":
    main()
