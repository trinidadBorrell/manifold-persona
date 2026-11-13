"""Download Han Xiao's public release (han-xiao/assistant-axis-shift, CC-BY-4.0) at a pinned revision: per-role
vectors, axes, judge scores and layer-16 per-response activations for SFT, DPO and RLVR. Anonymous download; no token.

    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python -I experiments/q1_emergence/261112_han_reanalysis/fetch_han.py
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
REPO_ID = "han-xiao/assistant-axis-shift"
REVISION = "bb109847996bcfad452b9fb24c9c9451cbe909ab"
PATTERNS = ["README.md"] + [f"{s}-sysprompt/{p}" for s in ("sft", "dpo", "rlvr")
                            for p in ("vectors-glm-5.3-flash/*", "axis-glm-5.3-flash.pt", "scores-glm-5.3-flash/*",
                                      "activations-layer16/*")]


def main():
    out = HERE / "data/han_release"
    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=PATTERNS, local_dir=out,
                      token=False, max_workers=4)
    files = sorted(p for p in out.rglob("*") if p.is_file() and ".cache" not in p.parts)
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / "data/han_release_manifest.json").write_text(json.dumps(dict(repo=REPO_ID, revision=REVISION,
                                                                         n_files=len(manifest), files=manifest), indent=1))
    print(len(manifest), "files")


if __name__ == "__main__":
    main()
