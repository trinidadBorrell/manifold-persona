"""Re-read every steered answer of one persona through the UNSTEERED model and keep its mean over
answer tokens (his measure: extract_acts.pooled, max_len 384) at a few layers, for the 3D PCA page.

Every answer is read with his empty-system-block context and its own question, whatever it was
generated under (his clamp cells, our add cells, the unsteered baseline, his prompted ceiling).

    python -m giovanni_repro.encode --root $PM/GIO --persona spy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAYERS = [18, 23, 28, 33, 36]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--persona", required=True)
    a = ap.parse_args()
    root, p = Path(a.root), a.persona
    from giovanni_repro.ours import load
    torch, tok, model = load(root)
    from persona_lora.data import encode_pair
    from persona_lora.extract_acts import pooled
    rows = []
    for f in sorted((root / p).glob("gen_steer_*.jsonl")) + [root / p / "gen_ceiling.jsonl"]:
        for line in open(f):
            x = json.loads(line)
            arm = "none" if x["cell"].startswith("add|") else "prompt" if x["cell"].startswith("prompt|") else "clamp"
            rows.append(dict(arm=arm, alpha=float(x["alpha"]), s=float(x["alpha"]) if arm == "clamp" else 0.0, qi=x["qi"],
                             question=x["question"], response=x["response"]))
    for f in sorted((root / "ours" / p).glob("gen_*.jsonl")):
        for line in open(f):
            x = json.loads(line)
            rows.append(dict(arm="add", alpha=float(x["alpha"]), s=float(x["s"]), qi=x["qi"],
                             question=x["question"], response=x["response"]))
    H = []
    for r in rows:
        pre, suf, start = encode_pair(tok, "", r["question"], r["response"])
        h = pooled(model, pre + suf, len(pre) + start, "cuda", 384)
        H.append(np.zeros((len(LAYERS), model.config.hidden_size), np.float32) if h is None else h[LAYERS])
    out = root / "pca"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / ("enc_%s.npz" % p), H=np.stack(H).astype(np.float16), layers=np.array(LAYERS))
    (out / ("enc_%s.json" % p)).write_text(json.dumps([{k: v for k, v in r.items() if k != "response"} for r in rows]))
    print(p, len(rows))


if __name__ == "__main__":
    main()
