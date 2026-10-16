"""3D PCA page: from the empty-prompt model to spy, ghost, vampire, pirate and musician, under his clamp
and our addition, on Qwen2.5-3B (answers re-read unsteered, mean over answer tokens).

    python -m giovanni_repro.pca_page --root $PM/GIO --out page.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

P = ["spy", "ghost", "vampire", "pirate", "musician"]
HERE = Path(__file__).resolve().parent


def ref_means(root, who, layers):
    fs = sorted((root / who / "shards").glob("acts_*.npz"))
    X = np.concatenate([np.load(f)["prompted"][:, layers] for f in fs]).astype(np.float32)
    return X.mean(0)                                                    # (n_layers, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    enc = {p: (np.load(root / "pca" / ("enc_%s.npz" % p)), json.loads((root / "pca" / ("enc_%s.json" % p)).read_text())) for p in P}
    layers = enc[P[0]][0]["layers"].tolist()
    refs = {w: ref_means(root, w, layers) for w in P + ["default"]}
    data = dict(layers=layers, personas=P, by_layer={})
    for li, L in enumerate(layers):
        cells, pts = [], []
        for p in P:
            H, meta = enc[p][0]["H"][:, li].astype(np.float32), enc[p][1]
            keys = sorted({(m["arm"], m["alpha"], m["s"]) for m in meta})
            for k in keys:
                idx = [i for i, m in enumerate(meta) if (m["arm"], m["alpha"], m["s"]) == k]
                cells.append(dict(persona=p, arm=k[0], alpha=k[1], s=k[2], mean=H[idx].mean(0), idx=idx))
            pts.append((p, H, meta))
        R = np.stack([refs[w][li] for w in P + ["default"]])
        C = np.stack([c["mean"] for c in cells])
        fit = np.concatenate([R, C])
        mu = fit.mean(0)
        U, S, Vt = np.linalg.svd(fit - mu, full_matrices=False)
        V = Vt[:3]
        ev = (S[:3] ** 2 / (S ** 2).sum()).tolist()
        proj = lambda X: ((X - mu) @ V.T).round(3).tolist()
        data["by_layer"][str(L)] = dict(
            var=[round(e, 3) for e in ev],
            refs={w: proj(refs[w][li][None])[0] for w in P + ["default"]},
            cells=[dict(persona=c["persona"], arm=c["arm"], alpha=c["alpha"], s=c["s"], xyz=proj(c["mean"][None])[0]) for c in cells],
            answers={p: dict(xyz=proj(H), arm=[m["arm"] for m in meta], alpha=[m["alpha"] for m in meta], s=[m["s"] for m in meta])
                     for p, H, meta in pts})
    s = json.dumps(data, allow_nan=False)
    tpl = (HERE / "pca_template.html").read_text()
    Path(a.out).write_text(tpl.replace("__DATA__", s))
    print("page -> %s (%.2f MB)" % (a.out, Path(a.out).stat().st_size / 1e6))


if __name__ == "__main__":
    main()
