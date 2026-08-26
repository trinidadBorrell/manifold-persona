"""3-level pilot verification: pod-generated slice vs the reference cloud.

Level 1  text        share of byte-identical responses (greedy decoding;
                     cross-hardware near-ties may flip a few)
Level 2  activation  cosine of primary-layer vectors on matching responses
Level 3  geometry    MLE + participation ratio on the pilot role's cloud,
                     both sides, relative difference

Pass bars: text >= 0.80, cosine > 0.999 (median), geometry within 5%.
(Text bar amended 2026-08-13 from 0.90: measured cross-hardware greedy
near-tie forks hit 16% on a 50-record pilot while cosine was 1.000000 and
geometry within 1.9% — the forks share 500+ char prefixes and are benign.
Inspect fork patterns whenever text < 0.90.)

Usage:
    python -m extraction.verify_pilot --pilot /tmp/pilot_instruct \\
        --reference data/embeddings_roles_resp_40q
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(d: Path):
    man = json.load(open(d / "manifest.json"))
    meta = pd.read_parquet(d / "metadata.parquet")
    avg = np.load(d / "prompt_avg.npy", mmap_mode="r")
    return man, meta, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    pman, pmeta, pavg = load(Path(args.pilot))
    rman, rmeta, ravg = load(Path(args.reference))
    n = len(pmeta)
    L = rman["primary_layer"]

    # The record list is deterministic (same seed + n_questions), so pilot
    # row i must be reference row i. Verify instead of trusting.
    for col in ("role", "instruction_idx", "question_idx"):
        if not (pmeta[col].values == rmeta[col].values[:n]).all():
            raise SystemExit(f"row alignment broken on {col} — pilot was not "
                             f"built with the reference's seed/n_questions.")

    # Level 1 — text
    same = (pmeta["response"].values == rmeta["response"].values[:n])
    t = float(same.mean())

    # Level 2 — activation cosine at the primary layer, matching rows only
    a = np.asarray(pavg[:n, L, :], dtype=np.float64)[same]
    b = np.asarray(ravg[:n, L, :], dtype=np.float64)[same]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    c_med, c_min = float(np.median(cos)), float(cos.min())

    # Level 3 — geometry on the pilot slice, both sides
    import skdim
    from sklearn.decomposition import PCA
    def geom(X):
        X = np.asarray(X, dtype=np.float64)
        P = PCA(n_components=min(50, min(X.shape) - 1), random_state=0
                ).fit_transform(X)
        mle = float(skdim.id.MLE().fit(P, n_neighbors=min(10, len(P) - 2))
                    .dimension_)
        lam = PCA().fit(P).explained_variance_
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        return mle, pr
    m_p, pr_p = geom(pavg[:n, L, :])
    m_r, pr_r = geom(ravg[:n, L, :])
    g_mle = abs(m_p - m_r) / m_r
    g_pr = abs(pr_p - pr_r) / pr_r

    ok1, ok2, ok3 = t >= 0.80, c_med > 0.999, max(g_mle, g_pr) <= 0.05
    print(f"level 1  text      {t:6.1%} identical ({int(same.sum())}/{n})"
          f"          {'PASS' if ok1 else '** FAIL **'}")
    print(f"level 2  cosine    median {c_med:.6f}  min {c_min:.6f}"
          f"        {'PASS' if ok2 else '** FAIL **'}")
    print(f"level 3  geometry  MLE {m_p:.2f} vs {m_r:.2f} ({g_mle:.1%})  "
          f"PR {pr_p:.2f} vs {pr_r:.2f} ({g_pr:.1%})  "
          f"{'PASS' if ok3 else '** FAIL **'}")
    print("\nVERDICT:", "PASS — pod reproduces the reference"
          if ok1 and ok2 and ok3 else "FAIL — do not start the full run")
    raise SystemExit(0 if ok1 and ok2 and ok3 else 1)


if __name__ == "__main__":
    main()
