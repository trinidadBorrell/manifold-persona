"""Stage 0: per layer and view, the quantities every panel job and both experiments share.

  centroids       (276, 4096) raw role centroids (mean over the 1,200 answers)
  default_cloud   the `default` role's grid-ordered answers are NOT stored (panel jobs read them from the mmap);
                  stored: its centroid, its top-10 right-singular vectors, and its 240 question means
  axis            Lu et al.: mean(default answers) - mean(all role answers), unit (`manifold_persona.common.assistant_axis`);
                  identical for raw and question-centred clouds (the centring is a common shift)
  axis_proj       <centroid_p - mean centroid, axis>  (the Experiment 1 target)
  response_chars  mean response length in characters per role (confounder)

    python -m cloud_predicts_position.axis --layer 25 --view avg
"""
from __future__ import annotations

import argparse

import numpy as np

from cloud_predicts_position import data as D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--view", default="avg")
    ap.add_argument("--out", default=str(D.OUT))
    a = ap.parse_args(argv)
    out = D.OUT if a.out == str(D.OUT) else __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = D.metadata(a.layer)
    A = D.open_layer(a.layer, a.view)
    roles = D.roles(meta)
    C = np.zeros((len(roles), A.shape[1]), np.float64)
    for i, r in enumerate(roles):
        X, _, _ = D.cloud(A, meta, r)
        C[i] = X.astype(np.float64).mean(0)
        if i % 40 == 0:
            print(i, r, flush=True)
    Xd, instr, quest = D.cloud(A, meta, D.DEFAULT)
    Xd = Xd.astype(np.float64)
    d_mean = Xd.mean(0)
    _, _, Vt = np.linalg.svd(Xd - d_mean, full_matrices=False)
    qmeans = np.stack([Xd[quest == q].mean(0) for q in np.unique(quest)])
    axis = d_mean - C.mean(0)
    axis /= np.linalg.norm(axis)
    proj = (C - C.mean(0)) @ axis
    chars = D.response_lengths(a.layer).reindex(roles).to_numpy()
    f = out / f"axis_{a.view}_L{a.layer}.npz"
    np.savez(f, roles=np.array(roles), centroids=C.astype(np.float32), default_centroid=d_mean.astype(np.float32),
             default_top10=Vt[:10].astype(np.float32), default_qmeans=qmeans.astype(np.float32),
             axis=axis.astype(np.float32), axis_proj=proj.astype(np.float32), response_chars=chars)
    i_def = roles.index(D.DEFAULT)
    print(f"wrote {f}; axis_proj default = {proj[i_def]:.2f}, range {proj.min():.2f}..{proj.max():.2f}, "
          f"|centroid| median {np.linalg.norm(C, axis=1).mean():.1f}")


if __name__ == "__main__":
    main()
