"""Phase 2 dose response (plan v2 s1 M4): clamp along each of the 35 axes (k = 1) toward
G = M_0 + s sigma_a a, s in {-3, ..., 3}, sigma_a = std of the 276 roles' coordinates on a.
10 questions x 1 seed per dose = 70 answers per axis, 2,450 in all. Same generation settings as ceiling.py.

    python -m persona_axes.dose --shard 0 --n-shards 5     # GPU; -> $PM/AX21/dose/gen_<axis>.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from persona_axes import AX, HERE, LAYER, QUESTIONS, VIEW
from persona_axes.ceiling import encode, generate

DOSES = [-3, -2, -1, 0, 1, 2, 3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="dose unit = scale x sigma_a (1 = pre-registered; 4 = addendum 2026-10-21)")
    ap.add_argument("--out-name", default="dose")
    a = ap.parse_args(argv)
    import torch
    from ace_steering import steer_gpu as AS
    V = AX / "vectors"
    names = json.loads((V / "axes_names.json").read_text())
    A = np.load(V / ("axes_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    R = np.load(V / ("roles_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    M0 = np.load(V / ("neutral_rawmean_%s_L%d.npy" % (VIEW, LAYER))).astype(np.float64)
    sigma = (R @ A.T).std(0) * a.scale
    out = AX / a.out_name
    out.mkdir(exist_ok=True)
    (out / "sigma.json").write_text(json.dumps(dict(scale=a.scale, unit=dict(zip(names, sigma.round(3).tolist())))))
    tok, model = AS.load_model(AS.MODEL)
    enc, P = encode(tok, model, "")
    for i in list(range(len(names)))[a.shard::a.n_shards]:
        f = out / ("gen_%s.jsonl" % names[i])
        if f.exists():
            continue
        u = A[i]; ut = torch.tensor(u, dtype=torch.float32)
        rows = []
        for s in DOSES:
            c = float(M0 @ u) + s * float(sigma[i])

            def hook(module, args, c=c):
                hs = args[0]
                uu = ut.to(hs.device)
                h = hs.float()
                h = h + (c - h @ uu).unsqueeze(-1) * uu
                return (h.to(hs.dtype),) + tuple(args[1:])
            ans = generate(tok, model, enc, P, hook, 1)[0]
            rows += [dict(axis=names[i], s=s, sigma=float(sigma[i]), qi=qi, question=QUESTIONS[qi], response=t) for qi, t in enumerate(ans)]
        print("%s sigma=%.1f | s=+3: %r" % (names[i], sigma[i], rows[-10]["response"][:90]), flush=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.rename(f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
