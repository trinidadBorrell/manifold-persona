"""A: is the Gaussian oracle any good?  B: can a persona be built from the others?

A 2048-d Gaussian from 120 points is singular, so everything happens in a
low-dimensional persona subspace fitted on the training half only.
"""
import argparse
import glob
import json

import numpy as np

def load():
    P, N = [], []
    for f in sorted(glob.glob("attr_acts/*.npz")):
        z = np.load(f, allow_pickle=True)
        P.append(z["prompted"].astype(np.float32))
        N.append(z["persona"].astype(str))
    return np.concatenate(P, 0), np.concatenate(N)


def fit_gaussians(pts_by_p, shrink):
    G = {}
    for p, Y in pts_by_p.items():
        mu = Y.mean(0)
        S = np.cov(Y.T)
        S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        sign, logdet = np.linalg.slogdet(S)
        G[p] = (mu, np.linalg.inv(S), logdet)
    return G


def logpdf(x, g):
    mu, Si, logdet = g
    d = x - mu
    return -0.5 * (d @ Si @ d + logdet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=0.25)
    ap.add_argument("--out", default="attr_analysis/oracle.json")
    args = ap.parse_args()
    X, who = load()
    personas = sorted(set(who))
    rng = np.random.default_rng(0)

    # ---------- A: can the oracle name a persona from one held-out response? ----------
    tr, te = {}, {}
    for p in personas:
        idx = np.where(who == p)[0]
        rng.shuffle(idx)
        tr[p], te[p] = idx[:80], idx[80:]
    Xtr = X[:, args.layer, :]
    asst_mu = Xtr[tr["default"]].mean(0)
    # subspace from TRAIN means only, so the test half never informs the basis
    Mtr = np.stack([Xtr[tr[p]].mean(0) - asst_mu for p in personas])
    _, _, vt = np.linalg.svd(Mtr - Mtr.mean(0), full_matrices=False)
    E = vt[:args.k].T

    G = fit_gaussians({p: (Xtr[tr[p]] - asst_mu) @ E for p in personas}, args.shrink)
    hit = top5 = n = 0
    conf_right, conf_wrong = [], []
    for p in personas:
        for x in (Xtr[te[p]] - asst_mu) @ E:
            ll = np.array([logpdf(x, G[q]) for q in personas])
            post = np.exp(ll - ll.max()); post /= post.sum()
            order = [personas[i] for i in np.argsort(-ll)]
            hit += order[0] == p
            top5 += p in order[:5]
            (conf_right if order[0] == p else conf_wrong).append(post.max())
            n += 1
    acc, acc5 = hit / n, top5 / n
    print(f"[A] oracle, {args.k}-d subspace, shrink {args.shrink}, layer {args.layer}")
    print(f"    held-out single-response accuracy  {acc:.3f}   (chance {1/len(personas):.3f})")
    print(f"    top-5 accuracy                     {acc5:.3f}")
    print(f"    mean posterior when right {np.mean(conf_right):.2f} | when wrong "
          f"{np.mean(conf_wrong) if conf_wrong else float('nan'):.2f}")

    # ---------- B: build each persona from the other 40 ----------
    Xa = X[:, args.layer, :]
    a = Xa[who == "default"].mean(0)
    D = {p: Xa[who == p].mean(0) - a for p in personas if p != "default"}
    names = sorted(D)
    recon = {}
    print(f"\n[B] leave-one-out reconstruction from the other {len(names)-1} personas")
    r2s, r2rand, nnz = [], [], []
    for t in names:
        others = [q for q in names if q != t]
        A = np.stack([D[q] for q in others], 1)
        y = D[t]
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A @ w) ** 2) / np.sum(y ** 2)
        # sparse: keep adding the component that most reduces residual
        sel, r, wsp = [], y.copy(), None
        for _ in range(6):
            c = [(abs(A[:, i] @ r) / np.linalg.norm(A[:, i]), i)
                 for i in range(len(others)) if i not in sel]
            sel.append(max(c)[1])
            As = A[:, sel]
            wsp, *_ = np.linalg.lstsq(As, y, rcond=None)
            r = y - As @ wsp
        r2sp = 1 - np.sum(r ** 2) / np.sum(y ** 2)
        # control: a random direction of the same length, same basis
        rv = rng.normal(size=y.shape); rv *= np.linalg.norm(y) / np.linalg.norm(rv)
        wr, *_ = np.linalg.lstsq(A, rv, rcond=None)
        r2r = 1 - np.sum((rv - A @ wr) ** 2) / np.sum(rv ** 2)
        r2s.append(r2); r2rand.append(r2r); nnz.append(r2sp)
        recon[t] = {"r2_full": round(float(r2), 3), "r2_sparse6": round(float(r2sp), 3),
                    "r2_random_control": round(float(r2r), 3),
                    "top": [[others[i], round(float(v), 2)]
                            for i, v in sorted(zip(sel, wsp), key=lambda z: -abs(z[1]))]}
    print(f"    full basis  R2 mean {np.mean(r2s):.3f}   random-vector control {np.mean(r2rand):.3f}")
    print(f"    sparse (6 components) R2 mean {np.mean(nnz):.3f}")
    print("\n    examples (6-component reconstructions):")
    for t in ["vampire", "ghost", "pirate", "toddler", "old_musician", "medieval_spy"]:
        if t in recon:
            terms = " + ".join(f"{w:+.2f}*{q}" for q, w in recon[t]["top"][:4])
            print(f"      {t:14} R2={recon[t]['r2_sparse6']:.2f}  {terms}")

    json.dump({"layer": args.layer, "k": args.k, "acc": acc, "acc5": acc5,
               "r2_full_mean": float(np.mean(r2s)),
               "r2_random_control": float(np.mean(r2rand)),
               "r2_sparse6_mean": float(np.mean(nnz)), "recon": recon},
              open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
