"""Where the judge classifier puts each persona, and how that compares with the steering goal.

The classifier is linear on PCA features of the layer-23 read-out, so for a persona t the
log-probability of t is maximised, inside a ball of radius R around the read-out centre,
by projected gradient ascent. The script reports that maximum, the direction that reaches
it, and its cosine with the persona-mean direction used for steering and with the
direction of the persona's own prompted answers. It also reports the classifier's mean
p(target) on the read-outs of every arm and strength, and writes the per-persona
directions for a steering run that aims at the classifier's optimum.
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

CACHE = "attr_analysis/_alllayer_cache.npz"


def parse(c):
    h, t = c.split("|")[:2]
    return ("prompt", t) if h == "prompt" else tuple(t.split("_", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attr_analysis/judge_classifier_all_bycell_model.npz")
    ap.add_argument("--acts", nargs="+", default=sorted(glob.glob("l23all_out/acts_*.npz")))
    ap.add_argument("--radii", default="5,10,15,20")
    ap.add_argument("--out", default="attr_analysis/clf_direction.json")
    ap.add_argument("--vectors", default="attr_analysis/clf_direction_vectors.npz")
    args = ap.parse_args()

    m = np.load(args.model, allow_pickle=True)
    mu, E, sd, W, b, T = m["mu"], m["E"], m["sd"], m["W"], m["b"], float(m["T"])
    classes = list(m["classes"].astype(str))
    cidx = {c: i for i, c in enumerate(classes)}
    L = int(m["layer"])

    def logits(x):
        return ((((x - mu) @ E) / sd) @ W + b) / T

    def logp(x, t):
        z = logits(x)
        return z[..., t] - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1)) - z.max(-1)

    def grad_logp(x, t):
        z = logits(x)
        p = np.exp(z - z.max())
        p /= p.sum()
        gf = (W[:, t] - W @ p) / T / sd
        return E @ gf

    z = np.load(CACHE, allow_pickle=True)
    names = list(z["names"].astype(str))
    M = z["M"][:, L, :].astype(np.float32)
    midx = {p: i for i, p in enumerate(names)}
    a = M[midx["default"]]

    X, cell, al = [], [], []
    for f in args.acts:
        za = np.load(f, allow_pickle=True)
        li = list(za["layers"]).index(L) if "layers" in za else L
        X.append(za["acts"][:, li, :].astype(np.float32))
        cell += list(za["cell"].astype(str))
        al += list(za["alpha"])
    X = np.concatenate(X)
    al = np.array(al)
    rows = defaultdict(list)
    for i, (c, s) in enumerate(zip(cell, al)):
        arm, t = parse(c)
        rows[(arm, t, float(s))].append(i)
    ro = {t: X[v].mean(0) for (arm, t, s), v in rows.items() if arm == "prompt"}
    centre = np.mean(list(ro.values()), 0)

    # the classifier's view of the read-outs
    print(f"{'arm':>7}{'s':>4}{'mean p(target)':>16}{'p>=0.1':>9}{'top-1 = target':>16}{'n':>7}")
    view = {}
    for (arm, s) in sorted({(arm, s) for arm, _, s in rows}):
        ps, hit = [], []
        for (arm2, t, s2), v in rows.items():
            if (arm2, s2) != (arm, s) or t not in cidx:
                continue
            P = logits(X[v])
            P = np.exp(P - P.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            ps += list(P[:, cidx[t]])
            hit += list(P.argmax(1) == cidx[t])
        ps, hit = np.array(ps), np.array(hit)
        view[f"{arm}|{s:g}"] = {"mean_p": float(ps.mean()), "p_ge_0.1": float((ps >= 0.1).mean()),
                                "top1": float(hit.mean()), "n": int(len(ps))}
        print(f"{arm:>7}{s:>4g}{ps.mean():>16.3f}{(ps >= 0.1).mean():>9.2f}{hit.mean():>16.2f}{len(ps):>7}")

    # the optimum inside a ball around the centre, per persona and radius
    radii = [float(r) for r in args.radii.split(",")]
    targets = [t for t in names if t != "default" and t in cidx]
    res = {r: [] for r in radii}
    best_dir = {}
    for t in targets:
        ti = cidx[t]
        u = M[midx[t]] - a
        u /= np.linalg.norm(u)
        pr = ro[t] - centre
        pr /= np.linalg.norm(pr)
        for R in radii:
            x = centre.copy()
            for _ in range(300):
                g = grad_logp(x, ti)
                x = x + 0.5 * g / (np.linalg.norm(g) + 1e-9)
                d = x - centre
                if np.linalg.norm(d) > R:
                    x = centre + d / np.linalg.norm(d) * R
            d = x - centre
            dn = d / np.linalg.norm(d)
            res[R].append((float(np.exp(logp(x, ti))), float(dn @ u), float(dn @ pr),
                           float(np.exp(logp(centre + R * u, ti))), float(np.exp(logp(centre + R * pr, ti)))))
            if R == radii[-1]:
                best_dir[t] = dn
    print(f"\nper persona, medians over {len(targets)} personas")
    print(f"{'R':>5}{'max p(target)':>15}{'cos(d*, u)':>12}{'cos(d*, prompted)':>19}{'p at R·u':>10}{'p at R·prompted':>17}")
    summary = {}
    for R in radii:
        v = np.array(res[R])
        med = np.median(v, 0)
        summary[f"R={R:g}"] = {"max_p": float(med[0]), "cos_u": float(med[1]), "cos_prompted": float(med[2]),
                               "p_along_u": float(med[3]), "p_along_prompted": float(med[4])}
        print(f"{R:>5g}{med[0]:>15.3f}{med[1]:>12.2f}{med[2]:>19.2f}{med[3]:>10.3f}{med[4]:>17.3f}")

    # where the actual read-outs sit along d* and along u, in the same units
    print(f"\nread-out displacement from the centre: along d* and along u (median over cells)")
    print(f"{'arm':>7}{'s':>4}{'|r|':>7}{'r·d*':>7}{'r·u':>7}{'cos(r,d*)':>11}{'cos(r,u)':>10}")
    for (arm, s) in sorted({(arm, s) for arm, _, s in rows}):
        vals = []
        for (arm2, t, s2), v in rows.items():
            if (arm2, s2) != (arm, s) or t not in best_dir:
                continue
            r = X[v].mean(0) - centre
            u = M[midx[t]] - a
            u /= np.linalg.norm(u)
            n = np.linalg.norm(r)
            vals.append((n, r @ best_dir[t], r @ u, r @ best_dir[t] / n, r @ u / n))
        med = np.median(np.array(vals), 0)
        print(f"{arm:>7}{s:>4g}{med[0]:>7.1f}{med[1]:>7.1f}{med[2]:>7.1f}{med[3]:>11.2f}{med[4]:>10.2f}")

    json.dump({"view": view, "optimum": summary, "radii": radii, "centre_to_a": float(np.linalg.norm(centre - a))},
              open(args.out, "w"), indent=1)
    np.savez(args.vectors, names=np.array(targets), dirs=np.stack([best_dir[t] for t in targets]),
             centre=centre, a=a, layer=L)
    print(f"\n-> {args.out}, {args.vectors}")


if __name__ == "__main__":
    main()
