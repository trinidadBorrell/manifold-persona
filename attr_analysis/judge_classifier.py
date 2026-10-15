"""A calibrated classifier from read-out activations to the judge's persona ranking.

Each judged answer gives one point: its mean response activation at one layer (encoded
with no steering and no system prompt) and the judge's ranked top-10 over the 276 role
names. The target is a soft distribution over the names, weight 1/rank over the ten
listed, with nonsense as an extra class. A multinomial logistic regression on PCA
features is fitted by cross-entropy on a training split and temperature-scaled on a
validation split; the test split asks whether its probabilities track the judge.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "attr_analysis")
from score_rank import load as load_judge

KS = (1, 5, 10)


def soft_target(rank, nonsense, classes, cidx):
    y = np.zeros(len(classes), np.float32)
    for i, name in enumerate(rank[:10]):
        if name in cidx:
            y[cidx[name]] += 1.0 / (i + 1)
    if nonsense:
        y[cidx["nonsense"]] += 1.0
    return y / y.sum() if y.sum() > 0 else y


def fit(X, Y, l2, epochs=600, lr=0.05):
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    W = torch.zeros(X.shape[1], Y.shape[1], requires_grad=True)
    b = torch.zeros(Y.shape[1], requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ W + b
        loss = -(Y * torch.log_softmax(logits, 1)).sum(1).mean() + l2 * (W ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def temperature(logits, Y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(Y * torch.log_softmax(logits / logT.exp(), 1)).sum(1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def ece(p_true, hit, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_true >= lo) & (p_true < hi + (hi == 1))
        if m.any():
            e += m.mean() * abs(p_true[m].mean() - hit[m].mean())
            rows.append((lo, hi, int(m.sum()), float(p_true[m].mean()), float(hit[m].mean())))
    return e, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", required=True)
    ap.add_argument("--judge", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="attr_analysis/judge_classifier.json")
    ap.add_argument("--dump", default=None, help="per-answer predictions for every answer")
    args = ap.parse_args()

    judged = {}
    roles = None
    for d in args.judge:
        rows, _ = load_judge(d)
        roles = roles or sorted(json.load(open(f"{d}/roles.json")))
        for r in rows:
            judged[(r["cell"], float(r["alpha"]), int(r["qi"]))] = r
    classes = roles + ["nonsense"]
    cidx = {c: i for i, c in enumerate(classes)}

    X, Y, meta = [], [], []
    for f in args.acts:
        z = np.load(f, allow_pickle=True)
        A = z["acts"][:, args.layer, :].astype(np.float32)
        for x, cell, al, qi in zip(A, z["cell"].astype(str), z["alpha"], z["qi"]):
            j = judged.get((cell, float(al), int(qi)))
            if j is None:
                continue
            X.append(x)
            Y.append(soft_target(j["rank"], j["nonsense"], classes, cidx))
            meta.append({"cell": cell, "s": float(al), "qi": int(qi), "target": j["target"],
                         "judge_rank": j["rank"], "nonsense": j["nonsense"]})
    X, Y = np.stack(X), np.stack(Y)
    print(f"{len(X)} judged answers, {len(classes)} classes, layer {args.layer}")

    # split by answer, stratified by cell: 60 / 20 / 20
    rng = np.random.default_rng(args.seed)
    by_cell = defaultdict(list)
    for i, m in enumerate(meta):
        by_cell[(m["cell"], m["s"])].append(i)
    tr, va, te = [], [], []
    for v in by_cell.values():
        v = list(v)
        rng.shuffle(v)
        n = len(v)
        tr += v[:int(0.6 * n)]
        va += v[int(0.6 * n):int(0.8 * n)]
        te += v[int(0.8 * n):]
    tr, va, te = np.array(tr), np.array(va), np.array(te)

    mu = X[tr].mean(0)
    _, S, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
    E = Vt[:args.pca].T
    sd = ((X[tr] - mu) @ E).std(0) + 1e-6
    feat = lambda idx: (((X[idx] - mu) @ E) / sd).astype(np.float32)

    W, b = fit(feat(tr), Y[tr], args.l2)
    lv = torch.tensor(feat(va)) @ W + b
    T = temperature(lv, torch.tensor(Y[va]))
    print(f"temperature {T:.2f}")

    def predict(idx):
        logits = (torch.tensor(feat(idx)) @ W + b) / T
        return torch.softmax(logits, 1).numpy()

    P = predict(te)
    tgt = np.array([cidx[meta[i]["target"]] for i in te])
    p_true = P[np.arange(len(te)), tgt]
    order = np.argsort(-P, 1)
    cls_rank = np.array([int(np.where(order[k] == tgt[k])[0][0]) + 1 for k in range(len(te))])
    jrank = np.array([(meta[i]["judge_rank"].index(meta[i]["target"]) + 1
                       if meta[i]["target"] in meta[i]["judge_rank"] else 11) for i in te])
    ent = -(P * np.log(P + 1e-12)).sum(1)

    print(f"\ntest answers: {len(te)}")
    print(f"{'k':>4}{'classifier r@k':>16}{'judge r@k':>12}{'agree':>8}")
    out = {"layer": args.layer, "pca": args.pca, "l2": args.l2, "T": T, "n_test": int(len(te)),
           "per_answer": {}, "per_cell": {}}
    for k in KS:
        c = cls_rank <= k
        j = jrank <= k
        out["per_answer"][f"@{k}"] = {"classifier": float(c.mean()), "judge": float(j.mean()),
                                      "agree": float((c == j).mean())}
        print(f"{k:>4}{c.mean():>16.2f}{j.mean():>12.2f}{(c == j).mean():>8.2f}")
    top1 = np.array([order[k][0] for k in range(len(te))])
    jtop1 = np.array([cidx.get(meta[i]["judge_rank"][0], -1) for i in te])
    out["per_answer"]["top1_matches_judge_top1"] = float((top1 == jtop1).mean())
    print(f"classifier top-1 equals judge top-1 in {(top1 == jtop1).mean():.2f} of answers")
    print(f"classifier top-1 inside judge top-5 in "
          f"{np.mean([classes[top1[k]] in meta[te[k]]['judge_rank'][:5] for k in range(len(te))]):.2f}")

    # calibration of p(target) against the judge's hit at 1 and at 5
    for k in (1, 5):
        hit = (jrank <= k).astype(float)
        e, rows = ece(p_true, hit)
        out[f"ece_vs_judge@{k}"] = e
        print(f"\np(target) against judge hit@{k}: ECE {e:.3f}, "
              f"AUC-like spearman {spearmanr(p_true, hit)[0]:+.2f}")
        print(f"{'bin':>12}{'n':>6}{'mean p':>9}{'judge hit':>11}")
        for lo, hi, n, mp, mh in rows:
            print(f"{f'[{lo:.1f},{hi:.1f})':>12}{n:>6}{mp:>9.2f}{mh:>11.2f}")
    print(f"\nentropy when the judge has the target in its top-5: "
          f"{ent[jrank <= 5].mean():.2f} nats; when it does not: {ent[jrank > 5].mean():.2f} nats")
    out["entropy_hit5"] = float(ent[jrank <= 5].mean())
    out["entropy_miss5"] = float(ent[jrank > 5].mean())

    # per cell on the test answers: mean p(target) and predicted recall against judged recall
    cells = defaultdict(list)
    for k, i in enumerate(te):
        cells[(meta[i]["cell"], meta[i]["s"])].append(k)
    print(f"\nper cell ({len(cells)} cells, {len(te) / len(cells):.0f} test answers each):")
    for kk in KS:
        pc = np.array([np.mean([cls_rank[k] <= kk for k in v]) for v in cells.values()])
        pj = np.array([np.mean([jrank[k] <= kk for k in v]) for v in cells.values()])
        pp = np.array([np.mean([p_true[k] for k in v]) for v in cells.values()])
        out["per_cell"][f"@{kk}"] = {"r_recall": float(pearsonr(pc, pj)[0]),
                                     "r_ptarget": float(pearsonr(pp, pj)[0]),
                                     "mae": float(np.abs(pc - pj).mean())}
        print(f"  recall@{kk:<3} r(classifier recall, judge recall) {pearsonr(pc, pj)[0]:+.2f}   "
              f"r(mean p(target), judge recall) {pearsonr(pp, pj)[0]:+.2f}   "
              f"MAE {np.abs(pc - pj).mean():.2f}")
    if args.dump:
        split = np.full(len(meta), "train", dtype=object)
        split[va] = "val"
        split[te] = "test"
        Pa = predict(np.arange(len(meta)))
        rows = []
        for i, m in enumerate(meta):
            o = np.argsort(-Pa[i])[:5]
            rows.append({**{k: m[k] for k in ("cell", "s", "qi", "target", "nonsense")},
                         "judge_top5": m["judge_rank"][:5], "split": split[i],
                         "p_target": float(Pa[i][cidx[m["target"]]]),
                         "clf_top5": [[classes[j], float(Pa[i][j])] for j in o]})
        json.dump(rows, open(args.dump, "w"))
        print(f"-> {args.dump} ({len(rows)} answers)")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
