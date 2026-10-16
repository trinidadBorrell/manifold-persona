"""A classifier that predicts the judge's persona ranking from a read-out activation.

Input: one judged answer = its mean response activation at one layer (read out with no
steering and no system prompt) plus the judge's ranked top-10 over the role names.

Model: PCA to a few dozen features, then multinomial logistic regression over the role
names plus a "nonsense" class. The target is soft: weight 1/rank on each of the ten
listed names, plus 1 on "nonsense" when the judge flagged the answer. The logits are
divided by a temperature fitted on a validation split so p(target) is on the judge's scale.

Steps in main():
  1. load_answers      pair each activation with its judge verdict
  2. split_answers     60/20/20 train, validation, test (by answer within a cell, or by persona)
  3. JudgeClassifier   fit PCA + weights on train, temperature on validation
  4. evaluate          recall@k, calibration and per-cell agreement with the judge on test
  5. save              the fitted model as .npz; load with JudgeClassifier.load(path)

Usage:
  python attr_analysis/judge_classifier.py --acts l23all_out/acts_*.npz --judge judge_rank_all \
      --layer 23 --split answer --balance --out X.json --save_model X_model.npz
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
SPLIT = (0.6, 0.8)  # train | validation | test


# ---------------------------------------------------------------- 1. data

def soft_target(rank, nonsense, cidx):
    """Judge's ranked names -> probability vector (1/rank each, nonsense = 1), normalised."""
    y = np.zeros(len(cidx), np.float32)
    for i, name in enumerate(rank[:10]):
        if name in cidx:
            y[cidx[name]] += 1.0 / (i + 1)
    if nonsense:
        y[cidx["nonsense"]] += 1.0
    return y / y.sum() if y.sum() > 0 else y


def load_answers(acts_files, judge_dirs, layer):
    """Return X (n, d), Y (n, classes), meta (list of dicts), classes (list of names)."""
    judged, roles = {}, None
    for d in judge_dirs:
        rows, _ = load_judge(d)
        roles = roles or sorted(json.load(open(f"{d}/roles.json")))
        for r in rows:
            judged[(r["cell"], float(r["alpha"]), int(r["qi"]))] = r
    classes = roles + ["nonsense"]
    cidx = {c: i for i, c in enumerate(classes)}

    X, Y, meta = [], [], []
    for f in acts_files:
        z = np.load(f, allow_pickle=True)
        li = list(z["layers"]).index(layer) if "layers" in z else layer
        A = z["acts"][:, li, :].astype(np.float32)
        for x, cell, al, qi in zip(A, z["cell"].astype(str), z["alpha"], z["qi"]):
            j = judged.get((cell, float(al), int(qi)))
            if j is None:
                continue
            X.append(x)
            Y.append(soft_target(j["rank"], j["nonsense"], cidx))
            meta.append({"cell": cell, "s": float(al), "qi": int(qi), "target": j["target"],
                         "judge_rank": j["rank"], "nonsense": j["nonsense"]})
    return np.stack(X), np.stack(Y), meta, classes


# ---------------------------------------------------------------- 2. split

def split_answers(meta, mode, seed):
    """Indices for train, validation, test.

    mode "answer": every (cell, strength) is split 60/20/20 across its answers, so all
    cells are seen in training and the test measures agreement on new answers.
    mode "persona": whole personas are held out, so the test measures transfer.
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["target"] if mode == "persona" else (m["cell"], m["s"])].append(i)
    tr, va, te = [], [], []
    if mode == "persona":
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        a, b = int(SPLIT[0] * n), int(SPLIT[1] * n)
        for part, ks in zip((tr, va, te), (keys[:a], keys[a:b], keys[b:])):
            for k in ks:
                part += groups[k]
        print(f"persona split: {a} / {b - a} / {n - b} personas")
    else:
        for v in groups.values():
            v = list(v)
            rng.shuffle(v)
            n = len(v)
            a, b = int(SPLIT[0] * n), int(SPLIT[1] * n)
            tr += v[:a]
            va += v[a:b]
            te += v[b:]
    return np.array(tr), np.array(va), np.array(te)


# ---------------------------------------------------------------- 3. model

class JudgeClassifier:
    """PCA -> standardise -> linear logits -> divide by temperature -> softmax."""

    def __init__(self, classes, layer):
        self.classes = list(classes)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.layer = int(layer)
        self.mu = self.E = self.sd = self.W = self.b = None
        self.T = 1.0

    # fitting ---------------------------------------------------------------
    def fit_features(self, X, n_pca):
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.E = Vt[:n_pca].T
        self.sd = ((X - self.mu) @ self.E).std(0) + 1e-6

    def features(self, X):
        return (((X - self.mu) @ self.E) / self.sd).astype(np.float32)

    def fit(self, X, Y, l2, weights=None, epochs=600, lr=0.05):
        """Weighted cross-entropy against the soft targets, ridge penalty l2 on W."""
        F = torch.tensor(self.features(X))
        Y = torch.tensor(Y)
        w = torch.ones(len(F)) if weights is None else torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        W = torch.zeros(F.shape[1], Y.shape[1], requires_grad=True)
        b = torch.zeros(Y.shape[1], requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = -(w * (Y * torch.log_softmax(F @ W + b, 1)).sum(1)).sum() + l2 * (W ** 2).sum()
            loss.backward()
            opt.step()
        self.W, self.b = W.detach().numpy(), b.detach().numpy()

    def calibrate(self, X, Y):
        """Fit one temperature on held-out answers so the softmax matches the soft targets."""
        logits = torch.tensor(self.features(X) @ self.W + self.b)
        Y = torch.tensor(Y)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(Y * torch.log_softmax(logits / logT.exp(), 1)).sum(1).mean()
            loss.backward()
            return loss
        opt.step(closure)
        self.T = float(logT.exp())

    # inference -------------------------------------------------------------
    def predict_proba(self, X):
        z = (self.features(X) @ self.W + self.b) / self.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def rank_of(self, P, targets):
        """1-based rank of each target class in its probability row."""
        order = np.argsort(-P, 1)
        return np.array([int(np.where(order[k] == t)[0][0]) + 1 for k, t in enumerate(targets)])

    # disk ------------------------------------------------------------------
    def save(self, path, **extra):
        np.savez(path, mu=self.mu, E=self.E, sd=self.sd, W=self.W, b=self.b, T=self.T,
                 classes=np.array(self.classes), layer=self.layer, **extra)

    @classmethod
    def load(cls, path):
        m = np.load(path, allow_pickle=True)
        self = cls(m["classes"].astype(str), int(m["layer"]))
        self.mu, self.E, self.sd, self.W, self.b = m["mu"], m["E"], m["sd"], m["W"], m["b"]
        self.T = float(m["T"])
        return self


def balance_weights(meta, idx, cidx):
    """Inverse frequency of the judge's top-1 name, mean 1, capped at 20."""
    top1 = np.array([cidx.get(meta[i]["judge_rank"][0], cidx["nonsense"]) for i in idx])
    freq = np.bincount(top1, minlength=len(cidx)).astype(np.float64)
    w = 1.0 / np.maximum(freq[top1], 1.0)
    return np.minimum(w / w.mean(), 20.0)


# ---------------------------------------------------------------- 4. evaluation

def ece(p_true, hit, bins=10):
    """Expected calibration error of p(target) against a 0/1 hit, with the per-bin rows."""
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_true >= lo) & (p_true < hi + (hi == 1))
        if m.any():
            e += m.mean() * abs(p_true[m].mean() - hit[m].mean())
            rows.append((lo, hi, int(m.sum()), float(p_true[m].mean()), float(hit[m].mean())))
    return e, rows


def evaluate(clf, P, meta_te):
    """Compare the classifier's ranking of the target with the judge's on the test answers."""
    out = {"per_answer": {}, "per_cell": {}}
    n = len(meta_te)
    tgt = np.array([clf.cidx[m["target"]] for m in meta_te])
    p_true = P[np.arange(n), tgt]
    cls_rank = clf.rank_of(P, tgt)
    jrank = np.array([m["judge_rank"].index(m["target"]) + 1 if m["target"] in m["judge_rank"] else 11
                      for m in meta_te])
    top1 = P.argmax(1)
    jtop1 = np.array([clf.cidx.get(m["judge_rank"][0], -1) for m in meta_te])

    # recall@k, classifier against judge
    print(f"\ntest answers: {n}")
    print(f"{'k':>4}{'classifier r@k':>16}{'judge r@k':>12}{'agree':>8}")
    for k in KS:
        c, j = cls_rank <= k, jrank <= k
        out["per_answer"][f"@{k}"] = {"classifier": float(c.mean()), "judge": float(j.mean()),
                                      "agree": float((c == j).mean())}
        print(f"{k:>4}{c.mean():>16.2f}{j.mean():>12.2f}{(c == j).mean():>8.2f}")
    out["per_answer"]["top1_matches_judge_top1"] = float((top1 == jtop1).mean())
    in5 = np.mean([clf.classes[top1[k]] in m["judge_rank"][:5] for k, m in enumerate(meta_te)])
    print(f"classifier top-1 equals judge top-1 in {(top1 == jtop1).mean():.2f} of answers")
    print(f"classifier top-1 inside judge top-5 in {in5:.2f}")

    # calibration of p(target) against the judge's hit at 1 and at 5
    for k in (1, 5):
        hit = (jrank <= k).astype(float)
        e, rows = ece(p_true, hit)
        out[f"ece_vs_judge@{k}"] = e
        print(f"\np(target) against judge hit@{k}: ECE {e:.3f}, "
              f"AUC-like spearman {spearmanr(p_true, hit)[0]:+.2f}")
        print(f"{'bin':>12}{'n':>6}{'mean p':>9}{'judge hit':>11}")
        for lo, hi, nn, mp, mh in rows:
            print(f"{f'[{lo:.1f},{hi:.1f})':>12}{nn:>6}{mp:>9.2f}{mh:>11.2f}")
    ent = -(P * np.log(P + 1e-12)).sum(1)
    out["entropy_hit5"] = float(ent[jrank <= 5].mean())
    out["entropy_miss5"] = float(ent[jrank > 5].mean())
    print(f"\nentropy when the judge has the target in its top-5: {out['entropy_hit5']:.2f} nats; "
          f"when it does not: {out['entropy_miss5']:.2f} nats")

    # per cell: classifier recall and mean p(target) against judged recall
    cells = defaultdict(list)
    for k, m in enumerate(meta_te):
        cells[(m["cell"], m["s"])].append(k)
    print(f"\nper cell ({len(cells)} cells, {n / len(cells):.0f} test answers each):")
    for k in KS:
        pc = np.array([np.mean(cls_rank[v] <= k) for v in cells.values()])
        pj = np.array([np.mean(jrank[v] <= k) for v in cells.values()])
        pp = np.array([p_true[v].mean() for v in cells.values()])
        out["per_cell"][f"@{k}"] = {"r_recall": float(pearsonr(pc, pj)[0]),
                                    "r_ptarget": float(pearsonr(pp, pj)[0]),
                                    "mae": float(np.abs(pc - pj).mean())}
        print(f"  recall@{k:<3} r(classifier recall, judge recall) {pearsonr(pc, pj)[0]:+.2f}   "
              f"r(mean p(target), judge recall) {pearsonr(pp, pj)[0]:+.2f}   "
              f"MAE {np.abs(pc - pj).mean():.2f}")
    return out


def dump_answers(clf, X, meta, tr, va, te, path):
    """Per-answer predictions for every loaded answer, with its split."""
    split = np.full(len(meta), "train", dtype=object)
    split[va], split[te] = "val", "test"
    P = clf.predict_proba(X)
    rows = []
    for i, m in enumerate(meta):
        o = np.argsort(-P[i])[:5]
        rows.append({**{k: m[k] for k in ("cell", "s", "qi", "target", "nonsense")},
                     "judge_top5": m["judge_rank"][:5], "split": split[i],
                     "p_target": float(P[i][clf.cidx[m["target"]]]),
                     "clf_top5": [[clf.classes[j], float(P[i][j])] for j in o]})
    json.dump(rows, open(path, "w"))
    print(f"-> {path} ({len(rows)} answers)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", required=True, help="read-out .npz files")
    ap.add_argument("--judge", nargs="+", required=True, help="judge directories (verdict_*.json)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="answer", choices=["answer", "persona"],
                    help="answer: 60/20/20 of answers within each cell; persona: whole personas held out")
    ap.add_argument("--balance", action="store_true",
                    help="weight answers by the inverse frequency of the judge's top-1 name")
    ap.add_argument("--out", default="attr_analysis/judge_classifier.json")
    ap.add_argument("--dump", default=None, help="per-answer predictions for every answer")
    ap.add_argument("--save_model", default=None, help="npz with the fitted PCA, weights and temperature")
    args = ap.parse_args()

    X, Y, meta, classes = load_answers(args.acts, args.judge, args.layer)
    print(f"{len(X)} judged answers, {len(classes)} classes, layer {args.layer}")
    tr, va, te = split_answers(meta, args.split, args.seed)

    clf = JudgeClassifier(classes, args.layer)
    clf.fit_features(X[tr], args.pca)
    w = balance_weights(meta, tr, clf.cidx) if args.balance else None
    clf.fit(X[tr], Y[tr], args.l2, weights=w)
    clf.calibrate(X[va], Y[va])
    print(f"temperature {clf.T:.2f}")

    out = {"layer": args.layer, "pca": args.pca, "l2": args.l2, "T": clf.T,
           "n_test": int(len(te)), "split": args.split, "balance": args.balance, "n_train": int(len(tr))}
    out.update(evaluate(clf, clf.predict_proba(X[te]), [meta[i] for i in te]))

    if args.save_model:
        clf.save(args.save_model, test_personas=np.array(sorted({meta[i]["target"] for i in te})))
    if args.dump:
        dump_answers(clf, X, meta, tr, va, te, args.dump)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
