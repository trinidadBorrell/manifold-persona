"""Task-length check on the saved coin-flip cells (CPU, no model).

Rules, fixed before the first run (owner go, 13/11/2026: "we can run 1"):
- Length = characters of the task description in tasks.json.
- z = (len harmless - len harmful) / 100 for each item's pair.
- Fit d = a + b z by OLS over the 400 items, with two-way cluster-robust SEs (harmless task, harmful task), t df 9.
  a = bias at equal length; b = change in bias per 100 extra characters on the harmless side.
- Reading: "length explains the lean" if b > 0 with its interval excluding 0 and a's interval including 0.
  "The lean survives equal length" if a's interval excludes 0 with the same sign as the cell bias. Else "unclear".
- Also reported: Pearson r over the 10 per-task means vs length, for each side, with a permutation p (20,000 draws),
  and the mean bias over pairs whose lengths differ by 25 characters or less (descriptive).
Exploratory: 10 tasks per side, so the check has low power and cannot separate harm from length causally.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from coinflip import T975_DF9, cluster_se, contributions  # noqa: E402

CELLS = ["base__plaintext", "sft__open_user_turn", "dpo__open_user_turn", "instruct__open_user_turn"]


def ols_cluster(d, z, h, x):
    X = np.column_stack([np.ones_like(z), z])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ d
    e = d - X @ beta

    def meat(g):
        _, idx = np.unique(g, return_inverse=True)
        G = idx.max() + 1
        S = np.zeros((G, 2))
        np.add.at(S, idx, X * e[:, None])
        return G / (G - 1) * S.T @ S

    h, x = np.asarray(h), np.asarray(x)
    V = XtX_inv @ (meat(h) + meat(x) - meat(h * 1000 + x)) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se


def perm_r(a, b, rng, n=20000):
    r = np.corrcoef(a, b)[0, 1]
    null = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n)])
    return float(r), float((np.abs(null) >= abs(r)).mean())


def main():
    tasks = json.loads((HERE / "tasks.json").read_text())
    Lh = np.array([len(s) for s in tasks["harmless"]])
    Lx = np.array([len(s) for s in tasks["harmful"]])
    rng = np.random.default_rng(0)
    out = {"lengths": {"harmless": Lh.tolist(), "harmful": Lx.tolist()}, "cells": {}}
    for c in CELLS:
        items = json.loads((HERE / "results/full" / f"{c}.json").read_text())["items"]
        d = contributions(items)[0]
        h = np.array([it["h"] for it in items])
        x = np.array([it["x"] for it in items])
        z = (Lh[h] - Lx[x]) / 100.0
        (a, b), (sa, sb) = ols_cluster(d, z, h, x)
        ci_a = [a - T975_DF9 * sa, a + T975_DF9 * sa]
        ci_b = [b - T975_DF9 * sb, b + T975_DF9 * sb]
        bias = float(d.mean())
        a_ex0 = ci_a[0] > 0 or ci_a[1] < 0
        b_pos = ci_b[0] > 0
        if b_pos and not a_ex0:
            reading = "length explains the lean"
        elif a_ex0 and np.sign(a) == np.sign(bias):
            reading = "the lean survives equal length"
        else:
            reading = "unclear"
        mean_x = np.array([d[x == k].mean() for k in range(10)])
        mean_h = np.array([d[h == k].mean() for k in range(10)])
        r_x, p_x = perm_r(Lx, mean_x, rng)
        r_h, p_h = perm_r(Lh, mean_h, rng)
        m = np.abs(Lh[h] - Lx[x]) <= 25
        se_m, _ = cluster_se(d[m], h[m], x[m])
        out["cells"][c] = dict(
            bias=bias, a_equal_length=float(a), a_ci=ci_a, b_per_100_chars=float(b), b_ci=ci_b, reading=reading,
            r_harmful_len_vs_task_mean=r_x, p_harmful=p_x, r_harmless_len_vs_task_mean=r_h, p_harmless=p_h,
            matched_n_pairs=int(m.sum() // 4), matched_bias=float(d[m].mean()), matched_se=se_m,
            per_harmful_task_mean=mean_x.tolist(), per_harmless_task_mean=mean_h.tolist())
    (HERE / "results/full/length_check.json").write_text(json.dumps(out, indent=1))
    for c, r in out["cells"].items():
        print(f"{c:26} bias {r['bias']:+.3f} | a(equal len) {r['a_equal_length']:+.3f} "
              f"[{r['a_ci'][0]:+.3f}, {r['a_ci'][1]:+.3f}] | b/100ch {r['b_per_100_chars']:+.3f} "
              f"[{r['b_ci'][0]:+.3f}, {r['b_ci'][1]:+.3f}] | {r['reading']}")
        print(f"{'':26} r harmful len {r['r_harmful_len_vs_task_mean']:+.2f} (p {r['p_harmful']:.3f}) | "
              f"r harmless len {r['r_harmless_len_vs_task_mean']:+.2f} (p {r['p_harmless']:.3f}) | "
              f"matched {r['matched_n_pairs']} pairs bias {r['matched_bias']:+.3f} (se {r['matched_se']:.3f})")


if __name__ == "__main__":
    main()
