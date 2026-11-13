"""E3 seed check: how much do the two seeds agree, and how much would a third seed narrow the interval?

Uses only saved judge answers (no new calls). Same cells, macro and crossed bootstrap as score_naming.py.
"""
import json, sys
import numpy as np

E3 = sys.argv[1]
sys.path.insert(0, E3)
import score_naming as sn  # noqa: E402

key, by_run, guesses = sn.load()
B = 4000
out = {}


def macro_frac(H):  # H: (20, 18, 5) floats in [0, 1]; equal weights, so a plain role mean
    return float(H.mean((1, 2)).mean())


def boot(H, rng, roles=True, qs=True):
    v = np.empty(B)
    for k in range(B):
        ri = rng.integers(0, 20, 20) if roles else np.arange(20)
        qi = rng.integers(0, 18, 18) if qs else np.arange(18)
        v[k] = macro_frac(H[ri][:, qi])
    return v


for cond in ("desc", "ex2"):
    for j in ("gpt", "claude"):
        rng = np.random.default_rng(20261023)
        H, F = [], []
        for run in ("run_main", "run_rep"):
            G, G1, Fl, donor, ids = sn.cube(key, by_run, guesses, run, cond)
            H.append(sn.target_hits(G[j]).astype(float))
            F.append(Fl["copied"] | Fl["selfname"])
        h1, h2 = H
        a, b = h1.astype(bool).ravel(), h2.astype(bool).ravel()
        agree = dict(same_outcome=float((a == b).mean()), p_hit2_if_hit1=float(b[a].mean()),
                     p_hit2_if_miss1=float(b[~a].mean()), phi=float(np.corrcoef(a, b)[0, 1]))
        # role-level agreement between seeds
        r1, r2 = h1.mean((1, 2)), h2.mean((1, 2))
        role_r = float(np.corrcoef(r1, r2)[0, 1])

        v1, v2 = boot(h1, rng), boot(h2, rng)
        avg = (h1 + h2) / 2
        v12 = boot(avg, rng)
        var1 = (v1.var() + v2.var()) / 2
        var2 = v12.var()
        Bn = max(2 * (var1 - var2), 0.0)   # SE^2(k) = A + Bn/k  -> var1 = A + Bn, var2 = A + Bn/2
        A = var1 - Bn
        pred = {k: float(np.sqrt(A + Bn / k)) for k in (1, 2, 3, 5)}
        roles_only, qs_only = boot(avg, rng, qs=False).var(), boot(avg, rng, roles=False).var()

        # rule 3 view: copied and self-naming replies removed (keep mask), per seed and pooled
        def lb_keep(Hs, Ks):
            v = np.empty(B)
            for k in range(B):
                ri, qi = rng.integers(0, 20, 20), rng.integers(0, 18, 18)
                num = sum((h[ri][:, qi] * kk[ri][:, qi]).sum((1, 2)) for h, kk in zip(Hs, Ks))
                den = sum(kk[ri][:, qi].sum((1, 2)) for kk in Ks)
                v[k] = np.mean(num[den > 0] / den[den > 0])
            return float(np.percentile(v, 2.5))
        K1, K2 = ~F[0], ~F[1]
        rule3 = dict(seed1_lb=lb_keep([h1], [K1]), seed2_lb=lb_keep([h2], [K2]), pooled_lb=lb_keep([h1, h2], [K1, K2]))

        out[f"{cond}/{j}"] = dict(
            macro=[macro_frac(h1), macro_frac(h2), macro_frac(avg)],
            ci_seed1=np.percentile(v1, [2.5, 97.5]).tolist(), ci_seed2=np.percentile(v2, [2.5, 97.5]).tolist(),
            ci_two_seeds=np.percentile(v12, [2.5, 97.5]).tolist(),
            se_by_seeds=pred, share_of_var_from_seed_noise_at_k1=float(Bn / var1) if var1 else None,
            var_roles_only=float(roles_only), var_questions_only=float(qs_only),
            item_agreement=agree, role_level_r=role_r, rule3_lower_bounds=rule3)

json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out.items():
    print(f"\n== {k}")
    print(" macro seed1/seed2/avg:", [round(x, 3) for x in v["macro"]])
    print(" 95% CI seed1:", [round(x, 3) for x in v["ci_seed1"]], " seed2:", [round(x, 3) for x in v["ci_seed2"]],
          " two seeds pooled:", [round(x, 3) for x in v["ci_two_seeds"]])
    print(" SE with 1/2/3/5 seeds:", {k2: round(x, 4) for k2, x in v["se_by_seeds"].items()},
          "| share of 1-seed variance from seed noise:", round(v["share_of_var_from_seed_noise_at_k1"], 3))
    print(" variance, roles resampled only:", round(v["var_roles_only"], 6), "| questions only:", round(v["var_questions_only"], 6))
    print(" item agreement:", {k2: round(x, 3) for k2, x in v["item_agreement"].items()}, "| role-level r:", round(v["role_level_r"], 3))
    print(" rule-3 lower bounds (copied+self-naming removed):", {k2: round(x, 3) for k2, x in v["rule3_lower_bounds"].items()})
