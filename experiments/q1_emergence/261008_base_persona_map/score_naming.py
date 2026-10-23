"""Part 1 of the E3 pilot: the frozen blind-naming test (THRESHOLDS.md, fixed 09/10/2026 before the run).

Per reply and judge: hit@5 = target role in the judge's 5 guesses; hit@1 = first guess. Data: 20 roles x
5 wordings x 18 shared questions per condition and seed. Macro = mean of the 20 role means.
  Rule 1  macro hit@5 crossed-bootstrap 95% lower bound >= 0.20 and wrong-role permutation p < 0.01
  Rule 2  >= 10 of 20 roles above their own null 99th percentile
  Rule 3  rules 1-2 hold after removing copied and self-naming replies
  Rule 4  invalid <= 10% and the complete-cell analysis gives the same decision
A condition passes a rule only if both judges pass it (owner decision A).
Bootstrap: 2,000 draws, roles and questions resampled independently, a role keeps all its wordings.
Null: 9,999 permutations of the role labels within each question, whole role blocks.
Invalid masks: "frozen" (token-limit stop or a 4-gram seen 3+ times); "none" (every reply scored);
for the strict check an invalid reply counts as a miss. Judge-based masks are added with --mask FILE.

    python score_naming.py [--mask NAME=FILE ...] [--out results/naming_sealed.json] [--show]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROLES = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
         "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
         "pirate", "vampire", "robot", "void"]
JUDGES = {"gpt": "answers_gpt", "claude": "answers_claude"}
CELLS = [("run_main", c) for c in ("desc", "ex2", "ex2_other", "ex2_filler")] + [("run_rep", "desc"), ("run_rep", "ex2")]
N_BOOT, N_PERM, SEED = 2000, 9999, 20261016
WORD = re.compile(r"[a-z']+")
NAME_TERMS = {  # auto_examples.py NAME_TERMS for the 20 pilot roles
    "assistant": ["assistant"], "chef": ["chef", "cook"], "detective": ["detective", "investigator"],
    "bartender": ["bartender", "barkeep", "mixologist"], "criminal": ["criminal", "outlaw"],
    "toddler": ["toddler"], "teenager": ["teenager", "teen", "teenage", "adolescent"],
    "grandparent": ["grandparent", "grandmother", "grandfather", "grandma", "grandpa"],
    "refugee": ["refugee", "displaced", "displacement", "asylum"], "cynic": ["cynic", "cynical"],
    "optimist": ["optimist", "optimistic"], "stoic": ["stoic", "stoicism"],
    "narcissist": ["narcissist", "narcissistic"], "pirate": ["pirate"], "vampire": ["vampire"],
    "robot": ["robot", "robotic"], "void": ["void", "between worlds", "between reality"],
    "translator": ["translator", "interpreter"], "counselor": ["counselor", "counsellor"],
    "consultant": ["consultant"]}


def grams(text, n):
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def rep4(text):
    w = re.findall(r"\w+", text.lower())
    return any(v >= 3 for v in collections.Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3)).values())


def examples_text(prompt, guest):
    body = prompt.split(f"About the guest: {guest}\n\n", 1)[-1]
    return body.rsplit("\n\nInterviewer:", 1)[0] if "\n\nInterviewer:" in body else ""


def load():
    key = json.loads((HERE / "judge" / "KEY.json").read_text())
    by_run = {}
    for run in ("run_main", "run_rep"):
        prompts = {json.loads(l)["i"]: json.loads(l) for l in open(HERE / run / "items.jsonl")}
        for l in open(HERE / run / "replies.jsonl"):
            r = json.loads(l)
            r["prompt"] = prompts[r["i"]]["prompt"]
            by_run[(run, r["i"])] = r
    guesses = {}
    for j, d in JUDGES.items():
        g = {}
        for f in (HERE / "judge" / d).glob("batch_*.json"):
            g.update({k: [x.lower() for x in v] for k, v in json.loads(f.read_text()).items()})
        if set(g) != set(key):
            raise SystemExit(f"{j}: {len(set(key) - set(g))} ids missing")
        guesses[j] = g
    return key, by_run, guesses


def flags(r):
    text = r["reply"] or ""
    ex = examples_text(r["prompt"], r["guest"]) if r["cond"] in ("ex2", "ex2_other") else ""
    copied = bool(ex and grams(text, 8) & grams(ex, 8)) or bool(grams(text, 5) & grams(r["guest"], 5))
    names = "|".join(re.escape(t) for t in NAME_TERMS[r["role"]])
    selfname = bool(re.search(rf"\b({names})(s|es)?\b", text, re.I))
    return dict(limit=r["ended_by"] == "limit", rep=rep4(text), empty=not text.strip(), copied=copied, selfname=selfname)


def cube(key, by_run, guesses, run, cond, donor_shift=None):
    """G[j][r, q, w, t]: judge j put role t in the top 5 for reply (role r, question q, wording w)."""
    rows = [v | {"id": k} for k, v in key.items() if v["run"] == run and v["cond"] == cond and v["role"] in ROLES]
    reps = {k: by_run[(run, v["i"])] for k, v in ((r["id"], r) for r in rows)}
    rows = [r for r in rows if reps[r["id"]]["shared_q"]]
    if donor_shift is not None:  # ex2_other: one donor per role, fixed by its shift along the pilot order
        rows = [r for r in rows if (ROLES.index(r["example_role"]) - ROLES.index(r["role"])) % 20 == donor_shift]
    qs = sorted({r["question_idx"] for r in rows})
    if len(qs) != 18 or len(rows) != 20 * 18 * 5:
        raise SystemExit(f"{run}/{cond}/{donor_shift}: {len(rows)} rows, {len(qs)} questions")
    G = {j: np.zeros((20, 18, 5, 20), bool) for j in guesses}
    G1 = {j: np.zeros((20, 18, 5), bool) for j in guesses}
    F = collections.defaultdict(lambda: np.zeros((20, 18, 5), bool))
    donor = np.zeros((20, 18, 5), int)
    ids = np.empty((20, 18, 5), object)
    for r in rows:
        a, b, w = ROLES.index(r["role"]), qs.index(r["question_idx"]), r["wording"]
        if ids[a, b, w] is not None:
            raise SystemExit(f"{run}/{cond}: two replies in cell {r['role']}, q{r['question_idx']}, w{w}")
        ids[a, b, w] = (run, r["i"])
        for j, g in guesses.items():
            top = g[r["id"]]
            for t in top:
                if t in ROLES:
                    G[j][a, b, w, ROLES.index(t)] = True
            G1[j][a, b, w] = top[0] == r["role"]
        for k, v in flags(reps[r["id"]]).items():
            F[k][a, b, w] = v
        if r["example_role"] in ROLES:
            donor[a, b, w] = ROLES.index(r["example_role"])
    return G, G1, dict(F), donor, ids


def macro(H, keep):
    """Mean over roles of the role mean over kept replies; H, keep: (20, 18, 5)."""
    num, den = (H * keep).sum((1, 2)), keep.sum((1, 2))
    return float(np.mean(num[den > 0] / den[den > 0])) if (den > 0).any() else float("nan")


def boot(H, keep, rng):
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        ri, qi = rng.integers(0, 20, 20), rng.integers(0, 18, 18)
        out[k] = macro(H[ri][:, qi], keep[ri][:, qi])
    return np.nanpercentile(out, [2.5, 97.5]).tolist()


def perm_null(G, keep, rng):
    """Per TARGET role and macro hit@5 when each question's role labels are permuted as whole blocks.

    Under a permutation the replies of role r at question q are scored against target P[r, q]; target t's
    null rate uses the replies that now carry t (role Pinv[t, q]), and their keep mask moves with them."""
    roles_null = np.empty((N_PERM, 20))
    q, w = np.arange(18)[None, :, None], np.arange(5)[None, None, :]
    t = np.arange(20)[:, None, None]
    for k in range(N_PERM):
        P = np.stack([rng.permutation(20) for _ in range(18)], 1)  # P[r, q] = permuted target of role r at q
        Pinv = np.argsort(P, axis=0)                               # Pinv[t, q] = role whose replies now carry t
        H = G[Pinv[:, :, None], q, w, t]
        K = keep[Pinv[:, :, None], q, w]
        num, den = (H * K).sum((1, 2)), K.sum((1, 2))
        roles_null[k] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return roles_null


def rules(G, keep, rng, strict_mask=None):
    H = G[np.arange(20)[:, None, None], np.arange(18)[None, :, None], np.arange(5)[None, None, :],
          np.arange(20)[:, None, None]]
    if strict_mask is not None:
        H = H & ~strict_mask
    obs = macro(H, keep)
    ci = boot(H, keep, rng)
    rn = perm_null(G if strict_mask is None else G & ~strict_mask[..., None], keep, rng)
    null_macro = np.nanmean(rn, 1)
    p = float((1 + np.sum(null_macro >= obs)) / (1 + N_PERM)) if np.isfinite(obs) else float("nan")
    num, den = (H * keep).sum((1, 2)), keep.sum((1, 2))
    role_obs = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    p99 = np.nanpercentile(rn, 99, axis=0)
    covered = int(np.sum(role_obs > p99))
    return dict(macro_hit5=obs, ci95=ci, perm_p=p, null_mean=float(null_macro.mean()),
                null_p99=float(np.percentile(null_macro, 99)), roles_covered=covered,
                role_hit5=dict(zip(ROLES, role_obs.round(4).tolist())),
                rule1=bool(ci[0] >= 0.20 and p < 0.01), rule2=bool(covered >= 10))


def target_hits(G):
    i = np.arange(20)
    return G[i[:, None, None], np.arange(18)[None, :, None], np.arange(5)[None, None, :], i[:, None, None]]


def score_cell(G, G1, F, masks, rng):
    """Per mask and judge: the four rules; then the frozen two-judge decision (both judges must pass)."""
    allkeep = np.ones((20, 18, 5), bool)
    base = {j: dict(macro_hit1=macro(G1[j], allkeep), all_replies=rules(G[j], allkeep, rng),
                    clean=rules(G[j], ~(F["copied"] | F["selfname"]), rng)) for j in G}
    res = {}
    for mname, inv in masks.items():
        rate = float(inv.mean())
        complete = np.broadcast_to(~inv.any(2, keepdims=True), inv.shape)
        res[mname] = {}
        for j in G:
            itt, clean = base[j]["all_replies"], base[j]["clean"]
            strict = rules(G[j], allkeep, rng, strict_mask=inv)
            cc = rules(G[j], complete & ~inv, rng)
            cc_clean = rules(G[j], complete & ~inv & ~(F["copied"] | F["selfname"]), rng)
            dec = itt["rule1"] and itt["rule2"] and clean["rule1"] and clean["rule2"]  # rules 1-3
            cc_dec = cc["rule1"] and cc["rule2"] and cc_clean["rule1"] and cc_clean["rule2"]
            res[mname][j] = dict(
                invalid_rate=rate, macro_hit1=base[j]["macro_hit1"], all_replies=itt, invalid_as_miss=strict,
                complete_cells=cc, complete_cells_clean=cc_clean, clean=clean,
                rule3=bool(clean["rule1"] and clean["rule2"]),
                rule4=bool(rate <= 0.10 and cc_dec == dec),
                strict_same_decision=bool((strict["rule1"] and strict["rule2"]) == (itt["rule1"] and itt["rule2"])))
        r = res[mname]
        res[mname]["both_judges"] = dict(
            rule1=all(r[j]["all_replies"]["rule1"] for j in G), rule2=all(r[j]["all_replies"]["rule2"] for j in G),
            rule3=all(r[j]["rule3"] for j in G), rule4=all(r[j]["rule4"] for j in G),
            strict_same_decision=all(r[j]["strict_same_decision"] for j in G),
            decision_p=max(max(r[j]["all_replies"]["perm_p"], r[j]["clean"]["perm_p"]) for j in G))
        bj = res[mname]["both_judges"]
        bj["plays"] = bool(bj["rule1"] and bj["rule2"] and bj["rule3"] and bj["rule4"])
    return res


def paired_ci(A, B, rng):
    """Crossed-bootstrap 95% CI of macro(A) - macro(B) on the same roles and questions."""
    keep = np.ones((20, 18, 5), bool)
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        ri, qi = rng.integers(0, 20, 20), rng.integers(0, 18, 18)
        out[k] = macro(A[ri][:, qi], keep) - macro(B[ri][:, qi], keep)
    return dict(diff=macro(A, keep) - macro(B, keep), ci95=np.percentile(out, [2.5, 97.5]).tolist())


def copy_baseline(by_run):
    """[RT 18] Copy-rule rate when each ex2 reply is checked against another role's examples (shift 10)."""
    ex2 = {(r["role"], r["question_idx"], r["wording"]): r for (run, _), r in by_run.items()
           if run == "run_main" and r["cond"] == "ex2" and r["shared_q"] and r["role"] in ROLES}
    own = other = 0
    for (role, q, w), r in ex2.items():
        o = ex2[(ROLES[(ROLES.index(role) + 10) % 20], q, w)]
        text = r["reply"] or ""
        own += bool(grams(text, 8) & grams(examples_text(r["prompt"], r["guest"]), 8))
        other += bool(grams(text, 8) & grams(examples_text(o["prompt"], o["guest"]), 8))
    return dict(n=len(ex2), own_examples_rate=own / len(ex2), unrelated_examples_rate=other / len(ex2))


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    adj, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * pvals[name]))
        adj[name] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", action="append", default=[], help="NAME=FILE: JSON list of [run, i] marked invalid")
    ap.add_argument("--out", default=str(HERE / "results" / "naming_sealed.json"))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    key, by_run, guesses = load()
    rng = np.random.default_rng(SEED)
    extra = {}
    for m in a.mask:
        name, f = m.split("=", 1)
        extra[name] = {tuple(x) for x in json.loads(Path(f).read_text())}
    hits = {}
    out = dict(settings=dict(n_boot=N_BOOT, n_perm=N_PERM, seed=SEED, judges=list(JUDGES), masks=["none", "frozen", *extra]))
    for run, cond in CELLS:
        shifts = [6, 10, 14] if cond == "ex2_other" else [None]
        for s in shifts:
            G, G1, F, donor, ids = cube(key, by_run, guesses, run, cond, s)
            masks = {"none": np.zeros((20, 18, 5), bool), "frozen": F["limit"] | F["rep"] | F["empty"]}
            for name, sel in extra.items():
                masks[name] = np.vectorize(lambda x: tuple(x) in sel, otypes=[bool])(ids)
            name = f"{run}/{cond}" + (f"/shift{s}" if s is not None else "")
            cell = dict(flags={k: float(v.mean()) for k, v in F.items()}, results=score_cell(G, G1, F, masks, rng))
            if cond == "ex2_other":
                Hd = {j: G[j][np.arange(20)[:, None, None], np.arange(18)[None, :, None], np.arange(5)[None, None, :], donor]
                      for j in G}
                cell["donor_hit5"] = {j: macro(Hd[j], np.ones((20, 18, 5), bool)) for j in G}
                cell["target_minus_donor"] = {j: paired_ci(target_hits(G[j]).astype(float), Hd[j].astype(float), rng)
                                              for j in G}
            out[name] = cell
            for j in G:
                hits.setdefault((run, cond), {}).setdefault(j, []).append(target_hits(G[j]).astype(float))
            print("scored", name)
    H = {c: {j: np.mean(v, 0) for j, v in d.items()} for c, d in hits.items()}  # ex2_other: mean over 3 donors
    pairs = [("ex2", "desc"), ("ex2_other", "desc"), ("ex2", "ex2_other"), ("ex2_filler", "desc"), ("ex2", "ex2_filler")]
    out["contrasts_run_main"] = {f"{x}-{y}": {j: paired_ci(H[("run_main", x)][j], H[("run_main", y)][j], rng)
                                             for j in JUDGES} for x, y in pairs}
    out["seed_agreement"] = {c: {j: paired_ci(H[("run_main", c)][j], H[("run_rep", c)][j], rng) for j in JUDGES}
                             for c in ("desc", "ex2")}
    pvals = {f"plays_{c}": out[f"run_main/{c}"]["results"]["none"]["both_judges"]["decision_p"] for c in ("desc", "ex2")}
    pvals.update(info_desc=1.0, info_ex2=1.0)  # Part 2 not run: counted as p = 1 in the family of four
    adj = holm(pvals)
    out["holm"] = dict(raw=pvals, adjusted=adj)
    out["primary_decisions"] = {}
    for c in ("desc", "ex2"):
        for m, r in out[f"run_main/{c}"]["results"].items():  # final call: all rules and Holm-adjusted p < 0.01
            r["both_judges"]["plays_holm"] = bool(r["both_judges"]["plays"] and adj[f"plays_{c}"] < 0.01)
        out["primary_decisions"][f"plays_{c}"] = {m: r["both_judges"]["plays_holm"]
                                                  for m, r in out[f"run_main/{c}"]["results"].items()}
    out["seed_rule_agreement"] = {c: {m: {k: (out[f"run_main/{c}"]["results"][m]["both_judges"][k],
                                              out[f"run_rep/{c}"]["results"][m]["both_judges"][k])
                                          for k in ("rule1", "rule2", "rule3", "rule4", "plays")}
                                      for m in out[f"run_main/{c}"]["results"]} for c in ("desc", "ex2")}
    out["copy_rule_unrelated_pairs"] = copy_baseline(by_run)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, "(sealed: no numbers printed)" if not a.show else "")
    if a.show:
        for name, cell in out.items():
            if "results" not in cell:
                continue
            for m, rj in cell["results"].items():
                for j, r in rj.items():
                    if j == "both_judges":
                        continue
                    print(f"{name:22s} {m:7s} {j:6s} hit5 {r['all_replies']['macro_hit5']:.3f} "
                          f"CI {r['all_replies']['ci95'][0]:.3f}-{r['all_replies']['ci95'][1]:.3f} "
                          f"p {r['all_replies']['perm_p']:.4f} cov {r['all_replies']['roles_covered']} hit1 {r['macro_hit1']:.3f}")


if __name__ == "__main__":
    main()
