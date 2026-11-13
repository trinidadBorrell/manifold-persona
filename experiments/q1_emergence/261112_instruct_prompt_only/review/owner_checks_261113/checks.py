"""Owner-review checks on the Instruct prompt-only control (rules: RULES.md). CPU, saved data only.

  .venv/bin/python -I checks.py            (from this folder; add --n_boot 100 for a quick engineering check)
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
POD_A = EXP.parent / "261029_instruct_persona_map"
D1 = Path("/Users/brendon/research-log/manifold-persona-log/Reference/2026-11-06 A D1 finish/d1_hit.json")
STATES_SAVED = [8, 9, 16, 17, 23, 24]
PERM_SEED = 20261113


def pod_a():
    spec = importlib.util.spec_from_file_location("_poda_analysis", POD_A / "analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def full_map(A, X, rows, roles, questions):
    """Role-pair distances from all questions and wordings (pod A's centring and cosine distance)."""
    S, N = A.cell_means(X, rows, roles, questions)
    side = A.M.Side(S, N, np.ones(len(questions), bool), np.ones(S.shape[2], bool))
    g = side.grams(np.ones((1, len(questions))))[0]
    n = np.sqrt(np.diag(g))
    d = 1 - g / np.outer(n, n)
    i, j = np.triu_indices(len(roles), 1)
    return d[i, j], i, j


def perm_p(x, y, n=9999, seed=PERM_SEED):
    rng = np.random.default_rng(seed)
    obs = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return float(obs), float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_halves", type=int, default=50)
    a = ap.parse_args()
    A = pod_a()
    M = A.M
    run = POD_A / "run_full"
    manifest = json.loads((run / "manifest.json").read_text())
    rows_r = [json.loads(line) for line in (run / "rows.jsonl").open()]
    roles = [r for r in manifest["roles"] if r != "default"]
    questions = sorted({r["question"] for r in rows_r})
    acts_r = np.load(run / "reply_mean_acts.npy", mmap_mode="r")
    c3 = EXP / "run_c3_full"
    keys = json.loads((c3 / "c3_keys.json").read_text())
    rows_c = [dict(role=k[0], format=k[1], wording=k[2], question=k[3], replicate=0, valid=True) for k in keys]

    # T: TF-IDF of the reply text, one mean vector per prompt
    use = [i for i, r in enumerate(rows_r) if r["valid"] and r["role"] in roles]
    tfidf = TfidfVectorizer(**A.TEXT["word"]).fit_transform([rows_r[i]["reply"] for i in use])
    groups = {}
    for pos, i in enumerate(use):
        r = rows_r[i]
        groups.setdefault((r["role"], r["format"], r["wording"], r["question"]), []).append(pos)
    gkeys = sorted(groups)
    XT = np.vstack([np.asarray(tfidf[groups[k]].mean(0)) for k in gkeys]).astype(np.float32)
    rows_t = [dict(role=k[0], format=k[1], wording=k[2], question=k[3], replicate=0, valid=True) for k in gkeys]

    feats = {
        "R": (np.asarray(acts_r[:, 17], dtype=np.float64), rows_r),
        "P": (np.asarray(np.load(c3 / "c3_acts.npy")[:, STATES_SAVED.index(17)], dtype=np.float64), rows_c),
        "E0": (np.asarray(acts_r[:, 0], dtype=np.float64), rows_r),
        "T": (XT, rows_t),
    }
    halves = A.make_halves(questions, n=a.n_halves)
    sides = {k: M.build_sides(*A.cell_means(X, rows, roles, questions), halves) for k, (X, rows) in feats.items()}
    rr, W = M.make_draws(len(roles), len(questions), n=a.n_boot, seed=A.SEED)
    comps = [("R", "R"), ("P", "P"), ("T", "T"), ("E0", "E0"), ("R", "P"), ("R", "T"), ("R", "E0"), ("P", "T"),
             ("P", "E0"), ("T", "E0")]
    panel = A.map_panel(sides, rr, W, comps, strict=True)
    out = dict(rules="RULES.md (13/11/2026 15:41 SAST)", config=dict(n_boot=a.n_boot, n_halves=a.n_halves),
               tfidf=dict(settings=A.TEXT["word"], n_replies=len(use), n_features=int(tfidf.shape[1])),
               agreement={f"{x}_with_{y}": A.summary(panel[(x, y)]["r"]) for x, y in comps}, paired={}, readings={})
    for x in ("P", "T", "E0"):
        diff = A.summary(panel[("R", "R")]["r"] - panel[("R", x)]["r"])
        out["paired"][f"RR_minus_R{x}"] = diff
        if x != "P":
            out["readings"][x] = ("the reply map holds structure that this feature does not carry"
                                  if diff["lo"] is not None and diff["lo"] > 0 else
                                  "the reply's words alone give the reply map")
    out["paired"]["RT_minus_RP"] = A.summary(panel[("R", "T")]["r"] - panel[("R", "P")]["r"])
    out["paired"]["RE0_minus_RP"] = A.summary(panel[("R", "E0")]["r"] - panel[("R", "P")]["r"])

    # check 3: judge naming scores against how far each role moves between the two maps
    d1 = json.loads(D1.read_text())["D1"]["judges"]
    hit = np.array([np.mean([j["per_role"][r]["hit5"] for j in d1.values()]) for r in roles])
    dR, i, j = full_map(A, *feats["R"], roles, questions)
    dP, _, _ = full_map(A, *feats["P"], roles, questions)
    rk_r, rk_p = rankdata(dR), rankdata(dP)
    move = np.array([np.abs(rk_r - rk_p)[(i == c) | (j == c)].mean() for c in range(len(roles))])
    ia = roles.index("assistant")
    others = [c for c in range(len(roles)) if c != ia]
    pair = {c: int(np.where(((i == c) & (j == ia)) | ((i == ia) & (j == c)))[0][0]) for c in others}
    toward = np.array([rk_r[pair[c]] - rk_p[pair[c]] for c in others])
    rho_move, p_move = perm_p(hit, move)
    rho_tow, p_tow = perm_p(hit[others], toward)
    out["check3"] = dict(
        judges=list(d1), per_role={r: dict(hit5=float(hit[c]), move=float(move[c]),
                                           toward_assistant=(None if c == ia else float(toward[others.index(c)])))
                                   for c, r in enumerate(roles)},
        hit5_vs_move=dict(spearman=rho_move, p=p_move, n=len(roles), predicted_sign="below 0"),
        hit5_vs_toward_assistant=dict(spearman=rho_tow, p=p_tow, n=len(others), predicted_sign="above 0"),
        full_map_spearman_R_P=float(spearmanr(dR, dP).statistic),
        reading=("the moved roles are the less-named roles"
                 if (rho_move < 0 and p_move < .05) or (rho_tow > 0 and p_tow < .05) else
                 "no evidence for reason 2 here"))
    (HERE / "results.json").write_text(json.dumps(A.clean(out), indent=1))
    print(json.dumps(A.clean({k: out[k] for k in ("agreement", "paired", "readings")}), indent=1))
    print(json.dumps(A.clean({k: v for k, v in out["check3"].items() if k != "per_role"}), indent=1))
    for r, v in sorted(out["check3"]["per_role"].items(), key=lambda kv: kv[1]["hit5"]):
        print(f"{r:>12} hit5 {v['hit5']:.2f} move {v['move']:5.1f} toward_assistant {v['toward_assistant']}")


if __name__ == "__main__":
    main()
