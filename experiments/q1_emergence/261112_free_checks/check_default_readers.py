"""Check 2 (rules in NOTE.md, fixed 12 Nov before any check 2 number): default position and spread when base, SFT,
DPO and Instruct read the same pod A Instruct replies in pod C frame F1 (question and reply, no role prompt), with a
text baseline (character 3-5-gram TF-IDF, 256-d truncated SVD). 1,000 draws shared by every source, 20 roles.
Exploratory, report only.

    .venv/bin/python -I experiments/q1_emergence/261112_free_checks/check_default_readers.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_role_count as C  # noqa: E402

READERS = ("base", "sft", "dpo", "instruct")
N_DRAWS = 1000
SEED = 20261113
TEXT_DIMS = 256


def text_features(rows_path):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [json.loads(line)["reply"] for line in open(rows_path)]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform(texts)
    return TruncatedSVD(TEXT_DIMS, random_state=0).fit_transform(tfidf).astype(np.float64), tfidf.shape


def draws(n_roles, nq):
    rng = np.random.default_rng(SEED)
    return [(rng.choice(n_roles, n_roles, replace=True),) + C.question_weights(rng, nq) for _ in range(N_DRAWS)]


def stats(X, rows, plan, halves):
    masks = dict(all=np.array([r["valid"] for r in rows]), limit=np.array([r["valid"] and r["limit"] for r in rows]))
    sums = {k: C.cell_sums(X, rows, m) for k, m in masks.items()}
    cs = sums["all"]
    n_roles = len(cs["clouds"]) - 1
    all_roles = np.arange(n_roles)
    full = np.ones(cs["S1"].shape[1])
    pos = np.array([abs(C.position(cs, wa, wb, idx)) for idx, wa, wb in plan])
    spr = {k: np.array([np.log((v := C.spread(s, wa + wb))[-1] / np.median(v[idx])) for idx, wa, wb in plan])
           for k, s in sums.items()}
    point = dict(position_abs=float(np.mean([abs(C.position(cs, a, b, all_roles)) for a, b in halves])),
                 spread_log_ratio={k: float(np.log(C.spread(s, full)[-1] / np.median(C.spread(s, full)[:-1])))
                                   for k, s in sums.items()})
    return dict(point=point, draws=dict(position_abs=pos, **{f"spread_{k}": v for k, v in spr.items()}),
                clouds=cs["clouds"])


def summary(d):
    return dict(point=d["point"], position_abs=C.interval(d["draws"]["position_abs"]),
                spread_all=C.interval(d["draws"]["spread_all"]), spread_limit=C.interval(d["draws"]["spread_limit"]))


def main():
    t0 = time.time()
    rows = C.grid_order(C.poda_rows())
    nq = len({r["question"] for r in rows})
    n_roles = len({r["role"] for r in rows}) - 1
    plan = draws(n_roles, nq)
    halves = []
    for i in range(C.N_POINT_SPLITS):
        perm = np.random.default_rng(SEED + 1 + i).permutation(nq)
        halves.append((np.isin(np.arange(nq), perm[: nq // 2]).astype(float),
                       np.isin(np.arange(nq), perm[nq // 2:]).astype(float)))
    results, raw = {}, {}
    for reader in READERS:
        X, rows_x = C.load(f"podc:{reader}")
        assert [r["role"] for r in rows_x] == [r["role"] for r in rows]
        raw[reader] = stats(X, rows, plan, halves)
        del X
        print(reader, "done", round(time.time() - t0), "s", flush=True)
    T, shape = text_features(C.PODA / "rows.jsonl")
    raw["text"] = stats(T, rows, plan, halves)
    print("text done", round(time.time() - t0), "s", flush=True)

    results = {k: summary(v) for k, v in raw.items()}
    paired = {}
    for reader in READERS:
        for stat in ("position_abs", "spread_all"):
            paired[f"{reader}_minus_text.{stat}"] = C.interval(raw[reader]["draws"][stat] - raw["text"]["draws"][stat])
    for a, b in (("base", "sft"), ("sft", "dpo"), ("dpo", "instruct"), ("base", "instruct")):
        for stat in ("position_abs", "spread_all"):
            paired[f"{b}_minus_{a}.{stat}"] = C.interval(raw[b]["draws"][stat] - raw[a]["draws"][stat])
    reading = {}
    for k, r in results.items():
        p, s = r["position_abs"], r["spread_all"]
        reading[k] = dict(outside_role_spread=bool(r["point"]["position_abs"] > 2 and p["lo"] > 2),
                          tighter_than_median_role=bool(s["hi"] < 0), wider_than_median_role=bool(s["lo"] > 0))
    out = dict(rules="NOTE.md, Check 2 (fixed 12 Nov 2026 before any check 2 number)", state=C.STATE, draws=N_DRAWS,
               seed=SEED, n_roles=n_roles, n_questions=nq, text_tfidf_shape=list(shape), text_dims=TEXT_DIMS,
               results=results, paired=paired, reading=reading, seconds=round(time.time() - t0, 1))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/check2_readers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(results={k: dict(position=v["position_abs"], spread=v["spread_all"]) for k, v in results.items()},
                          reading=reading), indent=1))


if __name__ == "__main__":
    main()
