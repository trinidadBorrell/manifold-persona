"""Check 2 (exploratory, NOTE.md): default position and spread with the role prompt hidden.

Pod C frame F1 states (question and Instruct reply, no role prompt) at base, SFT, DPO and Instruct, hidden state 17,
plus a text baseline on the same replies (character 3-5-gram TF-IDF, truncated SVD to 256 dimensions fitted on all
rows). All 20 roles. The same 1,000 crossed draws (question halves, questions resampled within each half, roles
resampled with replacement) go to every source, so differences between readers are paired.

    .venv/bin/python -I experiments/q1_emergence/261112_free_checks/check_hidden_prompt.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_role_count import (N_DRAWS, N_POINT_SPLITS, PODA, SEED, cell_sums, interval, load, poda_rows,  # noqa: E402
                              position, question_weights, spread)

READERS = ("base", "sft", "dpo", "instruct")
STAGE_PAIRS = (("sft", "base"), ("dpo", "sft"), ("instruct", "dpo"), ("instruct", "base"))
SVD_DIMS = 256
POSITION_BAR = 2.0


def text_source():
    rows = poda_rows()
    texts = [json.loads(line)["reply"] for line in open(PODA / "rows.jsonl")]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    X = TruncatedSVD(SVD_DIMS, random_state=SEED).fit_transform(tfidf)
    return np.asarray(X, dtype=np.float64), rows, tfidf.shape[1]


def draws(nq, n_roles):
    rng = np.random.default_rng(SEED + 2)
    out = []
    for _ in range(N_DRAWS):
        wa, wb = question_weights(rng, nq)
        out.append((wa, wb, rng.choice(n_roles, n_roles, replace=True)))
    return out


def evaluate(X, rows, plan, halves):
    masks = dict(all=np.array([r["valid"] for r in rows]), limit=np.array([r["valid"] and r["limit"] for r in rows]))
    sums = {k: cell_sums(X, rows, m) for k, m in masks.items()}
    cs = sums["all"]
    n_roles, nq = len(cs["clouds"]) - 1, cs["S1"].shape[1]
    all_roles = np.arange(n_roles)
    full_w = np.ones(nq)
    pos = np.array([abs(position(cs, wa, wb, ri)) for wa, wb, ri in plan])
    spr = {}
    for k, s in sums.items():
        vals = []
        for wa, wb, ri in plan:
            v = spread(s, wa + wb)
            vals.append(np.log(v[-1] / np.median(v[ri])))
        spr[k] = np.array(vals)
    point = dict(
        position_abs=float(np.mean([abs(position(cs, a, b, all_roles)) for a, b in halves])),
        spread_log_ratio={k: float(np.log(spread(s, full_w)[-1] / np.median(spread(s, full_w)[:-1])))
                          for k, s in sums.items()})
    return cs["clouds"], point, pos, spr


def main():
    t0 = time.time()
    raw, sources = {}, {}
    plan = halves = None
    for name in ("text",) + READERS:
        if name == "text":
            X, rows, vocab = text_source()
        else:
            X, rows = load(f"podc:{name}")
        if plan is None:
            nq = len({r["question"] for r in rows})
            n_roles = len({r["role"] for r in rows}) - 1
            plan = draws(nq, n_roles)
            halves = []
            for i in range(N_POINT_SPLITS):
                perm = np.random.default_rng(SEED + 1 + i).permutation(nq)
                halves.append((np.isin(np.arange(nq), perm[: nq // 2]).astype(float),
                               np.isin(np.arange(nq), perm[nq // 2:]).astype(float)))
        clouds, point, pos, spr = evaluate(X, rows, plan, halves)
        raw[name] = dict(position=pos, spread=spr)
        pi, si = interval(pos), {k: interval(v) for k, v in spr.items()}
        sources[name] = dict(
            point=point, position_abs=pi, spread_log_ratio=si,
            rule_outside_role_spread=bool(pi["point"] > POSITION_BAR and pi["lo"] > POSITION_BAR),
            rule_default_tighter=bool(si["all"]["hi"] < 0))
        print(name, f"pos {point['position_abs']:.2f} [{pi['lo']:.2f}, {pi['hi']:.2f}]",
              f"spread {point['spread_log_ratio']['all']:.3f} [{si['all']['lo']:.3f}, {si['all']['hi']:.3f}]",
              f"({time.time() - t0:.0f} s)", flush=True)
        del X

    def clears(a, b):
        return bool(a["lo"] > b["hi"] or a["hi"] < b["lo"])

    for name in READERS:
        s, t = sources[name], sources["text"]
        s["clears_text"] = dict(position=clears(s["position_abs"], t["position_abs"]),
                                spread_all=clears(s["spread_log_ratio"]["all"], t["spread_log_ratio"]["all"]))
    stage = {}
    for a, b in STAGE_PAIRS:
        stage[f"{a}-{b}"] = dict(position_abs=interval(raw[a]["position"] - raw[b]["position"]),
                                 spread_log_ratio_all=interval(raw[a]["spread"]["all"] - raw[b]["spread"]["all"]))
    out = dict(state=17, draws=N_DRAWS, clouds=clouds, text_baseline=dict(analyzer="char", ngram_range=[3, 5],
               vocabulary=vocab, svd_dims=SVD_DIMS, svd_fit="all rows"), sources=sources,
               stage_change_paired=stage, seconds=round(time.time() - t0, 1))
    path = HERE / "results" / "check2.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(clears_text={n: sources[n]["clears_text"] for n in READERS}, stage_change_paired=stage),
                     indent=1))


if __name__ == "__main__":
    main()
