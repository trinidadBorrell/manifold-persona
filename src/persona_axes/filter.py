"""Phase 0: judge-filter the trait pool and the axes (plan v2 s2.3). Every pos / neg answer is scored 0-100
with the trait's own evaluation prompt (upstream for the 240 traits, generated for the 35 axes) by Claude
Haiku 5.5. Keep rule (ours; App. C is inconsistent): a trait is kept if >= 3 of its 5 prompt pairs have
mean(pos) - mean(neg) >= 40.

    python -m persona_axes.filter --part traits [--dry-run]      # -> $PM/AX21/filter/traits_scores.csv, traits_kept.json
    python -m persona_axes.filter --part axes
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from persona_axes import AX, HERE
from persona_axes.api import HAIKU, Runner

MIN_PAIRS, MIN_DIFF = 3, 40


def eval_prompts(part):
    if part == "traits":
        return {t["name"]: t["eval_prompt"] for t in json.loads((HERE / "traits.json").read_text())["traits"]}
    return {k: v["eval_prompt"] for k, v in json.loads((AX / "prompts" / "axes_prompts.json").read_text()).items()}


def parse(raw):
    s = raw.strip()
    if "REFUSAL" in s.upper():
        return "REFUSAL"
    m = re.search(r"\b(\d{1,3})\b", s)
    return int(m.group(1)) if m and 0 <= int(m.group(1)) <= 100 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["traits", "axes"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    F = AX / "filter"
    F.mkdir(exist_ok=True)
    meta = pd.read_csv(AX / a.part / "meta.csv", usecols=["who", "pole", "pair", "question_idx", "question", "response", "unit"])
    meta = meta[meta.pole.isin(["pos", "neg"])].reset_index(drop=True)
    ev = eval_prompts(a.part)
    scores_f = F / (a.part + "_scores.jsonl")
    done = set()
    if scores_f.exists():                     # a unit counts as done only once it has a parsed score
        done = {d["unit"] for d in (json.loads(l) for l in scores_f.read_text().splitlines() if l.strip()) if d["score"] is not None}
    todo = meta[~meta.unit.isin(done)]

    def prompt(r):
        return ev[r.who].replace("{question}", str(r.question)).replace("{answer}", str(r.response))
    prompts = [prompt(r) for r in todo.itertuples()]
    R = Runner(HAIKU, workers=a.workers)
    print("%s: %d answers to score (%d done), estimate $%.2f" % (a.part, len(todo), len(done), R.estimate(prompts, 20)))
    if a.dry_run:
        print(prompts[0][:1200] if prompts else "")
        return
    lock = __import__("threading").Lock()
    fh = open(scores_f, "a")

    def one(r):
        pr = prompt(r)
        # Haiku 5.5 thinks before answering and the thinking counts against max_tokens: 20 left the text empty
        # (stop = max_tokens) on 23% of the calls, so the budget is 600 and the reply is still one number
        v, raw, stop = R.call(pr, max_tokens=600, effort="low", ok=lambda v: True, tries=1)
        sc = parse(raw)
        if sc is None and stop != "refusal":
            v, raw, stop = R.call(pr + "\nReply with the number only.", max_tokens=1000, effort="low", ok=lambda v: True, tries=1)
            sc = parse(raw)
        with lock:
            fh.write(json.dumps(dict(unit=int(r.unit), who=r.who, pole=r.pole, pair=int(r.pair), question_idx=int(r.question_idx),
                                     score=sc, stop=stop)) + "\n"); fh.flush()
        return sc is not None
    res = R.map(one, list(todo.itertuples()))
    fh.close()
    print("scored %d / %d, $%.2f" % (sum(res), len(res), R.spent))
    summarise(a.part)


def summarise(part):
    F = AX / "filter"
    df = pd.read_json(F / (part + "_scores.jsonl"), lines=True)
    df = df[df.score.notna()].drop_duplicates("unit", keep="last")       # re-scored units: keep the parsed one
    df["num"] = pd.to_numeric(df.score, errors="coerce")
    g = df.groupby(["who", "pair", "pole"]).num.mean().unstack("pole")
    g["diff"] = g["pos"] - g["neg"]
    per = g.reset_index()
    per.to_csv(F / (part + "_pairs.csv"), index=False)
    n_ok = per.groupby("who")["diff"].apply(lambda d: int((d >= MIN_DIFF).sum()))
    mean_diff = per.groupby("who")["diff"].mean()
    refusal = df.groupby("who").score.apply(lambda s: float((s == "REFUSAL").mean()))
    tab = pd.DataFrame(dict(pairs_ok=n_ok, mean_diff=mean_diff, refusal_share=refusal))
    tab["kept"] = tab.pairs_ok >= MIN_PAIRS
    tab.to_csv(F / (part + "_summary.csv"))
    kept = sorted(tab.index[tab.kept])
    (F / (part + "_kept.json")).write_text(json.dumps(dict(rule="pairs with mean(pos)-mean(neg) >= %d: at least %d of 5" % (MIN_DIFF, MIN_PAIRS),
                                                           kept=kept, dropped=sorted(tab.index[~tab.kept]))))
    print("%s: kept %d / %d; median pair diff %.0f; refusals %.1f%%" % (part, len(kept), len(tab), per["diff"].median(), 100 * (df.score == "REFUSAL").mean()))


if __name__ == "__main__":
    main()
