"""Can a cheaper judge replace GPT-5.6-Sol or Claude Sonnet 5.5 for D1 blind naming? (owner, 30/10: "1")

200 pod A replies (10 per scored role, replicates 0-2, fixed seed) go to six models as sync calls with the D1
prompt and list-only schema of or_naming.py. Rule, fixed before any answer: for each reply, "hit" = the true role
is among the judge's 5 names. A cheap judge matches if its hit agreement with Sol and with Sonnet is each no
more than 0.05 below the Sol-Sonnet agreement. The cheapest matching pair from two model families is proposed;
the owner decides. Answers stay in the work folder (outside the repo); results/ gets the aggregate only.

    python naming_judge_check.py WORK      # WORK: a prepared or_naming folder (its items and keys)
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import or_naming as O

N_PER_ROLE, SEED, MARGIN = 10, 20261030, 0.05
MODELS = {  # OpenAI models at reasoning effort high (as Sol), Anthropic models without thinking (as Sonnet)
    "openai/gpt-5.6-sol": dict(reasoning={"effort": "high"}, max_tokens=1200),
    "anthropic/claude-sonnet-5.5": dict(max_tokens=400),
    "openai/gpt-6.1-terra": dict(reasoning={"effort": "high"}, max_tokens=1200),
    "openai/gpt-5.6-luna": dict(reasoning={"effort": "high"}, max_tokens=1200),
    "anthropic/claude-haiku-5.5": dict(max_tokens=400),
    "deepseek/deepseek-v4.1-flash": dict(max_tokens=4000),  # at 1,200 it reasoned to the limit and wrote no answer
}
FAMILY = {m: m.split("/")[0] for m in MODELS}
BATCH_PRICE = {"openai/gpt-5.6-sol": (2.0, 10.0), "anthropic/claude-sonnet-5.5": (1.0, 5.0), "openai/gpt-6.1-terra": (0.5, 2.5),
               "openai/gpt-5.6-luna": (0.1, 0.6), "anthropic/claude-haiku-5.5": (0.05, 0.25),
               "deepseek/deepseek-v4.1-flash": (0.05, 0.15)}  # per 1M tokens, /api/v1/models, 30/10/2026
REF = ("openai/gpt-5.6-sol", "anthropic/claude-sonnet-5.5")


def ask(model, reply):
    body = dict(model=model, messages=[{"role": "user", "content": O.prompt(reply)}], response_format=O.SCHEMA,
                usage={"include": True}, **MODELS[model])
    for _ in range(2):  # one retry, as in the run
        for wait in (5, 10, 20, 40, 60, 60, 60, 60):  # the account's per-minute limit per model (HTTP 429)
            code, resp = O.orj.call("POST", "/chat/completions", body)
            if code != 429:
                break
            time.sleep(wait)
        if code in (401, 402, 403):  # key, credit or spend limit: nothing more can run
            raise SystemExit(f"{model}: HTTP {code}: {str(resp.get('error'))[:300]}")
        text, usage = O.orj.content_of(resp)
        names = O.parse(text)
        if names is not None:
            return dict(names=names, usage=usage, http=code)
    return dict(names=None, usage=usage, http=code, error=resp.get("error") if code != 200 else text)


def main():
    w = O.work_dir(sys.argv[1])
    m, items = O.load(w)
    by_role = {}
    for it in items:
        role = json.loads(it["key"])[0]
        if role != "default":
            by_role.setdefault(role, []).append(it)
    rng = random.Random(SEED)
    pick = [it for role in sorted(by_role) for it in rng.sample(by_role[role], N_PER_ROLE)]
    out = w / "judge_check"
    out.mkdir(exist_ok=True)
    (out / "rule.json").write_text(json.dumps(dict(rule=__doc__, models=MODELS, ids=[it["id"] for it in pick]), indent=1))
    answers = json.loads((out / "answers.json").read_text()) if (out / "answers.json").exists() else {}
    def run(model):  # 4 at a time per model; the models run side by side (each has its own rate limit)
        todo = [it for it in pick if (answers.get(f"{model}|{it['id']}") or {}).get("names") is None]
        with ThreadPoolExecutor(4) as ex:
            return model, list(zip(todo, ex.map(lambda it: ask(model, it["reply"]), todo)))
    with ThreadPoolExecutor(len(MODELS)) as ex:
        for model, got in ex.map(run, MODELS):
            for it, a in got:
                answers[f"{model}|{it['id']}"] = a
            (out / "answers.json").write_text(json.dumps(answers, indent=1))
            print(model, "done:", sum(a["names"] is not None for _, a in got), "of", len(got), "new answers valid", flush=True)
    truth = {it["id"]: json.loads(it["key"])[0] for it in pick}
    hit = {mo: {i: (answers[f"{mo}|{i}"]["names"] is not None and truth[i] in answers[f"{mo}|{i}"]["names"]) for i in truth}
           for mo in MODELS}
    agree = lambda a, b: sum(hit[a][i] == hit[b][i] for i in truth) / len(truth)
    base = agree(*REF)
    summary = {}
    for mo in MODELS:
        u = [answers[f"{mo}|{i}"]["usage"] or {} for i in truth]
        pin, pout = BATCH_PRICE[mo]
        per = sum((x.get("prompt_tokens") or 0) * pin + (x.get("completion_tokens") or 0) * pout for x in u) / len(u) / 1e6
        roles = sorted(set(truth.values()))
        macro = sum(sum(hit[mo][i] for i in truth if truth[i] == r) / N_PER_ROLE for r in roles) / len(roles)
        summary[mo] = dict(macro_hit5=macro, invalid=sum(answers[f"{mo}|{i}"]["names"] is None for i in truth),
                           agree_sol=agree(mo, REF[0]), agree_sonnet=agree(mo, REF[1]),
                           top1_same_as_sol=sum((answers[f"{mo}|{i}"]["names"] or [None])[0] == (answers[f"{REF[0]}|{i}"]["names"] or [""])[0] for i in truth) / len(truth),
                           full_run_batch_usd=per * (m["n_items"] + len(m["repeat_ids"])),
                           matches=mo in REF or (agree(mo, REF[0]) >= base - MARGIN and agree(mo, REF[1]) >= base - MARGIN))
    ok = [mo for mo in MODELS if summary[mo]["matches"]]
    pairs = sorted(((a, b) for a in ok for b in ok if a < b and FAMILY[a] != FAMILY[b]),
                   key=lambda p: summary[p[0]]["full_run_batch_usd"] + summary[p[1]]["full_run_batch_usd"])
    spent = sum(float((a["usage"] or {}).get("cost") or 0) for a in answers.values())
    result = dict(n=len(truth), sol_sonnet_agreement=base, margin=MARGIN, models=summary,
                  cheapest_matching_pair=pairs[0] if pairs else None, spent_usd=spent)
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    (O.HERE / "results").mkdir(exist_ok=True)
    (O.HERE / "results" / "naming_judge_check_261030.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
