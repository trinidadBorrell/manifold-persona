"""Calibrate the route judge on INTROSPECTIVE-question responses.

Why this file exists, and why the earlier calibration was measuring the wrong thing:

The first calibration drew its cases from the 240 extraction questions -- neutral task questions
like "What steps would you take to plan a marketing campaign?". The steering run does not use those.
The assistant-axis steering demo (notebooks/steer.ipynb) asks an identity question ("What is your
name?"), and run_steering.py asks the paper's five (Appendix D.1.2). A functional persona such as
`validator` is nearly invisible answering a marketing question and highly visible answering "Who are
you?", so the old calibration measured the judge on a distribution it will never see.

Ground truth is free and exact: the system prompt IS the persona.

It also answers the 512-vs-128 token question without extra generation. Responses are produced at
512 tokens; each is judged twice, once whole and once truncated to the first `--short-tokens`
tokens, so the effect of generation length on judge accuracy is measured directly.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

from steering.route_judge_v2 import (OpenRouter, build_system, instructions, judge, user_msg)

GEN = "/data/project/eeg_foundation/data/manifold_persona/introspective_calib/gen_qwen3_8b.parquet"


def truncate_tokens(text, tok, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tok.decode(ids[:n]).strip()


def dump(dirpath: Path, tag, i, row, system, res, truth):
    ok = "CORRECT" if res["score"] == truth else "WRONG"
    body = f"""# introspective case {i:03d} [{tag}] — {ok}

## Ground truth
- generated under persona: **{row['role']}** (instruction variant #{row['instruction_idx']})
- on the route under test: **{row['on_route']}**
- expected judge score: **{truth}**
- condition: **{tag}** ({row['n_tokens_judged']} tokens shown to the judge)

## Judge output
- score: **{res['score']}**   intensity: **{res['intensity']}**   position: **{res['position']}**
- persona_kind: **{res.get('persona_kind')}**
- evidence quoted: {res['evidence']!r}
- analysis: {res['analysis']}
- error: {res['error']}

---

## 1. SYSTEM PROMPT used to generate the response (the persona)
```
{row['system']}
```

## 2. QUESTION put to the model (introspective, Appendix D.1.2)
```
{row['question']}
```

## 3. The model's ANSWER, as shown to the judge
```
{row['judged_text']}
```

## 4. SYSTEM PROMPT given to the LLM judge
```
{system}
```

## 5. USER MESSAGE given to the LLM judge
```
{user_msg(row['question'], row['judged_text'])}
```

## 6. Raw judge reply
```
{res['raw']}
```
"""
    (dirpath / f"case_{i:03d}_{tag}_{ok.lower()}.md").write_text(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", default=GEN.replace(".parquet", ".csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-dir", default="prompts/ablation/introspective")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--k", type=int, default=16, choices=[8, 16])
    ap.add_argument("--pair", default="validator>vampire")
    ap.add_argument("--n-cases", type=int, default=40, help="responses judged (each judged twice)")
    ap.add_argument("--short-tokens", type=int, default=128)
    ap.add_argument("--max-calls", type=int, default=90)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    A, B = a.pair.split(">")
    routes = json.loads(Path("prompts/generated/routes_by_k.json").read_text())
    route = [A] + routes[f"{A}>{B}|{a.k}|distance"] + [B]
    on_route = set(route)
    system = build_system(route)

    df = (pd.read_csv(a.gen) if str(a.gen).endswith(".csv") else pd.read_parquet(a.gen))
    df = df[df.response.str.len() > 40].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    sel = df.iloc[rng.choice(len(df), size=min(a.n_cases, len(df)), replace=False)].reset_index(drop=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    planned = 2 * len(sel)
    print(f"route {A}>{B} k={a.k} ({len(route)} personas)")
    print(f"generations available {len(df)}  ·  cases {len(sel)}  ·  planned calls {planned} (cap {a.max_calls})")
    print(sel.role.value_counts().to_string())
    if planned > a.max_calls:
        sys.exit(f"planned {planned} exceeds --max-calls {a.max_calls}")

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    abl = Path(a.ablation_dir); abl.mkdir(parents=True, exist_ok=True)
    (abl / "_judge_system_prompt.txt").write_text(system)
    (abl / "_route.json").write_text(json.dumps({"pair": a.pair, "k": a.k, "route": route}, indent=1))
    sel.to_csv(outd / "cases.csv", index=False)
    if not a.yes:
        print("\nDRY RUN — no API calls. Pass --yes to spend.")
        return

    client = OpenRouter(Path("token/open-router.txt").read_text().strip())
    rows = []
    n = 0
    for i, r in sel.iterrows():
        truth = r.role if r.role in on_route else "other"
        full_n = len(tok(r.response, add_special_tokens=False)["input_ids"])
        for tag, text in (("full512", r.response),
                          (f"short{a.short_tokens}", truncate_tokens(r.response, tok, a.short_tokens))):
            nt = len(tok(text, add_special_tokens=False)["input_ids"])
            res = judge(client, a.model, system, r.question, text)
            n += 1
            row = dict(r); row["judged_text"] = text; row["n_tokens_judged"] = nt
            row["on_route"] = r.role in on_route
            dump(abl, tag, i, row, system, res, truth)
            rows.append(dict(case=i, condition=tag, role=r.role, question=r.question,
                             instruction_idx=int(r.instruction_idx), truth=truth,
                             n_tokens_full=full_n, n_tokens_judged=nt,
                             **{k2: v for k2, v in res.items() if k2 != "raw"}))
            mark = "ok " if res["score"] == truth else "MISS"
            print(f"  {n:3d}/{planned} {mark} {tag:9s} {r.role:12s} q={r.question[:22]:24s} -> {res['score']}",
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(outd / "introspective_calibration.csv", index=False)
    out["ok"] = out.score == out.truth
    print("\n=== ACCURACY BY CONDITION ===")
    print(out.groupby("condition").ok.agg(["mean", "sum", "count"]).round(3).to_string())
    print("\n=== BY ROLE (full512) ===")
    f = out[out.condition == "full512"]
    print(f.groupby("role").ok.agg(["mean", "count"]).round(2).to_string())
    print(f"\nparse failures: {out.score.isna().sum()}/{len(out)}")
    print(f"per-case dump -> {abl}")


if __name__ == "__main__":
    main()
