"""Judge the steering demo's generations and dump readable samples.

Separate from generation on purpose: the demo writes a CSV and spends nothing, this reads it and is
the only thing that costs money. So a generation rerun can never drag API spend along with it.

Writes one markdown file per generation under --ablation-dir containing the persona system prompt,
the question, the steered answer and the judge's full reply, so the run can be read by a human
rather than trusted.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

from steering.route_judge_v2 import OpenRouter, build_system, judge, user_msg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-dir", default="prompts/ablation/steer_demo")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--k", type=int, default=8, choices=[8, 16])
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    pair = df.pair.iloc[0]
    A, B = pair.split(">")
    routes = json.loads(Path("prompts/generated/routes_by_k.json").read_text())
    route = [A] + routes[f"{pair}|{a.k}|distance"] + [B]
    system = build_system(route)

    print(f"{pair} k={a.k}: {len(route)} personas, {len(route)+1} options")
    print(f"generations {len(df)}  ·  planned calls {len(df)} (cap {a.max_calls})")
    if len(df) > a.max_calls:
        sys.exit(f"planned {len(df)} exceeds --max-calls {a.max_calls}")
    if not a.yes:
        print("\nDRY RUN — no API calls. Pass --yes to spend.")
        return

    abl = Path(a.ablation_dir); abl.mkdir(parents=True, exist_ok=True)
    (abl / "_judge_system_prompt.txt").write_text(system)
    (abl / "_route.json").write_text(json.dumps({"pair": pair, "k": a.k, "route": route}, indent=1))

    client = OpenRouter(Path("token/open-router.txt").read_text().strip())
    rows = []
    for i, r in df.iterrows():
        res = judge(client, a.model, system, r.question, r.response)
        rows.append(dict(strategy=r.strategy, alpha=r.alpha, detour=r.detour,
                         has_system=bool(str(r.system).strip()),
                         **{k2: v for k2, v in res.items() if k2 != "raw"}))
        (abl / f"{r.strategy}_alpha{r.alpha}.md").write_text(f"""# {r.strategy} @ alpha={r.alpha}

- pair: **{pair}**  ·  k={r.k}  ·  hidden state {r.hidden_state}  ·  detour {r.detour}x
- system prompt present: **{bool(str(r.system).strip())}**

## Judge verdict
- score: **{res['score']}**   intensity: **{res['intensity']}**   position: **{res['position']}**
- evidence: {res['evidence']!r}
- analysis: {res['analysis']}

---

## 1. SYSTEM PROMPT (the source persona)
```
{r.system}
```

## 2. QUESTION
```
{r.question}
```

## 3. STEERED ANSWER
```
{r.response}
```

## 4. JUDGE SYSTEM PROMPT
```
{system}
```

## 5. JUDGE USER MESSAGE
```
{user_msg(r.question, r.response)}
```

## 6. RAW JUDGE REPLY
```
{res['raw']}
```
""")
        print(f"  [{r.strategy} a={r.alpha}] -> {res['score']} / {res['position']}", flush=True)

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print("\n=== VERDICTS ===")
    print(out.pivot_table(index="strategy", columns="alpha", values="score",
                          aggfunc="first").to_string())
    print(f"\nreadable samples -> {abl}")


if __name__ == "__main__":
    main()
