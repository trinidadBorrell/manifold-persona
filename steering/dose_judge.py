"""LLM judge for the dose-escalation grid (33 cells x 3 layers = 99 total).

Reuses the exact validated machinery from route_judge_v3/v2 -- build_system() with the
"examples_prioritize" configuration (worked examples + the exhaust-personas-first instruction),
the same OpenRouter client, the same judge() retry loop. Nothing about the judge prompt is
reinvented here; only the input shape differs (dose_L{L}.csv has no strategy/pair/system columns,
because the dose sweep is linear-only and single-strategy).

PERSONA LIST: the same fixed candidate set is used for L19, L25 AND L32 (the base
validator>vampire k=8 route, no layer suffix) rather than switching to the L32-specific list for
that one layer. The judge classifies TEXT, not activations -- it never sees a hidden state, only
the question and the response string -- so the candidate persona names are a linguistic frame, not
a claim about which layer's geometry produced them. Holding that frame fixed across layers is what
makes the resulting table comparable column-to-column; swapping it only for L32 would make "vampire"
mean a differently-anchored category at that layer and the three columns would no longer be the same
measurement at three depths.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd

from steering.route_judge_v3 import build_system, WORKED_EXAMPLES, PRIORITIZE  # noqa: F401
from steering.route_judge_v2 import OpenRouter, parse, user_msg

PAIR = "validator>vampire"
K = 8
ROUTE_KEY = f"{PAIR}|{K}|distance"          # fixed base route, used for all three layers


def judge(client, model, system, q, a, retries=4):
    last = None
    for k in range(retries):
        try:
            txt = client.complete(model, system, user_msg(q, a))
            d = parse(txt)
            return dict(persona_kind=d.get("persona_kind"), intensity=d.get("intensity"),
                        evidence=d.get("evidence", ""), analysis=d.get("analysis", ""),
                        error=None, raw=txt)
        except Exception as e:                       # noqa: BLE001
            last = repr(e)
            time.sleep(1.2 * (k + 1))
    return dict(persona_kind=None, intensity=None, evidence="", analysis="", error=last, raw="")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csvs", nargs="+", required=True, help="dose_L{L}.csv files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-dir", default="prompts/ablation/dose_judge")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    routes = json.loads(Path("prompts/generated/routes_by_k.json").read_text())
    A, B = PAIR.split(">")
    personas = [A] + routes[ROUTE_KEY] + [B]
    system = build_system(personas, with_examples=True, prioritize=True, persona_examples=False)

    abl = Path(a.ablation_dir); abl.mkdir(parents=True, exist_ok=True)
    (abl / "_judge_system_prompt.txt").write_text(system)
    (abl / "_personas.json").write_text(json.dumps({"pair": PAIR, "k": K, "personas": personas},
                                                    indent=1))

    frames = []
    for p in a.csvs:
        df = pd.read_csv(p)
        df["source_layer"] = int(df.layer.iloc[0])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"personas: {', '.join(personas)}")
    print(f"{len(all_df)} cells to judge (dry run shows plan only)")
    if not a.yes:
        print("\nDRY RUN -- no API calls. Pass --yes to spend.")
        return

    client = OpenRouter(Path("token/open-router.txt").read_text().strip())
    rows = []
    for i, r in all_df.iterrows():
        res = judge(client, a.model, system, r.question, r.response)
        rows.append(dict(layer=int(r.source_layer), kind=r.kind, question=r.question,
                         alpha=float(r.alpha), **{k: v for k, v in res.items() if k != "raw"}))
        (abl / f"L{int(r.source_layer)}__{r.kind}__a{r.alpha}.md").write_text(
            f"# L{int(r.source_layer)} {r.kind} alpha={r.alpha}\n\n"
            f"judge: **{res['persona_kind']}** intensity {res['intensity']}\n\n"
            f"evidence: {res['evidence']!r}\n\nanalysis: {res['analysis']}\n\n---\n\n"
            f"## question\n{r.question}\n\n## response\n{r.response}\n\n## raw\n{res['raw']}\n")
        print(f"  {i+1:3d}/{len(all_df)}  L{int(r.source_layer)} {r.kind:<8} a={r.alpha:<5g} -> "
              f"{res['persona_kind']}", flush=True)

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print(out.groupby(["layer", "persona_kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
