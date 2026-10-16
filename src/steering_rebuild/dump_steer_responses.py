"""Dump a steering run's generations (and any judge verdicts) as readable markdown.

One file per generation plus a single ALL_RESPONSES.md, so a run can be read end to end rather than
opened row by row in a CSV. If a judge results CSV is given its verdict is folded into each file,
keeping the generation and its judgement in one place.
"""
from __future__ import annotations
import argparse, textwrap
from pathlib import Path
import pandas as pd

TICK = chr(96) * 3


def block(text):
    return TICK + "\n" + str(text) + "\n" + TICK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="generations from jobs_condor/steer_demo.py")
    ap.add_argument("--judged", default=None, help="optional judge results CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Steering run")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["flat"] = d.response.apply(lambda s: " ".join(str(s).split()))
    jv = {}
    if a.judged and Path(a.judged).exists():
        j = pd.read_csv(a.judged)
        key = "persona_kind" if "persona_kind" in j.columns else "score"
        pos = "position" if "position" in j.columns else "alpha"
        for _, r in j.iterrows():
            jv.setdefault((r.strategy, round(float(r[pos]), 4)), []).append(
                dict(variant=r.get("variant", "default"), verdict=r[key],
                     intensity=r.get("intensity"), evidence=r.get("evidence", ""),
                     analysis=r.get("analysis", "")))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "RESULTS.csv", index=False)

    L = [f"# {a.title}", "",
         f"{len(d)} generations · pair **{d.pair.iloc[0]}** · k={d.k.iloc[0]} · "
         f"hidden state {d.hidden_state.iloc[0]} · question **{d.question.iloc[0]!r}**", ""]
    if "target_persona" in d.columns:
        L += ["Manifold strategies are sampled at **their own knot arc positions**, so each "
              "generation sits exactly on a persona centroid. The linear chord passes through no "
              "centroid, so it gets the same number of stops, equally spaced.", ""]
    L += ["---", ""]

    for s, g in d.groupby("strategy", sort=False):
        L += [f"## {s}", ""]
        for _, r in g.iterrows():
            t = r.get("target_persona")
            t = t if isinstance(t, str) and t else "(between personas)"
            al = round(float(r.alpha), 4)
            L += [f"### alpha = {al} &nbsp;→&nbsp; target: **{t}**", "",
                  block(textwrap.fill(r.flat, 96)), ""]
            for v in jv.get((s, al), []):
                L += [f"- judge [{v['variant']}] → **{v['verdict']}** "
                      f"(intensity {v['intensity']}) — {v['analysis']}", ""]
        L += ["---", ""]
    (out / "ALL_RESPONSES.md").write_text("\n".join(L))

    for _, r in d.iterrows():
        t = r.get("target_persona")
        t = t if isinstance(t, str) and t else "between"
        al = round(float(r.alpha), 4)
        body = [f"# {r.strategy} @ alpha={al} → target: {t}", "",
                f"- pair **{r.pair}** · k={r.k} · hidden state {r.hidden_state} · "
                f"detour {r.detour}x",
                f"- system prompt present: **{bool(str(r.system).strip())}**", "",
                "## 1. SYSTEM PROMPT given to the steered model", block(r.system),
                "", "## 2. QUESTION", block(r.question),
                "", "## 3. STEERED ANSWER", block(textwrap.fill(r.flat, 96)), ""]
        for v in jv.get((r.strategy, al), []):
            body += [f"## 4. JUDGE VERDICT — variant `{v['variant']}`",
                     f"- persona: **{v['verdict']}**   intensity: **{v['intensity']}**",
                     f"- evidence: {v['evidence']!r}",
                     f"- analysis: {v['analysis']}", ""]
        (out / f"{r.strategy}_alpha{al}.md").write_text("\n".join(body))

    print(f"wrote {len(d)} case files + ALL_RESPONSES.md + RESULTS.csv -> {out}")
    if jv:
        print(f"folded in judge verdicts for {len(jv)} cells")


if __name__ == "__main__":
    main()
