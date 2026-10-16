"""Fill page_template.html with a step's page_data.json -> <root>/step<N>/ace_step<N>.html.

    python -m ace_steering.page --step 1
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from ace_steering.common import (MAX_NEW_TOKENS, N_PROMPTS, N_SAMPLES, OUT_ROOT, QUESTIONS,
                                       TEMPERATURE, TOP_K, TOP_P)

HERE = Path(__file__).resolve().parent


def notes(step, meta, spend, judge_tag=""):
    per_cell = N_PROMPTS * len(QUESTIONS) * N_SAMPLES.get(step, 1)
    title = ("Four ways to steer from the Assistant to three characters" if step == 1
             else "Four ways to steer, across seven layers")
    lede = ("Qwen3-8B is steered from a source persona toward a target persona with three arms that share the same "
            "two endpoints. Each answer to one of five identity questions is labelled by an LLM judge as the source, "
            "one of the eight personas the manifold route passes through, the target, or none of them.")
    facts = [("model", "Qwen3-8B"),
             ("steered layer", "hidden_states[19]" if step == 1 else "hidden_states 15–32"),
             ("answers per cell", "%d (%d system prompts × %d questions × %d)" % (per_cell, N_PROMPTS, len(QUESTIONS), N_SAMPLES.get(step, 1))),
             ("sampling", "T %.1f · top-p %.1f · top-k %d · %d tokens" % (TEMPERATURE, TOP_P, TOP_K, MAX_NEW_TOKENS)),
             ("judge", "Claude Code subagents (Sonnet), 100 answers per agent" if judge_tag == "subagent"
              else "claude-sonnet-5, one call per answer"),
             ("answers", "{:,}".format(meta["n_answers"])),
             ("collapsed", str(meta["collapsed"])),
             ("judge spend", "none (Claude Code plan, ~4.3M tokens)" if judge_tag == "subagent" else "$%.2f" % spend)]
    body = """
<h2>How the four arms differ</h2>
<p>All three add a vector to the layer's output at every token, in the full 4096-dimensional space. μ_A and μ_B are the source and target
persona centroids (mean layer activation over each persona's role-playing answers), and r = μ_B − μ_A.</p>
<span class="formula">linear&nbsp;&nbsp;&nbsp;h ← h + α r</span>
<span class="formula">manifold h ← h + S(α) − S(0),&nbsp; S = spline through μ_A, 8 persona centroids, μ_B (arc-length α ∈ [0, 1])</span>
<span class="formula">ACE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;h ← h − ((h − μ_A)·r̂) r̂ + α r</span>
<span class="formula">ACE manifold&nbsp;h ← h + U Uᵀ (T(α) − h),&nbsp; U = orthonormal basis of {μ_B − μ_A, knots − μ_A} (9 directions), T(α) = S(α) on [0, 1], the chord beyond</span>
<p>ACE is affine concept editing (Marshall, Scherlis &amp; Belrose, arXiv 2411.09003, eq. 5) with the source centroid as the reference point.
It sets every token's coordinate along r to that of μ_A + α r and leaves everything orthogonal to r alone, so α = 0 is not "no steering":
it removes each token's own spread along r. Linear and manifold shift the state by a fixed vector, so where they land depends on where the
unsteered model already was. The reference implementation (<code>extra/steering-llama3</code>, <code>--leace orth</code>) centres on the
mean of both personas instead of μ_A, which makes its α equal to this α − 0.5.</p>
<p>ACE manifold applies the same idea to the whole route: every token's coordinates in the route's own directions are set to the route
point T(α), and the other 4,087 directions are left alone. With U = r̂ alone it is exactly ACE.</p>
<h2>Reading the judge curves</h2>
<p>The judge's options are the route's ten personas plus <code>other</code>. Answers whose most frequent 4-gram repeats more than 20 times
are labelled <code>collapsed</code> without a judge call. When the source is the assistant, <code>assistant</code> is a listed persona while
<code>other</code> also covers "a plain AI assistant answer", so the source curve and the <code>other</code> curve overlap in meaning on these routes.</p>
<p>The judge prompt is route_judge_v3's examples-and-prioritise prompt, the one used in the September 24–25 runs, with this route's options.
Each rendered prompt is saved under <code>output/ace-steering-29_09/judge/prompts/</code>.</p>
"""
    if judge_tag == "subagent":
        title += " (subagent judge)"
    return dict(eyebrow="Qwen3-8B · persona steering · step %d" % step, title=title, lede=lede,
                facts=[[html.escape(a), html.escape(b)] for a, b in facts], html=body)


def _no_nan(o):
    """NaN is not JSON: JSON.parse rejects the whole page on one bare NaN."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    return o


def _clip(t, n):
    t = t if isinstance(t, str) else ""
    return t if not n or len(t) <= n else t[:n].rstrip() + " … [cut for page size; %d characters in full]" % len(t)


def examples(root, sub, max_per_cell=None, judge_tag="", keep_keys=None, max_chars=None):
    """Every answer (or up to max_per_cell per cell) with its judge verdict, for the examples panel.

    Keyed like page_data's routes ("route|L19"); rows are compact lists:
    [arm, alpha, q_idx, prompt_idx, sample, label, intensity, evidence, text].
    """
    import pandas as pd
    from ace_steering.judge import key_of
    df = pd.read_csv(root / sub / "generations.csv")
    jl = root / "judge" / ("judgements_%s%s.jsonl" % (sub, "_" + judge_tag if judge_tag else ""))
    J = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            d = json.loads(line)
            if d.get("error") is None:
                J[d["key"]] = d
    out = {}
    for (route, layer, arm, al), g in df.groupby(["route", "layer", "arm", "alpha"]):
        if keep_keys is not None and "%s|L%d" % (route, layer) not in keep_keys:
            continue
        if max_per_cell:
            g = g.head(max_per_cell)
        rows = out.setdefault("%s|L%d" % (route, layer), [])
        for _, r in g.iterrows():
            j = J.get(key_of(r), {})
            rows.append([arm, float(al), int(r.q_idx), int(r.prompt_idx), int(r["sample"]),
                         j.get("persona_kind"), j.get("intensity"), j.get("evidence") or "",
                         _clip(r.response, max_chars)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--run-root", default=str(OUT_ROOT))
    ap.add_argument("--judge-tag", default="")
    a = ap.parse_args(argv)
    root = Path(a.run_root)
    sub = "smoke" if a.step == 0 else "step%d" % a.step
    outd = root / (sub + ("_" + a.judge_tag if a.judge_tag else ""))
    data = json.loads((outd / "page_data.json").read_text())
    sp = root / "judge" / "spend.json"
    spend = json.loads(sp.read_text())["spent_usd"] if sp.exists() else 0.0
    tpl = (HERE / "page_template.html").read_text()
    fp = outd / "findings.html"          # hand-written reading of the results, optional
    nt = notes(a.step, data["meta"], spend, a.judge_tag)
    if fp.exists():
        nt["html"] = fp.read_text() + nt["html"]
    step2 = a.step == 2
    data["examples"] = examples(root, sub, max_per_cell=3 if step2 else None, judge_tag=a.judge_tag,
                                keep_keys=set(data["routes"]), max_chars=800 if step2 else None)
    data = _no_nan(data)
    safe = lambda o: json.dumps(o, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")  # noqa: E731
    out = tpl.replace("__DATA__", safe(data)).replace("__NOTES__", safe(nt))
    p = outd / ("ace_%s.html" % outd.name)
    p.write_text(out)
    print("page -> %s (%.1f MB)" % (p, p.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
