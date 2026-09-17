# A->B steering: results

Run `2026-09-10T15-46-fig4`. Qwen3-8B, hidden state 19, additive steering,
eps=12, k=8, centripetal, lambda=0, temp 0.7 / top_p 0.9, max_new_tokens 256.
6,400 generations, all judged. Judge `claude-sonnet-5`.

Four arms: {linear chord, manifold spline} x {Assistant Axis, persona pair}.
Alpha grid `0, .1, .2, .3, .4, .5, .6, .7, .9, 1.0` -- **0.8 is absent**; ten
values with both endpoints requires dropping one interior point, and 0.8 is the
one that was dropped. Every figure is on that grid.

## 1. The axis arms reproduce the paper's Figure 4

Fraction judged `assistant` under the paper's Appendix D.1.3 rubric:

| arm | alpha=0 | alpha=1 |
|---|---|---|
| linear_axis | 74% | 12% |
| manifold_axis | 73% | 15% |

The identity does move, and the D.1.3 category mix shifts from `assistant` into
`nonhuman_role` and `weird_role` as alpha rises. This is the positive control:
the intervention works, the judge can see it, the pipeline is sound.

## 2. The pair arms do not move at all

Under D.1.3 the pair arms are `assistant` 90.9% (linear) and 90.2% (manifold)
at *every* alpha. That is true and uninformative -- D.1.3 asks "is this the
Assistant?", not "where along this route is it?".

So `steering/route_judge.py` rejudges the 2,400 pair rows against each route's
OWN persona sequence: source A, the centroids the spline threads, target B, and
`other`. The verdict is then a position on the route rather than a category.

Result, averaged over alpha (audited reading where one exists -- see 3):

| route | arm | A | waypoint | B | other |
|---|---|---|---|---|---|
| assistant->scout | linear | 99.0 | 1.0 | **0.0** | 0.0 |
| assistant->scout | manifold | 96.5 | 2.5 | **0.0** | 1.0 |
| assistant->veteran | linear | 98.5 | 0.0 | 1.5 | 0.0 |
| assistant->veteran | manifold | 96.5 | 2.0 | 1.5 | 0.0 |
| summarizer->scout | linear / manifold | 12.0 / 10.5 | 0.0 / 0.0 | 0.0 | 88.0 / 89.5 |
| summarizer->veteran | linear / manifold | 9.0 / 10.0 | 0.0 / 0.5 | 0.0 | 91.0 / 89.5 |
| validator->scout | linear / manifold | 14.5 / 11.0 | 0.0 / 0.0 | 0.0 | 85.5 / 89.0 |
| validator->veteran | linear / manifold | 5.0 / 5.0 | 0.0 / 0.0 | 0.0 | 95.0 / 95.0 |

The target persona is reached in at most 1.5% of responses, and in 0% on both
`scout` routes. On `assistant->veteran` not one of the 200 manifold responses
contains any veteran or military word, so even that 1.5% rests on
weathered-experience phrasing rather than the persona.

Manifold beats linear on waypoint hits (5 of 400 rows vs 2 of 400), which is
the direction the hypothesis predicts and far too small to claim.

## 3. The route judge is wrong on the two `assistant->` routes

616 of those 800 rows (77%) carry the wrong label. The rubric sends a generic
"I am an AI" response to `other` *unless a listed persona is itself that
identity* -- and the judge does not reliably read the listed `assistant` as
satisfying that clause. Its own analysis text repeatedly names a response
"standard AI assistant self-identification" and then emits `other`.

Two further defects found in the same audit:

- **Waypoint inflation.** Generic helper text containing "navigate" or
  "guidance" was scored `navigator`/`guide` with no persona claimed. On
  `assistant->scout` linear, 12 of 14 waypoint labels are spurious.
- **Non-determinism.** Byte-identical responses got different labels inside one
  slice; one group of five identical strings split 3x`assistant` / 2x`other`.

The table in section 2 uses the audited reading for those four cells. The four
`summarizer->`/`validator->` routes are unaffected: `assistant` is not an option
there, so an AI self-identification is correctly `other`.

Fixing this properly is one rubric line plus a re-judge of the 800 rows (~$2).
The audited numbers already answer the question, so it would confirm rather than
change the conclusion.

## 4. What this does and does not show

- It shows that steering toward a **midway** persona, at these alpha, does not
  produce a persona transition by either route.
- It does **not** show that A->B steering fails in general. Every pair route
  targets `scout` or `veteran`, both only mildly non-Assistant. `leviathan` and
  `aberration` were never steered toward. That is a selection error in the run.
- The arms are **not dose-matched**. The spline routes are 1.7x-4.3x the chord
  length, so at matched alpha the manifold arm perturbs the residual stream
  more. Where the arms differ, "the route matters" and "more perturbation
  matters" are both live; separating them needs a control this run lacks.

The obvious next run is these same routes retargeted at the far personas.

## Reproducing

    # generation (cluster, sharded)
    jobs_condor/path_full.submit

    # the paper's D.1.3 judge, all 6,400 rows
    python -m steering.judge --run <run_dir> --judge

    # the route judge, the 2,400 pair rows
    python steering/run_route_judge.py --data <run_dir>/data
    #   --dry-run              print the prompt and the per-route option lists
    #   --redo-empty-analysis  re-judge rows whose reasoning text was lost

    # rebuild the published artifact from a judged parquet
    python steering/build_route_artifact.py --data <run_dir>/data \
        --html <page>.html --audit <per_alpha>.json

Prompts for both judges, verbatim, are generated from the source at
`extra/judge_prompts.txt` (that directory is gitignored; regenerate rather than
expecting it in a clone).

## Figures

- Steering Dose Response -- https://claude.ai/code/artifact/707eafcd-5486-491a-814f-263b17a5c02e
- Persona Route Results -- https://claude.ai/code/artifact/39f96e40-0bd7-4a53-8e0b-c313550051cf
