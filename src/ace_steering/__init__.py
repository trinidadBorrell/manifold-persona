"""ACE vs linear vs manifold steering (09-29 to 10-02): step 1 (three routes at hidden state 19) and
step 2 (ten routes x seven layers).

Plan: ~/.claude/plans/piped-shimmying-simon.md (approved 2026-09-29).

A separate package, like `steering/followups`, so the modules it builds on
(`manifold_paths`, `interventions`, `activation_steering`, `route_judge_v3`,
`followups.judge_followups`) are imported, not rewritten. The only change to a
shared module is the new ACE hook in `src/steering_core/interventions.py`.

The GPU driver is `ace_steering.steer_gpu` (launched by jobs_condor/run_ace.sh).
"""
