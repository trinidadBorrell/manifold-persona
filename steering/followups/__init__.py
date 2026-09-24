"""Steering follow-ups: calibrated alpha, four routes, re-anchored start.

Plan: plans/2026-09-24-steering-followups.md (local-only; the contract).

A separate package, not edits to `steering/`, for one reason: the modules it
builds on (`manifold_paths`, `interventions`, `activation_steering`,
`route_judge_v3`) are imported, never modified, so everything the earlier runs
measured still describes the code that is in the tree. What is new here is
exactly what the plan adds -- the `nearest` knot rule, the alpha calibration,
the paper-faithful replacement hook, the judge wrapper and the figure tables.

The GPU driver lives in the gitignored `jobs_condor/followups_steer.py`
(cluster infra stays untracked); it imports the shared pieces from here.
"""
