# steering/archive/ — the v1 route judge, kept so run `2026-09-10T15-46-fig4` stays reproducible

`route_judge.py` (v1 rubric), `run_route_judge.py` (its concurrent driver) and
`build_route_artifact.py` (rewrites the "Persona Route Results" artifact from its output) are
**superseded by `steering/route_judge_v3.py`** (which builds on `route_judge_v2.py`). v1 is the
judge that `steering/RESULTS.md` §3 shows mislabelling 77% of the `assistant->` rows, inflating
waypoint hits on words like "navigate", and giving byte-identical responses different labels.
v2 fixed those failure modes and v3 dropped the positional labels that leaked the route
structure to the judge. Nothing outside this directory imports these three files, which is why
they were moved rather than left next to the judges in use. They were moved with `git mv`, not
deleted, because the section 2 table in RESULTS.md was produced by them and can only be
regenerated with them. The only edits were the import path (`steering.archive.route_judge`) and
the repo-root `sys.path` depth, so `python steering/archive/run_route_judge.py --data ...` still
runs. Do not use them for new judging.
