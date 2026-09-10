"""Fresh evaluation system prompts for the 50 near-Assistant roles.

Plan: docs/notes/paper-fidelity-plan.md (WP3).

WHY. arXiv:2601.10387 Appendix D.1.1: "We then repeated the data generation
process from our existing pipeline for these 50 roles to create NEW system
prompts for this evaluation." The role vectors were extracted from responses to
the ORIGINAL prompts, so evaluating steering on those same prompts tests on
training data - and Arms 2 and 3 aim at centroids built from exactly those
responses, which hands them an advantage that has nothing to do with geometry.

WHAT IS OURS, NOT THE PAPER'S. Appendix A holds their prompt-writing
meta-prompt, and it is NOT in our copy of the PDF (the text runs from the main
body to B.1 with no A.x). The instruction below is written from the section
2.1.1 description - "we relied on the same frontier model to generate five
system prompts designed to elicit each desired role" - and few-shots the role's
existing five so the new ones match the register while differing in wording.
This is a reconstruction. It is recorded as a deviation in the run manifest and
must be reported as one.

Output: <run_dir>/data/role_prompts_eval.json, consumed by
`run_steering.py --prompts`. The extraction cloud keeps using the ORIGINAL
prompts - it must, they are what the centroids are made of.

    .venv/bin/python -m steering.gen_role_prompts --out <run_dir> --yes
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering.judge import _client, load_api_key  # noqa: E402

WRITER_MODEL = "claude-sonnet-5"
N_PROMPTS = 5

WRITER_SYSTEM = """You write system prompts that elicit a specific character role from an instruct-tuned language model.

Given a [role] and the five [existing] system prompts already used for it, write FIVE NEW system prompts for the same role.

Requirements:
- Each must plainly direct the model to be, act as, or assume that role.
- They must be materially different from the [existing] ones and from each other: different phrasing, framing, and emphasis. Do not paraphrase.
- Match the register and length of the existing ones (one or two sentences, second person, no markdown, no examples, no meta-commentary).
- Describe the role itself. Never mention AI, language models, assistants, role-play, or the fact that this is a prompt.
- Return only the five prompts."""

WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        # NO minItems/maxItems: the API rejects any array minItems other than
        # 0 or 1 ("For 'array' type, 'minItems' values other than 0 or 1 are not
        # supported"), and every one of the 50 calls 400'd on it. The count is
        # enforced in `write_one`, which retries a short or long list anyway, so
        # the schema was asserting something the code already guarantees.
        "prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompts"],
    "additionalProperties": False,
}


def _params(role: str, existing: List[str]) -> dict:
    return {
        "model": WRITER_MODEL,
        "max_tokens": 1200,
        "system": WRITER_SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": WRITER_SCHEMA}},
        "messages": [{"role": "user",
                      "content": "[role]: %s\n\n[existing]:\n%s"
                                 % (role, "\n".join("- " + e for e in existing))}],
    }


def load_existing(role: str, instructions_dir: Path) -> List[str]:
    data = json.loads((instructions_dir / ("%s.json" % role)).read_text())
    return [i.get("pos", "") for i in data["instruction"]]


def write_one(client, role: str, existing: List[str], max_attempts: int = 5) -> dict:
    """Five new prompts for one role. Never raises; failures come back tagged."""
    for attempt in range(max_attempts):
        try:
            msg = client.messages.create(**_params(role, existing))
            text = next(b.text for b in msg.content if b.type == "text")
            prompts = json.loads(text)["prompts"]
            prompts = [p.strip() for p in prompts if p and p.strip()]
            if len(prompts) != N_PROMPTS:
                raise ValueError("got %d prompts, want %d" % (len(prompts), N_PROMPTS))
            overlap = set(p.lower() for p in prompts) & set(e.lower() for e in existing)
            if overlap:
                raise ValueError("model returned an existing prompt verbatim")
            return {"role": role, "prompts": prompts, "error": None}
        except Exception as exc:                       # noqa: BLE001
            if attempt == max_attempts - 1:
                return {"role": role, "prompts": None,
                        "error": "%s: %s" % (type(exc).__name__, exc)}
            # A 529 is the service being busy, not us being rate-limited, and it
            # can persist for minutes: 48 of 50 roles failed under one such
            # window. Wait meaningfully and jitter, so retries spread out.
            name = type(exc).__name__
            base = 10 if ("Overloaded" in name or "RateLimit" in name) else 2
            time.sleep(min(base * (2 ** attempt), 120) * (0.5 + random.random()))


def main() -> None:
    from manifold_persona.config import ROLE_INSTRUCTIONS_DIR

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--roles", default="near50",
                    help="'near50' (default), 'far50', 'all', or a comma-separated list")
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    # THE SAME GEOMETRY THE RUN WILL USE. `near50`/`far50` are read off the
    # Assistant Axis, and the axis differs depending on whether the
    # role-expression filter was applied — so calling load_geometry() bare here
    # while run_steering calls it with --labels produces two different role
    # lists. The run then dies hours in at load_system_prompts with
    # KeyError('no generated prompts for role X'), after the GPU time is spent.
    # WHICH CLOUD. Defaults to the resp240 (Qwen2.5-3B) cloud that
    # geometry.RESP240_DIR names. A run against a different model MUST pass
    # this: without it the axis, the centroids and the near-50 are built
    # from a different model's activations than the one being steered, and
    # nothing downstream can detect it.
    ap.add_argument("--resp-dir", default=None,
                    help="response cloud directory (default: the resp240 3B cloud)")
    ap.add_argument("--labels", default=None,
                    help="role_labels.parquet from steering.rolefilter (WP1); "
                         "MUST match the --labels run_steering will be given, "
                         "or near50 differs and the run fails on a missing role")
    ap.add_argument("--n-bar", default=None,
                    help="n_bar_lmsys.json from steering.lmsys_norm (WP2); "
                         "pass it for symmetry with run_steering (it does not "
                         "affect which roles are selected)")
    ap.add_argument("--yes", action="store_true",
                    help="required to spend money; without it this is a dry run")
    args = ap.parse_args()

    run_dir = Path(args.out)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)

    geom = None
    if args.roles in ("near50", "far50", "all"):
        from steering.geometry import RESP240_DIR, load_geometry
        geom = load_geometry(resp_dir=args.resp_dir or RESP240_DIR,
                         labels_path=args.labels, n_bar_path=args.n_bar)
        if args.labels is None:
            print("WARNING: --labels not given, so near50/far50 come from the "
                  "UNFILTERED axis. If run_steering is given --labels it will "
                  "select a different set of roles and fail on the first one "
                  "this bank does not cover.", flush=True)
        roles = {"near50": geom.near50, "far50": geom.far50,
                 "all": [r for r in geom.roles if r != "default"]}[args.roles]
    else:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    if not args.yes:
        raise SystemExit(
            "DRY RUN - nothing sent.\n  roles: %d\n  model: %s\n"
            "  rough cost: $%.2f\nRe-run with --yes." % (len(roles), args.model,
                                                         len(roles) * 0.01))
    if not load_api_key():
        raise SystemExit("no API key (ANTHROPIC_API_KEY or token/anthropic.txt)")

    instructions_dir = Path(ROLE_INSTRUCTIONS_DIR)
    client = _client()
    results: Dict[str, dict] = {}

    # RESUME. Each role costs one API call and the whole set is one file, so a
    # transient service outage that fails 48 of 50 roles must not force the two
    # that succeeded to be paid for and regenerated again — nor risk replacing
    # good prompts with a different draw on every retry.
    out_path = run_dir / "data" / "role_prompts_eval.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text()).get("prompts", {})
        for r, ps in prev.items():
            if ps:
                results[r] = {"role": r, "prompts": ps, "error": None}
        if results:
            print("resuming: %d roles already have prompts" % len(results))
    roles = [r for r in roles if r not in results]
    if not roles:
        print("every role already has prompts; nothing to do")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(write_one, client, r, load_existing(r, instructions_dir)): r
                for r in roles}
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results[res["role"]] = res
            if k % 10 == 0:
                print("[%d/%d]" % (k, len(roles)), flush=True)

    failed = [r for r, v in results.items() if v["prompts"] is None]
    payload = {
        "source": "generated by %s; reconstruction of the section 2.1.1 prompt-writing "
                  "step (Appendix A is absent from our PDF copy)" % args.model,
        "paper_ref": "arXiv:2601.10387 Appendix D.1.1",
        "deviation": "meta-prompt is ours, not the authors'",
        "n_prompts_per_role": N_PROMPTS,
        "n_roles": len(results),
        # Recorded so a later "no generated prompts for role X" is diagnosable:
        # it means run_steering built a different role list from a different
        # axis, and this line says which one the bank was built from.
        "roles_selection": args.roles,
        "labels": args.labels,
        "axis_definition": (geom.axis_report.get("definition") if geom else None),
        "failed": failed,
        "prompts": {r: v["prompts"] for r, v in sorted(results.items())
                    if v["prompts"] is not None},
    }
    p = out_path
    p.write_text(json.dumps(payload, indent=2))
    print("wrote %s  (%d roles, %d failed)" % (p, len(payload["prompts"]), len(failed)))
    if failed:
        print("FAILED roles (re-run to retry): %s" % ", ".join(failed))


if __name__ == "__main__":
    main()
