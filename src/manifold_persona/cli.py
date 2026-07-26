"""One command per stage.

    python -m manifold_persona.cli extract  [--response] [-- <extractor args>]
    python -m manifold_persona.cli explore  [-- <run_all args>]
    python -m manifold_persona.cli manifold (--run | --sweep | --local-id) [-- <args>]

A thin dispatcher: each subcommand forwards to an existing module's ``main`` (or
runs the exploratory ``run_all.py``), so behaviour is identical to invoking those
modules directly. Everything after ``--`` is passed through unchanged.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _split_passthrough(argv):
    """Return (own_args, passthrough) split on the first bare ``--``."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def cmd_extract(argv) -> None:
    own, rest = _split_passthrough(argv)
    ap = argparse.ArgumentParser(prog="manifold_persona.cli extract")
    ap.add_argument("--response", action="store_true",
                    help="generate answers and extract RESPONSE-token activations "
                         "(paper-matched). Default: prompt-token activations.")
    args = ap.parse_args(own)
    if args.response:
        from extraction import generate_and_extract_roles as m
    else:
        from extraction import build_and_extract_roles as m
    m.main(rest) if _accepts_argv(m.main) else _run_module_main(m, rest)


def cmd_explore(argv) -> None:
    _own, rest = _split_passthrough(argv)
    # run_all.py is a script, not a package module; run it by path with argv set.
    run_all = Path(__file__).resolve().parents[2] / "exploratory" / "assistant_axis" / "run_all.py"
    sys.argv = [str(run_all)] + rest
    runpy.run_path(str(run_all), run_name="__main__")


def cmd_manifold(argv) -> None:
    own, rest = _split_passthrough(argv)
    ap = argparse.ArgumentParser(prog="manifold_persona.cli manifold")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true", help="H1 decider (manifold.run)")
    g.add_argument("--sweep", action="store_true", help="role-count sweep (manifold.sweep)")
    g.add_argument("--local-id", action="store_true", help="local intrinsic dimension (manifold.local_id)")
    args = ap.parse_args(own)
    if args.run:
        from manifold import run as m
    elif args.sweep:
        from manifold import sweep as m
    else:
        from manifold import local_id as m
    m.main(rest)


def _accepts_argv(fn) -> bool:
    import inspect
    return len(inspect.signature(fn).parameters) >= 1


def _run_module_main(mod, rest) -> None:
    """For a module whose main() takes no argv: set sys.argv and call it."""
    sys.argv = [mod.__name__] + rest
    mod.main()


COMMANDS = {"extract": cmd_extract, "explore": cmd_explore, "manifold": cmd_manifold}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = (f"usage: manifold_persona.cli {{{','.join(sorted(COMMANDS))}}} ...\n"
             "  extract   build the role point cloud (--response for answer tokens)\n"
             "  explore   run the exploratory assistant-axis stage\n"
             "  manifold  fit/test the manifold (--run | --sweep | --local-id)")
    # Manual first-token dispatch, so `<stage> --help` reaches the SUBCOMMAND
    # parser rather than being swallowed by a top-level -h.
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return
    stage, rest = argv[0], argv[1:]
    if stage not in COMMANDS:
        print(usage, file=sys.stderr)
        sys.exit(f"manifold_persona.cli: error: unknown stage {stage!r}")
    COMMANDS[stage](rest)


if __name__ == "__main__":
    main()
