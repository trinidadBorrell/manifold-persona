"""ONE figure: where the identity goes, rollout by rollout, as the dose grows.

Plan: docs/notes/plan-new-run.md, step 7.

The Figure-4 plots average over roles and questions, which is what makes them
readable and also what hides the thing this figure is for: whether a response
moves from the prompted persona to the target one *gradually* or *all at once*,
and whether the 20 rollouts of a role move together or scatter.

So: no averaging. One cell per (rollout, alpha), coloured by what the
all-personas judge called it, reduced to five outcomes that answer the question
being asked:

    source      still the persona it was prompted as
    target      the persona we steered toward
    assistant   never left the Assistant
    other role  some third persona - the intervention moved it somewhere unasked
    broken      nonsensical or unclear

Rows are grouped by the 5 introspective questions so a question that behaves
differently from the others is visible as a band rather than averaged away.

    python -m steering.figure_transition --judged persona_judged.parquet --out RUN
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

LOG = logging.getLogger("figure_transition")

OUTCOMES = ["source", "target", "assistant", "other role", "broken"]
COLOURS = ["#55A868", "#C44E52", "#4C72B0", "#DD8452", "#8C8C8C"]


def classify(row) -> str:
    """Judge verdict -> one of five outcomes, relative to this row's own pair."""
    p = row.get("persona")
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "broken"
    if p in ("nonsensical", "unclear"):
        return "broken"
    if p == "assistant":
        return "assistant"
    if p == row.get("role"):
        return "source"
    if p == row.get("target_role") and row.get("target_role"):
        return "target"
    return "other role"


def make_figure(df: pd.DataFrame, out_path: Path, title: str = "",
                max_roles: int = 12) -> Optional[Path]:
    """One panel per source role (per target, when the arms have targets).

    THE CURRENT ARMS HAVE NO TARGET. `run_steering` writes
    `target_role = cell.target_role or cell.role`, and no arm sets one any more,
    so every row satisfies `target_role == role`: grouping by the pair yields 50
    self-pairs, the `target` outcome is unreachable, and the "source -> target"
    reading the figure was built for does not exist in this data. Detected from
    the frame rather than assumed, so a legacy target-directed run still draws
    the way it always did.
    """
    df = df.copy()
    df["outcome"] = df.apply(classify, axis=1)

    # Self-pairs: is there any row whose target differs from its prompted role?
    # Drop the conditions that cannot be a transition: the unsteered rows carry
    # target_role == role and arm == "unsteered", so leaving them in produced a
    # whole column plus two rows of "no data" panels and squeezed the actual
    # comparison into a third of the figure.
    if "arm" in df.columns:
        df = df[df["arm"] != "unsteered"]
    df = df[df["target_role"].notna() & (df["target_role"] != df["role"])]
    if df.empty:
        LOG.warning("no source->target rows to draw")
        return None

    has_targets = bool((df["target_role"].notna()
                        & (df["target_role"] != df["role"])).any())
    outcomes = OUTCOMES if has_targets else [o for o in OUTCOMES if o != "target"]
    colours = [COLOURS[OUTCOMES.index(o)] for o in outcomes]
    code = {o: i for i, o in enumerate(outcomes)}

    if has_targets:
        pairs = sorted(df.groupby(["role", "target_role"]).groups.keys())
    else:
        pairs = [(r, None) for r in sorted(df["role"].dropna().unique())]
    arms = sorted(df["arm"].dropna().unique())
    if not pairs or not arms:
        LOG.warning("nothing to draw")
        return None

    n_all = len(pairs)
    if n_all > max_roles:
        # 50 panels at 2.5in each is a 10-foot figure nobody can read. Take the
        # first `max_roles` in the frame's own order and SAY SO on the figure —
        # a silently truncated panel grid reads as the whole experiment.
        pairs = pairs[:max_roles]

    alphas = sorted(df["alpha"].dropna().unique())
    nrow, ncol = len(pairs), len(arms)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.8 * ncol + 1.6, 2.4 * nrow + 1.6),
                             sharex=True)
    cmap = ListedColormap(colours)

    for r, (src, tgt) in enumerate(pairs):
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            sub = df[(df["role"] == src) & (df["arm"] == arm)]
            if tgt is not None:
                sub = sub[sub["target_role"] == tgt]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#B00020", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # One row per rollout: (system prompt, question). Sorting by question
            # first keeps each question's 4 prompts adjacent, so a question-level
            # effect shows up as a contiguous band.
            #
            # Explicit column reads: `sub.get("question_idx", 0)` returns the
            # SCALAR 0 when the column is absent, and zip(0, 0) raises TypeError.
            zeros = np.zeros(len(sub), dtype=int)
            qi = (sub["question_idx"].to_numpy() if "question_idx" in sub
                  else zeros)
            si = (sub["system_idx"].to_numpy() if "system_idx" in sub else zeros)
            sub = sub.assign(_k=list(zip(qi, si)))
            keys = sorted(set(sub["_k"]))
            grid = np.full((len(keys), len(alphas)), np.nan)
            for i, k in enumerate(keys):
                for j, a in enumerate(alphas):
                    m = sub[(sub["_k"] == k) & (np.isclose(sub["alpha"], a))]
                    if len(m):
                        grid[i, j] = code[m.iloc[0]["outcome"]]

            ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(outcomes) - 1,
                      interpolation="nearest")
            ax.set_xticks(range(len(alphas)))
            # Bottom row only: sharex hides the others, and repeating them on
            # every panel just steals height from the data.
            if r == len(pairs) - 1:
                ax.set_xticklabels(["%g" % a for a in alphas], fontsize=8.5, rotation=0)
            else:
                ax.set_xticklabels([])
            # Question boundaries: the horizontal lines make the 5 questions
            # legible without a label per row.
            qs = [k[0] for k in keys]
            for i in range(1, len(qs)):
                if qs[i] != qs[i - 1]:
                    ax.axhline(i - 0.5, color="white", lw=1.1)
            ax.set_yticks([np.mean([i for i, q in enumerate(qs) if q == u])
                           for u in sorted(set(qs))])
            ax.set_yticklabels(["q%s" % u for u in sorted(set(qs))], fontsize=6.5)
            if r == 0:
                ax.set_title(arm, fontsize=9)
            if c == 0:
                ax.set_ylabel("%s\n-> %s" % (src, tgt) if tgt is not None
                              else str(src), fontsize=7.5)

    # The journey arms do not carry a dose: their `alpha` column holds f, the
    # fraction of the way from the activation to the target centroid, and it is
    # never negative. Labelling it "negative = away from the Assistant" would
    # describe the axis run, not this one.
    if df["arm"].astype(str).str.contains("journey").any():
        sub_label = ("f = fraction of the way to the target "
                     "(f = 1 replaces the activation with the target centroid)")
    else:
        sub_label = "steering strength alpha (negative = away from the Assistant)"
    if len(pairs) < n_all:
        sub_label += "   |   showing %d of %d roles" % (len(pairs), n_all)
    fig.supxlabel(sub_label, fontsize=9)
    fig.legend(handles=[Patch(facecolor=colours[i], label=o)
                        for i, o in enumerate(outcomes)],
               loc="upper center", ncol=len(outcomes), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    head = title or "EXPLORATORY - persona identity per rollout vs steering strength"
    if not has_targets:
        # Without this line an empty legend slot reads as "nothing ever reached
        # the target" rather than "no arm in this run had one".
        head += ("\nno arm in this run has a target role, so 'target' cannot "
                 "occur and is not drawn")
    fig.suptitle(head, fontsize=10, y=0.955)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    LOG.info("wrote %s", out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--max-roles", type=int, default=12,
                    help="panel rows to draw (default 12). All 50 roles would "
                         "be a 10-foot figure; the count drawn is stated on it.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_parquet(args.judged)
    p = Path(args.out) / "figures" / ("fig06_persona_transition_L%d.png" % args.layer)
    return 0 if make_figure(df, p, max_roles=args.max_roles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
