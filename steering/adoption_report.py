"""Which roles and which questions refuse to be adopted.

Plan: docs/notes/paper-fidelity-plan.md (WP1 / WP3b follow-up).

The role-expression judge gives every response a `fully` / `somewhat` / `no`
label. Pooled, that is one number. Split by ROLE and by QUESTION it answers a
different and more useful question: is a low adoption rate a property of the
model, or of a handful of roles the prompts never landed, or of a handful of
questions that pull every role back to the Assistant?

This matters downstream in two concrete ways:

  * A role whose responses are mostly `no` contributes a role vector built from
    responses that are not in role. It is inside the axis and inside its own
    centroid, so it degrades both. The paper drops exactly these (section 2.1.2,
    the >= 10 threshold); this report says which they are and by how much.
  * A question that pulls every role back to the Assistant is measuring the
    question, not the persona. In the eval that shows up as a flat alpha
    response for reasons that have nothing to do with steering.

Outliers are flagged against the POOLED rate with a Wilson interval, not by a
fixed cutoff: with 100 rows per role and 1,150 per question the sampling noise
differs by an order of magnitude between the two views, and a single threshold
would call one of them wrong.

    python -m steering.adoption_report --labels <role_labels.parquet> \
        --meta data/embeddings_roles_resp240/metadata.parquet --out <run_dir>

Eval mode reads the D.1.3 perspective judge instead, where "did not leave the
Assistant" is `judge_score == "assistant"`:

    python -m steering.adoption_report --judged <judged_L19.parquet> --out <run_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger("adoption_report")

Z = 1.959963984540054          # 95%
TOP_N = 25                     # rows drawn in the per-role figure


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval. Correct at p near 0 or 1, where normal-approx is not.

    Roles at 0% and 100% are exactly the cases this report exists to surface, and
    the textbook interval collapses to zero width there — it would present the
    least-evidenced rows as the most certain.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _rates(df: pd.DataFrame, by: str, flag_col: str) -> pd.DataFrame:
    """Per-group rate of `flag_col`, with a Wilson interval and the group's n."""
    rows = []
    for key, grp in df.groupby(by, sort=True):
        n = int(len(grp))
        k = int(grp[flag_col].sum())
        p, lo, hi = wilson(k, n)
        rows.append({by: key, "n": n, "k": k, "rate": p, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)
    return out


def _flag_outliers(tab: pd.DataFrame, pooled: float) -> pd.DataFrame:
    """A group is an outlier when its interval excludes the pooled rate.

    Direction is recorded, not just the fact: `high` groups are the ones failing
    to adopt the role, `low` groups adopt it more reliably than average, and only
    the first kind is a problem for the axis.
    """
    tab = tab.copy()
    tab["outlier"] = np.where(tab["lo"] > pooled, "high",
                              np.where(tab["hi"] < pooled, "low", ""))
    return tab


def _bar_figure(tab: pd.DataFrame, by: str, pooled: float, title: str,
                path: Path, top_n: int = TOP_N) -> Optional[Path]:
    """Horizontal bars for the worst `top_n` groups, with Wilson whiskers.

    Defensive by convention (steering/README.md): a plotting failure logs and
    returns, and never kills a run that already produced its numbers.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sub = tab.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * len(sub) + 1.5)))
        y = np.arange(len(sub))
        err = np.vstack([np.clip(sub["rate"] - sub["lo"], 0, None),
                         np.clip(sub["hi"] - sub["rate"], 0, None)])
        colours = ["#C44E52" if o == "high" else "#8C8C8C"
                   for o in sub.get("outlier", [""] * len(sub))]
        ax.barh(y, sub["rate"], color=colours, height=0.72)
        ax.errorbar(sub["rate"], y, xerr=err, fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=2)
        ax.axvline(pooled, color="#4C72B0", lw=1.4, ls="--",
                   label="pooled rate = %.1f%%" % (100 * pooled))
        ax.set_yticks(y)
        ax.set_yticklabels([str(v)[:52] for v in sub[by]], fontsize=7)
        ax.set_xlabel("fraction of responses")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    except Exception as exc:                       # noqa: BLE001
        LOG.warning("figure %s failed: %s", path.name, exc)
        return None


def build(df: pd.DataFrame, flag_col: str, label: str, out_dir: Path,
          tag: str) -> dict:
    """Per-role and per-question tables, figures, and a summary dict."""
    data_dir, fig_dir = out_dir / "data", out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pooled = float(df[flag_col].mean())
    summary = {"tag": tag, "criterion": label, "n_rows": int(len(df)),
               "pooled_rate": pooled}
    figures = []

    for by, fname, top in (("role", "fig04_adoption_by_role", TOP_N),
                           ("question", "fig05_adoption_by_question", 20)):
        if by not in df.columns:
            continue
        tab = _flag_outliers(_rates(df, by, flag_col), pooled)
        tab.to_csv(data_dir / ("adoption_by_%s_%s.csv" % (by, tag)), index=False)

        p = _bar_figure(tab, by, pooled,
                        "%s — worst %d by %s (%s)" % (label, top, by, tag),
                        fig_dir / ("%s_%s_L19.png" % (fname, tag)), top_n=top)
        if p:
            figures.append(str(p))

        high = tab[tab["outlier"] == "high"]
        summary["%s_n_groups" % by] = int(len(tab))
        summary["%s_n_flagged_high" % by] = int(len(high))
        summary["%s_worst" % by] = [
            {by: r[by], "rate": round(float(r["rate"]), 4), "n": int(r["n"])}
            for _, r in tab.head(10).iterrows()]

    (data_dir / ("adoption_summary_%s.json" % tag)).write_text(
        json.dumps(summary, indent=2))
    summary["figures"] = figures
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="role_labels.parquet from steering.rolefilter")
    src.add_argument("--judged", help="judged_L19.parquet from steering.judge")
    ap.add_argument("--meta", default="data/embeddings_roles_resp240/metadata.parquet",
                    help="cloud metadata, joined to --labels for role/question")
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.labels:
        lab = pd.read_parquet(args.labels)
        meta = pd.read_parquet(args.meta, columns=["role", "question", "instruction_idx"])
        df = meta.iloc[lab["i"].to_numpy()].reset_index(drop=True)
        df["label"] = lab["label"].to_numpy()
        df = df[df["label"].notna()]
        # The paper's discard: `no role-playing` responses never reach a role
        # vector. Everything here is about how much of the cloud that is, and
        # where it concentrates.
        df["flag"] = df["label"] == "no"
        tag = args.tag or "extraction"
        label = "no role-playing (dropped from the role vector)"
    else:
        df = pd.read_parquet(args.judged)
        df = df[df["judge_score"].notna()]
        # In the eval, `assistant` means the intervention did not move it.
        df["flag"] = df["judge_score"] == "assistant"
        tag = args.tag or "eval"
        label = "stayed the Assistant"
        if "alpha" in df.columns:
            print("NOTE: pooling over alpha. For the dose-response read fig01.")

    summary = build(df, "flag", label, out_dir, tag)
    print(json.dumps({k: v for k, v in summary.items() if k != "figures"}, indent=2))
    for f in summary.get("figures", []):
        print("wrote", f)


if __name__ == "__main__":
    main()
