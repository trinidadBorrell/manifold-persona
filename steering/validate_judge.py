"""Human-vs-judge agreement on the seven-category role-perspective rubric.

Plan: control 1 ("is the judge measuring what we think it measures?").

WHY THE BLIND / KEY SPLIT
-------------------------
The whole point of the control is an INDEPENDENT reading of the same responses.
If the human labeller can see `judge_score` while labelling, the number that
comes out is not agreement, it is anchoring: a disagreement now costs the
labeller an act of contradiction, and borderline cases collapse onto whatever
the model said. Kappa would then be biased upward by an unknown amount, and no
post-hoc analysis can undo it.

So the sample is written twice:

    judge_validation_blank_L19.csv   text + an empty `human_label` column
    judge_validation_key_L19.csv     row_id -> judge_score, nothing else

The labeller only ever opens the blank file; the key is joined afterwards by
`compute_kappa`. `row_id` is the only thing linking them, and it carries no
signal about the judge's answer. The rubric is copied into the blank file as
`#` comment lines (parsed out of `steering.judge.JUDGE_SYSTEM`, never retyped)
so the human is scored against the SAME seven definitions the model saw, and
does not have to open the judge module — where the labels are.

Stratification is across (arm, alpha) because agreement is not expected to be
uniform over the grid: strong steering produces the degenerate text where the
`weird_role` / `nonsensical` / `other` boundary is hardest, and a simple random
sample would under-represent exactly the cells the study leans on.

Usage:
    .venv/bin/python -m steering.validate_judge \
        --labels <filled blank csv> --key <key csv> --controls <controls_L19.json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from steering.judge import CATEGORIES

BLANK_NAME = "judge_validation_blank_L19.csv"
KEY_NAME = "judge_validation_key_L19.csv"
CONTROLS_KEY = "1_judge_validation"

BLANK_COLUMNS = ["row_id", "arm", "alpha", "target_role", "question", "response",
                 "human_label"]
KEY_COLUMNS = ["row_id", "judge_score"]


# --- rubric ---------------------------------------------------------------

def category_definitions() -> Dict[str, str]:
    """The seven definitions, lifted out of the judge's own system prompt.

    Parsed rather than transcribed: a second copy of the rubric is a second
    thing that can drift, and if it drifts the human and the model are no
    longer being asked the same question, which is precisely the assumption
    kappa is supposed to be testing.
    """
    from steering.judge import JUDGE_SYSTEM

    out = {}  # type: Dict[str, str]
    for line in JUDGE_SYSTEM.splitlines():
        m = re.match(r"^-\s*([A-Za-z_]+)\s*:\s*(.+)$", line.strip())
        if m and m.group(1) in CATEGORIES:
            out[m.group(1)] = " ".join(m.group(2).split())
    missing = [c for c in CATEGORIES if c not in out]
    if missing:
        raise RuntimeError(
            "could not parse definitions for %r out of JUDGE_SYSTEM; the rubric "
            "format changed and this parser must be updated rather than the "
            "definitions retyped here" % (missing,))
    return out


def _rubric_header() -> List[str]:
    defs = category_definitions()
    lines = [
        "# JUDGE VALIDATION - blind human labelling sheet.",
        "# Fill in the LAST column, human_label, with EXACTLY ONE of the seven",
        "# category names below. Leave a row blank to skip it; blank rows are",
        "# dropped, not counted as disagreement.",
        "# The judge's own labels are deliberately NOT in this file.",
        "#",
        "# CATEGORIES:",
    ]
    for c in CATEGORIES:
        lines.append("# - %s: %s" % (c, defs[c]))
    lines.append("#")
    return lines


# --- sampling -------------------------------------------------------------

def _row_ids(df: pd.DataFrame) -> pd.Series:
    """Stable join key. An existing `row_id` column wins; otherwise position.

    Position in the judged frame is stable because the judged parquet is
    written once and never reordered, and it is the only identifier available
    that is guaranteed unique (responses and (arm, alpha, role, question)
    tuples are not, across seeds).
    """
    if "row_id" in df.columns:
        ids = df["row_id"].astype(str)
        if ids.duplicated().any():
            raise ValueError("judged_df has duplicate row_id values; the blank "
                             "and key files could not be joined unambiguously")
        return ids
    return pd.Series(["r%06d" % i for i in range(len(df))], index=df.index)


def _allocate(sizes: List[int], n: int) -> List[int]:
    """Largest-remainder allocation of `n` draws over strata, capped by size."""
    total = int(sum(sizes))
    if n >= total:
        return list(sizes)
    quotas = [n * s / float(total) for s in sizes]
    take = [min(int(np.floor(q)), s) for q, s in zip(quotas, sizes)]
    order = sorted(range(len(sizes)),
                   key=lambda i: (-(quotas[i] - np.floor(quotas[i])), i))
    while sum(take) < n:
        progressed = False
        for i in order:
            if sum(take) == n:
                break
            if take[i] < sizes[i]:
                take[i] += 1
                progressed = True
        if not progressed:  # every stratum exhausted
            break
    return take


def stratified_sample(judged_df: pd.DataFrame, n: int = 100,
                      seed: int = 0) -> pd.DataFrame:
    """`n` rows spread over (arm, alpha) in proportion to cell size.

    Deterministic in `seed` so the same sample can be regenerated if the blank
    file is lost mid-labelling.
    """
    df = judged_df.reset_index(drop=True).copy()
    df["row_id"] = _row_ids(df).values
    if "arm" not in df.columns or "alpha" not in df.columns:
        raise ValueError("judged_df needs `arm` and `alpha` columns to stratify")

    groups = []  # type: List[tuple]
    for key, idx in df.groupby(["arm", "alpha"], dropna=False).groups.items():
        groups.append((str(key), np.asarray(idx)))
    groups.sort(key=lambda kv: kv[0])

    take = _allocate([len(idx) for _, idx in groups], int(n))
    rng = np.random.default_rng(seed)
    picked = []  # type: List[int]
    for (_, idx), k in zip(groups, take):
        if k <= 0:
            continue
        picked.extend(rng.choice(idx, size=k, replace=False).tolist())
    return df.loc[sorted(picked)].reset_index(drop=True)


def write_validation_files(judged_df: pd.DataFrame, out_dir, n: int = 100,
                           seed: int = 0) -> Dict[str, Path]:
    """Write the blind sheet and its separate key. Returns both paths.

    The two files are written together (so they cannot disagree about the
    sample) but are meant to be USED apart: hand over the blank one only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samp = stratified_sample(judged_df, n=n, seed=seed)

    if "judge_score" not in samp.columns:
        raise ValueError("judged_df has no `judge_score` column; there is "
                         "nothing to validate against")

    blank = samp.reindex(columns=BLANK_COLUMNS).copy()
    blank["human_label"] = ""
    blank_path = out / BLANK_NAME
    with open(blank_path, "w") as fh:
        fh.write("\n".join(_rubric_header()) + "\n")
        blank.to_csv(fh, index=False)

    key_path = out / KEY_NAME
    samp[KEY_COLUMNS].to_csv(key_path, index=False)
    return {"blank": blank_path, "key": key_path}


# --- agreement ------------------------------------------------------------

def _read_sheet(path) -> pd.DataFrame:
    """Read a CSV that may carry leading `#` rubric lines.

    Counted and skipped explicitly instead of `comment='#'`, which would also
    truncate any response text containing a hash.
    """
    n_comment = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                n_comment += 1
            else:
                break
    return pd.read_csv(path, skiprows=n_comment, dtype={"row_id": str})


def compute_kappa(blank_filled_path, key_path) -> Dict[str, object]:
    """Cohen's kappa between the human sheet and the judge key.

    The label set is pinned to `CATEGORIES` rather than inferred from the
    sample: a category neither rater used in 100 rows would otherwise vanish
    from the confusion matrix, silently changing its shape and making two runs
    of this control non-comparable.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    human = _read_sheet(blank_filled_path)
    key = _read_sheet(key_path)
    if "human_label" not in human.columns:
        raise ValueError("%s has no `human_label` column" % blank_filled_path)

    merged = human.merge(key[["row_id", "judge_score"]], on="row_id", how="inner")
    if len(merged) == 0:
        raise ValueError("no row_id overlap between the labels and the key")

    lab = merged["human_label"].fillna("").astype(str).str.strip()
    merged = merged.loc[lab != ""].copy()
    merged["human_label"] = lab.loc[merged.index]

    bad_h = sorted(set(merged["human_label"]) - set(CATEGORIES))
    bad_j = sorted(set(merged["judge_score"].astype(str)) - set(CATEGORIES))
    if bad_h or bad_j:
        raise ValueError("labels outside CATEGORIES - human %r, judge %r"
                         % (bad_h, bad_j))

    h = merged["human_label"].tolist()
    j = merged["judge_score"].astype(str).tolist()

    kappa = float(cohen_kappa_score(h, j, labels=CATEGORIES))
    raw = float(np.mean([a == b for a, b in zip(h, j)])) if h else float("nan")
    cm = confusion_matrix(h, j, labels=CATEGORIES)
    confusion = {row: {col: int(cm[i][k]) for k, col in enumerate(CATEGORIES)}
                 for i, row in enumerate(CATEGORIES)}  # outer = human, inner = judge

    return {
        "n_labelled": int(len(merged)),
        "cohens_kappa": kappa,
        "raw_agreement": raw,
        "confusion": confusion,
        "confusion_orientation": "outer key = human label, inner key = judge label",
        "per_category_counts": {
            "human": {c: int(sum(1 for x in h if x == c)) for c in CATEGORIES},
            "judge": {c: int(sum(1 for x in j if x == c)) for c in CATEGORIES},
        },
    }


def merge_into_controls(result: Dict[str, object], controls_path) -> Path:
    """Read-modify-write so the controls file accumulates across controls.

    Every control writes into the same JSON; clobbering it would delete the
    others, which are expensive to recompute.
    """
    p = Path(controls_path)
    existing = {}  # type: Dict[str, object]
    if p.exists():
        text = p.read_text().strip()
        if text:
            existing = json.loads(text)
            if not isinstance(existing, dict):
                raise ValueError("%s is not a JSON object" % p)
    existing[CONTROLS_KEY] = result
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(existing, indent=2) + "\n")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="the filled-in blank sheet")
    ap.add_argument("--key", required=True, help="judge_validation_key_L19.csv")
    ap.add_argument("--controls", required=True,
                    help="controls_L19.json (created if absent, merged if present)")
    args = ap.parse_args(argv)

    res = compute_kappa(args.labels, args.key)
    path = merge_into_controls(res, args.controls)

    print("judge validation (control 1)")
    print("  n labelled       %d" % res["n_labelled"])
    print("  Cohen's kappa    %.3f" % res["cohens_kappa"])
    print("  raw agreement    %.3f" % res["raw_agreement"])
    print("  category         human  judge")
    for c in CATEGORIES:
        print("    %-14s %5d  %5d" % (c, res["per_category_counts"]["human"][c],
                                      res["per_category_counts"]["judge"][c]))
    off = [(res["confusion"][a][b], a, b) for a in CATEGORIES for b in CATEGORIES
           if a != b and res["confusion"][a][b]]
    off.sort(reverse=True)
    if off:
        print("  top disagreements (human -> judge)")
        for cnt, a, b in off[:5]:
            print("    %-14s -> %-14s %d" % (a, b, cnt))
    print("  written to %s under %r" % (path, CONTROLS_KEY))


if __name__ == "__main__":
    main()
