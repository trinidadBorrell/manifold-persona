"""Read the dose-escalation grids and answer the two questions they were run to answer.

  1. Is there a dose at which behaviour changes?
  2. Does the model break before or after that dose?

The second question is why the canary is here. "Behaviour changed at alpha=8" means nothing on its
own -- a model reduced to repeating one token has also changed its behaviour. So every dose is
scored on both axes and the two alphas are reported side by side:

  alpha_move   first dose whose response differs from the alpha=0 response
  alpha_break  first dose where the CANARY collapses (loses Paris, or degenerates)

alpha_break <= alpha_move is a null: we damaged the model without ever steering it.
alpha_move < alpha_break is the only ordering in which a behavioural change is worth reading.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REP4_MAX = 4          # a 4-gram repeating 5+ times in <=96 tokens is a loop, not prose
DISTINCT_MIN = 0.45   # type/token ratio below this is vocabulary collapse


def collapsed(r):
    return bool(r.deg_rep4 > REP4_MAX or r.deg_distinct < DISTINCT_MIN or r.deg_n_tok < 3)


def main(paths):
    for p in paths:
        df = pd.read_csv(p)
        L = int(df.layer.iloc[0])
        print("=" * 78)
        print(f"hidden state {L}   chord {df.chord.iloc[0]:.1f}   "
              f"||h|| ~ {df.steer_h_norm.iloc[0]:.0f}")
        print("=" * 78)

        can = df[df.kind == "canary"].sort_values("alpha")
        keeps = can.response.str.contains("Paris", case=False, na=False)
        bad = can[[collapsed(r) for _, r in can.iterrows()] | ~keeps]
        a_break = bad.alpha.min() if len(bad) else None

        print("\n  canary (capital of France):")
        print("  %7s %9s %7s %9s %8s %s" % ("alpha", "push", "rep4", "distinct", "Paris?", "first 45 chars"))
        for _, r in can.iterrows():
            print("  %7g %9.0f %7d %9.2f %8s %r"
                  % (r.alpha, r.push_norm, r.deg_rep4, r.deg_distinct,
                     "yes" if "paris" in str(r.response).lower() else "NO",
                     str(r.response)[:45]))

        for q, g in df[df.kind == "identity"].groupby("question", sort=False):
            g = g.sort_values("alpha")
            base = g[g.alpha == 0].response.iloc[0]
            moved = g[(g.alpha > 0) & (g.response != base)]
            a_move = moved.alpha.min() if len(moved) else None
            print(f"\n  identity: {q!r}")
            print("  %7s %9s %8s %7s %9s %s"
                  % ("alpha", "d_tgt", "t", "rep4", "distinct", "first 45 chars"))
            for _, r in g.iterrows():
                print("  %7g %9.1f %+8.2f %7d %9.2f %r"
                      % (r.alpha, r.text_d_tgt, r.text_t, r.deg_rep4, r.deg_distinct,
                         str(r.response)[:45]))
            print(f"    alpha_move  = {a_move}   (first dose whose text differs from alpha=0)")
            print(f"    alpha_break = {a_break}   (first dose the canary fails)")
            if a_move is None:
                print("    -> NULL: no dose up to the maximum changed the answer at all.")
            elif a_break is not None and a_break <= a_move:
                print("    -> NULL: the model broke at or before it moved. Damage, not steering.")
            else:
                print("    -> there is a window. Read the responses in that range by hand.")
        print()


if __name__ == "__main__":
    main([Path(x) for x in sys.argv[1:]])
