"""Persistence barcodes, one figure per role, beside the Assistant's.

WHAT A BARCODE IS
-----------------
Put a ball around every point in a role's cloud and grow the radius. As the
balls swell and overlap, topological features appear and later disappear. Each
feature gets one horizontal bar: it starts at the radius where the feature is
born and ends where it dies. Long bars are real structure; short ones are
sampling noise. The x-axis is therefore a *distance*, in the same units as the
cloud's own diameter.

WHAT EACH ROW TRACKS
--------------------
  H0  CONNECTED COMPONENTS — "how many separate clumps?"
      Every point starts as its own component and dies when it first links to
      another, so an H0 bar's length is the distance to that point's nearest
      already-joined neighbour. One bar never dies (the component everything
      eventually joins); it is drawn to the diameter and marked. H0 is really a
      readout of how sparsely the points are spread, not of connectivity — the
      panel metric `betti0` inherits that caveat.

  H1  LOOPS — "does the behaviour close back on itself?"
      A ring of points that encircles a gap. Born when the ring closes, dies
      when the gap fills in. This is the row that says whether a persona's
      responses form a cycle rather than a blob.

  H2  VOIDS — "is there a hollow region the behaviour surrounds but never
      enters?" A shell, like the surface of a ball with nothing inside. Born
      when the shell seals, dies when the interior fills.

Computed at MAXDIM=2 as of 2026-08-04. The earlier runs stopped at H1 on the
belief that H2 was unaffordable here; measured, it is 0.3 s per role.

THE THRESHOLD LINE
------------------
The dashed vertical marker is `LIFETIME_FRAC` (10%) of the role's own diameter —
the rule the `betti*` counts use. A feature is counted only if its bar is longer
than that. It is drawn on each row so you can see how far the longest bar is
from qualifying, which on this data is the whole story: almost no role clears it
in H1.

WHY EVERY ROLE IS PLOTTED BESIDE `default`
------------------------------------------
A barcode alone is nearly unreadable — bar counts and scales differ per role and
there is no reference for "long". Putting the Assistant's barcode in the right
column on a SHARED x-axis makes the comparison direct: same filtration scale,
same rows, so a difference in bar length is a real difference and not a change
of units.

Usage:
    .venv/bin/python exploratory/per_persona/study_barcodes.py --outdir <run>
    .venv/bin/python exploratory/per_persona/study_barcodes.py --outdir <run> \\
        --roles poet accountant leviathan
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import C_REAL, C_DESIGN, C_INSTR, C_QUEST
from topology import LIFETIME_FRAC

REFERENCE = "default"
DIM_COLORS = {0: C_QUEST, 1: C_REAL, 2: C_INSTR}
DIM_NAMES = {0: "H0 — connected components (clumps)",
             1: "H1 — loops (behaviour closing back on itself)",
             2: "H2 — voids (hollow regions the cloud surrounds)"}
DPI = 150


def load_diagrams(pers_dir: Path, role: str) -> dict:
    """{dim: [n,2] birth/death} for one role, in ascending dimension."""
    with np.load(pers_dir / f"{role}.npz") as z:
        return {int(k[1:]): np.asarray(z[k], float) for k in sorted(z.files)}


def _draw(ax, bars: np.ndarray, diameter: float, color: str, xmax: float):
    """One barcode row. Bars sorted by birth, infinite deaths cut at diameter."""
    if len(bars) == 0:
        ax.text(.5, .5, "no features", ha="center", va="center", fontsize=8,
                color="0.5", transform=ax.transAxes)
        n_inf = 0
    else:
        birth, death = bars[:, 0].copy(), bars[:, 1].copy()
        infinite = ~np.isfinite(death)
        n_inf = int(infinite.sum())
        death[infinite] = diameter
        order = np.lexsort((death - birth, birth))
        birth, death, infinite = birth[order], death[order], infinite[order]
        y = np.arange(len(birth))
        ax.hlines(y, birth, death, color=color, lw=1.1)
        # The essential class is drawn in a different colour: it does not die,
        # and reading it as a very long-lived feature would be wrong.
        if n_inf:
            ax.hlines(y[infinite], birth[infinite], death[infinite],
                      color="#111111", lw=1.6)
        ax.set_ylim(-1, max(len(birth), 1))
    ax.axvline(LIFETIME_FRAC * diameter, color=C_DESIGN, ls="--", lw=1.2, zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    return n_inf


def barcode_figure(role: str, dgm_r: dict, dgm_ref: dict, diam_r: float,
                   diam_ref: float, out: Path, L: int, meta: str = ""):
    dims = sorted(set(dgm_r) | set(dgm_ref))
    # ONE x-limit for both columns. The whole point is visual comparison, and
    # per-panel autoscaling would make two different scales look alike.
    xmax = max(diam_r, diam_ref) * 1.02
    fig, axes = plt.subplots(len(dims), 2, figsize=(13, 2.6 * len(dims)),
                             squeeze=False)
    for i, dim in enumerate(dims):
        for jcol, (nm, dgm, diam) in enumerate(
                ((role, dgm_r, diam_r), (REFERENCE, dgm_ref, diam_ref))):
            ax = axes[i][jcol]
            bars = dgm.get(dim, np.empty((0, 2)))
            n_inf = _draw(ax, bars, diam, DIM_COLORS.get(dim, C_REAL), xmax)
            life = (bars[:, 1] - bars[:, 0]) if len(bars) else np.array([0.0])
            life = life[np.isfinite(life)]
            longest = float(life.max()) if life.size else 0.0
            counted = int((life > LIFETIME_FRAC * diam).sum())
            ax.set_title(f"{nm}   ·   {len(bars)} bars"
                         + (f" ({n_inf} never dies)" if n_inf else "")
                         + f"   ·   longest {longest:.2f} "
                           f"= {longest/diam:.1%} of diameter"
                         + f"   ·   counted: {counted}", fontsize=8)
            if jcol == 0:
                ax.set_ylabel(DIM_NAMES[dim].split(" — ")[0], fontsize=10,
                              fontweight="bold")
        axes[i][0].text(-0.02, 1.30, DIM_NAMES[dim], transform=axes[i][0].transAxes,
                        fontsize=9, color="0.3", ha="left", va="bottom")
    for ax in axes[-1]:
        ax.set_xlabel("filtration scale (distance)", fontsize=9)
    axes[0][1].plot([], [], color=C_DESIGN, ls="--",
                    label=f"{LIFETIME_FRAC:.0%} of diameter — the counting rule")
    axes[0][1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Persistence barcodes — {role}  vs  {REFERENCE} (the Assistant)"
                 f"{meta}\nshared x-axis; diameters {diam_r:.1f} and {diam_ref:.1f}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    ap.add_argument("--roles", nargs="*", default=None,
                    help="subset of roles; default is every role in the run")
    args = ap.parse_args()
    run_dir, L = Path(args.outdir), args.label_layer
    pers = run_dir / "data" / "persistence"
    # Barcodes are the topology family's raw evidence, so they live with it.
    import families as _F
    outdir = (run_dir / "figures" / "families"
              / _F.folder("topology") / "barcodes")
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    panel = pd.read_csv(run_dir / "data" / f"per_role_panel_L{L}.csv")
    diam = dict(zip(panel.role, panel.cloud_diameter))
    axis = dict(zip(panel.role, panel.axis_proj))
    roles = args.roles or sorted(p.stem for p in pers.glob("*.npz"))
    if REFERENCE not in diam:
        raise SystemExit(f"`{REFERENCE}` not in the panel; nothing to compare against")

    dgm_ref = load_diagrams(pers, REFERENCE)
    print(f"{len(roles)} roles -> {outdir}")
    print(f"  reference `{REFERENCE}`: "
          + ", ".join(f"H{d}={len(v)} bars" for d, v in sorted(dgm_ref.items())))
    if max(dgm_ref) < 2:
        print("  NOTE: these diagrams stop at H1. Re-run study_panel.py to get H2 "
              "(topology.MAXDIM is now 2).")

    for i, r in enumerate(roles):
        try:
            dgm = load_diagrams(pers, r)
        except FileNotFoundError:
            print(f"  [skip] no diagram saved for {r!r}")
            continue
        meta = (f"\naxis_proj {axis[r]:+.2f}  (Assistant sits at "
                f"{axis[REFERENCE]:+.2f})" if r in axis else "")
        barcode_figure(r, dgm, dgm_ref, diam[r], diam[REFERENCE],
                       outdir / f"{r}_vs_{REFERENCE}_L{L}.png", L, meta)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(roles)}  ({time.time()-t0:.0f}s)")
    print(f"wrote {len(list(outdir.glob('*.png')))} barcode figures "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
