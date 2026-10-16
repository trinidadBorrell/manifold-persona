"""Do the two arms actually differ? Measured, in full space, not in a 2-D shadow.

The PCA trajectory figure (fig07) makes the arms look alike, and it is partly
right and partly an artefact: PC1+PC2 hold about half the variance, and the arms
genuinely share their early path. This figure asks the question in the space the
steering happens in, so neither excuse applies.

Two panels, both computed on the REPLAYED activations (`capture_activations.py`),
never on the intended delta:

  A  cos( act(alpha) - act(0), -a_hat ) -- how much of the measured displacement
     lies along the Assistant Axis. A fixed push along the axis should keep this
     roughly constant; an arm that leaves the axis shows it falling.

  B  || act_linear(alpha) - act_manifold(alpha) || as a fraction of ||act||, at
     matched dose. Zero would mean the arms are the same intervention.

    python -m steering_rebuild.figures_divergence --acts a.npy --meta a.meta.parquet \\
        --cloud <dir> --labels labels.parquet --outdir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

LIN, MAN = "linear_axis", "manifold_axis"


def curves(acts, meta, roles, a_hat):
    """Per-role, per-alpha: axis alignment for each arm, and the arms' separation."""
    align = {LIN: {}, MAN: {}}
    sep = {}
    for role in roles:
        b = meta[(meta.role == role) & (meta.arm == "unsteered")].index
        if not len(b):
            continue
        base = acts[np.asarray(b)].mean(0)
        alphas = sorted(meta[(meta.role == role) & (meta.arm == LIN)].alpha.unique())
        for a in alphas:
            ends = {}
            for arm in (LIN, MAN):
                i = meta[(meta.role == role) & (meta.arm == arm)
                         & (np.isclose(meta.alpha, a))].index
                if not len(i):
                    continue
                m = acts[np.asarray(i)].mean(0)
                ends[arm] = m
                d = m - base
                n = np.linalg.norm(d)
                if n > 0:
                    align[arm].setdefault(float(a), []).append(float(d @ (-a_hat) / n))
            if len(ends) == 2:
                s = np.linalg.norm(ends[LIN] - ends[MAN]) / np.linalg.norm(ends[LIN])
                sep.setdefault(float(a), []).append(float(s))
    return align, sep


def make(align, sep, out: Path, n_roles: int):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for arm, col, lab in ((LIN, "#4C72B0", "linear"), (MAN, "#C44E52", "manifold")):
        xs = sorted(align[arm])
        mu = [float(np.mean(align[arm][a])) for a in xs]
        lo = [float(np.percentile(align[arm][a], 25)) for a in xs]
        hi = [float(np.percentile(align[arm][a], 75)) for a in xs]
        ax.plot(xs, mu, "-o", color=col, lw=2.4, ms=6, label=lab)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0)
    ax.axhline(0, color="#888", lw=0.9)
    ax.set_title("A — how much of the measured move is along the axis", fontsize=11)
    ax.set_xlabel("alpha (fraction of avg. residual norm)")
    ax.set_ylabel(r"cos$\,($act$(\alpha)-$act$(0)$, $-\hat a)$")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=10)

    ax = axes[1]
    xs = sorted(sep)
    mu = [100 * float(np.mean(sep[a])) for a in xs]
    lo = [100 * float(np.percentile(sep[a], 25)) for a in xs]
    hi = [100 * float(np.percentile(sep[a], 75)) for a in xs]
    ax.plot(xs, mu, "-o", color="#8172B3", lw=2.4, ms=6)
    ax.fill_between(xs, lo, hi, color="#8172B3", alpha=0.18, lw=0)
    ax.axhline(0, color="#888", lw=0.9)
    ax.set_title("B — distance between the two arms at matched dose", fontsize=11)
    ax.set_xlabel("alpha (fraction of avg. residual norm)")
    ax.set_ylabel(r"$\|$linear $-$ manifold$\|$  (% of $\|$act$\|$)")
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("The two arms are the same intervention only at low dose — "
                 "measured in full 4096-D space", fontsize=13.5, y=0.985)
    fig.text(0.5, 0.012,
             "mean over %d roles, band = inter-quartile range across roles. Computed on "
             "REPLAYED activations of the generated text, not on the intended delta." % n_roles,
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    return out


def make_html(align, sep, out, n_roles: int):
    """Interactive twin of fig08. Per-role points behind the mean, hoverable.

    Written from the same `align`/`sep` dicts the PNG uses, so the two cannot
    drift apart: if one is regenerated, the other is regenerated from the same
    numbers in the same call.
    """
    import json as _j

    def series(d, colour, name, scale=1.0):
        xs = sorted(d)
        return {"x": xs, "y": [scale * float(np.mean(d[a])) for a in xs],
                "type": "scatter", "mode": "lines+markers", "name": name,
                "line": {"color": colour, "width": 3}, "marker": {"size": 8},
                "hovertemplate": "alpha %{x:+.3f}<br>%{y:.3f}<extra>" + name + "</extra>"}

    a_traces = [series(align[LIN], "#4C72B0", "linear"),
                series(align[MAN], "#C44E52", "manifold")]
    b_traces = [series(sep, "#8172B3", "|linear - manifold|", scale=100.0)]

    doc = """<!doctype html><meta charset="utf-8"><title>Arm divergence</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>body{font:14px system-ui,sans-serif;margin:0;color:#111}
h1{font-size:18px;margin:14px 18px 2px}p{margin:0 18px 8px;color:#555;font-size:12.5px}
.row{display:flex;flex-wrap:wrap}.c{flex:1 1 480px;height:60vh}</style>
<h1>The two arms are the same intervention only at low dose</h1>
<p>Measured in full 4096-D space on the replayed activations of the generated text, not on the
intended delta. Mean over %d roles. <b>A</b>: how much of the measured move lies along the
Assistant Axis. <b>B</b>: distance between the arms at matched dose, as %% of the activation norm.</p>
<div class="row"><div class="c" id="a"></div><div class="c" id="b"></div></div><script>
Plotly.newPlot('a', %s, {title:'A - alignment with the Assistant Axis',
 xaxis:{title:'alpha'}, yaxis:{title:'cos(act(a)-act(0), -axis)'}, hovermode:'x unified',
 margin:{l:60,r:20,t:40,b:50}}, {responsive:true, displaylogo:false});
Plotly.newPlot('b', %s, {title:'B - distance between the arms',
 xaxis:{title:'alpha'}, yaxis:{title:'%% of |act|'}, hovermode:'x unified',
 margin:{l:60,r:20,t:40,b:50}}, {responsive:true, displaylogo:false});
</script>""" % (n_roles, _j.dumps(a_traces), _j.dumps(b_traces))
    out.write_text(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--cloud", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--roles", default=None, help="default: every role present")
    a = ap.parse_args()

    from steering_core.geometry import load_geometry
    from steering_core import interventions as IV

    acts = np.load(a.acts).astype(np.float64)
    meta = pd.read_parquet(a.meta)
    roles = ([r.strip() for r in a.roles.split(",")] if a.roles
             else sorted(meta.role.dropna().unique()))
    g = load_geometry(resp_dir=a.cloud, labels_path=a.labels)
    align, sep = curves(acts, meta, roles, IV.unit(g.axis_unit))

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    p = make(align, sep, out / "fig08_arm_divergence_L19.png", len(roles))
    h = make_html(align, sep, out / "interactive_arm_divergence_L19.html", len(roles))
    json.dump({"n_roles": len(roles),
               "axis_alignment_mean": {k: {str(a_): float(np.mean(v))
                                           for a_, v in d.items()} for k, d in align.items()},
               "separation_frac_mean": {str(a_): float(np.mean(v)) for a_, v in sep.items()}},
              open(out / "fig08_arm_divergence.json", "w"), indent=2)
    print("wrote", p)
    print("wrote", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
