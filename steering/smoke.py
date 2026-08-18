"""Smoke test: prove the plumbing before spending GPU hours.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Outputs -> Smoke test).

Checks, in order, each of which has failed silently in some pipeline somewhere:

  1. spline port          matches scipy's independent natural cubic, and
                          interpolates its control points exactly at lam = 0
  2. HOOK LAYER OFF-BY-ONE  the hooked module's output IS hidden_states[19].
                          This is the one that would corrupt everything while
                          looking fine: steering one layer away from the space
                          the targets were defined in.
  3. delta norms          every arm realises ||delta|| = alpha * N_bar per
                          position, to tolerance — the dose matching the whole
                          comparison rests on
  4. arms differ          arms 2 and 3 point in measurably different directions
                          (if they did not, the experiment would be vacuous)
  5. generation           all three arms produce text end to end
  6. rate                 measured generations/sec, so the full run's wall-clock
                          is a measurement rather than a guess

Usage:
    .venv/bin/python -m steering.smoke                 # no API calls at all
    .venv/bin/python -m steering.smoke --judge         # + <=50 interactive judge calls
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from steering.geometry import LAYER_HS_INDEX, load_geometry
from steering.activation_steering import ActivationSteering, hook_layer_for_hidden_state
from steering import interventions as IV
from steering.spline1d import CubicSpline1D

TOL_DOSE = 0.02          # relative tolerance on ||delta|| / (alpha * N_bar)


def check_spline() -> dict:
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(0)
    u = np.sort(rng.uniform(-3, 3, 12))
    y = rng.normal(size=(12, 4))
    ours, ref = CubicSpline1D(u, y, lam=0.0), CubicSpline(u, y, axis=0, bc_type="natural")
    q = np.linspace(u[0], u[-1], 200)
    interior = float(np.abs(ours.evaluate(q) - ref(q)).max())
    knots = float(np.abs(ours.evaluate(u) - y).max())
    ok = interior < 1e-10 and knots < 1e-10
    return {"check": "spline_port", "passed": ok,
            "max_abs_diff_vs_scipy": interior, "max_knot_interp_error": knots}


def check_hook_layer(model, tokenizer, hook_layer: int) -> dict:
    """The hooked module's output must equal hidden_states[LAYER_HS_INDEX].

    `output_hidden_states=True` returns L+1 tensors: index 0 is the embedding
    output and index i>=1 is the output of model.layers[i-1]. If this is off by
    one, every arm steers in a space one layer away from the geometry that
    defines its targets, and nothing downstream would notice.
    """
    import torch

    enc = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    captured = {}

    def cb(activations, layer_idx):
        t = activations[0] if isinstance(activations, (tuple, list)) else activations
        captured["h"] = t.detach().float().cpu().numpy().copy()
        return torch.zeros_like(t)          # no-op delta: pure observation

    with ActivationSteering(model, intervention_type="dynamic", delta_fn=cb,
                            layer_indices=[hook_layer], positions="all"):
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

    ref = out.hidden_states[LAYER_HS_INDEX].detach().float().cpu().numpy()
    got = captured["h"]
    diff = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).max())

    # Also confirm the neighbours are NOT equal, so a match cannot be accidental.
    n_lo = float(np.abs(got - out.hidden_states[LAYER_HS_INDEX - 1].detach().float().cpu().numpy()).max())
    n_hi = (float(np.abs(got - out.hidden_states[LAYER_HS_INDEX + 1].detach().float().cpu().numpy()).max())
            if LAYER_HS_INDEX + 1 < len(out.hidden_states) else float("inf"))

    return {"check": "hook_layer_offset",
            "passed": bool(diff <= 1e-3 * max(scale, 1.0) and n_lo > 1e-3 and n_hi > 1e-3),
            "hook_layer": hook_layer, "hidden_state_index": LAYER_HS_INDEX,
            "max_abs_diff": diff, "activation_absmax": scale,
            "diff_to_layer_below": n_lo, "diff_to_layer_above": n_hi}


def check_delta_norms(geom, model, tokenizer, hook_layer: int, alpha: float = 1.0) -> dict:
    """Every arm must realise ||delta|| = alpha * N_bar at every position."""
    import torch

    enc = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    target = geom.centroid(geom.far50[0])
    want = alpha * geom.n_bar
    results, dirs = {}, {}

    for name, fn in [
        ("arm2_linear", IV.make_arm2_delta_fn(target, alpha, geom.n_bar)),
        ("arm3_manifold", IV.make_arm3_delta_fn(geom.spline, geom.axis_unit, target,
                                                alpha, geom.n_bar)),
    ]:
        seen = {}

        def probe(activations, layer_idx, _fn=fn, _seen=seen):
            t = activations[0] if isinstance(activations, (tuple, list)) else activations
            d = _fn(t, layer_idx)
            _seen["d"] = d.detach().float().cpu().numpy().copy()
            return torch.zeros_like(t)

        with ActivationSteering(model, intervention_type="dynamic", delta_fn=probe,
                                layer_indices=[hook_layer], positions="all"):
            with torch.no_grad():
                model(**enc)
        norms = np.linalg.norm(seen["d"], axis=-1).reshape(-1)
        results[name] = {"mean": float(norms.mean()), "min": float(norms.min()),
                         "max": float(norms.max()),
                         "rel_err": float(np.abs(norms - want).max() / want)}
        dirs[name] = seen["d"].reshape(-1, seen["d"].shape[-1])

    # Arm 1 is a static vector; its norm is exact by construction.
    v1 = IV.arm1_axis_vector(geom.axis_unit, alpha, geom.n_bar)
    results["arm1_axis"] = {"mean": float(np.linalg.norm(v1)), "min": float(np.linalg.norm(v1)),
                            "max": float(np.linalg.norm(v1)),
                            "rel_err": float(abs(np.linalg.norm(v1) - want) / want)}

    a2, a3 = dirs["arm2_linear"], dirs["arm3_manifold"]
    cos = float(np.mean(np.sum(a2 * a3, 1) /
                        (np.linalg.norm(a2, axis=1) * np.linalg.norm(a3, axis=1) + 1e-12)))
    passed = all(r["rel_err"] < TOL_DOSE for r in results.values())
    return {"check": "delta_norms", "passed": bool(passed), "alpha": alpha,
            "n_bar": geom.n_bar, "target_norm": want, "per_arm": results,
            "mean_cos_arm2_arm3": cos,
            "arms_are_distinct": bool(abs(cos) < 0.999)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="run dir (default: timestamped smoke dir)")
    ap.add_argument("--judge", action="store_true",
                    help="also exercise the judge on <=50 interactive calls")
    ap.add_argument("--max-calls", type=int, default=50)
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from manifold_persona.config import MODEL_NAME
    from steering.runmeta import new_run_dir

    run_dir = Path(args.out) if args.out else new_run_dir("smoke")
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    checks = []

    print("== 1. spline port")
    checks.append(check_spline())
    print("   ", checks[-1])

    print("== geometry")
    t0 = time.time()
    geom = load_geometry()
    print("    loaded in %.1fs | N_bar %.3f | %d roles" % (time.time() - t0, geom.n_bar, len(geom.roles)))

    model_name = args.model or MODEL_NAME
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("== loading %s on %s" % (model_name, device))
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    hook_layer = hook_layer_for_hidden_state(LAYER_HS_INDEX)

    print("== 2. hook layer off-by-one")
    checks.append(check_hook_layer(model, tokenizer, hook_layer))
    print("   ", checks[-1])

    print("== 3/4. delta norms + arm distinctness")
    checks.append(check_delta_norms(geom, model, tokenizer, hook_layer))
    print("   ", json.dumps(checks[-1], indent=2))

    if not args.skip_generation:
        print("== 5/6. generation")
        from steering.run_steering import build_grid, generate_cell, default_batch_size
        cells = build_grid(geom, smoke=True)
        bs = default_batch_size(device)
        print("    batch_size=%d on %s" % (bs, device))
        # One cell from every arm, so a broken arm cannot hide behind a working one.
        pick, seen = [], set()
        for c in cells:
            if c.arm not in seen:
                seen.add(c.arm); pick.append(c)
        t0, rows = time.time(), 0
        frames = []
        for c in pick:
            df = generate_cell(c, geom, model, tokenizer, hook_layer, batch_size=bs)
            frames.append(df)
            rows += len(df)
            print("    %-42s %2d rows  %.2f gen/s" % (c.shard_name, len(df),
                                                      len(df) / df.attrs["elapsed_s"]))
        rate = rows / (time.time() - t0)
        import pandas as pd
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_parquet(run_dir / "data" / "generations_L19.parquet", index=False)
        # `rows > 0` is NOT a check. The first version of this file passed 4/4
        # while every response was the string "!!!!!!!!" (Observations O3), so
        # the check now looks at the TEXT: degenerate output is one character
        # repeated, or near-empty, and it must be zero in the unsteered arm.
        def _degenerate(t: str) -> bool:
            t = (t or "").strip()
            return len(t) < 10 or len(set(t.replace(" ", ""))) <= 2

        deg = all_df["response"].map(_degenerate)
        base = all_df["arm"] == "unsteered"
        n_deg_base = int(deg[base].sum())
        examples = all_df.loc[deg, "response"].head(2).tolist()
        checks.append({"check": "generation", "rows": rows,
                       "passed": bool(rows > 0 and n_deg_base == 0),
                       "degenerate_unsteered": n_deg_base,
                       "degenerate_total": int(deg.sum()),
                       "degenerate_examples": [e[:60] for e in examples],
                       "generations_per_second": rate,
                       "projected_hours_full_grid": 31750 / rate / 3600,
                       "batch_size": bs, "device": device})
        print("   ", checks[-1])
        for arm in all_df["arm"].unique():
            r = all_df[all_df["arm"] == arm].iloc[-1]
            print("\n    %s alpha=%.2f role=%s q=%r" % (arm, r["alpha"], r["role"], r["question"]))
            print("       ", repr(r["response"][:170]))

    if args.judge:
        from steering.judge import smoke_judge
        checks.append(smoke_judge(run_dir, max_calls=args.max_calls))
        print("   ", checks[-1])

    (run_dir / "data" / "smoke_checks.json").write_text(json.dumps(checks, indent=2))
    failed = [c for c in checks if not c.get("passed", False)]
    print("\n%s  %d/%d checks passed -> %s"
          % ("FAIL" if failed else "PASS", len(checks) - len(failed), len(checks), run_dir))
    if failed:
        raise SystemExit("failed: " + ", ".join(c["check"] for c in failed))


if __name__ == "__main__":
    main()
