"""Descriptive uncertainty checks on A's existing, verified prompt cells.

This script does not change A's original result, intervals, gates or role panel.
The conditional spreads below are diagnostics, not additive variance shares.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def interval(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("undefined or nonfinite axis draw")
    lo, hi = np.percentile(values[1:], [2.5, 97.5])
    return dict(point=float(values[0]), lo=float(lo), hi=float(hi), n_nonfinite=0)


def cosine(a, b):
    aa, bb = np.sum(a * a, axis=-1), np.sum(b * b, axis=-1)
    if np.any(aa <= 1e-20) or np.any(bb <= 1e-20):
        raise ValueError("undefined zero-length axis")
    return np.clip(np.sum(a * b, axis=-1) / np.sqrt(aa * bb), -1, 1)


def axis_draws(S, halves, rr, W, mode="joint"):
    """Same balanced-cell contrast as A, with conditional role/question draws."""
    S, rr, W = np.asarray(S, float), np.asarray(rr), np.asarray(W, float)
    if not np.isfinite(S).all() or not np.isfinite(W).all():
        raise ValueError("axis inputs must be finite")
    if S.ndim != 4 or rr.ndim != 2 or W.shape != (len(rr), S.shape[1]):
        raise ValueError("axis input shapes differ")
    R = S.shape[0] - 1
    if rr.shape[1] != R or np.any(rr < 0) or np.any(rr >= R) or np.any(W < 0):
        raise ValueError("invalid resampling weights")
    if mode not in {"joint", "roles_only", "questions_only"}:
        raise ValueError("unknown conditional mode")
    counts = np.array([np.bincount(r, minlength=R) / R for r in rr])
    role_weights = np.column_stack([-counts, np.ones(len(rr))])
    reference = S.mean((0, 1, 2))
    output = []
    for half in halves:
        features, coefficients = [], []
        for qm, wm in half:
            raw = S[:, qm][:, :, wm].mean(2) - reference
            if mode == "roles_only":
                features.append(raw.mean(1))
                coefficients.append(role_weights)
            else:
                qw = W[:, qm]
                den = qw.sum(1, keepdims=True)
                if np.any(den <= 0):
                    raise ValueError("undefined empty resampled half")
                qw = qw / den
                if mode == "questions_only":
                    features.append(raw[-1] - raw[:-1].mean(0))
                    coefficients.append(qw)
                else:
                    features.append(raw.reshape(-1, S.shape[-1]))
                    coefficients.append((role_weights[:, :, None] * qw[:, None, :]).reshape(len(rr), -1))
        a, b = features
        ca, cb = coefficients
        aa = np.sum((ca @ (a @ a.T)) * ca, axis=1)
        bb = np.sum((cb @ (b @ b.T)) * cb, axis=1)
        ab = np.sum((ca @ (a @ b.T)) * cb, axis=1)
        if np.any(aa <= 1e-20) or np.any(bb <= 1e-20):
            raise ValueError("undefined zero-length axis")
        output.append(np.clip(ab / np.sqrt(aa * bb), -1, 1))
    return np.mean(output, axis=0)


def influences(S, halves):
    """Leave-one-out point estimates only; no selection or new pass decisions."""
    S = np.asarray(S, float)
    R, Q, NW = S.shape[0] - 1, S.shape[1], S.shape[2]
    base, role, question, wording, single, all_words = [], [], [], [], [], []
    for half in halves:
        sides, role_axes, question_axes, wording_axes, single_axes, all_axes = [], [], [], [], [], []
        for qm, wm in half:
            qs = np.flatnonzero(qm)
            raw = S[:, qs][:, :, wm]
            vectors = raw.mean((1, 2))
            axis = vectors[-1] - vectors[:-1].mean(0)
            sides.append(axis)
            role_axes.append(vectors[-1] - (vectors[:-1].sum(0) - vectors[:-1]) / (R - 1))
            deltas = raw[-1].mean(1) - raw[:-1].mean((0, 2))
            qaxes = np.tile(axis, (Q, 1))
            qaxes[qs] = (len(qs) * axis - deltas) / (len(qs) - 1)
            question_axes.append(qaxes)
            dwords = S[-1, qs].mean(0)
            rmean = vectors[:-1].mean(0)
            waxes = np.tile(axis, (NW, 1))
            for w in np.flatnonzero(wm):
                kept = wm.copy()
                kept[w] = False
                waxes[w] = dwords[kept].mean(0) - rmean
            wording_axes.append(waxes)
            single_axes.append(dwords - rmean)
            all_axes.append(dwords.mean(0) - rmean)
        base.append(cosine(*sides))
        role.append(cosine(*role_axes))
        question.append(cosine(*question_axes))
        wording.append(cosine(*wording_axes))
        single.append(cosine(*single_axes))
        all_words.append(cosine(*all_axes))
    full_axis = S[-1].mean((0, 1)) - S[:-1].mean((0, 1, 2))
    role_means = S[:-1].mean((1, 2))
    drop_role = S[-1].mean((0, 1)) - (role_means.sum(0) - role_means) / (R - 1)
    qdelta = S[-1].mean(1) - S[:-1].mean((0, 2))
    drop_q = (Q * full_axis - qdelta) / (Q - 1)
    return dict(observed_cosine=float(np.mean(base)),
                leave_role_out=np.mean(role, axis=0).tolist(),
                leave_question_out=np.mean(question, axis=0).tolist(),
                leave_default_wording_out=np.mean(wording, axis=0).tolist(),
                default_wording_alone=np.mean(single, axis=0).tolist(),
                all_default_wordings=float(np.mean(all_words)),
                full_axis_alignment_leave_role_out=cosine(drop_role, full_axis).tolist(),
                full_axis_alignment_leave_question_out=cosine(drop_q, full_axis).tolist())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--original-results", type=Path, required=True)
    p.add_argument("--raw-read-note", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if sys.platform != "linux":
        p.error("real-data diagnostic runs on the existing pod")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(stage):
        write(args.out / "status.json", dict(stage=stage, elapsed_seconds=time.monotonic()-started,
              updated_utc=datetime.now(timezone.utc).isoformat()))
        print(stage, flush=True)
    original_hash = digest(args.original_results)
    original = json.loads(args.original_results.read_text())
    expected_source = next(h for name, h in original["provenance"]["code"].items()
                           if name.endswith("/instruct_persona_map/analysis.py") or
                           name.endswith("/261029_instruct_persona_map/analysis.py"))
    if digest(args.analysis_source) != expected_source:
        raise ValueError("analysis source differs from the original result")
    spec = importlib.util.spec_from_file_location("frozen_poda_axis", args.analysis_source)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    expected_maps = next(h for name, h in original["provenance"]["code"].items() if name.endswith("/maps.py"))
    if digest(A.MAPS_PATH) != expected_maps:
        raise ValueError("map helper differs from the original result")
    write(args.out / "spec.json", dict(question="Which sampled units drive A's axis uncertainty?",
          hypotheses=["Role-panel composition drives the interval", "Questions drive the interval",
                      "Default-prompt wordings affect the point estimate", "Finite bootstrap draws explain the miss"],
          mode="de-risk: existing data, conditional resampling and point sensitivities only",
          budget="two CPU threads, 1200 seconds; no GPU, generation, judge call or HF upload",
          original_results_sha256=original_hash, analysis_sha256=expected_source,
          maps_sha256=expected_maps, diagnostic_sha256=digest(__file__), command=sys.argv,
          limits="Exploratory diagnostic; original rules and outputs unchanged; spreads are not additive variance shares"))
    progress("verify_original_inputs")
    run = A.load_run(args.run, args.raw_read_note)
    if run.provenance["input_hashes"] != original["provenance"]["input_hashes"]:
        raise ValueError("input hashes differ from original analysis")
    progress("construct_balanced_prompt_cells")
    S, N = A.cell_means(run.X[:, A.STATE], run.rows, run.roles + ["default"], run.questions)
    if not np.all(N == 1):
        raise ValueError("diagnostic requires complete balanced prompt-cell support")
    rr, W = A.M.make_draws(len(run.roles), len(run.questions), n=A.N_BOOT, seed=A.SEED)
    halves = A.make_halves(run.questions, n=A.N_HALVES)
    progress("reproduce_joint_axis_interval")
    scores = {"joint": axis_draws(S, halves, rr, W)}
    actual, expected = interval(scores["joint"]), original["primary"]["D3"]["cosine"]
    for k in ("point", "lo", "hi"):
        if not np.isclose(actual[k], expected[k], atol=2e-10, rtol=0):
            raise ValueError(f"original {k} does not reproduce: {actual[k]} != {expected[k]}")
    reference, _ = A.axis_statistics(S, N, halves, rr[:65], W[:65])
    for k in ("point", "lo", "hi"):
        if not np.isclose(interval(scores["joint"][:65])[k], reference["cosine"][k], atol=2e-10, rtol=0):
            raise ValueError("bounded original-function equivalence failed")
    for mode in ("roles_only", "questions_only"):
        progress(mode)
        scores[mode] = axis_draws(S, halves, rr, W, mode)
    progress("repeat_joint_draws_for_monte_carlo_check")
    r2, w2 = A.M.make_draws(len(run.roles), len(run.questions), n=A.N_BOOT, seed=A.SEED+1)
    scores["joint_independent_draw_seed"] = axis_draws(S, halves, r2, w2)
    progress("role_question_and_default_wording_influences")
    influence = influences(S, halves)
    summaries = {}
    for k, values in scores.items():
        summaries[k] = dict(interval(values), width=float(np.ptp(np.percentile(values[1:], [2.5, 97.5]))),
                            bootstrap_sd=float(np.std(values[1:], ddof=1)))
    point = summaries["joint"]["point"]
    roles = [dict(role=name, cosine_without=float(value), delta=float(value-point),
                  full_axis_alignment=influence["full_axis_alignment_leave_role_out"][i])
             for i, (name, value) in enumerate(zip(run.roles, influence["leave_role_out"]))]
    questions = [dict(question=q, cosine_without=float(value), delta=float(value-point),
                      full_axis_alignment=influence["full_axis_alignment_leave_question_out"][i])
                 for i, (q, value) in enumerate(zip(run.questions, influence["leave_question_out"]))]
    wordings = [dict(wording=i, cosine_without=influence["leave_default_wording_out"][i],
                    delta=influence["leave_default_wording_out"][i]-point,
                    cosine_using_only_this_default_wording=influence["default_wording_alone"][i]) for i in range(5)]
    examples = []
    for w in range(5):
        candidates = sorted((r for r in run.rows if r["role"] == "default" and r["wording"] == w and r["replicate"] == 0),
                            key=lambda r: r["question"])
        for row in candidates[:2]:
            examples.append({k: row[k] for k in ("role", "wording", "question", "replicate", "reply", "valid")})
    write(args.out / "default_raw_examples.json", examples)
    np.savez_compressed(args.out / "draws.npz", **scores, roles=rr, question_weights=W)
    if digest(args.original_results) != original_hash:
        raise ValueError("original result changed during diagnostic")
    result = dict(status="descriptive_diagnostic_complete", original_results_sha256=original_hash,
                  original_axis_result=original["primary"]["D3"], summaries=summaries,
                  role_influence=sorted(roles, key=lambda x: abs(x["delta"]), reverse=True),
                  question_influence=sorted(questions, key=lambda x: abs(x["delta"]), reverse=True),
                  default_wording_influence=wordings, all_default_wordings_on_each_side=influence["all_default_wordings"],
                  input_provenance=run.provenance, versions=dict(python=platform.python_version(), numpy=np.__version__),
                  elapsed_seconds=time.monotonic()-started,
                  limits=["Same fixed question/wording halves and original bootstrap multiplicities",
                          "Conditional role/question spreads are not additive variance shares",
                          "Leaving out a role, question or default wording is a sensitivity, not a revised pass result",
                          "Default-wording checks change the descriptive contrast and do not amend the primary rule",
                          "No new replies or validated behavior labels; no population-wide causal claim"])
    write(args.out / "results.json", result)
    write(args.out / "SHA256SUMS.json", {p.name: digest(p) for p in args.out.iterdir() if p.is_file() and p.name not in ("status.json", "SHA256SUMS.json")})
    progress("complete")


if __name__ == "__main__":
    main()
