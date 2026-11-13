"""Pod A analysis under THRESHOLDS v1.3. Heavy analysis runs on a pod.

No model generation or judge calls occur here. C's Gram helpers are loaded
read-only. A's bootstrap re-centres distinct sampled roles, as E3 does, and its
null permutes one half only. Dependency hashes enter the runtime fingerprint. Reduced counts are engineering
checks and cannot produce a confirmatory pass.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAPS_PATH = HERE.parent / "261029_fixed_text_read_through/maps.py"
_spec = importlib.util.spec_from_file_location("_poda_map_statistics", MAPS_PATH)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
N_BOOT, N_PERM, N_HALVES = 2000, 9999, 50
MODEL = "allenai/OLMo-2-1124-7B-Instruct"
REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
STATE, OTHER_STATES = 17, (9, 23)
SEED = 20261030
EMBED_MODEL = ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130")
TEXT = {"word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)}


class AnalysisError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def phase(out, name, fingerprint, compute):
    """A completed phase is reused only with identical code, data and settings."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / (name + ".json")
    if p.exists():
        saved = json.loads(p.read_text())
        if saved.get("fingerprint") != fingerprint:
            raise AnalysisError(f"{p}: checkpoint fingerprint differs")
        actual = hashlib.sha256(json.dumps(saved["result"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        if saved.get("result_sha256") != actual:
            raise AnalysisError(f"{p}: cached result checksum differs")
        for name, h in saved["result"].get("artifact_hashes", {}).items():
            artifact = out / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not artifact.is_file() or sha256(artifact) != h:
                raise AnalysisError(f"{p}: cached artifact differs: {name}")
        return saved["result"]
    try:
        result = clean(compute())
    except Exception as error:
        write_json(out / (name + "_error.json"), dict(fingerprint=fingerprint,
                   error_type=type(error).__name__, message=str(error)))
        raise
    checksum = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write_json(p, dict(fingerprint=fingerprint, result=result, result_sha256=checksum))
    return result


def confirmatory_settings(n_boot, n_perm, n_halves):
    return (n_boot, n_perm, n_halves) == (N_BOOT, N_PERM, N_HALVES)


def precision_ok(s, half_width):
    vals = [s.get(k) for k in ("point", "lo", "hi")]
    return bool(all(v is not None and np.isfinite(v) for v in vals)
                and vals[1] - 1e-12 <= vals[0] <= vals[2] + 1e-12
                and vals[2] - vals[1] <= 2 * half_width + 1e-12)


def summary(values):
    """A v1.3 freezes percentile intervals; C's later amendment is separate."""
    values = np.asarray(values, float)
    if len(values) == 0:
        raise AnalysisError("summary requires an observed value")
    bad = int((~np.isfinite(values[1:])).sum())
    lo, hi = np.percentile(values[1:], [2.5, 97.5]) if len(values) > 1 and not bad else (None, None)
    return clean(dict(point=float(values[0]), lo=lo, hi=hi, n_nonfinite=bad))


def make_halves(questions, n=N_HALVES, seed=0, n_wordings=5):
    """The E3/C fixed masks: disjoint questions AND disjoint wording ids."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        qa = np.isin(np.arange(len(questions)), rng.choice(len(questions), len(questions) // 2, replace=False))
        nw = rng.choice([n_wordings // 2, n_wordings - n_wordings // 2])
        wa = np.isin(np.arange(n_wordings), rng.choice(n_wordings, nw, replace=False))
        out.append(((qa, wa), (~qa, ~wa)))
    return out


def cell_means(X, rows, roles, questions, replicates=None):
    keys = [json.dumps([r["role"], r["format"], r["wording"], r["question"], r["replicate"]]) for r in rows]
    use = np.array([bool(r["valid"]) and (replicates is None or r["replicate"] in replicates) for r in rows])
    if not np.isfinite(np.asarray(X)[use]).all():
        raise AnalysisError("a valid row has nonfinite features")
    return M.cells(X, keys, roles, questions, mask=use)


def one_half_null(sides, n_q, n_perm=N_PERM, seed=SEED + 1):
    """A D2 null: the right half alone gets one role permutation per draw.

    C's cross-checkpoint two-direction null is a different statistic and is
    intentionally not used for this within-model decision.
    """
    from scipy.stats import rankdata
    ones = np.ones((1, n_q))
    d = np.array([[M.dist(side.grams(ones))[0] for side in h] for h in sides])
    C = d.shape[-1]
    i, j = np.triu_indices(C, 1)
    left = rankdata(d[:, 0, i, j], axis=-1)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for start in range(0, n_perm, 500):
        perms = np.array([rng.permutation(C) for _ in range(min(500, n_perm - start))])
        right = rankdata(d[:, 1][:, perms[:, i], perms[:, j]], axis=-1)
        null[start:start + len(perms)] = M.corr(left[:, None], right).mean(0)
    return null


def map_panel(sides, rr, W, comparisons, strict=True):
    """Fixed halves, E3 distinct-role centring, duplicated pair multiplicity.

    Re-centring a subset of a centred Gram matrix is equivalent to re-centring
    its raw vectors. C's panel helper centres all roles before resampling, so
    it cannot implement A's E3 bootstrap without this extra step.
    """
    from scipy.stats import rankdata
    out = {pair: dict(r=np.full(len(rr), np.nan), undefined_halves=np.zeros(len(rr), int))
           for pair in comparisons}
    for start in range(0, len(rr), 50):
        stop = min(start + 50, len(rr))
        grams = {key: np.array([[side.grams(W[start:stop]) for side in h] for h in values])
                 for key, values in sides.items()}
        for b in range(start, stop):
            unique, pos = np.unique(rr[b], return_inverse=True)
            i, j = np.triu_indices(len(rr[b]), 1)
            keep = rr[b, i] != rr[b, j]
            ranks = {}
            for key, value in grams.items():
                g = value[:, :, b - start][..., unique, :][..., unique]
                g = g - g.mean(-1, keepdims=True) - g.mean(-2, keepdims=True) + g.mean((-1, -2), keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    norm = np.sqrt(np.maximum(np.diagonal(g, axis1=-2, axis2=-1), 0))
                    d = 1 - g / (norm[..., :, None] * norm[..., None, :])
                ranks[key] = rankdata(d[..., pos[i[keep]], pos[j[keep]]], axis=-1)
            for a, c in comparisons:
                values = (M.corr(ranks[a][:, 0], ranks[c][:, 1]) + M.corr(ranks[c][:, 0], ranks[a][:, 1])) / 2
                out[(a, c)]["r"][b] = np.mean(values) if strict else M._nanmean(values)
                out[(a, c)]["undefined_halves"][b] = int((~np.isfinite(values)).sum())
    return out


def map_statistics(X, rows, roles, questions, n_boot=N_BOOT, n_perm=N_PERM, n_halves=N_HALVES,
                   replicates=None):
    S, N = cell_means(X, rows, roles, questions, replicates)
    halves = make_halves(questions, n=n_halves)
    sides = M.build_sides(S, N, halves)
    rr, W = M.make_draws(len(roles), len(questions), n=n_boot, seed=SEED)
    stats = map_panel({"a": sides}, rr, W, [("a", "a")], strict=True)[("a", "a")]
    values = stats["r"]
    null = one_half_null(sides, len(questions), n_perm=n_perm, seed=SEED + 1)
    if not np.isfinite(null).all():
        raise AnalysisError("map null contains undefined draws")
    s = summary(values)
    p = M.p_value(values[0], null) if np.isfinite(values[0]) else None
    precision = precision_ok(s, .10)
    criterion = s["lo"] is not None and s["lo"] >= .5 and p is not None and p < .01
    return dict(raw_r=s, p=p, spearman_brown=float(2 * values[0] / (1 + values[0]))
                if np.isfinite(values[0]) and values[0] > -1 else None,
                precision_met=precision, criterion_met=bool(criterion),
                passed=bool(criterion and precision), null_method="one half role labels permuted",
                null_p99=float(np.percentile(null, 99)), undefined_halves=stats["undefined_halves"].tolist(),
                bootstrap_values=clean(values), null_values=clean(null))


def axis_statistics(S, N, halves, rr, W):
    """Crossed-bootstrap contrast cosines with train-only projections.

    Last class is the default anchor. Role draws exclude that anchor. Gram
    contractions avoid holding all bootstrap axes in full feature space.
    A common translation reduces cancellation and does not change an axis.
    """
    C, Q, _, D = S.shape
    if rr.shape[1] != C - 1:
        raise AnalysisError("axis role bootstrap must exclude the default anchor")
    counts = np.array([np.bincount(r, minlength=C - 1) / (C - 1) for r in rr])
    weights = np.column_stack([-counts, np.ones(len(rr))])
    reference = S.sum((0, 1, 2)) / N.sum()
    cosines, origins, units, scores = [], [], [], []
    for half in halves:
        Ts, coefs, vectors = [], [], []
        for qm, wm in half:
            qs = np.flatnonzero(qm)
            ns = N[:, qs][:, :, wm].sum(2)
            raw = S[:, qs][:, :, wm].sum(2)
            T = (raw - ns[..., None] * reference).reshape(C * len(qs), D)
            den = np.einsum("bq,cq->bc", W[:, qs], ns)
            with np.errstate(divide="ignore", invalid="ignore"):
                coef = (weights[:, :, None] * W[:, None, qs] / den[:, :, None]).reshape(len(rr), -1)
                V = raw.sum(1) / ns.sum(1)[:, None]
            Ts.append(T)
            coefs.append(coef)
            vectors.append(V)
        ca, cb = coefs
        Ta, Tb = Ts
        aa = np.einsum("bi,ij,bj->b", ca, Ta @ Ta.T, ca, optimize=True)
        bb = np.einsum("bi,ij,bj->b", cb, Tb @ Tb.T, cb, optimize=True)
        ab = np.einsum("bi,ij,bj->b", ca, Ta @ Tb.T, cb, optimize=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = ab / np.sqrt(np.maximum(aa, 0) * np.maximum(bb, 0))
        cv[(aa <= 1e-20) | (bb <= 1e-20)] = np.nan
        cosines.append(np.clip(cv, -1, 1))
        oo, uu, ss = [], [], []
        for side in range(2):
            train, test = vectors[side], vectors[1 - side]
            o = train[:-1].mean(0)
            a = train[-1] - o
            u = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-10 else np.full(D, np.nan)
            oo.append(o)
            uu.append(u)
            ss.append((test - o) @ u)
        origins.append(oo)
        units.append(uu)
        scores.append(ss)
    vals = np.asarray(cosines).mean(0)
    sm = summary(vals)
    criterion = sm["point"] is not None and sm["point"] >= .8
    precision = precision_ok(sm, .05)
    result = dict(cosine=sm, criterion_met=bool(criterion), precision_met=precision,
                  passed=bool(criterion and precision), scope="repeatable default-vs-role contrast",
                  undefined_draws=int((~np.isfinite(vals)).sum()),
                  projection_rule="train-half axis and role-mean origin; other-half vectors only")
    return result, dict(origins=np.array(origins), units=np.array(units), test_scores=np.array(scores))


def prompt_index(rows, roles, questions):
    ri, qi = {r: i for i, r in enumerate(roles)}, {q: i for i, q in enumerate(questions)}
    groups = {}
    for i, r in enumerate(rows):
        if r["valid"] and r["role"] in ri:
            groups.setdefault((ri[r["role"]], qi[r["question"]], r["wording"]), []).append(i)
    keys = sorted(groups)
    if not keys:
        raise AnalysisError("no valid prompt means")
    return dict(y=np.array([k[0] for k in keys]), q=np.array([k[1] for k in keys]),
                w=np.array([k[2] for k in keys]), rows=[groups[k] for k in keys],
                C=len(roles), Q=len(questions))


def decode_splits(meta, n=N_HALVES, seed=20261029):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        qs = rng.choice(meta["Q"], meta["Q"] // 2, replace=False)
        ws = rng.choice(5, 3, replace=False)
        tr = np.isin(meta["q"], qs) & np.isin(meta["w"], ws)
        te = ~np.isin(meta["q"], qs) & ~np.isin(meta["w"], ws)
        if any(len(np.unique(meta["y"][m])) != meta["C"] for m in (tr, te)):
            raise AnalysisError("a role has no train or test prompt")
        out.append((tr, te))
    return out


def synthetic_gate():
    """Deterministic calibration only; no language model or experiment data."""
    rng = np.random.default_rng(14)
    C, Q, D = 12, 16, 20
    halves = make_halves(range(Q), n=5)
    rr, W = M.make_draws(C, Q, n=0)
    ps = []
    for k in range(24):
        panels = {s: M.synthetic_cells(rng.normal(size=(C, D)), np.zeros((Q, D)), .7, rng) for s in "ab"}
        sides = {s: M.build_sides(*v, halves) for s, v in panels.items()}
        hybrid = [(a[0], b[1]) for a, b in zip(sides["a"], sides["b"])]
        obs = map_panel({"a": hybrid}, rr, W, [("a", "a")], strict=True)[("a", "a")]["r"][0]
        ps.append(M.p_value(obs, one_half_null(hybrid, Q, n_perm=99, seed=k + 51)))
    R = rng.normal(size=(C, D))
    S, N = M.synthetic_cells(R, np.zeros((Q, D)), .1, rng)
    sides = M.build_sides(S, N, halves)
    obs = map_panel({"a": sides}, rr, W, [("a", "a")], strict=True)[("a", "a")]["r"][0]
    planted_p = M.p_value(obs, one_half_null(sides, Q, n_perm=99, seed=3))
    Z = np.zeros((C + 1, Q, 5, D))
    Z[-1, :, :, 0] = 1
    axis, _ = axis_statistics(Z, np.ones(Z.shape[:3]), halves, rr, W)
    uniform = .25 < np.mean(ps) < .75 and np.mean(np.asarray(ps) <= .1) < .3
    return dict(passed=bool(uniform and planted_p <= .01 and axis["cosine"]["point"] == 1),
                independent_null_mean_p=float(np.mean(ps)), independent_null_low_tail=float(np.mean(np.asarray(ps) <= .1)),
                planted_p=planted_p, axis_offset_recorded=axis["cosine"]["point"],
                axis_offset_limit="This passes with no role differences; it is not evidence of persona structure.")


def row_key(row):
    return json.dumps([row[k] for k in ("role", "format", "wording", "question", "replicate")], separators=(",", ":"))


class Run:
    pass


def verify_generation_identity(manifest):
    """Historical generation bytes stay fixed when an analysis helper changes."""
    approved = {"gen_extract.py": "fd030a35a7abf7900d5dfe3cdc25ba2a298d3a679a77a75b3099405c9bbcfa8b",
                "rows.py": "b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a"}
    if manifest.get("code") != approved:
        raise AnalysisError("generation source hashes differ from the approved archived A run")


def verify_snapshot(manifest, grid):
    """Offline G5 and row-check prerequisites; never download a tokenizer."""
    from manifold_persona import rows as R
    if manifest.get("sampling") != R.sampling_settings(.5, 1.1, 128):
        raise AnalysisError("sampling differs from frozen A settings")
    markers = ["<|user|>", "<|system|>", "<|assistant|>"]
    if manifest.get("markers") != markers or manifest.get("model_name_fill") != "OLMo":
        raise AnalysisError("chat markers or model-name fill differ")
    if manifest.get("grid_hash") != R.freeze_grid(grid)[1]:
        raise AnalysisError("grid hash differs from manifest")
    verify_generation_identity(manifest)
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    expected_files = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    if set(manifest.get("tokenizer_files", {})) != expected_files:
        raise AnalysisError("tokenizer snapshot hash coverage differs")
    for name, value in manifest["tokenizer_files"].items():
        path = cached_file(MODEL, name, revision=REVISION, local_files_only=True)
        if path is None or sha256(path) != value:
            raise AnalysisError(f"cached tokenizer snapshot differs: {name}")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != manifest.get("chat_template"):
        raise AnalysisError("chat template differs from manifest")
    fields = ("model", "revision", "tokenizer_files", "chat_template", "code", "dtype", "sampling", "markers",
              "model_name_fill", "grid_hash", "batch_size")
    if R.fingerprint({key: manifest[key] for key in fields}) != manifest.get("fingerprint"):
        raise AnalysisError("run fingerprint differs from its recorded parts")
    return tok


def load_run(path, raw_read_note, engineering=False):
    """Read verified local inputs on the pod, using a memmap for large states.

    This does not access HF. The separate HF check precedes a real invocation.
    A provided receipt is recorded and checked against the local files; it is
    not presented as a new network verification.
    """
    run = Run()
    run.path = Path(path).resolve()
    note = Path(raw_read_note)
    if not note.is_file() or not note.read_text().strip():
        raise AnalysisError("a raw-read attestation note is required")
    required = {"manifest.json", "grid.json", "rows.jsonl", "reply_mean_acts.npy"}
    checks = {}
    for line in (run.path / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise AnalysisError("checksum path escapes the run")
        if name in checks or sha256(run.path / name) != h:
            raise AnalysisError(f"checksum mismatch or duplicate: {name}")
        checks[name] = h
    if not required <= checks.keys():
        raise AnalysisError("checksum coverage is incomplete")
    run.manifest = json.loads((run.path / "manifest.json").read_text())
    if (run.manifest.get("model"), run.manifest.get("revision")) != (MODEL, REVISION):
        raise AnalysisError("model/revision differs from approved A run")
    run.rows = [json.loads(l) for l in (run.path / "rows.jsonl").read_text().splitlines() if l.strip()]
    grid = json.loads((run.path / "grid.json").read_text())["keys"]
    keys = [row_key(r) for r in run.rows]
    if keys != grid or len(set(keys)) != len(keys):
        raise AnalysisError("rows do not match the unique ordered grid")
    run.roles = [r for r in run.manifest["roles"] if r != "default"]
    run.questions = sorted({r["question"] for r in run.rows})
    reps = run.manifest["replicates"]
    expected = {(c, q, w, r) for c in run.roles + ["default"] for q in run.questions for w in range(5) for r in range(reps)}
    actual = {(r["role"], r["question"], r["wording"], r["replicate"]) for r in run.rows}
    if actual != expected or any(r["format"] != "chat" for r in run.rows):
        raise AnalysisError("incomplete or changed role/question/wording/replicate grid")
    if any(type(r["valid"]) is not bool for r in run.rows):
        raise AnalysisError("validity flags must be boolean")
    if not engineering:
        approved_q = json.loads((HERE.parent / "261008_base_persona_map/questions_40.json").read_text())
        if isinstance(approved_q, dict):
            approved_q = approved_q.get("questions", approved_q.get("ids"))
        approved_roles = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
                          "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
                          "pirate", "vampire", "robot", "void"]
        if set(run.roles) != set(approved_roles) or run.questions != sorted(approved_q) or reps != 5:
            raise AnalysisError("A primary analysis requires the approved full grid")
        tok = verify_snapshot(run.manifest, grid)
        from manifold_persona import rows as R
        for row in run.rows:
            try:
                R.check_row(row, {tok.eos_token_id, tok.pad_token_id}, tok, run.manifest["markers"])
            except (ValueError, TypeError) as error:
                raise AnalysisError(f"fresh G1 row check failed: {error}") from error
    gates = json.loads((run.path / "gates.json").read_text())
    if not gates.get("passed") or any(gates.get("gates", {}).get(f"G{i}", {}).get("pass") is not True for i in range(1, 6)):
        raise AnalysisError("saved G1-G5 gates must all pass")
    valid = np.array([bool(r["valid"]) for r in run.rows])
    if gates["gates"]["G1"].get("n_rows") != len(run.rows) or gates["gates"]["G1"].get("invalid") != int((~valid).sum()):
        raise AnalysisError("saved G1 row counts differ")
    g3 = gates["gates"]["G3"]
    if g3.get("n_rows", 0) < 40 or g3.get("identical_same_batches") != g3["n_rows"]:
        raise AnalysisError("saved same-batch seed check is incomplete")
    for state in (9, 17, 23):
        cell = gates["gates"]["G4"].get("cells", {}).get(f"L{state}", {})
        if not cell.get("pass") or cell.get("n") != int(valid.sum()):
            raise AnalysisError("saved reconstruction coverage is incomplete")
        if not engineering and (cell.get("min_cos", -1) < .95 or cell.get("median_cos", -1) < .99
                                or cell.get("median_rel_norm", 1) > .02):
            raise AnalysisError("saved reconstruction tolerances fail")
    if not engineering and gates["gates"]["G5"].get("mismatch") != {}:
        raise AnalysisError("saved snapshot gate reports a mismatch")
    run.X = np.load(run.path / "reply_mean_acts.npy", mmap_mode="r", allow_pickle=False)
    if run.X.ndim != 3 or run.X.shape[:2] != (len(run.rows), 33) or run.X.dtype.kind != "f":
        raise AnalysisError("state array shape/dtype differs from model")
    if not engineering and run.X.shape[-1] != 4096:
        raise AnalysisError("model width differs from approved full run")
    for lo in range(0, len(run.rows), 256):
        a, v = run.X[lo:lo + 256], valid[lo:lo + 256]
        if not np.isfinite(a[v]).all() or not np.isnan(a[~v]).all():
            raise AnalysisError("valid states must be finite and invalid states NaN")
    for c in run.roles + ["default"]:
        m = np.array([r["role"] == c for r in run.rows])
        if valid[m].mean() < .95:
            raise AnalysisError(f"valid rate below G2 for {c}")
    for r in run.rows:
        n = len(r["kept_ids"])
        if r["span"] != [len(r["prompt_ids"]), len(r["prompt_ids"]) + n] or (r["valid"] and not n):
            raise AnalysisError("row token span or validity differs")
    receipt_path = run.path / "hf_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt is not None:
        if not receipt.get("verified") or not all(f.get("match") and f.get("sha256") == f.get("remote_sha256") for f in receipt["files"]):
            raise AnalysisError("provided HF receipt is not verified")
        for name, h in checks.items():
            matches = [f for f in receipt["files"] if f["path"].endswith("/" + name)]
            if len(matches) != 1 or matches[0]["sha256"] != h:
                raise AnalysisError("provided HF receipt differs from checksum inputs")
    run.provenance = dict(input_hashes=checks, gates_hash=sha256(run.path / "gates.json"),
                          raw_read_note_hash=sha256(note), model=MODEL, revision=REVISION,
                          hf_receipt_revision=receipt.get("revision") if receipt else None,
                          receipt_check="local metadata check only; no new HF network verification",
                          saved_gate_limit="Saved G3/G4 evidence is trusted only after input hashes and full coverage checks.")
    return run


def decode_features(meta, feature_fn, splits, kind="linear", cache=None, fingerprint=None, reader="states"):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    hit, count = np.zeros((meta["C"], meta["Q"])), np.zeros((meta["C"], meta["Q"]))
    accs = []
    for fold, (tr, te) in enumerate(splits):
        def one():
            Xtr, Xte = feature_fn(tr, te)
            if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
                raise AnalysisError("nonfinite decoder features")
            scaler = StandardScaler().fit(Xtr)
            model = (LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
                     if kind == "linear" else KNeighborsClassifier(n_neighbors=5))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    model.fit(scaler.transform(Xtr), meta["y"][tr])
                except ConvergenceWarning as error:
                    raise AnalysisError("decoder did not converge; no score is reported") from error
            return dict(pred=model.predict(scaler.transform(Xte)).tolist())
        saved = phase(cache, f"{reader}_{fold}", fingerprint, one) if cache else one()
        pred, y = np.asarray(saved["pred"]), meta["y"][te]
        if pred.shape != y.shape:
            raise AnalysisError("cached fold predictions have wrong shape")
        accs.append(float(np.mean([np.mean(pred[y == c] == c) for c in range(meta["C"])])))
        np.add.at(hit, (y, meta["q"][te]), pred == y)
        np.add.at(count, (y, meta["q"][te]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = hit / count
    return dict(mean_fold_accuracy=float(np.mean(accs)), cell=cell, counts=count,
                never_tested_cells=int((count == 0).sum()), classifier=kind)


def accuracy_draws(cell, rr, W):
    """No refits in the decoding bootstrap, as frozen for A."""
    cell = np.asarray(cell, float)
    if not np.isfinite(cell).all():
        return np.full(len(rr), np.nan)
    means = np.einsum("bq,cq->bc", W, cell) / W.sum(1)[:, None]
    return np.take_along_axis(means, rr, axis=1).mean(1)


def prompt_features(X, meta):
    return np.array([np.asarray(X[g], float).mean(0) for g in meta["rows"]])


def text_factory(rows, meta, params):
    """Fit vocabulary on training replies only, then average replicates per prompt."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["reply"] for r in rows]
    ii, jj, vv = [], [], []
    for j, group in enumerate(meta["rows"]):
        ii.extend([j] * len(group))
        jj.extend(group)
        vv.extend([1 / len(group)] * len(group))
    mean = sparse.csr_matrix((vv, (ii, jj)), shape=(len(meta["rows"]), len(rows)))

    def features(tr, te):
        train_rows = [i for j in np.flatnonzero(tr) for i in meta["rows"][j]]
        v = TfidfVectorizer(**params).fit([texts[i] for i in train_rows])
        P = mean @ v.transform(texts)
        return P[tr].toarray(), P[te].toarray()
    return features


def holm_decisions(ps, criteria, alpha=None):
    keys = list(ps)
    if any(ps[k] is None or not np.isfinite(ps[k]) for k in keys):
        return dict(status="missing_primary_p", adjusted_p=None, decisions=None, alpha=alpha)
    order = sorted(keys, key=lambda k: ps[k])
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, (len(keys) - i) * ps[key])
        adjusted[key] = min(1.0, float(running))
    if alpha is None:
        return dict(status="pending_owner_alpha", adjusted_p=adjusted, decisions=None, alpha=None)
    if not 0 < alpha < 1:
        raise AnalysisError("Holm alpha must be strictly between 0 and 1")
    return dict(status="specified_alpha", adjusted_p=adjusted,
                decisions={k: bool(criteria[k] and adjusted[k] < alpha) for k in keys}, alpha=alpha)


def save_npz(path, **arrays):
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def core_analysis(run, out, config, label="full5", replicates=None):
    X = run.X[:, STATE]
    d2 = map_statistics(X, run.rows, run.roles, run.questions, replicates=replicates, **config)
    S, N = cell_means(X, run.rows, run.roles + ["default"], run.questions, replicates)
    halves = make_halves(run.questions, n=config["n_halves"])
    rr, W = M.make_draws(len(run.roles), len(run.questions), n=config["n_boot"], seed=SEED)
    d3, details = axis_statistics(S, N, halves, rr, W)
    path = Path(out) / f"axis_{label}.npz"
    save_npz(path, **details, role_names=np.array(run.roles + ["default"]),
             question_ids=np.array(run.questions), half_questions=np.array([h[0][0] for h in halves]),
             half_wordings=np.array([h[0][1] for h in halves]), bootstrap_roles=rr, bootstrap_question_weights=W)
    control = dict(available=False, reason="assistant role absent")
    if "assistant" in run.roles:
        T, P = S.copy(), N.copy()
        i = run.roles.index("assistant")
        T[-1], P[-1] = S[i], N[i]
        ac, ad = axis_statistics(T, P, halves, rr, W)
        cos = np.sum(details["units"] * ad["units"], axis=-1)
        control = dict(available=True, contrast=ac, cosine_to_default_contrast=clean(float(cos.mean())),
                       scope="same contrast with assistant replacing default; report only")
    V = S[:-1].sum((1, 2)) / N[:-1].sum((1, 2))[:, None]
    V -= V.mean(0)
    eig = np.maximum(np.linalg.eigvalsh(V @ V.T)[::-1], 0)
    spectrum = dict(eigenvalues=eig.tolist(), max_centred_rank=len(run.roles) - 1,
                    components_for_70_percent=int(np.searchsorted(np.cumsum(eig) / eig.sum(), .7) + 1) if eig.sum() else None,
                    PC_end_interpretation=dict(available=False, reason="No numerical eigengap/subspace-stability pass rules are frozen."))
    return dict(D2=d2, D3=d3, assistant_control=control, spectrum=spectrum,
                artifact_hashes={path.name: sha256(path)}, replicates=replicates or list(range(run.manifest["replicates"])))


def replicate_reports(run, n_halves=N_HALVES):
    X = run.X[:, STATE]
    halves = make_halves(run.questions, n=n_halves)
    rr, W = M.make_draws(len(run.roles), len(run.questions), n=0)
    cells = [cell_means(X, run.rows, run.roles, run.questions, [r]) for r in range(run.manifest["replicates"])]
    sides = {str(r): M.build_sides(*v, halves) for r, v in enumerate(cells)}
    comparisons = [(str(a), str(b)) for a in range(len(cells)) for b in range(a, len(cells))]
    stats = map_panel(sides, rr, W, comparisons, strict=True)
    per_rep = {str(r): clean(float(stats[(str(r), str(r))]["r"][0])) for r in range(len(cells))}
    between = {f"{a}:{b}": clean(float(v["r"][0])) for (a, b), v in stats.items() if a != b}
    held = {}
    for r, (S, N) in enumerate(cells):
        others = [v for j, v in enumerate(cells) if j != r]
        count = sum(v[1] for v in others)
        pooled = sum(v[0] * v[1][..., None] for v in others) / np.maximum(count, 1)[..., None]
        ss = M.build_sides(pooled, (count > 0).astype(float), halves)
        st = map_panel({"train": ss, "test": sides[str(r)]}, rr, W, [("train", "test")], strict=True)
        held[str(r)] = dict(map_r=clean(float(st[("train", "test")]["r"][0])),
                            training_replicates=[j for j in range(len(cells)) if j != r], test_replicate=r)
    groups = prompt_index(run.rows, run.roles, run.questions)
    variances = []
    for g in groups["rows"]:
        if len(g) >= 2:
            variances.append(float(np.var(np.asarray(X[g], float), axis=0, ddof=1).sum()))
    return dict(per_replicate_map_r=per_rep, replicate_pair_map_r=between, replicate_held_out=held,
                within_prompt_sample_variance=dict(prompts_with_two_or_more=len(variances),
                                                   mean_trace=float(np.mean(variances)) if variances else None,
                                                   denominator="valid replicates minus one; equal prompt weights"),
                scope="reported sensitivity; no separate pass rule")


def noise_reports(run, n_halves=N_HALVES):
    """Separate wording and generation sensitivities; no extra decision rules.

    A held-out replicate decoder learns on the other replicates and predicts
    that replicate on disjoint question AND wording folds. Only common prompt
    keys enter that comparison; missing support is reported, not filled.
    """
    X = run.X[:, STATE]
    rr, W = M.make_draws(len(run.roles), len(run.questions), n=0)
    halves = make_halves(run.questions, n=n_halves)

    def subset(predicate):
        return [dict(row, valid=bool(row["valid"] and predicate(row))) for row in run.rows]

    def describe(rows):
        S, N = cell_means(X, rows, run.roles, run.questions)
        sides = M.build_sides(S, N, halves)
        stat = map_panel({"x": sides}, rr, W, [("x", "x")])[("x", "x")]
        Sa, Na = cell_means(X, rows, run.roles + ["default"], run.questions)
        axis, details = axis_statistics(Sa, Na, halves, rr, W)
        try:
            meta = prompt_index(rows, run.roles, run.questions)
            features = prompt_features(X, meta)
            decoded = decode_features(meta, lambda tr, te: (features[tr], features[te]),
                                      decode_splits(meta, n=n_halves))
            dec = dict(available=True, accuracy=decoded["mean_fold_accuracy"])
        except AnalysisError as error:
            dec = dict(available=False, reason=str(error))
        return dict(map_r=clean(stat["r"][0]), axis_cosine=axis["cosine"]["point"],
                    activation_decode=dec, missing_prompts=int((N == 0).sum())), details

    per_rep, held, wordings = {}, {}, {}
    for rep in range(run.manifest["replicates"]):
        test_rows = subset(lambda row, r=rep: row["replicate"] == r)
        train_rows = subset(lambda row, r=rep: row["replicate"] != r)
        per_rep[str(rep)], test_axis = describe(test_rows)
        held[str(rep)], train_axis = describe(train_rows)
        held[str(rep)]["axis_cosine_to_heldout_replicate"] = clean(
            float(np.sum(train_axis["units"] * test_axis["units"], axis=-1).mean()))
        try:
            train_meta, test_meta = [prompt_index(rows, run.roles, run.questions)
                                     for rows in (train_rows, test_rows)]
            def lookup(meta):
                return {(int(c), int(q), int(w)): group for c, q, w, group in
                        zip(meta["y"], meta["q"], meta["w"], meta["rows"])}
            train, test = lookup(train_meta), lookup(test_meta)
            keys = sorted(train.keys() & test.keys())
            meta = dict(y=np.array([k[0] for k in keys]), q=np.array([k[1] for k in keys]),
                        w=np.array([k[2] for k in keys]), C=len(run.roles), Q=len(run.questions))
            ftrain = np.array([np.asarray(X[train[k]], float).mean(0) for k in keys])
            ftest = np.array([np.asarray(X[test[k]], float).mean(0) for k in keys])
            decoded = decode_features(meta, lambda tr, te: (ftrain[tr], ftest[te]),
                                      decode_splits(meta, n=n_halves))
            held[str(rep)]["heldout_decode_accuracy"] = dict(available=True,
                accuracy=decoded["mean_fold_accuracy"], common_prompts=len(keys),
                missing_train_or_test_prompts=len(train.keys() | test.keys()) - len(keys))
        except AnalysisError as error:
            held[str(rep)]["heldout_decode_accuracy"] = dict(available=False, reason=str(error))
    for wording in range(5):
        wordings[str(wording)], _ = describe(subset(lambda row, w=wording: row["wording"] != w))
    return dict(per_replicate=per_rep, replicate_held_out=held, wording_held_out=wordings,
                scope="Descriptive point estimates on the fixed folds; no primary verdict or precision claim.")


def embedding_features(run, out, model_path):
    """Only an already-downloaded immutable snapshot may be used."""
    if model_path is None:
        return None, dict(available=False, reason="Provide the local pinned embedding model snapshot on the pod.")
    path = Path(model_path).resolve()
    if path.name != EMBED_MODEL[1] or not path.is_dir():
        raise AnalysisError("embedding model path must be the pinned snapshot directory")
    from sentence_transformers import SentenceTransformer
    text_hash = hashlib.sha256(json.dumps([r["reply"] for r in run.rows], ensure_ascii=False).encode()).hexdigest()
    cache, meta = Path(out) / "reply_embeddings.npy", Path(out) / "reply_embeddings.meta.json"
    expected = dict(model=EMBED_MODEL[0], revision=EMBED_MODEL[1], texts_sha256=text_hash)
    if cache.exists() or meta.exists():
        if not cache.exists() or not meta.exists():
            raise AnalysisError("incomplete embedding cache")
        m = json.loads(meta.read_text())
        if any(m.get(k) != v for k, v in expected.items()) or m.get("array_sha256") != sha256(cache):
            raise AnalysisError("embedding cache provenance differs")
        X = np.load(cache, mmap_mode="r", allow_pickle=False)
    else:
        model = SentenceTransformer(str(path), local_files_only=True, device="cpu")
        X = model.encode([r["reply"] for r in run.rows], batch_size=128, normalize_embeddings=True,
                         show_progress_bar=True).astype(np.float32)
        with cache.with_suffix(".tmp").open("wb") as f:
            np.save(f, X)
        os.replace(cache.with_suffix(".tmp"), cache)
        write_json(meta, dict(expected, array_sha256=sha256(cache)))
    if X.shape[0] != len(run.rows) or X.ndim != 2 or not np.isfinite(X).all():
        raise AnalysisError("embedding rows or values differ")
    return X, dict(expected, available=True, array_sha256=sha256(cache))


def decoding_reports(run, out, config, fingerprint, model_path=None):
    P = prompt_index(run.rows, run.roles, run.questions)
    splits = decode_splits(P, n=config["n_halves"])
    rr, W = M.make_draws(len(run.roles), len(run.questions), n=config["n_boot"], seed=SEED)
    save_npz(Path(out) / "decoder_resampling.npz", roles=rr, question_weights=W,
             train=np.array([x[0] for x in splits]), test=np.array([x[1] for x in splits]))
    raw_features = {f"state_{k}": prompt_features(run.X[:, k], P) for k in (STATE, 0) + OTHER_STATES}
    features = {k: (lambda tr, te, x=X: (x[tr], x[te])) for k, X in raw_features.items()}
    features.update({"text_" + name: text_factory(run.rows, P, params) for name, params in TEXT.items()})
    emb, emb_info = embedding_features(run, out, model_path)
    if emb is not None:
        E = prompt_features(emb, P)
        features["text_embed"] = lambda tr, te: (E[tr], E[te])
    reports, draws = {}, {}
    for name, fn in features.items():
        result = decode_features(P, fn, splits, cache=Path(out) / "folds", fingerprint=fingerprint, reader=name)
        values = accuracy_draws(result["cell"], rr, W)
        draws[name] = values
        reports[name] = dict(accuracy=summary(values), mean_fold_accuracy=result["mean_fold_accuracy"],
                             never_tested_cells=result["never_tested_cells"], classifier="E3 balanced logistic C=1; train-fold scaler")
    knn = decode_features(P, features[f"state_{STATE}"], splits, kind="knn", cache=Path(out) / "folds",
                          fingerprint=fingerprint, reader="knn_state17")
    reports["knn_state17"] = dict(accuracy=summary(accuracy_draws(knn["cell"], rr, W)),
                                 mean_fold_accuracy=knn["mean_fold_accuracy"], method="train-fold scaler, k=5; exploratory")
    baseline_names = ["text_word", "text_char", "state_0"] + (["text_embed"] if emb is not None else [])
    best = np.max([draws[k] for k in baseline_names], axis=0)
    gain = summary(draws[f"state_{STATE}"] - best)
    return dict(readers=reports, activation_minus_best_text=gain, embedding=emb_info,
                complete_text_baselines=emb is not None,
                matched_rows="valid replicate means of the same prompt; identical folds for every reader",
                interval_scope="crossed role/question resampling of cached test predictions; no refits")


def _naming_module():
    name = "_poda_naming_analysis"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "analysis_naming.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute-on-pod", action="store_true")
    parser.add_argument("--raw-read-note", type=Path)
    parser.add_argument("--engineering", action="store_true", help="Allow synthetic small grids/reduced counts; no confirmatory decisions")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-halves", type=int, default=N_HALVES)
    parser.add_argument("--naming", type=Path, help="Supplied approved judge outputs only; never invokes a judge")
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--skip-decoding", action="store_true", help="Explicit partial report; missing readers are recorded")
    parser.add_argument("--holm-alpha", type=float, help="Owner-specified family level; omitted means no family verdict")
    args = parser.parse_args(argv)
    if args.synthetic_only:
        result = synthetic_gate()
        print(json.dumps(clean(result), allow_nan=False))
        return 0 if result["passed"] else 3
    if any(getattr(args, k) is None for k in ("run", "out", "raw_read_note")) or not args.execute_on_pod:
        parser.error("real analysis requires --run, --out, --raw-read-note and --execute-on-pod")
    if sys.platform != "linux" and not args.engineering:
        parser.error("heavy analysis must run on a pod; use --synthetic-only on this Mac")
    if sys.platform != "linux" and args.engineering:
        header = np.load(args.run / "reply_mean_acts.npy", mmap_mode="r", allow_pickle=False)
        if header.shape[0] > 2000 or header.shape[-1] > 64:
            parser.error("Mac engineering mode accepts small synthetic arrays only")
    if min(args.n_boot, args.n_perm, args.n_halves) < 1:
        parser.error("resampling counts must be positive")
    frozen = confirmatory_settings(args.n_boot, args.n_perm, args.n_halves) and not args.engineering
    if not frozen and not args.engineering:
        parser.error("reduced counts require --engineering")
    run = load_run(args.run, args.raw_read_note, engineering=args.engineering)
    args.out.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), MAPS_PATH, HERE / "analysis_naming.py", HERE / "THRESHOLDS.md",
                  HERE / "gen_extract.py", HERE.parents[2] / "src/manifold_persona/rows.py"]
    naming = _naming_module()
    code_paths += [naming.CANDIDATES_PATH, naming.PROMPT_PATH, naming.QUESTIONS_PATH]
    config = dict(n_boot=args.n_boot, n_perm=args.n_perm, n_halves=args.n_halves)
    import importlib.metadata
    import platform
    import subprocess
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "sentence-transformers", "joblib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
    provenance = dict(run.provenance, code={str(p): sha256(p) for p in code_paths}, settings=clean(vars(args)),
                      versions=packages, python=platform.python_version(), machine=platform.machine(),
                      git_head=head.stdout.strip() if head.returncode == 0 else None,
                      code_state="Uncommitted analysis and dependency bytes are recorded by the code hashes.",
                      confirmatory_settings=frozen, analysis_scope="context-conditioned reply states; no persona identity claim")
    if args.naming:
        provenance["naming_sha256"] = sha256(args.naming)
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()
    manifest = args.out / "analysis_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise AnalysisError("analysis output fingerprint differs; use a separate output folder")
    write_json(manifest, dict(fingerprint=fingerprint, provenance=provenance))
    g6 = phase(args.out, "G6", fingerprint, synthetic_gate)
    if not g6["passed"]:
        raise AnalysisError("G6 synthetic calibration failed; no real metric computed")
    result = dict(G6=g6, confirmatory_settings=frozen, provenance=provenance)
    result["primary"] = phase(args.out, "primary", fingerprint, lambda: core_analysis(run, args.out, config))
    result["replicates_0_2"] = phase(args.out, "replicates_0_2", fingerprint,
                                       lambda: core_analysis(run, args.out, config, "replicates_0_2", [0, 1, 2]))
    result["replicate_sensitivities"] = phase(args.out, "replicate_sensitivities", fingerprint,
                                                 lambda: replicate_reports(run, args.n_halves))
    result["noise_sensitivities"] = phase(args.out, "noise_sensitivities", fingerprint,
                                            lambda: noise_reports(run, args.n_halves))
    result["exploratory_layers"] = {}
    for state in OTHER_STATES:
        result["exploratory_layers"][str(state)] = phase(args.out, f"map_state{state}", fingerprint,
            lambda s=state: map_statistics(run.X[:, s], run.rows, run.roles, run.questions, **config))
    if args.skip_decoding:
        result["decoding"] = dict(available=False, reason="explicit --skip-decoding; report incomplete")
    else:
        result["decoding"] = phase(args.out, "decoding", fingerprint,
            lambda: decoding_reports(run, args.out, config, fingerprint, args.embedding_model_path))
    if args.naming:
        naming = _naming_module()
        result["D1"] = phase(args.out, "D1_naming", fingerprint, lambda: naming.score_naming(
            run.rows, run.roles, run.questions, naming.load_payload(args.naming),
            n_boot=args.n_boot, n_perm=args.n_perm, seed=SEED))
    else:
        result["D1"] = dict(available=False, reason="No approved judge panel/output supplied; no judge call made", p=None)
    d2, d3, d1 = result["primary"]["D2"], result["primary"]["D3"], result["D1"]
    result["family"] = holm_decisions({"D1": d1.get("p"), "D2": d2["p"]},
        {"D1": bool(d1.get("passed")), "D2": bool(d2["passed"])}, args.holm_alpha)
    if not frozen:
        result["family"]["decisions"] = None
        result["family"]["status"] = "engineering_only"
    result["fidelity"] = dict(available=False, reason="Requires the separately validated fidelity rubric and approved labels")
    result["D3_axis_gate"] = dict(passed=d3["passed"] if frozen else None,
                                      reason="precision and repeatability gate; not part of Holm")
    result["limits"] = ["Wordings and observed replicate means stay fixed within intervals.",
                        "No automatic additional samples follow an unmet precision target.",
                        "Holm family alpha is not frozen; no family verdict is invented.",
                        "PC-end interpretation has no approved numerical stability criterion.",
                        "Missing D1, embedding reader or fidelity labels remain explicit."]
    write_json(args.out / "results.json", result)
    print(json.dumps(dict(out=str(args.out), confirmatory_settings=frozen, family=result["family"]["status"])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"analysis refused: {error}", file=sys.stderr)
        raise SystemExit(3)
