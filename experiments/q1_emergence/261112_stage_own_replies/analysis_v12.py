"""Own-reply analysis under THRESHOLDS.md v1.2 (owner approved 13/11/2026): the default's position and spread on each
stage's own replies (SFT, DPO, Instruct pod A), on full replies and on the first 16 kept tokens, with a text baseline.

Statistics: ../261112_free_checks/check_role_count.py (imported). Map gate: pod A's map functions
(../261029_instruct_persona_map/analysis.py). Lu scale: ../261112_free_checks/check3_lu_scale.py. Han's axis:
his persona_axes.py. Runs on Modal CPU through modal_analysis_v12.py; ROOT is the outputs volume.

Steps:
  prep     input checks, common first16 mask and support, text features, reference profile, shared draws
  source   one (stage, view, feature): points, draws, gates, planted shifts
  power    P1-P3 from the sources; prints no real estimate
  measure  the readings; refuses unless the power check passed

Revision b (13/11/2026, after review/codex_analysis_v12_261113.md; same rules, seed and draws): each stage uses its
own valid flags; Lu's scale uses the same row-weighted means as the position statistic; power.json holds no absolute
response; P3 gives both directions; sources carry a run fingerprint and an array hash; non-finite draws are counted;
report-only comparisons carry their gates; Han's change is saved for all 20 roles; the end test is strict.
The reviewed code is in review/code_before_fixes_261113/.
"""
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(Q / "261112_free_checks"))
CR = load_module("check_role_count", Q / "261112_free_checks/check_role_count.py")
LU = load_module("check3_lu_scale", Q / "261112_free_checks/check3_lu_scale.py")
PA = load_module("poda_analysis", Q / "261029_instruct_persona_map/analysis.py")
HAN_AXES = Q / "261112_han_reanalysis/data/han_repo/src/persona_shift/persona_axes.py"

STAGES = ("sft", "dpo", "instruct")
PAIRS = (("sft", "dpo"), ("dpo", "instruct"))
REPORT_PAIRS = (("sft", "instruct"),)
VIEWS = ("full", "first16")
STATE = 17
FEATURES = ("s17", "s9", "s23", "text")
SOURCE_VIEWS = {"full": FEATURES, "first16": FEATURES, "full_m16": ("s17", "text")}
CUT = 16
N_DRAWS, N_SPLITS, N_MAP_DRAWS = 4000, 50, 1000
SEED = 2026111312
SHIFTS = (0.5, 1.0, 2.0)
SCALES = (0.67, 1.5)
P3_GRID = {"position": (0.25, 0.5, 0.75, 1.0, 1.5), "spread": (0.05, 0.10, 0.15, 0.20, 0.30)}
CONF, REPORT_CONF, P3_RATE = 99.5, 95.0, 0.8
END = 2.0
D2_FLOOR, PROFILE_FLOOR, MAX_DROPPED = 0.5, 0.5, 0.01
DESIRABLE = ("assistant", "translator", "counselor", "consultant", "stoic")
UNDESIRABLE = ("criminal", "cynic", "narcissist", "pirate", "vampire")
BORDERLINE = ("stoic", "cynic")
TEXT = dict(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
TEXT_DIMS = 256
RUN_DIRS = {"instruct": "poda_dl/experiments/q1_emergence/261029_instruct_persona_map/run_full",
            "sft": "own_replies/sft_full", "dpo": "own_replies/dpo_full"}
PODA_SUMS_SHA256 = "b079b00d16228f79be18f73c138dfe1e42ccc050b62759f9db3b91015b814db3"
CODE = {"analysis_v12.py": Path(__file__).resolve(), "check_role_count.py": Path(CR.__file__),
        "check3_lu_scale.py": Path(LU.__file__), "poda_analysis.py": Path(PA.__file__),
        "maps.py": Path(PA.MAPS_PATH), "persona_axes.py": HAN_AXES}


class InputError(RuntimeError):
    pass


# ---------- small helpers ----------

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes():
    return {k: sha256(p) for k, p in CODE.items()}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def row_key(r):
    return json.dumps([r["role"], r["format"], r["wording"], r["question"], r["replicate"]], separators=(",", ":"))


def run_dir(root, stage):
    return Path(root) / RUN_DIRS[stage]


def lc_dir(root, stage):
    return Path(root) / "length_control/full" / stage


def source_name(stage, view, feature):
    return f"{stage}_{view}_{feature}"


def all_sources():
    return [(s, v, f) for s in STAGES for v, fs in SOURCE_VIEWS.items() for f in fs]


def interval(v, conf):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [(100 - conf) / 2, 100 - (100 - conf) / 2])
    return [float(lo), float(hi)]


def fixed_splits(nq):
    """check_role_count's 50 fixed question splits (same seeds), as weight vectors."""
    out = []
    for i in range(N_SPLITS):
        perm = np.random.default_rng(CR.SEED + 1 + i).permutation(nq)
        out.append((np.isin(np.arange(nq), perm[: nq // 2]).astype(float),
                    np.isin(np.arange(nq), perm[nq // 2:]).astype(float)))
    return out


def make_draws(n_roles, nq, n=None, seed=SEED):
    """Crossed draws shared by every source: roles with replacement, random question halves, questions resampled
    within each half (check_role_count.question_weights)."""
    rng = np.random.default_rng(seed)
    idx, wa, wb = [], [], []
    for _ in range(N_DRAWS if n is None else n):
        idx.append(rng.choice(n_roles, n_roles, replace=True))
        a, b = CR.question_weights(rng, nq)
        wa.append(a)
        wb.append(b)
    return dict(role_idx=np.array(idx), wa=np.array(wa), wb=np.array(wb))


# ---------- inputs ----------

def receipt_hashes(paths):
    out = {}
    for p in paths:
        for f in json.loads(Path(p).read_text())["files"]:
            out[f["volume_path"]] = f["sha256"]
    return out


def input_files(root):
    files = []
    for s in STAGES:
        files += [run_dir(root, s) / n for n in ("rows.jsonl", "reply_mean_acts.npy", "manifest.json")]
        files += [lc_dir(root, s) / n for n in ("first16_acts.npy", "keys.json", "support.json", "manifest.json")]
    return files


def check_inputs(root, receipts, volume_root="/out"):
    """SHA-256 of every input against its HF receipt; pod A against its SHA256SUMS.txt, whose own hash is pinned."""
    expected = {}
    pa_dir = run_dir(root, "instruct")
    if sha256(pa_dir / "SHA256SUMS.txt") != PODA_SUMS_SHA256:
        raise InputError("pod A SHA256SUMS.txt differs from its HF receipt")
    for line in (pa_dir / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        expected[str(pa_dir / name)] = h
    for vp, h in receipt_hashes(receipts).items():
        expected[str(Path(root) / Path(vp).relative_to(volume_root))] = h
    files = input_files(root)
    missing = [str(f) for f in files if str(f) not in expected]
    if missing:
        raise InputError(f"no receipt hash for {missing}")
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(map(str, files), ex.map(sha256, files)))
    bad = [f for f, h in got.items() if h != expected[f]]
    if bad:
        raise InputError(f"SHA-256 differs from the receipt: {bad}")
    return got


def load_rows(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append(dict(role=r["role"], format=r["format"], wording=r["wording"], question=r["question"],
                        replicate=r["replicate"], valid=bool(r["valid"]), n_kept=len(r["kept_ids"]),
                        first_ids=r["kept_ids"][:CUT], reply=r["reply"]))
    return out


def check_keys(rows, keys_json):
    """The full row key, in order, is the same in the three stages and in each length-control keys.json."""
    keys = {s: [row_key(r) for r in rows[s]] for s in STAGES}
    ref = keys[STAGES[0]]
    if len(set(ref)) != len(ref):
        raise InputError("duplicate row keys")
    for s in STAGES:
        if keys[s] != ref:
            raise InputError(f"{s}: row keys differ from {STAGES[0]}")
        kj = keys_json[s]
        if kj["row_index"] != list(range(len(ref))) or kj["keys"] != ref:
            raise InputError(f"{s}: keys.json differs from rows.jsonl")
    return len(ref)


def finite_rows(A, chunk=2000):
    """Rows of a [n, states, dim] array that are finite; a partly finite row is an error."""
    fin = np.zeros(A.shape[0], bool)
    for i in range(0, A.shape[0], chunk):
        a = np.asarray(A[i:i + chunk], np.float32).reshape(min(chunk, A.shape[0] - i), -1)
        all_f, any_f = np.isfinite(a).all(1), np.isfinite(a).any(1)
        if (all_f != any_f).any():
            raise InputError("a first16 row is partly finite")
        fin[i:i + chunk] = all_f
    return fin


def check_first16(rows, fin, support):
    """Codex length review, finding 1: the finite rows must be exactly the eligible rows (valid, >= 16 kept tokens),
    and their count and role shares must equal support.json."""
    eligible = np.array([r["valid"] and r["n_kept"] >= CUT for r in rows])
    if not (fin == eligible).all():
        raise InputError(f"first16 finite rows differ from eligible rows at {int((fin != eligible).sum())} rows")
    if int(fin.sum()) != support["n_finite_rows"]:
        raise InputError("first16 finite count differs from support.json")
    roles = np.array([r["role"] for r in rows])
    for role, share in support["share"].items():
        if abs(float(fin[roles == role].mean()) - share) > 0.0005 + 1e-9:
            raise InputError(f"first16 share of {role} differs from support.json")


def support_check(meta, mask):
    """Every cloud keeps rows in every wording, and in each half of the 50 fixed question splits."""
    clouds = sorted({r["role"] for r in meta} - {CR.DEFAULT}) + [CR.DEFAULT]
    qs = sorted({r["question"] for r in meta})
    ws = sorted({r["wording"] for r in meta})
    cnt = np.zeros((len(clouds), len(qs), len(ws)), int)
    ci, qi, wi = ({v: i for i, v in enumerate(a)} for a in (clouds, qs, ws))
    for r, m in zip(meta, mask):
        if m:
            cnt[ci[r["role"]], qi[r["question"]], wi[r["wording"]]] += 1
    by_wording = cnt.sum(1)
    fails = [f"{clouds[c]} wording {ws[w]}" for c, w in zip(*np.where(by_wording == 0))]
    for i, (a, b) in enumerate(fixed_splits(len(qs))):
        for name, h in (("a", a), ("b", b)):
            per = (cnt.sum(2) * h).sum(1)
            fails += [f"{clouds[c]} split {i} half {name}" for c in np.where(per == 0)[0]]
    return dict(passed=not fails, failures=fails[:20], n_failures=len(fails),
                min_rows_per_cloud_wording=int(by_wording.min()),
                rows_per_cloud={c: int(cnt[i].sum()) for i, c in enumerate(clouds)})


def fit_text(texts_by_stage):
    """Character 3-5-gram TF-IDF and a 256-d truncated SVD, fitted once on the texts of the three stages together."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    sizes = [len(t) for t in texts_by_stage]
    tf = TfidfVectorizer(**TEXT).fit_transform([t for ts in texts_by_stage for t in ts])
    Z = TruncatedSVD(TEXT_DIMS, random_state=0).fit_transform(tf).astype(np.float64)
    return np.split(Z, np.cumsum(sizes)[:-1]), list(tf.shape)


# ---------- statistics on one source ----------

def role_pc1(cs, n_roles):
    ma = CR.means(cs, np.ones(cs["S1"].shape[1]))[:n_roles]
    c = ma - ma.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0], c @ vt[0], float(s[0] ** 2 / (s ** 2).sum())


def orient(u, prof, clouds, ref=None):
    """Instruct full state 17 (ref None): sign so the assistant role projects above the role mean. Every other
    source: sign so its 20 role projections correlate positively with that Instruct profile."""
    if ref is None:
        r = 1.0
        if prof[clouds.index("assistant")] < 0:
            u, prof = -u, -prof
    else:
        r = float(np.corrcoef(prof, ref)[0, 1])
        if r < 0:
            u, prof, r = -u, -prof, -r
    return u, prof, r


def shift_default(cs, v):
    """check_role_count's planted shift: v added to every default row, through the sums."""
    d = len(cs["clouds"]) - 1
    S1, S2 = cs["S1"].copy(), cs["S2"].copy()
    S2[d] = cs["S2"][d] + 2 * np.einsum("qwd,d->qw", cs["S1"][d], v) + cs["N"][d] * (v @ v)
    S1[d] = cs["S1"][d] + cs["N"][d][..., None] * v
    return dict(cs, S1=S1, S2=S2, S1q=S1.sum(2), Nq=cs["N"].sum(2))


def scale_default(cs, k):
    """Each default row's distance to its cloud-and-wording mean (over all questions) times sqrt(k)."""
    d = len(cs["clouds"]) - 1
    s = np.sqrt(k)
    N, S1d, S2d = cs["N"][d], cs["S1"][d], cs["S2"][d]
    m = S1d.sum(0) / np.maximum(N.sum(0), 1)[:, None]
    S1, S2 = cs["S1"].copy(), cs["S2"].copy()
    S1[d] = s * S1d + (1 - s) * N[..., None] * m[None]
    S2[d] = s * s * S2d + 2 * s * (1 - s) * np.einsum("qwd,wd->qw", S1d, m) + (1 - s) ** 2 * N * (m * m).sum(-1)[None]
    return dict(cs, S1=S1, S2=S2, S1q=S1.sum(2), Nq=cs["N"].sum(2))


def spread_log_ratio(cs, w, role_idx=None):
    v = CR.spread(cs, w)
    ref = v[:-1] if role_idx is None else v[role_idx]
    return float(np.log(v[-1] / np.median(ref)))


def points(cs, u, splits, n_roles):
    roles = np.arange(n_roles)
    full = np.ones(cs["S1"].shape[1])
    return dict(position=float(np.mean([CR.position(cs, a, b, roles, u) for a, b in splits])),
                position_abs=float(np.mean([abs(CR.position(cs, a, b, roles)) for a, b in splits])),
                spread=spread_log_ratio(cs, full))


def split_pc1_cosine(cs, splits, n_roles):
    out = []
    for a, b in splits:
        us = []
        for w in (a, b):
            m = CR.means(cs, w)[:n_roles]
            us.append(np.linalg.svd(m - m.mean(0), full_matrices=False)[2][0])
        out.append(abs(float(us[0] @ us[1])))
    return float(np.mean(out))


def draw_stats(cs, u, draws):
    n = len(draws["role_idx"])
    pos, spr = np.full(n, np.nan), np.full(n, np.nan)
    Nq = cs["Nq"]
    empty = 0
    for k in range(n):
        idx, wa, wb = draws["role_idx"][k], draws["wa"][k], draws["wb"][k]
        if ((wa @ Nq.T) == 0).any() or ((wb @ Nq.T) == 0).any():
            empty += 1
            continue
        pos[k] = CR.position(cs, wa, wb, idx, u)
        spr[k] = spread_log_ratio(cs, wa + wb, idx)
    nonfinite = int((~(np.isfinite(pos) & np.isfinite(spr))).sum()) - empty
    return pos, spr, empty, nonfinite


def planted(cs, u, prof, splits, n_roles, base):
    sd = float(np.std(prof, ddof=1))
    out = dict(role_sd_full=sd, shifts={}, scales={})
    for delta in SHIFTS:
        p = points(shift_default(cs, delta * sd * u), u, splits, n_roles)
        out["shifts"][str(delta)] = dict(moved_signed=p["position"] - base["position"],
                                         moved_abs=p["position_abs"] - base["position_abs"],
                                         spread_moved=p["spread"] - base["spread"])
    for k in SCALES:
        p = spread_log_ratio(scale_default(cs, k), np.ones(cs["S1"].shape[1]))
        out["scales"][str(k)] = dict(spread_moved=p - base["spread"], target=float(np.log(k)))
    return out


def map_gate(X, meta, mask, roles, questions):
    """Pod A's map rule (D2): fixed question-and-wording halves, role-role cosine-distance maps, raw Spearman r over
    distinct sampled role pairs, mean over 50 halves; 95% interval from 1,000 draws (pod A's strict summary)."""
    rows = [dict(r, valid=bool(m and r["valid"])) for r, m in zip(meta, mask)]
    S, N = PA.cell_means(X, rows, roles, questions)
    halves = PA.make_halves(questions, n=N_SPLITS)
    sides = PA.M.build_sides(S, N, halves)
    rr, W = PA.M.make_draws(len(roles), len(questions), n=N_MAP_DRAWS, seed=PA.SEED)
    st = PA.map_panel({"a": sides}, rr, W, [("a", "a")], strict=True)[("a", "a")]
    s = PA.summary(st["r"])
    return dict(raw_r=s["point"], lo=s["lo"], hi=s["hi"], n_nonfinite=s["n_nonfinite"],
                undefined_halves=int(np.sum(st["undefined_halves"])),
                passed=bool(s["lo"] is not None and s["lo"] >= D2_FLOOR))


def lu_scale(cs, ref, draws):
    """Report only: Lu's 0-1 scale on the roles-only PC1 of the same row-weighted cloud means as the position
    statistic (CR.means), through check3_lu_scale.lu_position. Signed to the Instruct profile, so 1 is the end the
    assistant role leans to. 95% interval: the shared draws' question weights, roles fixed."""
    nq = cs["S1"].shape[1]

    def one(w):
        return LU.lu_position(CR.means(cs, w)[:, None, :], np.ones(1))

    dist, outside = one(np.ones(nq))
    M = CR.means(cs, np.ones(nq))
    roles, d = M[:-1], M[-1]
    c = roles.mean(0)
    u = np.linalg.svd(roles - c, full_matrices=False)[2][0]
    pr, pd = (roles - c) @ u, (d - c) @ u
    if np.corrcoef(pr, ref)[0, 1] < 0:
        pr, pd = -pr, -pd
    x = float((pd - pr.min()) / (pr.max() - pr.min()))
    W = draws["wa"] + draws["wb"]
    dd = np.full(len(W), np.nan)
    for k, w in enumerate(W):
        if ((w @ cs["Nq"].T) > 0).all():
            dd[k] = one(w)[0]
    return dict(available=True, scaled_position=x, distance_to_nearer_end=dist, outside_role_range=outside,
                distance_ci95=interval(dd, REPORT_CONF), n_draws=int(np.isfinite(dd).sum()),
                weighting="rows equal within the view (CR.means), as the position statistic"), dd


def source_stats(X, meta, mask, ref, draws, map_check=True):
    """All statistics of one source. ref None marks the Instruct full state-17 source that defines the sign."""
    cs = CR.cell_sums(X, meta, mask)
    clouds = cs["clouds"]
    n_roles = len(clouds) - 1
    nq = cs["S1"].shape[1]
    splits = fixed_splits(nq)
    u, prof, share = role_pc1(cs, n_roles)
    u, prof, r = orient(u, prof, clouds, ref)
    base = points(cs, u, splits, n_roles)
    pos, spr, empty, nonfinite = draw_stats(cs, u, draws)
    out = dict(clouds=clouds, n_rows=int(mask.sum()), rows_per_cloud={c: int(cs["N"][i].sum()) for i, c in enumerate(clouds)},
               point=base, profile_corr=r, pc1_share=share, split_pc1_cosine=split_pc1_cosine(cs, splits, n_roles),
               dropped_draws=empty, nonfinite_draws=nonfinite, dropped_share=(empty + nonfinite) / len(pos),
               planted=planted(cs, u, prof, splits, n_roles, base))
    out["lu"], lu_draws = lu_scale(cs, prof if ref is None else ref, draws)
    if map_check:
        out["map_gate"] = map_gate(X, meta, mask, clouds[:-1], sorted({r["question"] for r in meta}))
    arrays = dict(position=pos, spread=spr, profile=prof, cloud_means=CR.means(cs, np.ones(nq)), lu_distance=lu_draws)
    return out, arrays


# ---------- steps ----------

ARTIFACTS = ("mask_first16.npy", "valid.npy", "meta.json", "text_full.npy", "text_first16.npy",
             "reference_profile.npy", "draws.npz")


def fingerprint(rec):
    """One hash over what a result depends on: code, inputs, prep artifacts, runtime, seed and draw counts."""
    keys = ("code", "inputs_sha256", "artifacts_sha256", "runtime", "seed", "n_draws", "n_map_draws")
    return hashlib.sha256(json.dumps({k: rec.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def prep(root, out, receipts, tokenizers=None, volume_root="/out", runtime=None):
    """tokenizers: {stage: tokenizer} for the first16 text; None decodes nothing (tests). runtime: package versions
    and tokenizer pins, recorded and bound into the fingerprint."""
    t0 = time.time()
    out = Path(out)
    if (out / "prep.json").exists():
        raise InputError(f"{out}/prep.json exists; prep runs once")
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(rules="THRESHOLDS.md v1.2, approved 13/11/2026", code_revision="b", code=code_hashes(), seed=SEED,
               n_draws=N_DRAWS, n_map_draws=N_MAP_DRAWS, runtime=runtime)
    rec["inputs_sha256"] = check_inputs(root, receipts, volume_root)
    rows = {s: load_rows(run_dir(root, s) / "rows.jsonl") for s in STAGES}
    n = check_keys(rows, {s: json.loads((lc_dir(root, s) / "keys.json").read_text()) for s in STAGES})
    for s in STAGES:
        man = json.loads((run_dir(root, s) / "manifest.json").read_text())
        if man.get("acts_shape", [n, 33, 4096])[:2] != [n, 33]:
            raise InputError(f"{s}: reply_mean_acts is not [rows, 33 states, dim]")
    fin = {}
    for s in STAGES:
        A = np.load(lc_dir(root, s) / "first16_acts.npy", mmap_mode="r")
        fin[s] = finite_rows(A)
        check_first16(rows[s], fin[s], json.loads((lc_dir(root, s) / "support.json").read_text())["first16"])
    valid = np.array([[r["valid"] for r in rows[s]] for s in STAGES])
    m16 = np.logical_and.reduce([fin[s] for s in STAGES])
    meta = [{k: r[k] for k in ("role", "format", "wording", "question", "replicate")} for r in rows[STAGES[0]]]
    rec["n_rows"] = n
    rec["n_valid"] = {s: int(valid[j].sum()) for j, s in enumerate(STAGES)}
    rec["n_valid_all_stages"] = int(valid.all(0).sum())
    np.save(out / "valid.npy", valid)
    rec["first16"] = dict(n_finite={s: int(fin[s].sum()) for s in STAGES}, n_common=int(m16.sum()),
                          support=support_check(meta, m16))
    np.save(out / "mask_first16.npy", m16)
    write_json(out / "meta.json", meta)

    texts_full = [[r["reply"] for r in rows[s]] for s in STAGES]
    Z, shape = fit_text(texts_full)
    np.save(out / "text_full.npy", np.stack(Z))
    rec["text"] = dict(params={**TEXT, "ngram_range": list(TEXT["ngram_range"])}, dims=TEXT_DIMS, full_tfidf_shape=shape)
    if tokenizers is not None:
        keep = np.where(m16)[0]
        texts16 = [[tokenizers[s].decode(rows[s][i]["first_ids"], skip_special_tokens=False) for i in keep]
                   for s in STAGES]
        Z16, shape16 = fit_text(texts16)
        T16 = np.full((len(STAGES), n, TEXT_DIMS), np.nan)
        for j in range(len(STAGES)):
            T16[j, keep] = Z16[j]
        np.save(out / "text_first16.npy", T16)
        rec["text"]["first16_tfidf_shape"] = shape16
        rec["text"]["first16_decode"] = "tok.decode(kept_ids[:16], skip_special_tokens=False), as rows.reply"

    X = load_features(root, out, "instruct", "full", "s17")
    cs = CR.cell_sums(X, meta, valid[STAGES.index("instruct")])
    u, prof, _ = role_pc1(cs, len(cs["clouds"]) - 1)
    u, prof, _ = orient(u, prof, cs["clouds"], None)
    np.save(out / "reference_profile.npy", prof)
    rec["reference"] = dict(source="instruct full state 17", roles=cs["clouds"][:-1],
                            sign_rule="assistant projects above the role mean")
    draws = make_draws(len(cs["clouds"]) - 1, cs["S1"].shape[1])
    np.savez(out / "draws.npz", **draws)
    rec["artifacts_sha256"] = {a: sha256(out / a) for a in ARTIFACTS if (out / a).exists()}
    rec["fingerprint"] = fingerprint(rec)
    rec["seconds"] = round(time.time() - t0, 1)
    write_json(out / "prep.json", rec)
    return dict(n_rows=n, n_valid_all_stages=rec["n_valid_all_stages"], first16=rec["first16"]["n_finite"],
                n_common=rec["first16"]["n_common"], support_passed=rec["first16"]["support"]["passed"],
                support_failures=rec["first16"]["support"]["failures"], text=rec["text"], seconds=rec["seconds"])


def load_features(root, out, stage, view, feature):
    if feature == "text":
        T = np.load(Path(out) / ("text_first16.npy" if view == "first16" else "text_full.npy"), mmap_mode="r")
        return np.asarray(T[STAGES.index(stage)], np.float64)
    s = int(feature[1:])
    if view == "first16":
        man = json.loads((lc_dir(root, stage) / "manifest.json").read_text())
        A = np.load(lc_dir(root, stage) / "first16_acts.npy", mmap_mode="r")
        return np.asarray(A[:, man["states"].index(s), :], np.float64)
    A = np.load(run_dir(root, stage) / "reply_mean_acts.npy", mmap_mode="r")
    if A.shape[1] != 33:
        raise InputError("reply_mean_acts must hold 33 hidden states (index 0 = embeddings)")
    return np.asarray(A[:, s, :], np.float64)


def read_prep(out, artifacts=()):
    """prep.json, checked: same code, a fingerprint that matches its content, and unchanged artifacts."""
    out = Path(out)
    rec = json.loads((out / "prep.json").read_text())
    if rec["code"] != code_hashes():
        raise InputError("prep ran other code")
    if rec["fingerprint"] != fingerprint(rec):
        raise InputError("prep.json does not match its fingerprint")
    bad = [a for a in artifacts if sha256(out / a) != rec["artifacts_sha256"].get(a)]
    if bad:
        raise InputError(f"prep artifacts changed since prep: {bad}")
    return rec


def source(root, out, stage, view, feature, runtime=None):
    t0 = time.time()
    out = Path(out)
    text_file = "text_first16.npy" if view == "first16" else "text_full.npy"
    prep_rec = read_prep(out, ("mask_first16.npy", "valid.npy", "meta.json", "reference_profile.npy", "draws.npz")
                         + ((text_file,) if feature == "text" else ()))
    if runtime is not None and runtime.get("versions") != (prep_rec["runtime"] or {}).get("versions"):
        raise InputError("this container's package versions differ from prep's")
    valid = np.load(out / "valid.npy")[STAGES.index(stage)]
    meta = [dict(r, valid=bool(v)) for r, v in zip(json.loads((out / "meta.json").read_text()), valid)]
    if view == "full":
        mask = valid
    else:
        if not prep_rec["first16"]["support"]["passed"]:
            raise InputError("first16 support failed; the first16 view stops")
        mask = valid & np.load(out / "mask_first16.npy")
    is_ref = (stage, view, feature) == ("instruct", "full", f"s{STATE}")
    ref = np.load(out / "reference_profile.npy")
    draws = dict(np.load(out / "draws.npz"))
    X = load_features(root, out, stage, "full" if view == "full_m16" else view, feature)
    rec, arrays = source_stats(X, meta, mask, None if is_ref else ref, draws,
                               map_check=feature in (f"s{STATE}", "text") and view in VIEWS)
    if is_ref and not np.allclose(arrays["profile"], ref):
        raise InputError("the reference source does not reproduce the prep profile")
    name = source_name(stage, view, feature)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    tmp = out / "sources" / f"{name}.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(out / "sources" / f"{name}.npz")
    rec.update(stage=stage, view=view, feature=feature, code=code_hashes(), fingerprint=prep_rec["fingerprint"],
               arrays_sha256=sha256(out / "sources" / f"{name}.npz"), seconds=round(time.time() - t0, 1))
    write_json(out / "sources" / f"{name}.json", rec)  # written last: a source counts only with its JSON
    return dict(source=name, dropped_draws=rec["dropped_draws"], nonfinite_draws=rec["nonfinite_draws"],
                seconds=rec["seconds"])


def source_problem(out, prep_rec, stage, view, feature):
    """None if this source is complete and belongs to this run; else the reason."""
    out = Path(out)
    name = source_name(stage, view, feature)
    j, z = out / "sources" / f"{name}.json", out / "sources" / f"{name}.npz"
    if not j.exists() or not z.exists():
        return "missing"
    rec = json.loads(j.read_text())
    if rec.get("fingerprint") != prep_rec["fingerprint"] or rec.get("code") != prep_rec["code"]:
        return "other run or code"
    if rec.get("arrays_sha256") != sha256(z):
        return "arrays differ from the recorded hash"
    return None


def load_sources(out):
    out = Path(out)
    prep_rec = read_prep(out)
    S = {}
    for st, v, f in all_sources():
        name = source_name(st, v, f)
        why = source_problem(out, prep_rec, st, v, f)
        if why:
            raise InputError(f"{name}: {why}")
        S[(st, v, f)] = (json.loads((out / "sources" / f"{name}.json").read_text()),
                         dict(np.load(out / "sources" / f"{name}.npz")))
    return S


def p3_rates(d_full, d_16, grid):
    """Simulated rejection rate of a planted change under the frozen reading rule (each view's 99.5% interval
    excludes 0, and both views give the same reading), from the paired difference draws centred on their median, so
    the real difference is hidden. One simulated result = the planted change + one centred draw (the same draw in
    both views); its interval = that result + the centred percentile offsets. Rates are given for a planted increase
    and a planted decrease; `rate` is the lower of the two. The gates are taken as passed; they are not simulated."""
    ok = np.isfinite(d_full) & np.isfinite(d_16)
    c = [d[ok] - np.median(d[ok]) for d in (d_full, d_16)]
    q = [np.percentile(x, [(100 - CONF) / 2, 100 - (100 - CONF) / 2]) for x in c]
    rates = {}
    for delta in grid:
        per = {}
        for name, sign in (("planted_increase", 1), ("planted_decrease", -1)):
            inc, dec = np.ones(int(ok.sum()), bool), np.ones(int(ok.sum()), bool)
            for x, (lo, hi) in zip(c, q):
                obs = sign * delta + x
                inc &= obs + lo > 0
                dec &= obs + hi < 0
            per[name] = dict(rate=float((inc | dec).mean()), wrong_direction=float((dec if sign > 0 else inc).mean()))
        rates[str(delta)] = dict(per, rate=min(v["rate"] for v in per.values()))
    smallest = next((d for d in grid if rates[str(d)]["rate"] >= P3_RATE), None)
    return dict(rates=rates, smallest_with_80pct=smallest, smallest_rule="the lower of the two directions' rates",
                n_draws=int(ok.sum()))


def power(out):
    """P1-P3. power.json holds pass flags, signed responses (they do not depend on the real position) and centred P3
    rates only; the absolute responses stay in the source files until measure()."""
    S = load_sources(out)
    p1, p2 = {}, {}
    gate_ok = True
    for (st, v, f), (rec, _) in S.items():
        pl = rec["planted"]
        name = source_name(st, v, f)
        p1[name] = {d: dict(moved_signed=x["moved_signed"], spread_moved=x["spread_moved"],
                            within_10pct=abs(x["moved_signed"] - float(d)) <= 0.1 * float(d),
                            spread_unchanged=abs(x["spread_moved"]) < 0.01) for d, x in pl["shifts"].items()}
        p2[name] = {k: dict(spread_moved=x["spread_moved"], target=x["target"],
                            within_001=abs(x["spread_moved"] - x["target"]) <= 0.01) for k, x in pl["scales"].items()}
        if f == f"s{STATE}" and v in VIEWS:
            gate_ok &= all(x["within_10pct"] and x["spread_unchanged"] for x in p1[name].values())
            gate_ok &= all(x["within_001"] for x in p2[name].values())
    p3 = {}
    for a, b in PAIRS + REPORT_PAIRS:
        for meas in ("position", "spread"):
            d = {v: S[(b, v, f"s{STATE}")][1][meas] - S[(a, v, f"s{STATE}")][1][meas] for v in VIEWS}
            p3[f"{a}_to_{b}_{meas}"] = p3_rates(d["full"], d["first16"], P3_GRID[meas])
    rec = dict(rules="THRESHOLDS.md v1.2", passed=bool(gate_ok), gated_sources=[source_name(s, v, f"s{STATE}")
               for s in STAGES for v in VIEWS], P1=p1, P2=p2, P3=p3, code=code_hashes(),
               fingerprint=read_prep(out)["fingerprint"])
    write_json(Path(out) / "power.json", rec)
    return rec


def reading_position(lo, hi, point):
    if point > END and lo > END:
        return "high end"
    if point < -END and hi < -END:
        return "low end"
    return "not at one end"


def reading_sign(lo, hi, up="above 0", down="below 0"):
    if lo > 0:
        return up
    if hi < 0:
        return down
    return "interval includes 0"


def combine(views):
    """Both views must give the same reading; a disagreement is length-dependent (no claim)."""
    vals = set(views.values())
    if "not read" in vals:
        return "not read"
    return vals.pop() if len(vals) == 1 else "length-dependent"


def han_group_test(means_by_stage, roles, pairs):
    pax = load_module("han_persona_axes", HAN_AXES)
    proj = {s: np.asarray(pax.axis_projections(M[:-1], M[-1])) for s, M in means_by_stage.items()}
    out = dict(axis="persona_axes.axis_projections: default minus mean role, origin mean role, unit direction")
    for a, b in pairs:
        for units in ("raw", "role_sd"):
            pa_, pb_ = proj[a], proj[b]
            if units == "role_sd":
                pa_, pb_ = pa_ / np.std(pa_, ddof=1), pb_ / np.std(pb_, ddof=1)
            change = pb_ - pa_
            out[f"{a}_to_{b}_{units}_change_all_roles"] = {r: float(x) for r, x in zip(roles, change)}
            for drop in (False, True):
                good = [r for r in DESIRABLE if not (drop and r in BORDERLINE)]
                bad = [r for r in UNDESIRABLE if not (drop and r in BORDERLINE)]
                lab = good + bad
                vals = change[[roles.index(r) for r in lab]]
                obs = vals[:len(good)].mean() - vals[len(good):].mean()
                null = np.array([vals[list(c)].mean() - np.delete(vals, list(c)).mean()
                                 for c in itertools.combinations(range(len(lab)), len(good))])
                out[f"{a}_to_{b}_{units}_{'no_borderline' if drop else 'all'}"] = dict(
                    gap=float(obs), desirable_mean=float(vals[:len(good)].mean()),
                    undesirable_mean=float(vals[len(good):].mean()),
                    p_exact=float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), n_splits=len(null),
                    change_per_role={r: float(x) for r, x in zip(lab, vals)})
    out["projection_by_stage"] = {s: dict(zip(roles, map(float, p))) for s, p in proj.items()}
    return out


def measure(out):
    out = Path(out)
    pw = json.loads((out / "power.json").read_text())
    if not pw["passed"]:
        raise InputError("the power check failed; no real number is read")
    if pw["code"] != code_hashes() or pw.get("fingerprint") != read_prep(out)["fingerprint"]:
        raise InputError("the power check ran other code or another run")
    S = load_sources(out)

    def n_ok(*arrays):
        return int(np.logical_and.reduce([np.isfinite(a) for a in arrays]).sum())

    summ = {}
    for key, (rec, arr) in S.items():
        p = rec["point"]
        summ[source_name(*key)] = dict(
            position=dict(point=p["position"], ci995=interval(arr["position"], CONF),
                          ci95=interval(arr["position"], REPORT_CONF)),
            position_abs=dict(point=p["position_abs"], ci95=interval(np.abs(arr["position"]), REPORT_CONF)),
            spread=dict(point=p["spread"], ci995=interval(arr["spread"], CONF), ci95=interval(arr["spread"], REPORT_CONF)),
            profile_corr=rec["profile_corr"], pc1_share=rec["pc1_share"], split_pc1_cosine=rec["split_pc1_cosine"],
            dropped_share=rec["dropped_share"], dropped_draws=rec["dropped_draws"],
            nonfinite_draws=rec["nonfinite_draws"], n_draws=n_ok(arr["position"], arr["spread"]),
            map_gate=rec.get("map_gate"), lu=rec["lu"], n_rows=rec["n_rows"])

    def gate(stage, view, meas, feature=f"s{STATE}"):
        s = summ[source_name(stage, view, feature)]
        ok = s["map_gate"]["passed"] and s["dropped_share"] <= MAX_DROPPED
        return bool(ok and (meas != "position" or s["profile_corr"] >= PROFILE_FLOOR))

    tests = {}
    n = 0
    for st in STAGES:
        n += 1
        per = {}
        for v in VIEWS:
            s = summ[source_name(st, v, f"s{STATE}")]["position"]
            per[v] = reading_position(*s["ci995"], s["point"]) if gate(st, v, "position") else "not read"
        tests[f"{n}_position_{st}"] = dict(views=per, reading=combine(per))
    for st in STAGES:
        n += 1
        per = {}
        for v in VIEWS:
            s = summ[source_name(st, v, f"s{STATE}")]["spread"]
            per[v] = reading_sign(*s["ci995"], "wider", "tighter") if gate(st, v, "spread") else "not read"
        tests[f"{n}_spread_{st}"] = dict(views=per, reading=combine(per))
    diffs = {}
    for meas in ("position", "spread"):
        for a, b in PAIRS + REPORT_PAIRS:
            per, info = {}, {}
            for v in VIEWS:
                ra, rb = S[(a, v, f"s{STATE}")], S[(b, v, f"s{STATE}")]
                d = rb[1][meas] - ra[1][meas]
                pt = rb[0]["point"][meas] - ra[0]["point"][meas]
                info[v] = dict(point=pt, ci995=interval(d, CONF), ci95=interval(d, REPORT_CONF), n_paired=n_ok(d))
                per[v] = (reading_sign(*info[v]["ci995"], "increase", "decrease")
                          if gate(a, v, meas) and gate(b, v, meas) else "not read")
            diffs[f"{a}_to_{b}_{meas}"] = dict(views=info, readings=per, reading=combine(per),
                                               p3=pw["P3"][f"{a}_to_{b}_{meas}"])
        for a, b in PAIRS:
            n += 1
            tests[f"{n}_{meas}_change_{a}_to_{b}"] = diffs[f"{a}_to_{b}_{meas}"]
    state_minus_text = {}
    for st in STAGES:
        for v in VIEWS:
            a, t = S[(st, v, f"s{STATE}")][1], S[(st, v, "text")][1]
            state_minus_text[f"{st}_{v}"] = {m: dict(point=S[(st, v, f"s{STATE}")][0]["point"][m] - S[(st, v, "text")][0]["point"][m],
                                                     ci95=interval(a[m] - t[m], REPORT_CONF), n_paired=n_ok(a[m] - t[m]),
                                                     gates_passed=dict(state=gate(st, v, m), text=gate(st, v, m, "text")))
                                             for m in ("position", "spread")}
    roles = S[(STAGES[0], "full", f"s{STATE}")][0]["clouds"][:-1]
    han = {v: han_group_test({st: S[(st, v, f"s{STATE}")][1]["cloud_means"] for st in STAGES}, roles,
                             PAIRS + REPORT_PAIRS) for v in VIEWS}
    planted_abs = {source_name(*k): {d: x["moved_abs"] for d, x in r["planted"]["shifts"].items()} for k, (r, _) in S.items()}
    rec = dict(rules="THRESHOLDS.md v1.2", code_revision="b", confirmatory=tests, report_only=dict(
        stage_changes=diffs, sources=summ, state_minus_text=state_minus_text, han_group_test=han,
        planted_absolute_response=planted_abs),
        n_confirmatory=n, interval=f"{CONF}% percentile, {N_DRAWS} crossed draws", code=code_hashes(),
        fingerprint=pw["fingerprint"])
    write_json(out / "results_v12.json", rec)
    return rec
