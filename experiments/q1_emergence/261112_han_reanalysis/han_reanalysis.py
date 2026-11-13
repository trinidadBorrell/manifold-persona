"""Re-analysis of Han Xiao's public release (rules: THRESHOLDS.md v1.0, Parts A-D). CPU only.

Every result describes Olmo 3 7B (Han Xiao's release), not OLMo-2. Part A is a reproduction gate; Parts B-D are
exploratory. Han Xiao's own code (data/han_repo, commit 6f28e91) supplies the axis, split-half and permutation
functions; his notebooks/q1_robustness.py fixes the Part A/B procedure (min 50 score-3 replies, seed 0 per split
run, 100 halves, 10,000 shuffles). Every .pt file is loaded with weights_only=True.

  python -I han_reanalysis.py a  --data DATA --out OUT          # gate
  python -I han_reanalysis.py b  --data DATA --out OUT
  python -I han_reanalysis.py cd --data DATA --out OUT
  python -I han_reanalysis.py report --out OUT
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data/han_repo/src"))
import torch  # noqa: E402
from persona_shift import persona_axes as pa  # noqa: E402
from persona_shift.robustness import permutation as perm  # noqa: E402
from persona_shift.robustness import split_half as sh  # noqa: E402

LABEL = "Olmo 3 7B (Han Xiao's release)"
LAYER = 16
LAYER_ROW = pa.LAYERS.index(LAYER)
MIN_COUNT, N_REPEATS, N_PERM, SEED = 50, 100, 10_000, 0  # q1_robustness.py
N_BOOT, BOOT_SEED = 1000, 20261112
STAGES = list(pa.STAGES)  # SFT, DPO, RLVR
PAIRS = [("SFT", "DPO"), ("DPO", "RLVR")]
FIG_DESIRABLE, FIG_P = 0.183, 0.066
LABELS = HERE / "data/han_repo/configs/role_desirability.json"


# ---------- loading (Han's functions, with weights_only=True) ----------

def stage_dir(data, stage):
    return Path(data) / pa.STAGES[stage]


def load_released_vectors(data, stage):
    out = {}
    for p in sorted((stage_dir(data, stage) / f"vectors-{pa.JUDGE_TAG}").glob("*.pt")):
        out[p.stem] = torch.load(p, weights_only=True)["vector"].float().numpy()[LAYER_ROW]
    return out


def passed_ids(data, stage):
    return pa.judge_score_3_response_ids(pa.load_judge_scores(Path(data), stage))


def supported(passed):
    return {r: ids for r, ids in passed.items() if len(ids) >= MIN_COUNT}


def load_acts(data, stage, role):
    d = torch.load(stage_dir(data, stage) / f"activations-layer{LAYER}" / f"{role}.pt", weights_only=True)
    return list(d["keys"]), d["acts"].float().numpy()


def activation_roles(data, stage):
    return sorted(p.stem for p in (stage_dir(data, stage) / f"activations-layer{LAYER}").glob("*.pt")
                  if p.stem != "default")


def key_parts(keys):
    p = np.array([int(k.split("_p", 1)[1].split("_q", 1)[0]) for k in keys])
    q = np.array([int(k.rsplit("_q", 1)[1]) for k in keys])
    return p, q


def sums_from_acts(keys, acts, keep_ids):
    """One role's per-question sums and counts, as pa.sum_activations_by_question (keep_ids None keeps all)."""
    _, q = key_parts(keys)
    kept = np.ones(len(keys), bool) if keep_ids is None else np.array([k in keep_ids for k in keys])
    one_hot = (q[kept][None, :] == np.arange(pa.N_QUESTIONS)[:, None]).astype(np.float32)
    return one_hot @ acts[kept], one_hot.sum(axis=1)


def sums_by_question(data, stage, keep):
    """pa.sum_activations_by_question with weights_only=True; keep[role] = set of ids, or None for all replies."""
    roles = sorted(keep) + ["default"]
    sums, counts = [], []
    for role in roles:
        keys, acts = load_acts(data, stage, role)
        s, c = sums_from_acts(keys, acts, None if role == "default" else keep[role])
        sums.append(s)
        counts.append(c)
    return roles, np.stack(sums), np.stack(counts)


def labels_table(path):
    t = pa.load_desirability(Path(path)).set_index("role")
    return t


def group_mask(labels, roles, drop_borderline=False):
    lab = labels.reindex(roles)
    label = lab["label"].fillna("neutral")
    border = lab["borderline"].fillna(False).astype(bool)
    m = label.isin(["desirable", "undesirable"])
    if drop_borderline:
        m &= ~border
    m = m.to_numpy()
    return m, (label.to_numpy()[m] == "desirable")


def write(out, name, obj):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=1, default=float))


def pct(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


# ---------- Part A: reproduction gate ----------

def part_a(data, labels_path=LABELS, n_perm=N_PERM):
    proj, checks = {}, {}
    for s in ("SFT", "DPO"):
        vec = load_released_vectors(data, s)
        roles = sorted(r for r in vec if r != "default")
        R, d = np.stack([vec[r] for r in roles]), vec["default"]
        proj[s] = dict(zip(roles, pa.axis_projections(R, d)))
        axis_file = stage_dir(data, s) / f"axis-{pa.JUDGE_TAG}.pt"
        cos = None
        if axis_file.exists():
            axis = torch.load(axis_file, weights_only=True).float().numpy()[LAYER_ROW]
            mine = d - R.mean(0)
            cos = float(axis @ mine / (np.linalg.norm(axis) * np.linalg.norm(mine)))
        sup = set(supported(passed_ids(data, s)))
        checks[s] = dict(n_roles_with_vectors=len(roles), axis_file_cosine=cos,
                         roles_with_vector_not_supported=sorted(set(roles) - sup),
                         supported_without_vector=sorted(sup - set(roles)))
    common = sorted(set(proj["SFT"]) & set(proj["DPO"]))
    shift = np.array([proj["DPO"][r] - proj["SFT"][r] for r in common])
    m, good = group_mask(labels_table(labels_path), common)
    obs, _, p = perm.permutation_test(shift[m], good, n_perm, np.random.default_rng(SEED))
    res = dict(label=LABEL, part="A", stages="SFT->DPO", layer=LAYER, n_common=len(common),
               n_desirable=int(good.sum()), n_undesirable=int((~good).sum()),
               desirable_shift=float(shift[m][good].mean()), undesirable_shift=float(shift[m][~good].mean()),
               gap=float(obs), permutation_p=float(p), n_permutations=n_perm, checks=checks,
               reference=dict(desirable_shift=FIG_DESIRABLE, permutation_p=FIG_P))
    res["gate_pass"] = bool(abs(res["desirable_shift"] - FIG_DESIRABLE) <= 0.01 and abs(p - FIG_P) <= 0.02)
    return res


# ---------- Part B: raw vs role-SD units ----------

def stage_run(data, stage, n_repeats=N_REPEATS):
    keep = supported(passed_ids(data, stage))
    roles, sums, counts = sums_by_question(data, stage, keep)
    vectors = sums.sum(axis=1) / counts.sum(axis=1)[:, None]
    a, b = sh.split_half(sums, counts, n_repeats, np.random.default_rng(SEED))
    return dict(roles=roles[:-1], projection=pa.axis_projections(vectors[:-1], vectors[-1]), half_a=a, half_b=b,
                reliability=float(np.mean(sh.reliability(a, b))), vectors=vectors)


def units_view(x, idx, units):
    if units == "raw":
        return x[..., idx]
    if units == "z_common":  # THRESHOLDS v1.0: z over the roles present in both stages, within each stage
        y = x[..., idx]
        return (y - y.mean(-1, keepdims=True)) / y.std(-1, keepdims=True)
    if units == "z_stage":  # Han's notebook: z over all of the stage's roles, then the common roles
        y = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
        return y[..., idx]
    raise ValueError(units)


def pair_stats(run_a, run_b, labels, units, drop_borderline, n_perm=N_PERM):
    common = sorted(set(run_a["roles"]) & set(run_b["roles"]))
    ia = [run_a["roles"].index(r) for r in common]
    ib = [run_b["roles"].index(r) for r in common]
    shift = units_view(run_b["projection"], ib, units) - units_view(run_a["projection"], ia, units)
    noise = ((units_view(run_b["half_a"], ib, units) - units_view(run_b["half_b"], ib, units))
             - (units_view(run_a["half_a"], ia, units) - units_view(run_a["half_b"], ia, units)))
    m, good = group_mask(labels, common, drop_borderline)
    s, nz = shift[m], noise[:, m]
    obs, _, p = perm.permutation_test(s, good, n_perm, np.random.default_rng(SEED))

    def se(per_repeat):
        return float(sh.full_data_se(np.asarray(per_repeat)[:, None])[0])

    gap_se = se([perm.mean_difference(row, good) for row in nz])
    return dict(n_common=len(common), n_desirable=int(good.sum()), n_undesirable=int((~good).sum()),
                desirable_shift=float(s[good].mean()), desirable_se=se(nz[:, good].mean(axis=1)),
                undesirable_shift=float(s[~good].mean()), undesirable_se=se(nz[:, ~good].mean(axis=1)),
                gap=float(obs), gap_se=gap_se, gap_z=float(obs / gap_se) if gap_se else None, permutation_p=float(p))


def part_b(data, labels_path=LABELS, n_repeats=N_REPEATS, n_perm=N_PERM):
    labels = labels_table(labels_path)
    runs = {s: stage_run(data, s, n_repeats) for s in STAGES}
    out = dict(label=LABEL, part="B", exploratory=True, layer=LAYER, n_repeats=n_repeats, n_permutations=n_perm,
               reliability={s: runs[s]["reliability"] for s in STAGES},
               n_roles={s: len(runs[s]["roles"]) for s in STAGES}, pairs={})
    for a, b in PAIRS:
        common = sorted(set(runs[a]["roles"]) & set(runs[b]["roles"]))
        sd = {s: float(np.std(units_view(runs[s]["projection"], [runs[s]["roles"].index(r) for r in common], "raw")))
              for s in (a, b)}
        cell = dict(role_sd_raw=sd)
        for units in ("raw", "z_common", "z_stage"):
            for drop in (False, True):
                cell[f"{units}{'_no_borderline' if drop else ''}"] = pair_stats(runs[a], runs[b], labels, units, drop, n_perm)
        out["pairs"][f"{a}->{b}"] = cell
    return out


# ---------- Parts C and D: default position and spread, judge-filter sensitivity ----------

def boot_weights(n_boot=N_BOOT, seed=BOOT_SEED, n_q=pa.N_QUESTIONS):
    """Row 0 = the observed data (all ones); rows 1.. = question resamples, shared by every stage and part."""
    rng = np.random.default_rng(seed)
    W = np.ones((n_boot + 1, n_q))
    for b in range(1, n_boot + 1):
        W[b] = np.bincount(rng.integers(0, n_q, n_q), minlength=n_q)
    return W


def pc1_position(sums, counts, W, chunk=100):
    """Per draw: PC1 of the centred role vectors (roles only); the default's place on it.

    Returns per draw: distance to the nearer end of the role range (0..1; 0 when outside), outside flag,
    signed z (sign: each draw aligned to the observed PC1 of this stage), PC1 variance share, roles dropped,
    and the observed PC1 unit vector.
    """
    n = sums.shape[0] - 1
    dist, outside, z, share, dropped = [], [], [], [], []
    pc_ref = None
    for c0 in range(0, len(W), chunk):
        Wc = W[c0:c0 + chunk]
        num = np.tensordot(Wc, sums, axes=([1], [1]))  # (c, roles+1, D)
        den = Wc @ counts.T  # (c, roles+1)
        for j in range(len(Wc)):
            ok = den[j, :n] > 0
            R = num[j, :n][ok] / den[j, :n][ok, None]
            d = num[j, n] / den[j, n]
            mu = R.mean(0)
            Rc = R - mu
            evals, evecs = np.linalg.eigh(Rc @ Rc.T)
            pc = Rc.T @ evecs[:, -1]
            pc /= np.linalg.norm(pc)
            if pc_ref is None:
                pc_ref = pc.copy()
            if pc @ pc_ref < 0:
                pc = -pc
            pr, pdft = Rc @ pc, (d - mu) @ pc
            lo, hi = pr.min(), pr.max()
            t = (pdft - lo) / (hi - lo)
            out = bool(t < 0 or t > 1)
            dist.append(0.0 if out else float(min(t, 1 - t)))
            outside.append(out)
            z.append(float((pdft - pr.mean()) / pr.std()))
            share.append(float(evals[-1] / evals.sum()))
            dropped.append(int((~ok).sum()))
    return dict(distance=np.array(dist), outside=np.array(outside), z=np.array(z), pc1_share=np.array(share),
                dropped=np.array(dropped), pc_obs=pc_ref)


def cloud_variance(keys, acts, W):
    """Per draw: within-system-prompt variance of one cloud (all replies), mean over the 5 prompts."""
    p, q = key_parts(keys)
    P = 5
    if len(set(zip(p, q))) != len(keys) or p.min() < 0 or p.max() >= P:
        raise SystemExit("a cloud has a duplicate reply or an unknown prompt index")
    M = np.zeros((P, pa.N_QUESTIONS))  # 1 where the reply exists (some SFT clouds miss 1-2 of 1,200)
    M[p, q] = 1
    A = np.zeros((P, pa.N_QUESTIONS, acts.shape[1]))
    A[p, q] = acts
    A -= (A.sum(axis=1, keepdims=True) / M.sum(1)[:, None, None]) * M[..., None]  # centre present cells only
    sq = (A ** 2).sum(-1)  # (P, Q); 0 at missing cells
    tot = W @ M.T  # (B, P): missing cells carry no weight
    mean = np.tensordot(W, A, axes=([1], [1])) / tot[..., None]  # (B, P, D)
    msq = (W @ sq.T) / tot  # (B, P)
    return (msq - (mean ** 2).sum(-1)).mean(1)  # (B,)


def summarise_draws(x):
    x = np.asarray(x, float)
    return dict(observed=float(x[0]), ci95=pct(x[1:]))


def part_cd(data, labels_path=LABELS, n_boot=N_BOOT, n_perm=N_PERM, stages=STAGES):
    labels = labels_table(labels_path)
    W = boot_weights(n_boot)
    per = {}
    for s in stages:
        t0 = time.time()
        keep_f = supported(passed_ids(data, s))
        roles_all = activation_roles(data, s)
        sf, cf, su, cu, var = [], [], [], [], {}
        missing = {}
        for role in roles_all + ["default"]:
            keys, acts = load_acts(data, s, role)
            if len(keys) != 5 * pa.N_QUESTIONS:
                missing[role] = 5 * pa.N_QUESTIONS - len(keys)
            if role == "default" or role in keep_f:
                a, b = sums_from_acts(keys, acts, None if role == "default" else keep_f[role])
                sf.append((role, a, b))
            a, b = sums_from_acts(keys, acts, None)
            su.append((role, a, b))
            var[role] = cloud_variance(keys, acts, W)
        order_f = [r for r, _, _ in sf]
        assert order_f[-1] == "default" and [r for r, _, _ in su][-1] == "default"
        S_f, C_f = np.stack([a for _, a, _ in sf]), np.stack([b for _, _, b in sf])
        S_u, C_u = np.stack([a for _, a, _ in su]), np.stack([b for _, _, b in su])
        c1f, c1u = pc1_position(S_f, C_f, W), pc1_position(S_u, C_u, W)
        role_var = np.stack([var[r] for r in roles_all])  # (roles, B)
        logratio = np.log(var["default"]) - np.log(np.median(role_var, axis=0))
        # Part D shifts need the unfiltered observed vectors
        vec_u = S_u.sum(1) / C_u.sum(1)[:, None]
        proj_u = pa.axis_projections(vec_u[:-1], vec_u[-1])
        per[s] = dict(roles_f=order_f[:-1], roles_u=roles_all, c1f=c1f, c1u=c1u, logratio=logratio,
                      proj_u=proj_u, seconds=round(time.time() - t0, 1), missing=missing)
        print(f"{s}: {len(order_f) - 1} filtered roles, {len(roles_all)} roles, {per[s]['seconds']}s", flush=True)
    ref = "RLVR" if "RLVR" in per else stages[-1]

    def stage_sign(s, key):
        return float(np.sign(per[s][key]["pc_obs"] @ per[ref][key]["pc_obs"]))

    out = dict(label=LABEL, part="C and D", exploratory=True, layer=LAYER, n_boot=n_boot, boot_seed=BOOT_SEED,
               sign_reference=ref, C1={}, C1_unfiltered={}, C2={}, C3={}, D_shifts={})
    zs = {}
    for key, name in (("c1f", "C1"), ("c1u", "C1_unfiltered")):
        g = float(np.sign(per[ref][key]["z"][0])) or 1.0
        zs[key] = {}
        for s in per:
            r = per[s][key]
            z = r["z"] * stage_sign(s, key) * g
            zs[key][s] = z
            dist_ci = pct(r["distance"][1:])
            share_out = float(r["outside"][1:].mean())
            out[name][s] = dict(
                n_roles=len(per[s]["roles_f" if key == "c1f" else "roles_u"]),
                distance_to_nearer_end=dict(observed=float(r["distance"][0]), ci95=dist_ci),
                outside_role_range=dict(observed=bool(r["outside"][0]), share_of_draws=share_out),
                signed_z=summarise_draws(z), pc1_variance_share=summarise_draws(r["pc1_share"]),
                cos_pc1_with_reference=float(abs(r["pc_obs"] @ per[ref][key]["pc_obs"])),
                draws_with_dropped_roles=int((r["dropped"][1:] > 0).sum()),
                reading_extreme=bool((dist_ci[1] is not None and dist_ci[1] <= 0.05) or share_out >= 0.95))
    for s in per:
        lr = per[s]["logratio"]
        ci = pct(lr[1:])
        out["C2"][s] = dict(log_var_default_over_median_role=summarise_draws(lr),
                            reading_tighter=bool(ci[1] is not None and ci[1] < 0))
    for a, b in PAIRS:
        if a in per and b in per:
            dz = zs["c1f"][b] - zs["c1f"][a]
            dl = per[b]["logratio"] - per[a]["logratio"]
            cz, cl = pct(dz[1:]), pct(dl[1:])
            out["C3"][f"{a}->{b}"] = dict(
                signed_z_change=dict(observed=float(dz[0]), ci95=cz, present=bool(cz[0] > 0 or cz[1] < 0)),
                logratio_change=dict(observed=float(dl[0]), ci95=cl, present=bool(cl[0] > 0 or cl[1] < 0)))
            common = sorted(set(per[a]["roles_u"]) & set(per[b]["roles_u"]))
            ia = [per[a]["roles_u"].index(r) for r in common]
            ib = [per[b]["roles_u"].index(r) for r in common]
            cell = {}
            for units in ("raw", "z_common"):
                shift = units_view(per[b]["proj_u"], ib, units) - units_view(per[a]["proj_u"], ia, units)
                m, good = group_mask(labels, common)
                obs, _, p = perm.permutation_test(shift[m], good, n_perm, np.random.default_rng(SEED))
                cell[units] = dict(n_desirable=int(good.sum()), n_undesirable=int((~good).sum()),
                                   desirable_shift=float(shift[m][good].mean()),
                                   undesirable_shift=float(shift[m][~good].mean()), gap=float(obs),
                                   permutation_p=float(p))
            out["D_shifts"][f"{a}->{b}"] = cell
    out["seconds_per_stage"] = {s: per[s]["seconds"] for s in per}
    out["clouds_with_missing_replies"] = {s: dict(n_clouds=len(per[s]["missing"]),
                                                  n_missing_replies=int(sum(per[s]["missing"].values())),
                                                  clouds=per[s]["missing"]) for s in per}
    return out


# ---------- report ----------

def fmt(x, n=3):
    return "n/a" if x is None else f"{x:+.{n}f}"


def report(out):
    out = Path(out)
    L = [f"# Re-analysis of Han Xiao's release: results", "", f"All numbers: {LABEL}, layer 16. Parts B-D are exploratory.", ""]
    if (out / "part_a.json").exists():
        a = json.loads((out / "part_a.json").read_text())
        L += ["## Part A (gate)", "",
              f"- Desirable shift SFT->DPO {fmt(a['desirable_shift'])} (reference +0.183); undesirable {fmt(a['undesirable_shift'])}; "
              f"gap {fmt(a['gap'])}; permutation p {a['permutation_p']:.4f} (reference 0.066); "
              f"{a['n_desirable']} desirable, {a['n_undesirable']} undesirable. Gate: {'PASS' if a['gate_pass'] else 'FAIL'}.", ""]
    if (out / "part_b.json").exists():
        b = json.loads((out / "part_b.json").read_text())
        L += ["## Part B (units)", "", "| pair | units | desirable (se) | undesirable (se) | gap (se) | p |", "|---|---|---|---|---|---|"]
        for pair, cell in b["pairs"].items():
            for k, v in cell.items():
                if k == "role_sd_raw":
                    continue
                L.append(f"| {pair} | {k} | {fmt(v['desirable_shift'])} ({v['desirable_se']:.3f}) | "
                         f"{fmt(v['undesirable_shift'])} ({v['undesirable_se']:.3f}) | {fmt(v['gap'])} ({v['gap_se']:.3f}) | "
                         f"{v['permutation_p']:.3f} |")
            L.append(f"| {pair} | role SD (raw) | {cell['role_sd_raw']} | | | |")
        L.append("")
    if (out / "part_cd.json").exists():
        c = json.loads((out / "part_cd.json").read_text())
        L += ["## Part C1 (default position on roles-only PC1) and D (unfiltered)", "",
              "| version | stage | distance obs [CI] | outside share | signed z obs [CI] | extreme |", "|---|---|---|---|---|---|"]
        for name in ("C1", "C1_unfiltered"):
            for s, v in c[name].items():
                dd, zz = v["distance_to_nearer_end"], v["signed_z"]
                L.append(f"| {name} | {s} | {dd['observed']:.3f} [{dd['ci95'][0]:.3f}, {dd['ci95'][1]:.3f}] | "
                         f"{v['outside_role_range']['share_of_draws']:.2f} | {fmt(zz['observed'], 2)} "
                         f"[{fmt(zz['ci95'][0], 2)}, {fmt(zz['ci95'][1], 2)}] | {v['reading_extreme']} |")
        L += ["", "## Part C2 (spread) and C3 (stage changes)", ""]
        for s, v in c["C2"].items():
            lr = v["log_var_default_over_median_role"]
            L.append(f"- C2 {s}: log(var default / median role) {fmt(lr['observed'])} [{fmt(lr['ci95'][0])}, {fmt(lr['ci95'][1])}]; tighter: {v['reading_tighter']}")
        for pair, v in c["C3"].items():
            L.append(f"- C3 {pair}: signed z change {fmt(v['signed_z_change']['observed'], 2)} {v['signed_z_change']['ci95']} "
                     f"present {v['signed_z_change']['present']}; log-ratio change {fmt(v['logratio_change']['observed'])} "
                     f"{v['logratio_change']['ci95']} present {v['logratio_change']['present']}")
        L += ["", "## Part D shifts (unfiltered vectors)", ""]
        for pair, cell in c["D_shifts"].items():
            for units, v in cell.items():
                L.append(f"- {pair} {units}: desirable {fmt(v['desirable_shift'])}, undesirable {fmt(v['undesirable_shift'])}, "
                         f"gap {fmt(v['gap'])}, p {v['permutation_p']:.3f}")
    (out / "results.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "b", "cd", "report"])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--n_repeats", type=int, default=N_REPEATS)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    t0 = time.time()
    if a.part == "a":
        r = part_a(a.data, a.labels, a.n_perm)
        write(a.out, "part_a.json", r)
        print(json.dumps({k: r[k] for k in ("desirable_shift", "undesirable_shift", "gap", "permutation_p", "gate_pass")}))
        if not r["gate_pass"]:
            raise SystemExit("Part A gate failed; stop (THRESHOLDS.md)")
    elif a.part == "b":
        write(a.out, "part_b.json", part_b(a.data, a.labels, a.n_repeats, a.n_perm))
    elif a.part == "cd":
        write(a.out, "part_cd.json", part_cd(a.data, a.labels, a.n_boot, a.n_perm))
    print(report(a.out) if a.part == "report" else f"part {a.part} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
