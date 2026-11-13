"""Check 1 (design check, NOTE.md): precision of the default-position and spread statistics at 10, 15 and 20 roles.

Position: PC1 of the centred role means on one question half; on the other half, |default projection - mean role
projection| / SD of the role projections. Spread: within-wording variance per cloud, log(default / median role),
on all rows and on rows cut at the 128-token limit. 1,000 draws per role count: random question halves,
questions resampled within each half, roles resampled with replacement from a random subset of n roles.

    .venv/bin/python -I experiments/q1_emergence/261112_free_checks/check_role_count.py --source poda
    .venv/bin/python -I experiments/q1_emergence/261112_free_checks/check_role_count.py --source podc:instruct
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parent
PODA = Q / "261029_instruct_persona_map/run_full"
PODC_DATA = HERE / "data/experiments/q1_emergence/261029_fixed_text_read_through/run_full"
STATE = 17
N_DRAWS = 1000
ROLE_COUNTS = (10, 15, 20)
SHIFTS = (0.5, 1.0, 2.0)
N_POINT_SPLITS = 50
SEED = 20261112
DEFAULT = "default"


def poda_rows():
    out = []
    for line in open(PODA / "rows.jsonl"):
        r = json.loads(line)
        out.append(dict(role=r["role"], wording=r["wording"], question=r["question"], replicate=r["replicate"],
                        limit=r["stop_reason"] == "limit", valid=r["valid"]))
    return out


def grid_order(rows):
    """Pod C stores pod A rows in pod A's frozen grid order; map them back to the row metadata."""
    keys = json.loads((PODA / "grid.json").read_text())["keys"]
    by = {json.dumps([r["role"], "chat", r["wording"], r["question"], r["replicate"]], separators=(",", ":")): r
          for r in rows}
    missing = [k for k in keys if k not in by]
    if missing:
        raise SystemExit(f"grid keys not matched to rows, first {missing[0]}")
    return [by[k] for k in keys]


def load(source):
    rows = poda_rows()
    if source == "poda":
        X = np.load(PODA / "reply_mean_acts.npy", mmap_mode="r")[:, STATE, :]
    else:
        ckpt = source.split(":", 1)[1]
        meta = json.loads((PODC_DATA / ckpt / "pod_a.F1.json").read_text())
        X = np.load(PODC_DATA / ckpt / "pod_a.F1.reply.npy", mmap_mode="r")[:, meta["layers"]["reply"].index(STATE), :]
        rows = grid_order(rows)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] != len(rows) or not np.isfinite(X).all():
        raise SystemExit("states and rows differ in length, or states are not finite")
    return X, rows


def cell_sums(X, rows, mask):
    clouds = sorted({r["role"] for r in rows} - {DEFAULT}) + [DEFAULT]
    qs = sorted({r["question"] for r in rows})
    ws = sorted({r["wording"] for r in rows})
    ci, qi, wi = ({v: i for i, v in enumerate(a)} for a in (clouds, qs, ws))
    S1 = np.zeros((len(clouds), len(qs), len(ws), X.shape[1]))
    S2 = np.zeros((len(clouds), len(qs), len(ws)))
    N = np.zeros((len(clouds), len(qs), len(ws)))
    for i, r in enumerate(rows):
        if mask[i]:
            c, q, w = ci[r["role"]], qi[r["question"]], wi[r["wording"]]
            S1[c, q, w] += X[i]
            S2[c, q, w] += X[i] @ X[i]
            N[c, q, w] += 1
    return dict(clouds=clouds, S1=S1, S2=S2, N=N, S1q=S1.sum(2), Nq=N.sum(2))


def means(cs, wq):
    """Cloud means [cloud, dim] for question weights wq [q]."""
    num = np.tensordot(wq, cs["S1q"], axes=([0], [1]))
    return num / (wq @ cs["Nq"].T)[:, None]


def spread(cs, wq):
    """Within-wording variance per cloud for question weights wq."""
    s1w = np.einsum("q,cqwd->cwd", wq, cs["S1"])
    nw = np.einsum("q,cqw->cw", wq, cs["N"])
    s2 = np.einsum("q,cqw->c", wq, cs["S2"])
    between = np.where(nw > 0, (s1w ** 2).sum(-1) / np.maximum(nw, 1), 0).sum(1)
    return (s2 - between) / nw.sum(1)


def position(cs, wa, wb, role_idx, u_ref=None):
    ma, mb = means(cs, wa), means(cs, wb)
    d = len(cs["clouds"]) - 1
    ra = ma[role_idx]
    u = np.linalg.svd(ra - ra.mean(0), full_matrices=False)[2][0]
    if u_ref is not None and u @ u_ref < 0:
        u = -u
    rb = mb[role_idx]
    centre = rb.mean(0)
    pr, pd = (rb - centre) @ u, (mb[d] - centre) @ u
    return (pd - pr.mean()) / pr.std(ddof=1)


def question_weights(rng, nq):
    perm = rng.permutation(nq)
    a, b = perm[: nq // 2], perm[nq // 2:]
    return (np.bincount(rng.choice(a, len(a)), minlength=nq).astype(float),
            np.bincount(rng.choice(b, len(b)), minlength=nq).astype(float))


def interval(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(point=float(np.median(v)), lo=float(lo), hi=float(hi), width=float(hi - lo), se=float(np.std(v, ddof=1)))


def run(source):
    t0 = time.time()
    X, rows = load(source)
    masks = dict(all=np.array([r["valid"] for r in rows]), limit=np.array([r["valid"] and r["limit"] for r in rows]))
    sums = {k: cell_sums(X, rows, m) for k, m in masks.items()}
    cs = sums["all"]
    n_roles, nq = len(cs["clouds"]) - 1, cs["S1"].shape[1]
    out = dict(source=source, state=STATE, n_rows=len(rows), n_roles=n_roles, n_questions=nq, draws=N_DRAWS,
               rows_per_cloud={c: int(cs["N"][i].sum()) for i, c in enumerate(cs["clouds"])},
               limit_rows_per_cloud={c: int(sums["limit"]["N"][i].sum()) for i, c in enumerate(cs["clouds"])})

    rng = np.random.default_rng(SEED)
    full_w = np.ones(nq)
    all_roles = np.arange(n_roles)
    ma = means(cs, full_w)[all_roles]
    u_full = np.linalg.svd(ma - ma.mean(0), full_matrices=False)[2][0]
    halves = []
    for i in range(N_POINT_SPLITS):
        perm = np.random.default_rng(SEED + 1 + i).permutation(nq)
        halves.append((np.isin(np.arange(nq), perm[: nq // 2]).astype(float),
                       np.isin(np.arange(nq), perm[nq // 2:]).astype(float)))
    out["point"] = dict(
        position_abs=float(np.mean([abs(position(cs, a, b, all_roles)) for a, b in halves])),
        position_signed_full_pc1=float(np.mean([position(cs, a, b, all_roles, u_full) for a, b in halves])),
        spread_log_ratio={k: float(np.log(spread(s, full_w)[-1] / np.median(spread(s, full_w)[:-1]))) for k, s in sums.items()},
        spread_by_cloud={c: float(v) for c, v in zip(cs["clouds"], spread(cs, full_w))})

    res = {}
    for n in ROLE_COUNTS:
        pos, spr = [], {k: [] for k in sums}
        for _ in range(N_DRAWS):
            subset = rng.choice(all_roles, n, replace=False)
            role_idx = rng.choice(subset, n, replace=True)
            wa, wb = question_weights(rng, nq)
            pos.append(abs(position(cs, wa, wb, role_idx)))
            for k, s in sums.items():
                v = spread(s, wa + wb)
                spr[k].append(np.log(v[-1] / np.median(v[role_idx])))
        res[n] = dict(position_abs=interval(np.array(pos)),
                      spread_log_ratio={k: interval(np.array(v)) for k, v in spr.items()})
    out["by_role_count"] = {str(n): r for n, r in res.items()}

    def fit40(widths):
        ns = np.array(ROLE_COUNTS, float)
        a = float(np.sum(widths / np.sqrt(ns)) / np.sum(1 / ns))
        return dict(a=a, width_at_40=a / np.sqrt(40), width_at_20=a / np.sqrt(20))
    out["extrapolation_1_over_sqrt_n"] = dict(
        position_abs=fit40(np.array([res[n]["position_abs"]["width"] for n in ROLE_COUNTS])),
        spread_log_ratio_all=fit40(np.array([res[n]["spread_log_ratio"]["all"]["width"] for n in ROLE_COUNTS])))
    out["detectable_change_between_two_stages"] = {
        str(n): dict(position=2.8 * np.sqrt(2) * res[n]["position_abs"]["se"],
                     spread_all=2.8 * np.sqrt(2) * res[n]["spread_log_ratio"]["all"]["se"]) for n in ROLE_COUNTS}

    planted = {}
    for delta in SHIFTS:
        p = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in cs.items()}
        base_proj = (means(cs, full_w)[all_roles] - means(cs, full_w)[all_roles].mean(0)) @ u_full
        v = delta * base_proj.std(ddof=1) * u_full
        d = n_roles
        p["S2"][d] = cs["S2"][d] + 2 * np.einsum("qwd,d->qw", cs["S1"][d], v) + cs["N"][d] * (v @ v)
        p["S1"][d] = cs["S1"][d] + cs["N"][d][..., None] * v
        p["S1q"], p["Nq"] = p["S1"].sum(2), p["N"].sum(2)
        planted[str(delta)] = dict(
            position_signed=float(np.mean([position(p, a, b, all_roles, u_full) for a, b in halves])),
            spread_log_ratio=float(np.log(spread(p, full_w)[-1] / np.median(spread(p, full_w)[:-1]))))
    out["planted_shift_sanity"] = dict(original_signed=out["point"]["position_signed_full_pc1"], shifts=planted)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="poda")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = run(args.source)
    path = HERE / args.out_dir / f"check1_{args.source.replace(':', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("source", "rows_per_cloud", "point", "extrapolation_1_over_sqrt_n",
                                         "detectable_change_between_two_stages", "planted_shift_sanity", "seconds")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
