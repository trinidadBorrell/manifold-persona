"""Checks that answer Codex findings 3, 4 and 5 on the coin-flip run (CPU, saved files only; coinflip.py is unchanged).

Written after the results were seen (13/11/2026); report only.
- Finding 5: all 17 cells are local, match the HF receipt (SHA-256, size), and come from one run (same code, tasks,
  item order, outcome ids, library versions; one pinned revision per stage).
- Finding 4: the bias from the model's own output probabilities (model_pH, model_pT), beside the float32 lens value
  that the analysis uses. This comparison can fail: the tolerance is 0.005 on the bias scale.
- Finding 3: Holm-adjusted p values (t, df 9) for three test families.

  .venv/bin/python experiments/q1_emergence/261112_coinflip_stages/codex_fix_checks_261113.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coinflip import MODES, STAGES, T975_DF9, cluster_se, modes_for  # noqa: E402

RUN = HERE / "results/full"
RECEIPT = HERE.parent / "261112_modal_gpu/hf_receipt_261113.json"
TOL = 0.005


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def bias_from(items, kh, kt, last):
    pH = np.array([it[kh][-1] if last else it[kh] for it in items])
    pT = np.array([it[kt][-1] if last else it[kt] for it in items])
    q = pH / (pH + pT)
    d = 2 * np.where([it["harmless_heads"] for it in items], q, 1 - q) - 1
    se, _ = cluster_se(d, [it["h"] for it in items], [it["x"] for it in items])
    return d, float(d.mean()), se


def main():
    cells = [f"{s}__{m}" for s in STAGES for m in modes_for(s)]
    assert len(cells) == 17
    receipt = {Path(f["path"]).name: f for f in json.loads(RECEIPT.read_text())["files"]
               if "261112_coinflip_stages/run_full/" in f["path"]}
    out = dict(note="post-hoc checks for Codex findings 3, 4, 5; report only", provenance={}, native={}, holm={})
    metas, ids, ok_all = {}, {}, True
    for c in cells:
        f = RUN / f"{c}.json"
        raw = f.read_bytes()
        rec = json.loads(raw)
        metas[c], ids[c] = rec["meta"], [it["id"] for it in rec["items"]]
        r = receipt.get(f.name)
        match = r is not None and r["sha256"] == hashlib.sha256(raw).hexdigest() and r["size"] == len(raw)
        ok_all &= match
        out["provenance"][c] = dict(local=True, matches_hf_receipt=bool(match), n_items=rec["meta"]["n_items"],
                                    revision=rec["meta"]["revision"])
    same = {k: len({json.dumps(m[k], sort_keys=True) for m in metas.values()}) == 1
            for k in ("coinflip_py_sha256", "tasks_json_sha256", "outcome_ids", "torch", "transformers", "n_items",
                      "dtype", "batch_size", "tokenizer_json_sha256", "platform", "device")}
    per_stage_rev = {s: sorted({m["revision"] for c, m in metas.items() if c.startswith(s + "__")}) for s in STAGES}
    pinned = {e["stage"]: e["revision"] for e in json.loads((HERE / "stages.json").read_text())["7b"]}
    out["one_run"] = dict(all_17_local_and_match_hf=bool(ok_all), same_across_cells=same,
                          same_item_order=len({tuple(v) for v in ids.values()}) == 1,
                          one_pinned_revision_per_stage=all(v == [pinned[s]] for s, v in per_stage_rev.items()),
                          code_hash_is_current_coinflip_py=next(iter(metas.values()))["coinflip_py_sha256"]
                          == hashlib.sha256((HERE / "coinflip.py").read_bytes()).hexdigest())
    summ = json.loads((RUN / "summary.json").read_text())
    worst = 0.0
    for c in cells:
        items = json.loads((RUN / f"{c}.json").read_text())["items"]
        dl, bl, sl = bias_from(items, "pH", "pT", True)
        dn, bn, sn = bias_from(items, "model_pH", "model_pT", False)
        gap = abs(bl - bn)
        worst = max(worst, gap)
        assert abs(bl - summ["cells"][c]["bias"]) < 1e-12
        out["native"][c] = dict(lens_bias=bl, lens_ci=[bl - T975_DF9 * sl, bl + T975_DF9 * sl],
                                native_bias=bn, native_ci=[bn - T975_DF9 * sn, bn + T975_DF9 * sn],
                                abs_gap=gap, max_item_gap=float(np.abs(dl - dn).max()),
                                same_reading=bool((bl - T975_DF9 * sl > 0 or bl + T975_DF9 * sl < 0)
                                                  == (bn - T975_DF9 * sn > 0 or bn + T975_DF9 * sn < 0)),
                                within_tolerance=bool(gap <= TOL))
    out["native_check"] = dict(tolerance=TOL, worst_abs_gap=worst, passed=bool(worst <= TOL),
                               all_readings_same=all(v["same_reading"] for v in out["native"].values()))
    p_cell = {c: float(2 * stats.t.sf(abs(v["bias"] / v["se"]), 9)) for c, v in summ["cells"].items()}
    p_chg = {c: float(2 * stats.t.sf(abs(v["change"] / v["se"]), 9)) for c, v in summ["changes"].items()}
    prim_cells = [f"{s}__{m}" for m in ("plaintext", "open_user_turn") for s in STAGES if f"{s}__{m}" in p_cell]
    prim_chg = [c for c in p_chg if c.split("__", 1)[1] in ("plaintext", "open_user_turn")]
    fams = {"7 primary cells": {c: p_cell[c] for c in prim_cells},
            "12 primary cells and changes": {**{c: p_cell[c] for c in prim_cells}, **{c: p_chg[c] for c in prim_chg}},
            "29 cells and changes": {**p_cell, **p_chg}}
    for name, fam in fams.items():
        keys = list(fam)
        adj = holm(np.array([fam[k] for k in keys]))
        out["holm"][name] = dict(n=len(keys), **{k: dict(p=fam[k], holm_p=float(a)) for k, a in zip(keys, adj)
                                                  if k in ("dpo__open_user_turn", "instruct__open_user_turn",
                                                           "sft->dpo__open_user_turn", "sft->dpo__plaintext")})
    (HERE / "results/codex_fix_checks_261113.json").write_text(json.dumps(out, indent=1))
    print("FINDING 5", json.dumps(out["one_run"], indent=1))
    print("FINDING 4: native output vs lens")
    for c, v in out["native"].items():
        print(f" {c:32} lens {v['lens_bias']:+.4f} [{v['lens_ci'][0]:+.3f},{v['lens_ci'][1]:+.3f}] | native {v['native_bias']:+.4f} "
              f"[{v['native_ci'][0]:+.3f},{v['native_ci'][1]:+.3f}] | gap {v['abs_gap']:.4f} item max {v['max_item_gap']:.3f} | same reading {v['same_reading']}")
    print(" ", out["native_check"])
    print("FINDING 3: Holm")
    for name, v in out["holm"].items():
        print(f" {name} (n {v['n']}):", {k: (round(x['p'], 4), round(x['holm_p'], 4)) for k, x in v.items() if k != 'n'})


if __name__ == "__main__":
    main()
