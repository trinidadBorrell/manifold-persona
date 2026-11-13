"""Challenge checks for the coin-flip and preference-probe claims (CPU, saved files only; post-hoc, report only).

Written after the results were seen, for /challenge-result on 13/11/2026. Nothing here is a pre-registered test.
It recomputes the reported numbers with its own code (no import from coinflip.py or pref_probe.py for the
point estimates) and adds checks for contrary evidence and open alternatives.

  .venv/bin/python experiments/q1_emergence/261112_coinflip_stages/challenge_checks_261113.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from length_check import ols_cluster  # noqa: E402  (inference only)

T = 2.2621571627409915
STAGES = ["base", "sft", "dpo", "instruct"]
tasks = json.loads((HERE / "tasks.json").read_text())
Lh = np.array([len(s) for s in tasks["harmless"]])
Lx = np.array([len(s) for s in tasks["harmful"]])


def load(path):
    return json.loads(Path(path).read_text())["items"]


def probe_pairs(items):
    """Own recompute: per pair, log-odds in each order and their mean. Returns arrays indexed h*10 + x."""
    a, b = np.full(100, np.nan), np.full(100, np.nan)  # harmless as A, harmless as B
    for it in items:
        k = it["h"] * 10 + it["x"]
        if it["harmless_first"]:
            a[k] = np.log(it["pA"]) - np.log(it["pB"])
        else:
            b[k] = np.log(it["pB"]) - np.log(it["pA"])
    assert not np.isnan(a).any() and not np.isnan(b).any()
    return a, b, (a + b) / 2


def coin_pairs(items):
    """Own recompute: per pair, mean of 2 P(harmless) - 1 over its 4 items; also the two harmless_first halves."""
    tot, h1, h0 = np.zeros(100), np.zeros(100), np.zeros(100)
    for it in items:
        q = it["pH"][-1] / (it["pH"][-1] + it["pT"][-1])
        d = 2 * (q if it["harmless_heads"] else 1 - q) - 1
        k = it["h"] * 10 + it["x"]
        tot[k] += d / 4
        (h1 if it["harmless_first"] else h0)[k] += d / 2
    return tot, h1, h0


def sb(r):
    return 2 * r / (1 + r)


def mean_cluster(v):
    """Two-way cluster SE of a mean over the 100 pairs (own code)."""
    h, x = np.arange(100) // 10, np.arange(100) % 10
    e = v - v.mean()
    vh = 10 / 9 * sum(e[h == g].sum() ** 2 for g in range(10)) / 100 ** 2
    vx = 10 / 9 * sum(e[x == g].sum() ** 2 for g in range(10)) / 100 ** 2
    vp = 100 / 99 * (e ** 2).sum() / 100 ** 2
    var = vh + vx - vp
    se = float(np.sqrt(var if var > 0 else max(vh, vx)))
    return float(v.mean()), se


def main():
    out = dict(note="post-hoc challenge checks, report only", probe={}, coin={}, link={})
    h, x = np.arange(100) // 10, np.arange(100) % 10
    z = (Lh[h] - Lx[x]) / 100.0
    for run in ("full", "interview"):
        for s in STAGES:
            items = load(HERE / f"results/pref_probe/{run}/{s}.json")
            a, b, m = probe_pairs(items)
            mean, se = mean_cluster(m)
            by_x = np.array([m[x == k].mean() for k in range(10)])
            by_h = np.array([m[h == k].mean() for k in range(10)])
            drop_x = [m[x != k].mean() for k in range(10)]
            drop_h = [m[h != k].mean() for k in range(10)]
            (ia, ib), (sa, sbb) = ols_cluster(m, z, h, x)
            r_orders = float(np.corrcoef(a, b)[0, 1])
            wins = int(sum((it["pA"] > it["pB"]) == (it["harmless_letter"] == "A") for it in items))
            lose = [(tasks["harmless"][it["h"]][:40], tasks["harmful"][it["x"]][:40], it["harmless_letter"])
                    for it in items if (it["pA"] > it["pB"]) != (it["harmless_letter"] == "A")]
            out["probe"][f"{run}__{s}"] = dict(
                mean=mean, ci=[mean - T * se, mean + T * se], wins=wins,
                mean_harmless_as_A=float(a.mean()), mean_harmless_as_B=float(b.mean()),
                per_harmful_task=by_x.round(3).tolist(), per_harmless_task=by_h.round(3).tolist(),
                n_harmful_tasks_negative=int((by_x < 0).sum()), n_harmless_tasks_negative=int((by_h < 0).sum()),
                drop_one_task_range=[float(min(drop_x + drop_h)), float(max(drop_x + drop_h))],
                at_equal_length=float(ia), at_equal_length_ci=[float(ia - T * sa), float(ia + T * sa)],
                slope_per_100_chars=float(ib), slope_ci=[float(ib - T * sbb), float(ib + T * sbb)],
                pair_sd=float(m.std(ddof=1)), r_between_orders=r_orders, reliability_pair_mean=float(sb(r_orders)),
                harmful_letter_wins_examples=lose[:6], n_harmful_letter_wins=len(lose))
    summ = json.loads((HERE / "results/full/summary.json").read_text())
    for f in sorted((HERE / "results/full").glob("*__*.json")):
        tot, h1, h0 = coin_pairs(load(f))
        mean, se = mean_cluster(tot)
        r_half = float(np.corrcoef(h1, h0)[0, 1])
        by_x = np.array([tot[x == k].mean() for k in range(10)])
        by_h = np.array([tot[h == k].mean() for k in range(10)])
        out["coin"][f.stem] = dict(
            mean=mean, ci=[mean - T * se, mean + T * se], reported=summ["cells"][f.stem]["bias"],
            reported_ci=summ["cells"][f.stem]["ci"], pair_sd=float(tot.std(ddof=1)),
            r_between_halves=r_half, reliability_pair_mean=float(sb(r_half)),
            n_harmful_tasks_negative=int((by_x < 0).sum()), n_harmless_tasks_negative=int((by_h < 0).sum()))
    out["coin_changes_sft_to_dpo"] = {k: dict(change=v["change"], ci=v["ci"], excludes_0=v["excludes_0"])
                                      for k, v in summ["changes"].items() if k.startswith("sft->dpo")}
    out["coin_all_cells"] = {k: dict(bias=v["bias"], ci=v["ci"], present=v["present"]) for k, v in summ["cells"].items()}
    t80 = stats.t.ppf(0.80, 9)
    for run in ("full", "interview"):
        for s in ("sft", "dpo", "instruct"):
            _, _, pref = probe_pairs(load(HERE / f"results/pref_probe/{run}/{s}.json"))
            coin, _, _ = coin_pairs(load(HERE / f"results/full/{s}__open_user_turn.json"))
            r = float(np.corrcoef(pref, coin)[0, 1])
            zp, zc = (pref - pref.mean()) / pref.std(), (coin - coin.mean()) / coin.std()
            (_, sl), (_, se) = ols_cluster(zc, zp, h, x)
            rel = (out["probe"][f"{run}__{s}"]["reliability_pair_mean"]
                   * out["coin"][f"{s}__open_user_turn"]["reliability_pair_mean"])
            out["link"][f"{run}__{s}"] = dict(
                pair_r=r, spearman=float(stats.spearmanr(pref, coin)[0]), se=float(se),
                r_needed_for_80pct_power=float((T + t80) * se),
                attenuation_factor=float(np.sqrt(rel)) if rel > 0 else None,
                r_if_true_r_were_0p31=float(0.31 * np.sqrt(rel)) if rel > 0 else None)
    (HERE / "results/challenge_checks_261113.json").write_text(json.dumps(out, indent=1))
    print("PROBE (own recompute)")
    for k, v in out["probe"].items():
        print(f" {k:20} mean {v['mean']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] wins {v['wins']} | asA {v['mean_harmless_as_A']:+.2f} "
              f"asB {v['mean_harmless_as_B']:+.2f} | neg tasks x {v['n_harmful_tasks_negative']} h {v['n_harmless_tasks_negative']} | "
              f"drop1 [{v['drop_one_task_range'][0]:+.2f},{v['drop_one_task_range'][1]:+.2f}] | eq-len {v['at_equal_length']:+.2f} "
              f"[{v['at_equal_length_ci'][0]:+.2f},{v['at_equal_length_ci'][1]:+.2f}] slope {v['slope_per_100_chars']:+.2f} "
              f"[{v['slope_ci'][0]:+.2f},{v['slope_ci'][1]:+.2f}] | sd {v['pair_sd']:.2f} rel {v['reliability_pair_mean']:.2f}")
    print("COIN (own recompute vs summary.json)")
    for k, v in out["coin"].items():
        print(f" {k:32} {v['mean']:+.4f} vs {v['reported']:+.4f} | ci [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] vs "
              f"[{v['reported_ci'][0]:+.3f},{v['reported_ci'][1]:+.3f}] | pair sd {v['pair_sd']:.3f} rel {v['reliability_pair_mean']:.2f} | "
              f"neg tasks x {v['n_harmful_tasks_negative']} h {v['n_harmless_tasks_negative']}")
    print("COIN sft->dpo changes, all frames")
    for k, v in out["coin_changes_sft_to_dpo"].items():
        print(f" {k:36} {v['change']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] excl0 {v['excludes_0']}")
    print("LINK")
    for k, v in out["link"].items():
        print(f" {k:20} r {v['pair_r']:+.2f} spearman {v['spearman']:+.2f} se {v['se']:.2f} | r for 80% power {v['r_needed_for_80pct_power']:.2f} | "
              f"attenuation {v['attenuation_factor']} | observed r if true r 0.31: {v['r_if_true_r_were_0p31']}")
    for k in ("interview__sft", "full__sft"):
        print(k, "harmful letter wins:", out["probe"][k]["n_harmful_letter_wins"], out["probe"][k]["harmful_letter_wins_examples"][:4])


if __name__ == "__main__":
    main()
