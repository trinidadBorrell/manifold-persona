"""Numbers and example replies for the 28/10/2026 ex2 explainer page (no new scoring; reads saved results).

    python figure_data.py   # from 261008_base_persona_map; writes ../261023_judge_designs/results/figure_data.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import score_naming as S  # noqa: E402

JD = Path("../261023_judge_designs/results")
LOOP = json.loads((JD / "naming_loop_nonsense.json").read_text())
B16 = json.loads((JD / "naming_broken_masks.json").read_text())
SEEDS = json.loads((JD / "boot_seed_loop.json").read_text())
LABELS = {tuple(r["reply_key"].split("/")): r for r in map(json.loads, open("results/broken/finish_261016/labels.jsonl"))}
MASKS = {m: {tuple(x) for x in json.loads((JD / "masks" / f"{m}.json").read_text())}
         for m in ("final", "loop_nonsense")}


def label(run, i):
    return LABELS[(run, str(i))]


out = {}

# 1. hit@5 by condition, run and judge (all replies)
out["by_condition"] = [
    dict(run=run, cond=cond, judge=j, hit5=r["all_replies"]["macro_hit5"], ci=r["all_replies"]["ci95"],
         null_mean=r["all_replies"]["null_mean"])
    for run in ("run_main", "run_rep") for cond in ("desc", "ex2") for j in ("gpt", "claude")
    for r in [LOOP[f"{run}/{cond}"]["results"]["none"][j]]]

# 4. rule-4 check (complete cells, copied + self-naming removed) per broken list, GPT and Claude judge
lists = [("none", "No reply counted as broken", LOOP), ("frozen", "Pre-set list (token limit or repeated 4-gram)", LOOP),
         ("loop_nonsense", "Loops and nonsense only", LOOP), ("final", "16 Oct definition (owner + GPT)", B16),
         ("gpt", "16 Oct definition (GPT)", B16), ("jev", "16 Oct definition (Jev)", B16),
         ("any", "16 Oct definition (any judge)", B16)]
out["by_list"] = []
for key, name, src in lists:
    for run in ("run_main", "run_rep"):
        res = src[f"{run}/ex2"]["results"][key]
        out["by_list"].append(dict(
            list=key, name=name, run=run,
            gpt_lb=res["gpt"]["complete_cells_clean"]["ci95"][0], claude_lb=res["claude"]["complete_cells_clean"]["ci95"][0],
            gpt_point=res["gpt"]["complete_cells_clean"]["macro_hit5"], broken_rate=res["gpt"]["invalid_rate"],
            rules=[res["both_judges"][k] for k in ("rule1", "rule2", "rule3", "rule4")], plays=res["both_judges"]["plays"]))

out["seeds"] = SEEDS
out["primary"] = LOOP["primary_decisions"]
out["holm"] = LOOP["holm"]["adjusted"]

key, by_run, guesses = S.load()
ids_of = {(v["run"], v["i"]): k for k, v in key.items()}

# 2. hit rate of 16-Oct-broken vs fine ex2 replies, 3. reasons, 6. per-role
out["broken_vs_fine"], out["reasons"], out["roles"] = [], {}, {}
for run in ("run_main", "run_rep"):
    for cond in ("ex2", "desc"):
        G, G1, F, donor, ids = S.cube(key, by_run, guesses, run, cond)
        H = {j: S.target_hits(G[j]) for j in G}
        if cond == "ex2":
            inv = np.vectorize(lambda x: tuple(x) in MASKS["final"], otypes=[bool])(ids)
            for j in G:
                out["broken_vs_fine"].append(dict(run=run, judge=j, n_broken=int(inv.sum()), n_fine=int((~inv).sum()),
                                                  hit_broken=float(H[j][inv].mean()), hit_fine=float(H[j][~inv].mean())))
            reasons = collections.Counter(label(*x)["reason"] for x in ids.ravel() if tuple(x) in MASKS["final"])
            out["reasons"][run] = dict(reasons)
            out["translator_kept"] = out.get("translator_kept", {})
            comp = ~inv.any(2, keepdims=True)
            clean = ~(F["copied"] | F["selfname"])
            t = S.ROLES.index("translator")
            out["translator_kept"][run] = dict(total=90, kept=int((np.broadcast_to(comp, inv.shape) & ~inv & clean)[t].sum()),
                                               broken=int(inv[t].sum()))
        if run == "run_main":
            out["roles"][cond] = {j: dict(zip(S.ROLES, H[j].mean((1, 2)).round(4).tolist())) for j in G}

# 7. example replies (run_main unless noted); first match by item index, reply length 120-900 characters


def example(tag, run, cond, role=None, reason=None, hit=None, broken=None, selfname=None, copied=None):
    G, G1, F, donor, ids = S.cube(key, by_run, guesses, run, cond)
    rows = []
    for a in range(20):
        for b in range(18):
            for w in range(5):
                r = by_run[ids[a, b, w]]
                lab = label(*ids[a, b, w])
                g = guesses["gpt"][ids_of[ids[a, b, w]]]
                is_hit = S.ROLES[a] in g
                if role and S.ROLES[a] != role:
                    continue
                if reason and lab["reason"] != reason:
                    continue
                if broken is not None and lab["broken"] != broken:
                    continue
                if hit is not None and is_hit != hit:
                    continue
                if selfname is not None and bool(F["selfname"][a, b, w]) != selfname:
                    continue
                if copied is not None and bool(F["copied"][a, b, w]) != copied:
                    continue
                if not 120 <= len(r["reply"] or "") <= 900:
                    continue
                rows.append((r["i"], a, b, w, r, lab, g))
    i, a, b, w, r, lab, g = sorted(rows, key=lambda x: x[0])[0]
    question = r["prompt"].rsplit("Interviewer:", 1)[-1].rsplit("Guest:", 1)[0].strip()
    return dict(tag=tag, run=run, cond=cond, i=i, role=S.ROLES[a], wording=w, question=question, prompt=r["prompt"],
                reply=r["reply"], ended_by=r["ended_by"], gpt_top5=g, claude_top5=guesses["claude"][ids_of[(run, i)]],
                gpt_hit=S.ROLES[a] in g, claude_hit=S.ROLES[a] in guesses["claude"][ids_of[(run, i)]],
                broken_16oct=bool(lab["broken"]), reason_16oct=lab["reason"], label_source=lab["label_source"],
                broken_loop_list=(run, i) in MASKS["loop_nonsense"], copied=bool(F["copied"][a, b, w]),
                selfname=bool(F["selfname"][a, b, w]), limit=bool(F["limit"][a, b, w]), rep4=bool(F["rep"][a, b, w]))


out["examples"] = [
    example("played", "run_main", "ex2", role="pirate", hit=True, broken=False, selfname=False, copied=False),
    example("played", "run_main", "ex2", role="toddler", hit=True, broken=False, selfname=False, copied=False),
    example("not_played", "run_main", "ex2", role="detective", hit=False, broken=False),
    example("broken_16oct_hit", "run_main", "ex2", role="translator", reason="format", hit=True),
    example("broken_16oct_hit", "run_main", "ex2", role="narcissist", reason="off_question", hit=True),
    example("broken_16oct_hit", "run_main", "ex2", role="bartender", reason="off_question", hit=True),
    example("real_loop", "run_main", "ex2", reason="loop"),
    example("selfname", "run_main", "ex2", role="robot", hit=True, broken=False, selfname=True),
    example("desc_miss", "run_main", "desc", role="chef", hit=False, broken=False),
]
(JD / "figure_data.json").write_text(json.dumps(out, indent=1))
print("wrote", JD / "figure_data.json", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
