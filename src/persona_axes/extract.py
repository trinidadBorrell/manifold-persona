"""Phase 0 extraction (plan v2 s2.3, s3): answers + residual states for the trait pool, the 35 axes and
the set (a) personas. Same engine as E1 (pm_geometry.extract_states: greedy, Qwen3 template, thinking off,
teacher-forced replay with hooks on hidden_states 12-32, `last` and `mean` views kept per answer).

    python -m persona_axes.extract --part traits --shard 0 --n-shards 32      # 240 x 2 poles x 40 q, cap 256
    python -m persona_axes.extract --part axes   --shard 0 --n-shards 4       # 35 x 2 poles x 40 q + 40 neutral, cap 256
    python -m persona_axes.extract --part seta   --shard 0 --n-shards 4       # 41 personas x 40 E1 questions (rot40), cap 1024

traits / axes: question j (sorted id) answered under prompt pair j mod 5, pole pos = pole_a, neg = pole_b.
seta: exactly the E1 design (40 E1 questions, instruction j mod 5, cap 1024) so the E1 neutral answers apply.
Output: $PM/AX21/<part>/chunks/*.npz|csv, resumable by chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from manifold_persona.prompts_roles import load_questions

from persona_axes import AX, HERE, LAYERS, NEUTRAL

CAP = {"traits": 256, "axes": 256, "seta": 1024}


def _questions(part):
    q = json.loads((HERE / "questions.json").read_text())
    ids = q["e1_question_ids"] if part == "seta" else q["trait_question_ids"]
    bank = load_questions()
    return [(i, bank[i]["question"]) for i in ids]


def build_units(part):
    qs = _questions(part)
    units = []
    if part == "traits":
        for t in json.loads((HERE / "traits.json").read_text())["traits"]:
            for pole in ("pos", "neg"):
                for j, (qi, q) in enumerate(qs):
                    units.append(dict(who=t["name"], pole=pole, pair=j % 5, question_idx=qi, question=q,
                                      system=t["pairs"][j % 5][pole]))
    elif part == "axes":
        for j, (qi, q) in enumerate(qs):                       # neutral answers to the trait questions
            units.append(dict(who=NEUTRAL, pole="", pair=-1, question_idx=qi, question=q, system=None))
        gen = json.loads((AX / "prompts" / "axes_prompts.json").read_text())
        for a in json.loads((HERE / "axes.json").read_text())["axes"]:
            pairs = gen[a["name"]]["instruction"]
            for pole in ("pos", "neg"):
                for j, (qi, q) in enumerate(qs):
                    units.append(dict(who=a["name"], pole=pole, pair=j % 5, question_idx=qi, question=q,
                                      system=pairs[j % 5][pole]))
    elif part == "seta":
        gen = json.loads((AX / "prompts" / "personas_a_prompts.json").read_text())
        for p in json.loads((HERE / "personas_a.json").read_text())["personas"]:
            ins = [x["pos"] for x in gen[p["name"]]["instruction"]]
            for j, (qi, q) in enumerate(qs):
                units.append(dict(who=p["name"], pole="", pair=j % 5, question_idx=qi, question=q, system=ins[j % 5]))
    for i, u in enumerate(units):
        u["unit"] = i
    return units


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(CAP))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true", help="count units and show two, no model")
    a = ap.parse_args(argv)
    units = build_units(a.part)
    out = AX / a.part
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    if a.shard == 0:
        (out / "units.json").write_text(json.dumps(dict(part=a.part, n_units=len(units), layers=list(LAYERS),
                                                        max_new_tokens=CAP[a.part], greedy=True)))
    lo, hi = len(units) * a.shard // a.n_shards, len(units) * (a.shard + 1) // a.n_shards
    todo = [units[s:min(s + a.chunk, hi)] for s in range(lo, hi, a.chunk)]
    print("%s: %d units, shard %d/%d = %d units in %d chunks" % (a.part, len(units), a.shard, a.n_shards, hi - lo, len(todo)))
    if a.dry_run:
        print(json.dumps(units[0], indent=1)[:600]); print(json.dumps(units[-1], indent=1)[:600])
        return
    from pm_geometry import extract_states as E                 # adds jobs_condor shims to sys.path
    assert tuple(E.LAYERS) == tuple(LAYERS), E.LAYERS
    tok, model = E.load_model()
    t_all, n_all = time.time(), 0
    for ch in todo:
        name = "u%06d-%06d" % (ch[0]["unit"], ch[-1]["unit"] + 1)
        if (out / "chunks" / (name + ".npz")).exists():
            continue
        rows, L, M = [], [], []
        t0 = time.time()
        for s in range(0, len(ch), a.batch_size):
            r, l, m = E.run_batch(tok, model, ch[s:s + a.batch_size], CAP[a.part])
            rows += r; L.append(l); M.append(m)
        L, M = np.concatenate(L), np.concatenate(M)
        tmp = out / "chunks" / (name + ".tmp.npz")
        np.savez(tmp, last=L, mean=M)
        os.replace(tmp, out / "chunks" / (name + ".npz"))
        pd.DataFrame(rows).to_csv(out / "chunks" / (name + ".csv"), index=False)
        n_all += len(ch)
        nn = np.array([r["n_new"] for r in rows])
        print("%s: %d units %.0fs (%.2f ans/s) | n_new median %d p90 %d | hit_cap %.1f%%" % (
            name, len(ch), time.time() - t0, len(ch) / (time.time() - t0), np.median(nn), np.percentile(nn, 90),
            100 * np.mean([r["hit_cap"] for r in rows])), flush=True)
    print("done: %d units in %.0fs" % (n_all, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
