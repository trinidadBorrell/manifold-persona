"""Reproduce Giovanni Marraffini's layer-23 pinned steering for one persona, with HIS code.

His branch (origin/persona-definition @ b860780) is exported unchanged to <out>/repo (git archive)
and his modules are run from there:
  targets   persona_lora.gen_targets  --limit_questions 120 --n_phrasings 5 --max_new_tokens 160
                                      --dtype fp16 --seed 0          (63_encode.sbatch)
  acts      persona_lora.extract_acts --max_len 384 --dtype fp16    (63_encode.sbatch)
  vectors   attr_analysis/build_l23.py, arm L23 only: M[p] = mean of p's prompted response-token
            points (loader.load_all); u = (M[t,23] - M[default,23]) / norm; a = M[default,23] . u;
            d = ||M[t,23] - M[default,23]||
  steer     persona_lora.steer_pin  --layers 23 --dtype fp16, s in {3, 4, 6, 8}  (l23b.sbatch)
            + an unsteered baseline (add-mode cell with a zero vector, s = 0)
  ceiling   persona_lora.steer_pin2 cell prompt|<persona> (role instruction 0, no steering)  (pb.sbatch)

The only change: this cluster's transformers (4.53) takes `torch_dtype`, his code passes `dtype`;
a shim maps one to the other before his modules import transformers.

One condor DAG runs it in parallel (jobs_condor/gio_repro.dag): targets+acts per persona and for
`default` at once, then vectors, then the steering (5 shards) and the ceiling at once.
Layout: <root>/repo (his tree), <root>/<who>/{targets,acts}_<who>.*, <root>/<persona>/{vectors,gen_*}.

    python -m giovanni_repro.run --root $PM/GIO --persona spy --stage targets --who spy
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

BRANCH = "origin/persona-definition"
REPO = Path(__file__).resolve().parents[2]
STRENGTHS = [3.0, 4.0, 6.0, 8.0]
L = 23


def export(out):
    d = out / "repo"
    if not (d / "persona_lora").exists():
        d.mkdir(parents=True, exist_ok=True)
        ar = subprocess.run(["git", "-C", str(REPO), "archive", BRANCH], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=ar, check=True)
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", BRANCH], check=True, capture_output=True, text=True).stdout
        (d / "EXPORTED_FROM.txt").write_text("%s %s\n" % (BRANCH, sha.strip()))
    return d


def shim():
    import transformers
    from transformers import AutoModelForCausalLM
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 56):
        orig = AutoModelForCausalLM.from_pretrained.__func__

        def fp(cls, *a, **k):
            if "dtype" in k and "torch_dtype" not in k:
                k["torch_dtype"] = k.pop("dtype")
            return orig(cls, *a, **k)
        AutoModelForCausalLM.from_pretrained = classmethod(fp)


def his(repo, module, argv):
    """Run one of his modules as `python -m module argv`, from his tree."""
    sys.path[:0] = [str(repo), str(repo / "src")]
    os.chdir(repo)
    sys.argv = [module] + argv
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--shard", default="0/1", help="steer: his --shard i/n")
    ap.add_argument("--stage", required=True, choices=["export", "targets", "acts", "vectors", "steer", "ceiling"])
    ap.add_argument("--who", default=None, help="targets/acts: persona or default")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / a.persona
    out.mkdir(parents=True, exist_ok=True)
    repo = export(root)
    if a.stage == "export":
        print("exported", repo); return
    shim()
    t = a.persona
    w = root / (a.who or t)
    w.mkdir(exist_ok=True)
    if a.stage == "targets":
        (w / ("personas_%s.json" % a.who)).write_text(json.dumps([a.who]))
        his(repo, "persona_lora.gen_targets", ["--model", "Qwen/Qwen2.5-3B-Instruct",
            "--personas", str(w / ("personas_%s.json" % a.who)), "--limit_questions", "120", "--n_phrasings", "5",
            "--max_new_tokens", "160", "--dtype", "fp16", "--seed", "0", "--out", str(w / ("targets_%s.jsonl" % a.who))])
    elif a.stage == "acts":
        his(repo, "persona_lora.extract_acts", ["--model", "Qwen/Qwen2.5-3B-Instruct",
            "--targets", str(w / ("targets_%s.jsonl" % a.who)), "--out", str(w / ("acts_%s.npz" % a.who)),
            "--max_len", "384", "--dtype", "fp16"])
    elif a.stage == "vectors":
        def pts(p):   # question shards (shard.py) if present, else his single extract_acts file
            fs = sorted((root / p / "shards").glob("acts_*.npz"))
            return np.concatenate([np.load(f)["prompted"] for f in fs]) if fs else np.load(root / p / ("acts_%s.npz" % p))["prompted"]
        X = {p: pts(p) for p in (t, "default")}
        M = {p: X[p].astype(np.float32).mean(0) for p in (t, "default")}
        base = M["default"][L]
        true = M[t][L] - base
        u = true / np.linalg.norm(true)
        k = "pin|L23_%s|L23" % t
        Z = {k: u, k + "|a": np.array(float(base @ u)), k + "|d": np.array(float(np.linalg.norm(true))),
             "add|L23_none|L23": np.zeros_like(u)}
        np.savez_compressed(out / "vectors.npz", **Z)
        (out / "cells.json").write_text(json.dumps([["pin|L23_%s" % t, STRENGTHS], ["add|L23_none", [0.0]]]))
        (out / "cells_ceiling.json").write_text(json.dumps([["prompt|%s" % t, [0.0]]]))
        (out / "questions.json").write_text((repo / "pb" / "pb_questions.json").read_text())
        meta = dict(persona=t, layer=L, d=float(np.linalg.norm(true)), a=float(base @ u),
                    n_points={p: int(X[p].shape[0]) for p in (t, "default")})
        (out / "vectors_meta.json").write_text(json.dumps(meta, indent=1))
        print(meta)
    elif a.stage == "steer":
        his(repo, "persona_lora.steer_pin", ["--vectors", str(out / "vectors.npz"), "--cells", str(out / "cells.json"),
            "--layers", "23", "--dtype", "fp16", "--shard", a.shard,
            "--out", str(out / ("gen_steer_%s.jsonl" % a.shard.split("/")[0]))])
    elif a.stage == "ceiling":
        his(repo, "persona_lora.steer_pin2", ["--vectors", str(out / "vectors.npz"), "--cells", str(out / "cells_ceiling.json"),
            "--questions", str(out / "questions.json"), "--layers", "23", "--dtype", "fp16",
            "--out", str(out / "gen_ceiling.jsonl")])


if __name__ == "__main__":
    main()
