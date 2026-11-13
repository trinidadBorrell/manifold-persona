"""Modal CPU launcher for the own-reply analysis (rules: THRESHOLDS.md v1.2, approved 13/11/2026; code:
analysis_v12.py). Inputs and outputs live on the outputs volume; outputs go to /out/analysis_v12/. Tokenizers for the
first16 text come from the weights volume (offline). Each step runs inside a remote orchestrator, so a dropped Mac
client does not stop it (run with --detach).

  .venv/bin/modal run --detach experiments/q1_emergence/261112_stage_own_replies/modal_analysis_v12.py --step power
  .venv/bin/modal run --detach experiments/q1_emergence/261112_stage_own_replies/modal_analysis_v12.py --step measure

power: prep (once), the 30 sources, then P1-P3; returns only the power result (no real estimate).
measure: refuses unless the power check passed; returns the readings and copies the small files to results/.

Revision b (after the Codex review): outputs go to /out/analysis_v12b/ and results/analysis_v12b/. The first run's
outputs stay in /out/analysis_v12/. A source is reused only if its fingerprint and array hash match this run.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if modal.is_local() else Path("/root/mp")
ROOT = "/root/mp"
E = "experiments/q1_emergence"
FILES = [f"{E}/261112_stage_own_replies/analysis_v12.py",
         f"{E}/261112_free_checks/check_role_count.py",
         f"{E}/261112_free_checks/check3_lu_scale.py",
         f"{E}/261112_free_checks/check_default_readers.py",
         f"{E}/261029_instruct_persona_map/analysis.py",
         f"{E}/261029_fixed_text_read_through/maps.py",
         f"{E}/261112_han_reanalysis/data/han_repo/src/persona_shift/persona_axes.py",
         f"{E}/261112_modal_gpu/hf_receipt_261113_full.json",
         f"{E}/261112_modal_gpu/hf_receipt_261113_length.json",
         f"{E}/261112_coinflip_stages/stages.json"]
OUT = "/out/analysis_v12b"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.9.0", "pandas==3.0.5",
                      "transformers==4.46.3", "tokenizers==0.20.3")
         .env({"HF_HOME": "/weights/hf", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}))
for f in FILES:
    image = image.add_local_file(REPO / f, f"{ROOT}/{f}")
app = modal.App("persona-own-reply-analysis-261113", image=image)
weights = modal.Volume.from_name("persona-olmo2-7b-weights-261112", version=2)
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)
VOL = {"/weights": weights, "/out": outputs}


def analysis():
    import importlib.util
    import sys
    if "analysis_v12" in sys.modules:
        return sys.modules["analysis_v12"]
    spec = importlib.util.spec_from_file_location("analysis_v12", f"{ROOT}/{FILES[0]}")
    A = importlib.util.module_from_spec(spec)
    sys.modules["analysis_v12"] = A
    spec.loader.exec_module(A)
    return A


def versions():
    import numpy, scipy, sklearn, pandas, torch, transformers, tokenizers, platform
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, pandas=pandas.__version__, torch=torch.__version__,
                transformers=transformers.__version__, tokenizers=tokenizers.__version__)


@app.function(cpu=8, memory=65536, timeout=7200, retries=0, volumes=VOL)
def prep_job():
    from transformers import AutoTokenizer
    A = analysis()
    stages = {e["stage"]: e for e in json.loads(Path(f"{ROOT}/{FILES[9]}").read_text())["7b"]}
    toks = {s: AutoTokenizer.from_pretrained(stages[s]["repo"], revision=stages[s]["revision"], local_files_only=True)
            for s in A.STAGES}
    runtime = dict(versions=versions(),
                   tokenizers={s: dict(repo=stages[s]["repo"], revision=stages[s]["revision"]) for s in A.STAGES})
    try:
        r = A.prep("/out", OUT, [f"{ROOT}/{FILES[7]}", f"{ROOT}/{FILES[8]}"], toks, runtime=runtime)
    finally:
        outputs.commit()
    return r


@app.function(cpu=2, memory=8192, timeout=3600, retries=0, max_containers=30, volumes=VOL)
def source_job(args):
    A = analysis()
    try:
        return A.source("/out", OUT, *args, runtime=dict(versions=versions()))
    finally:
        outputs.commit()


@app.function(cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOL)
def power_job():
    A = analysis()
    try:
        return A.power(OUT)
    finally:
        outputs.commit()


@app.function(cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOL)
def measure_job():
    A = analysis()
    try:
        A.measure(OUT)
    finally:
        outputs.commit()
    return small_files()


def small_files():
    out = {}
    for p in sorted(Path(OUT).rglob("*.json")):
        if p.name != "meta.json":
            out[str(p.relative_to(OUT))] = p.read_text()
    return out


def power_summary(pw):
    """Only what the power check may show: pass flags and P3 sizes; no real estimate."""
    gated = {k: dict(P1=all(x["within_10pct"] and x["spread_unchanged"] for x in pw["P1"][k].values()),
                     P2=all(x["within_001"] for x in pw["P2"][k].values())) for k in pw["gated_sources"]}
    others = sorted(k for k in pw["P1"] if k not in pw["gated_sources"]
                    and not all(x["within_10pct"] for x in pw["P1"][k].values()))
    return dict(passed=pw["passed"], gated=gated, report_only_sources_failing_P1=others,
                P3={k: dict(rates=v["rates"], smallest_with_80pct=v["smallest_with_80pct"]) for k, v in pw["P3"].items()})


@app.function(cpu=1, memory=2048, timeout=10800, retries=0, volumes=VOL)
def orchestrate(step):
    A = analysis()
    if step == "power":
        outputs.reload()
        prep = None
        if not Path(OUT, "prep.json").exists():
            prep = prep_job.remote()
            print("prep:", json.dumps(prep, default=str), flush=True)
        outputs.reload()
        if not json.loads(Path(OUT, "prep.json").read_text())["first16"]["support"]["passed"]:
            raise RuntimeError("first16 support failed; see prep.json")
        prep_rec = A.read_prep(OUT)
        todo = [s for s in A.all_sources() if A.source_problem(OUT, prep_rec, *s)]
        print(f"sources to run: {len(todo)} of {len(A.all_sources())}", flush=True)
        failed = []
        for r in source_job.map(todo, return_exceptions=True):
            if isinstance(r, BaseException):
                failed.append(repr(r))
            print("source:", r if not isinstance(r, BaseException) else repr(r), flush=True)
        if failed:
            raise RuntimeError(f"{len(failed)} sources failed: {failed[:3]}")
        pw = power_job.remote()
        outputs.reload()
        return dict(prep=prep, prep_json=Path(OUT, "prep.json").read_text(), power=power_summary(pw),
                    power_json=Path(OUT, "power.json").read_text())
    if step == "measure":
        return measure_job.remote()
    raise ValueError(step)


@app.local_entrypoint()
def main(step: str = "power"):
    res = orchestrate.remote(step)
    dest = HERE / "results/analysis_v12b"
    dest.mkdir(parents=True, exist_ok=True)
    if step == "power":
        (dest / "power.json").write_text(res["power_json"])
        (dest / "prep.json").write_text(res["prep_json"])
        prep = json.loads(res["prep_json"])
        print(json.dumps(dict(prep={k: prep.get(k) for k in ("n_rows", "n_valid_all_stages", "text", "seconds")},
                              first16={k: v for k, v in prep["first16"].items() if k != "support"},
                              support={k: v for k, v in prep["first16"]["support"].items() if k != "rows_per_cloud"},
                              power=res["power"]), indent=1, default=str))
    else:
        for name, text in res.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_text(text)
        print(f"wrote {len(res)} files to {dest}")
