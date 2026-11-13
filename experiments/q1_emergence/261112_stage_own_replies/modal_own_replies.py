"""Modal launcher: SFT and DPO write their own replies on pod A's grid (rules: THRESHOLDS.md), in parallel on two
A100-80GB, then pod A's gates. Weights come from the volume that ../261112_modal_gpu/modal_runs.py filled; outputs go
to the shared outputs volume under own_replies/.

  .venv/bin/modal run experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py --step smoke
  .venv/bin/modal run --detach experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py --step full
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if modal.is_local() else Path("/root/mp")
ROOT = "/root/mp"
EXP = f"{ROOT}/experiments/q1_emergence"
GPU = "A100-80GB"
TIMEOUT = {"smoke": 1800, "full": 10800}
POD_A = "experiments/q1_emergence/261029_instruct_persona_map"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0+cu129", index_url="https://download.pytorch.org/whl/cu129")
         .pip_install("transformers==4.47.1", "tokenizers==0.21.0", "numpy==2.5.2", "huggingface_hub==0.36.2",
                      "safetensors==0.8.0", "hf-xet==1.7.1")
         .env({"HF_HOME": "/weights/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
         .add_local_file(REPO / "src/manifold_persona/rows.py", f"{ROOT}/src/manifold_persona/rows.py")
         .add_local_file(REPO / POD_A / "gen_extract.py", f"{ROOT}/{POD_A}/gen_extract.py")
         .add_local_file(REPO / POD_A / "gates.py", f"{ROOT}/{POD_A}/gates.py")
         .add_local_file(REPO / "experiments/q1_emergence/261008_base_persona_map/questions_40.json",
                         f"{EXP}/261008_base_persona_map/questions_40.json")
         .add_local_file(REPO / "experiments/q1_emergence/261112_coinflip_stages/stages.json",
                         f"{EXP}/261112_coinflip_stages/stages.json")
         .add_local_file(REPO / "refs/assistant-axis/extraction_questions.jsonl",
                         f"{ROOT}/refs/assistant-axis/extraction_questions.jsonl")
         .add_local_dir(REPO / "refs/assistant-axis/roles/instructions", f"{ROOT}/refs/assistant-axis/roles/instructions")
         .add_local_file(HERE / "stage_run.py", f"{EXP}/261112_stage_own_replies/stage_run.py"))
app = modal.App("persona-own-replies-261112", image=image)
weights = modal.Volume.from_name("persona-olmo2-7b-weights-261112", version=2)
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)


SMOKE = "smoke_v2"  # the code test for the v1.1 wrapper; the first smoke (old wrapper) stays on the volume and HF


def run(cmd, log):
    """Stream a child process to the Modal log and a file; commit the outputs volume every 5 minutes. Any non-zero
    exit, or a failed commit, raises."""
    import os
    import subprocess
    import sys
    import threading
    stop, errors = threading.Event(), []

    def committer():
        while not stop.wait(300):
            try:
                outputs.commit()
            except Exception as e:  # noqa: BLE001  (reported after the child ends)
                errors.append(repr(e))

    t = threading.Thread(target=committer, daemon=True)
    t.start()
    print("$", " ".join(cmd), flush=True)
    try:
        with open(log, "a") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env={**os.environ, "HF_HUB_OFFLINE": "1"})
            for line in p.stdout:
                if "warn" not in line.lower():
                    sys.stdout.write(line)
                f.write(line)
            code = p.wait()
    finally:
        stop.set()
        t.join(timeout=60)
        outputs.commit()
    if code:
        raise RuntimeError(f"exit {code}: {' '.join(cmd[2:6])}; log {log}")
    if errors:
        raise RuntimeError(f"volume commit failed during the run: {errors[:3]}")


def stage_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage_run", f"{EXP}/261112_stage_own_replies/stage_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate_summary(out):
    g = json.loads((out / "gates.json").read_text())
    return g["passed"], {name: v["pass"] for name, v in g["gates"].items()}


def require_smoke(stage):
    """The full run needs a passed v1.1 code test of the same stage, with the same code and runtime."""
    smoke = Path(f"/out/own_replies/{stage}_{SMOKE}")
    receipt = smoke.parent / f"{smoke.name}.stage_receipt.json"
    if not (smoke / "gates.json").exists() or not receipt.exists():
        raise RuntimeError(f"no {SMOKE} result for {stage}; run --step smoke first")
    passed, gates = gate_summary(smoke)
    if not passed:
        raise RuntimeError(f"{stage} {SMOKE} gates failed: {gates}")
    S, rec = stage_module(), json.loads(receipt.read_text())
    now = dict(code=S.code_hashes(), runtime=S.runtime())
    bad = [k for k in now if rec.get(k) != now[k]]
    if bad:
        raise RuntimeError(f"{stage}: {SMOKE} ran with different {bad}; run the smoke again")


def job(stage, step):
    import sys
    if step == "full":
        require_smoke(stage)
    tag = SMOKE if step == "smoke" else "full"
    out = Path(f"/out/own_replies/{stage}_{tag}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = str(out.parent / f"{stage}_{tag}.log")
    wrapper = f"{EXP}/261112_stage_own_replies/stage_run.py"
    extra = ["--roles", "canary", "--limit", "40"] if step == "smoke" else []
    run([sys.executable, "-I", wrapper, "--stage", stage, "gen", "--out", str(out)] + extra, log)
    try:
        run([sys.executable, "-I", wrapper, "--stage", stage, "gates", "--run", str(out)], log)
    finally:
        passed, gates = gate_summary(out) if (out / "gates.json").exists() else (False, None)
    if not passed:
        raise RuntimeError(f"{stage} {tag}: gates did not pass: {gates}")
    man = json.loads((out / "manifest.json").read_text())
    return dict(stage=stage, step=step, out=str(out), n_rows=man["n_rows"], n_invalid=man["n_invalid"],
                seconds=man["seconds"], stop_reasons=man["stop_reasons"], gates_passed=passed, gates=gates)


VOL = {"/weights": weights, "/out": outputs}


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=TIMEOUT["smoke"], retries=0, max_containers=2, volumes=VOL)
def smoke_job(stage):
    return job(stage, "smoke")


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=TIMEOUT["full"], retries=0, max_containers=2, volumes=VOL)
def full_job(stage):
    return job(stage, "full")


@app.local_entrypoint()
def main(step: str = "smoke", stages: str = "sft,dpo"):
    fn = {"smoke": smoke_job, "full": full_job}[step]
    failed = False
    for r in fn.map(stages.split(","), return_exceptions=True):
        failed |= isinstance(r, BaseException)
        print(json.dumps(r, indent=1, default=repr))
    if failed:
        raise SystemExit("at least one stage failed; see above")
