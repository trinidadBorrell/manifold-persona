"""One Modal app for the two GPU runs of 12/11/2026: the OLMo-2-7B coin-flip test (../261112_coinflip_stages) and the
Instruct prompt-only control (../261112_instruct_prompt_only). Weights and outputs stay on Modal volumes; only small
summaries come back to the Mac. Public models, anonymous download; no HF token is used.

  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/modal_runs.py --step prefetch   # CPU, 4 checkpoints
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/modal_runs.py --step smoke      # GPU code test
  .venv/bin/modal run experiments/q1_emergence/261112_modal_gpu/modal_runs.py --step full       # GPU, owner go
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if modal.is_local() else Path("/root/mp")  # inside the container the file sits in /root
ROOT = "/root/mp"
EXP = f"{ROOT}/experiments/q1_emergence"
GPU = "A100-80GB"  # the pod A and pod C card
GPU_TIMEOUT = {"smoke": 1800, "full": 5400}  # hard stop in seconds; the worst-case bill is this time on one GPU
STAGES_7B = (json.loads((REPO / "experiments/q1_emergence/261112_coinflip_stages/stages.json").read_text())["7b"]
             if modal.is_local() else [])
IGNORE = ["**/__pycache__", "**/results", "**/*.log"]

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0+cu129", index_url="https://download.pytorch.org/whl/cu129")
         .pip_install("transformers==4.47.1", "tokenizers==0.21.0", "numpy==2.5.2", "huggingface_hub==0.36.2",
                      "safetensors==0.8.0", "hf-xet==1.7.1")
         .env({"HF_HOME": "/weights/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
         .add_local_file(REPO / "src/manifold_persona/rows.py", f"{ROOT}/src/manifold_persona/rows.py")
         .add_local_dir(REPO / "experiments/q1_emergence/261112_coinflip_stages", f"{EXP}/261112_coinflip_stages",
                        ignore=IGNORE)
         .add_local_dir(REPO / "experiments/q1_emergence/261112_instruct_prompt_only",
                        f"{EXP}/261112_instruct_prompt_only", ignore=IGNORE))
app = modal.App("persona-gpu-runs-261112", image=image)
weights = modal.Volume.from_name("persona-olmo2-7b-weights-261112", create_if_missing=True, version=2)
outputs = modal.Volume.from_name("persona-gpu-runs-261112", create_if_missing=True, version=2)


@app.function(cpu=4, memory=8192, timeout=3600, retries=0, volumes={"/weights": weights})
def prefetch(entry):
    import os
    from huggingface_hub import snapshot_download
    path = snapshot_download(entry["repo"], revision=entry["revision"], token=False, max_workers=8,
                             allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"])
    weights.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return dict(stage=entry["stage"], path=path, gb=round(size / 1e9, 2))


def run(cmd, log):
    import os
    import subprocess
    import sys
    print("$", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})
        for line in p.stdout:
            if "warn" not in line.lower():
                sys.stdout.write(line)
            f.write(line)
        if p.wait():
            raise RuntimeError(f"{cmd[2]} exited {p.returncode}; log {log}")


def gpu_job(step):
    import sys
    from pathlib import Path as P
    out = P(f"/out/{step}")
    out.mkdir(parents=True, exist_ok=True)
    log = str(out / "run.log")
    smoke = step == "smoke"
    cf = f"{EXP}/261112_coinflip_stages/coinflip.py"
    c3 = f"{EXP}/261112_instruct_prompt_only/extract_c3.py"
    try:
        run([sys.executable, "-I", cf, "score", "--set", "7b", "--device", "cuda", "--out", str(out / "coinflip"),
             "--stages", "base,instruct" if smoke else "base,sft,dpo,instruct"] + (["--limit", "40"] if smoke else []), log)
        run([sys.executable, "-I", cf, "analyse", "--run", str(out / "coinflip")], log)
        run([sys.executable, "-I", c3, "extract", "--inputs", f"{EXP}/261112_instruct_prompt_only/inputs",
             "--out", str(out / "c3"), "--device", "cuda"] + (["--limit", "32"] if smoke else []), log)
    finally:
        outputs.commit()
    summary = (out / "coinflip/summary.md").read_text()
    c3_manifest = json.loads((out / "c3/manifest.json").read_text())
    return dict(step=step, coinflip_summary=summary, c3_check=c3_manifest["environment_check"],
                c3_n=c3_manifest["n_prompts"], c3_seconds=c3_manifest["seconds"], device=c3_manifest["versions"]["device"])


VOLUMES = {"/weights": weights, "/out": outputs}


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=GPU_TIMEOUT["smoke"], retries=0, max_containers=1, volumes=VOLUMES)
def smoke_job():
    return gpu_job("smoke")


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=GPU_TIMEOUT["full"], retries=0, max_containers=1, volumes=VOLUMES)
def full_job():
    return gpu_job("full")


@app.local_entrypoint()
def main(step: str = "smoke"):
    if step == "prefetch":
        for r in prefetch.map(STAGES_7B):
            print(r)
    elif step == "smoke":
        print(json.dumps(smoke_job.remote(), indent=1))
    elif step == "full":
        print(json.dumps(full_job.remote(), indent=1))
    else:
        raise SystemExit("step must be prefetch, smoke or full")
