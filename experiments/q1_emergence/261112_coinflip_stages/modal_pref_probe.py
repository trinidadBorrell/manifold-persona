"""Modal launcher for pref_probe.py (rules: THRESHOLDS_pref_probe.md v0.2). Same image, weights volume and output
volume as ../261112_modal_gpu/modal_runs.py. Public models, anonymous download; no HF token is used.

  .venv/bin/modal run experiments/q1_emergence/261112_coinflip_stages/modal_pref_probe.py --step smoke   # 2 pairs, base + instruct
  .venv/bin/modal run experiments/q1_emergence/261112_coinflip_stages/modal_pref_probe.py --step full    # 100 pairs, 4 stages
  .venv/bin/modal run experiments/q1_emergence/261112_coinflip_stages/modal_pref_probe.py --step interview  # v0.3: E3 frame, 4 stages
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = "/root/mp/experiments/q1_emergence/261112_coinflip_stages"
GPU = "A100-80GB"
IGNORE = ["**/__pycache__", "**/results", "**/review", "**/*.log"]

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0+cu129", index_url="https://download.pytorch.org/whl/cu129")
         .pip_install("transformers==4.47.1", "tokenizers==0.21.0", "numpy==2.5.2", "huggingface_hub==0.36.2",
                      "safetensors==0.8.0", "hf-xet==1.7.1")
         .env({"HF_HOME": "/weights/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(HERE, ROOT, ignore=IGNORE))
app = modal.App("persona-pref-probe-261113", image=image)
weights = modal.Volume.from_name("persona-olmo2-7b-weights-261112", version=2)
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)
VOLUMES = {"/weights": weights, "/out": outputs}


def job(step):
    import subprocess
    import sys
    out = Path(f"/out/pref_probe/{step}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", f"{ROOT}/pref_probe.py", "score", "--out", str(out), "--device", "cuda",
           "--stages", "base,instruct" if step == "smoke" else "base,sft,dpo,instruct"]
    cmd += ["--limit_pairs", "2"] if step == "smoke" else []
    cmd += ["--frame", "interview"] if step == "interview" else []
    print("$", " ".join(cmd), flush=True)
    try:
        with open(out / "run.log", "a") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                if "warn" not in line.lower():
                    sys.stdout.write(line)
                f.write(line)
            if p.wait():
                raise RuntimeError(f"pref_probe.py exited {p.returncode}")
    finally:
        outputs.commit()
    return sorted(str(f) for f in out.glob("*.json"))


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=1500, retries=0, max_containers=1, volumes=VOLUMES)
def smoke_job():
    return job("smoke")


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=2700, retries=0, max_containers=1, volumes=VOLUMES)
def full_job():
    return job("full")


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=2700, retries=0, max_containers=1, volumes=VOLUMES)
def interview_job():
    return job("interview")


@app.local_entrypoint()
def main(step: str = "smoke"):
    fn = dict(smoke=smoke_job, full=full_job, interview=interview_job).get(step)
    if fn is None:
        raise SystemExit("step must be smoke, full or interview")
    print(json.dumps(fn.remote(), indent=1))
