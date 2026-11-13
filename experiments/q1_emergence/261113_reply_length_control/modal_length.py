"""Modal launcher for the reply length control (rules: THRESHOLDS.md). Instruct, SFT and DPO re-read their own
replies in parallel on three A100-80GB. Weights: the volume that ../261112_modal_gpu/modal_runs.py filled. Outputs:
the shared outputs volume under length_control/.

  .venv/bin/modal run experiments/q1_emergence/261113_reply_length_control/modal_length.py --step fetch   # CPU: pod A rows + states from HF
  .venv/bin/modal run experiments/q1_emergence/261113_reply_length_control/modal_length.py --step smoke   # 64 rows per stage
  .venv/bin/modal run experiments/q1_emergence/261113_reply_length_control/modal_length.py --step full    # 21,000 rows per stage
  .venv/bin/modal run experiments/q1_emergence/261113_reply_length_control/modal_length.py --step verify  # CPU: verify_outputs.py

The 13/11 runs used the launcher in review/code_before_fixes_261113/ (its full step checked only the smoke's
length_control.py hash). The smoke gate and the verify step below were added after them (Codex review, findings 1, 2).
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if modal.is_local() else Path("/root/mp")
ROOT = "/root/mp"
EXP = f"{ROOT}/experiments/q1_emergence"
SCRIPT = f"{EXP}/261113_reply_length_control/length_control.py"
VERIFY = f"{EXP}/261113_reply_length_control/verify_outputs.py"
ROWS_PY = f"{ROOT}/src/manifold_persona/rows.py"
STAGES_JSON = f"{EXP}/261112_coinflip_stages/stages.json"
GPU = "A100-80GB"
TIMEOUT = {"smoke": 1800, "full": 3600}
HF_REPO = "brend0nc/manifold-persona-experiment-artifacts"
PODA_REV = "60a81b9b2e337a46a0a2c41862357164273da5ff"  # pod A run_full, HF receipt run_full/hf_receipt.json
PODA_PATH = "experiments/q1_emergence/261029_instruct_persona_map/run_full"
PODA_SUMS_SHA256 = "b079b00d16228f79be18f73c138dfe1e42ccc050b62759f9db3b91015b814db3"
RUNS = {"instruct": f"/out/poda_dl/{PODA_PATH}", "sft": "/out/own_replies/sft_full", "dpo": "/out/own_replies/dpo_full"}

def with_code(img):
    return (img.add_local_file(REPO / "src/manifold_persona/rows.py", ROWS_PY)
            .add_local_file(REPO / "experiments/q1_emergence/261112_coinflip_stages/stages.json", STAGES_JSON)
            .add_local_file(HERE / "length_control.py", SCRIPT)
            .add_local_file(HERE / "verify_outputs.py", VERIFY))


image = with_code(modal.Image.debian_slim(python_version="3.12")
                  .pip_install("torch==2.13.0+cu129", index_url="https://download.pytorch.org/whl/cu129")
                  .pip_install("transformers==4.47.1", "tokenizers==0.21.0", "numpy==2.5.2", "huggingface_hub==0.36.2",
                               "safetensors==0.8.0", "hf-xet==1.7.1")
                  .env({"HF_HOME": "/weights/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}))
cpu_image = with_code(modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.2"))
app = modal.App("persona-length-control-261113", image=image)
weights = modal.Volume.from_name("persona-olmo2-7b-weights-261112", version=2)
outputs = modal.Volume.from_name("persona-gpu-runs-261112", version=2)


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@app.function(cpu=4, memory=8192, timeout=3600, retries=0, volumes={"/out": outputs},
              secrets=[modal.Secret.from_name("hf-read", required_keys=["HF_TOKEN"])])
def fetch():
    """Pod A's rows, states and checksums at the pinned HF revision, checked against pod A's SHA256SUMS.txt."""
    import os
    from huggingface_hub import hf_hub_download
    names = ["SHA256SUMS.txt", "rows.jsonl", "reply_mean_acts.npy", "manifest.json"]
    for n in names:
        hf_hub_download(HF_REPO, f"{PODA_PATH}/{n}", repo_type="dataset", revision=PODA_REV, local_dir="/out/poda_dl",
                        token=os.environ["HF_TOKEN"])
    run = Path(RUNS["instruct"])
    if sha256(run / "SHA256SUMS.txt") != PODA_SUMS_SHA256:
        raise RuntimeError("pod A SHA256SUMS.txt differs from its HF receipt")
    sums = dict(reversed(line.split("  ", 1)) for line in (run / "SHA256SUMS.txt").read_text().splitlines())
    bad = [n for n in names[1:] if sha256(run / n) != sums[n]]
    outputs.commit()
    if bad:
        raise RuntimeError(f"pod A files differ from SHA256SUMS.txt: {bad}")
    return dict(run=str(run), files={n: (run / n).stat().st_size for n in names}, verified=True)


def run(cmd, log):
    """Stream a child process to the Modal log and a file; commit the outputs volume at the end. A non-zero exit raises."""
    import os
    import subprocess
    import sys
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
        outputs.commit()
    if code:
        raise RuntimeError(f"exit {code}: {' '.join(cmd[2:7])}; log {log}")


def verify_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_outputs", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(stage, step):
    import sys
    out = Path(f"/out/length_control/{step}/{stage}")
    if step == "full":
        V = verify_module()
        smoke = Path(f"/out/length_control/smoke/{stage}/manifest.json")
        if not smoke.exists():
            raise RuntimeError(f"{stage}: no smoke; run --step smoke first")
        smoke_man = json.loads(smoke.read_text())
        input_sums = V.read_sums(Path(RUNS[stage]) / "SHA256SUMS.txt")
        entry = {e["stage"]: e for e in json.loads(Path(STAGES_JSON).read_text())["7b"]}[stage]
        bad = V.smoke_gate(smoke_man, dict(length_control_py=sha256(SCRIPT), rows_py=sha256(ROWS_PY)), input_sums, entry)
        if bad:
            raise RuntimeError(f"{stage}: the smoke differs from this full run ({bad}); run --step smoke again")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    log = str(out.parent / f"{stage}.log")
    run([sys.executable, "-I", SCRIPT, "extract", "--stage", stage, "--run", RUNS[stage], "--out", str(out)]
        + (["--limit", "64"] if step == "smoke" else []), log)
    man = json.loads((out / "manifest.json").read_text())
    if step == "full" and V.fingerprint_diff(smoke_man, man):
        raise RuntimeError(f"{stage}: the full run's fingerprint differs from the smoke's "
                           f"({V.fingerprint_diff(smoke_man, man)}); its outputs in {out} are not usable")
    chk = json.loads((out / "check.json").read_text())
    sup = json.loads((out / "support.json").read_text())
    return dict(stage=stage, step=step, n_rows=man["n_rows"], seconds=man["seconds"], check_passed=man["check_passed"],
                check={k: {m: round(v[m], 5) for m in ("min_cos", "median_cos", "median_rel_norm")}
                       for k, v in chk["cells"].items()},
                support={c: dict(available=s["available"], lowest=s["lowest"], n_finite_rows=s["n_finite_rows"])
                         for c, s in sup.items()})


VOL = {"/weights": weights, "/out": outputs}


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=TIMEOUT["smoke"], retries=0, max_containers=3, volumes=VOL)
def smoke_job(stage):
    return job(stage, "smoke")


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=TIMEOUT["full"], retries=0, max_containers=3, volumes=VOL)
def full_job(stage):
    return job(stage, "full")


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=1800, retries=0, max_containers=3, volumes={"/out": outputs})
def verify_job(stage):
    """verify_outputs.py on the full outputs of one stage; the record goes to /out/length_control/verify/."""
    rec = verify_module().verify_stage(stage, RUNS[stage], f"/out/length_control/full/{stage}",
                                       f"/out/length_control/smoke/{stage}")
    out = Path(f"/out/length_control/verify/{stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    outputs.commit()
    return rec


@app.local_entrypoint()
def main(step: str = "smoke", stages: str = "instruct,sft,dpo"):
    if step == "fetch":
        print(json.dumps(fetch.remote(), indent=1))
        return
    if step == "verify":
        (HERE / "verify").mkdir(exist_ok=True)
        failed = False
        for rec in verify_job.map(stages.split(","), return_exceptions=True):
            if isinstance(rec, BaseException):
                print(repr(rec))
                failed = True
                continue
            (HERE / "verify" / f"{rec['stage']}.json").write_text(json.dumps(rec, indent=1))
            print(json.dumps(dict(stage=rec["stage"], passed=rec["passed"], failures=rec["failures"]), indent=1))
            failed |= not rec["passed"]
        if failed:
            raise SystemExit("verify failed; see above")
        return
    fn = {"smoke": smoke_job, "full": full_job}[step]
    failed = False
    for r in fn.map(stages.split(","), return_exceptions=True):
        failed |= isinstance(r, BaseException)
        print(json.dumps(r, indent=1, default=repr))
    if failed:
        raise SystemExit("at least one stage failed; see above")
