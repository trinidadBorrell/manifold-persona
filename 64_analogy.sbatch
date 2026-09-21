#!/bin/bash
#SBATCH -J mp_analog
#SBATCH -p gpu-best,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|volta|turing|ampere"
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --array=0-3%4
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/analog_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/analog_%A_%a.err
set -euo pipefail
S=/data/parietal/store3/work/gmarraff/repos/manifold-persona
W=/data/parietal/store3/work/gmarraff/manifold-persona
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$S
V=$S/.venv_pascal
cd $S
# preflight: a job can hold a --gres allocation and still see no device
$V/bin/python - <<'PRE'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("PREFLIGHT FAIL: no CUDA device despite --gres")
cap = torch.cuda.get_device_capability()
sm = f"sm_{cap[0]}{cap[1]}"
arch = torch.cuda.get_arch_list()
print(f"preflight ok: {torch.cuda.get_device_name(0)} {sm} in {arch}")
if sm not in arch:
    sys.exit(f"PREFLIGHT FAIL: {sm} not in {arch}")
PRE
I=$SLURM_ARRAY_TASK_ID
mkdir -p $W/analogy
$V/bin/python -m persona_lora.steer_analogy \
  --model Qwen/Qwen2.5-3B-Instruct \
  --vectors $S/attr_analysis/analogy_vectors.npz \
  --cells $S/attr_analysis/analogy_cells.json \
  --shard $I --nshards 4 --dtype fp16 \
  --out $W/analogy/shard${I}.jsonl
echo "done=$(date)"
