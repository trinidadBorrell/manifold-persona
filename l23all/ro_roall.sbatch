#!/bin/bash
#SBATCH -J mp_roroall
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=0-7%8
#SBATCH -o /scratch/gmarraff/mp/logs/roroall_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/roroall_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/roall
S=$SLURM_ARRAY_TASK_ID
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
PY
.venv/bin/python -m persona_lora.readout_acts \
  --gen $O/gen_*.jsonl --layers 20,23,26,28 --shard "${S}/8" --dtype fp16 \
  --out "$O/acts_${S}.npz"
echo "done=$(date)"
