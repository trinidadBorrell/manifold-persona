#!/bin/bash
#SBATCH -J mp_tokstats
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH --array=0-3%4
#SBATCH -o /scratch/gmarraff/mp/logs/tokstats_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/tokstats_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/tokstats
mkdir -p $O
S=$SLURM_ARRAY_TASK_ID
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
PY
.venv/bin/python -m persona_lora.token_stats \
  --gen /scratch/gmarraff/mp/l23all/gen_prompt_*.jsonl --layer 23 --shard "${S}/4" --dtype fp16 \
  --out "$O/stats_${S}.npz"
echo "done=$(date)"
