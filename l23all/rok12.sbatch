#!/bin/bash
#SBATCH -J mp_rok12
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH --array=0-7%8
#SBATCH -o /scratch/gmarraff/mp/logs/rok12_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/rok12_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/rok12
mkdir -p $O
I=$SLURM_ARRAY_TASK_ID
# tasks 0-3 k8 (4 shards), 4-7 k32 (4 shards)
if [ "$I" -lt 4 ]; then ARM=k8; S=$I; else ARM=k32; S=$((I - 4)); fi
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
PY
.venv/bin/python -m persona_lora.steer_pin \
  --vectors "$R/l23all/ro_all_vectors.npz" --cells "$R/l23all/cells_ro12_${ARM}.json" \
  --layers 23 --shard "${S}/4" --dtype fp16 \
  --out "$O/gen_${ARM}_${S}.jsonl"
echo "done=$(date)"
