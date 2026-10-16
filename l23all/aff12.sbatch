#!/bin/bash
#SBATCH -J mp_aff12
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-3%4
#SBATCH -o /scratch/gmarraff/mp/logs/aff12_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/aff12_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/aff12
mkdir -p $O
S=$SLURM_ARRAY_TASK_ID
N=4
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
cap = torch.cuda.get_device_capability()
arch = torch.cuda.get_arch_list()
ok = [a for a in arch if a.startswith(f"sm_{cap[0]}") and int(a.split("_")[1][-1]) <= cap[1]]
assert ok, f"sm_{cap[0]}{cap[1]} not in {arch}"
print(torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}", flush=True)
PY
.venv/bin/python -m persona_lora.steer_pin \
  --vectors "$R/l23all/aff12_vectors.npz" --cells "$R/l23all/cells_aff12.json" \
  --layers 23 --shard "${S}/${N}" --dtype fp16 \
  --out "$O/gen_aff_${S}.jsonl"
echo "done=$(date)"
