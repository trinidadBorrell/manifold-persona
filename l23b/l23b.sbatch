#!/bin/bash
#SBATCH -J mp_l23b
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=0-8%9
#SBATCH -o /scratch/gmarraff/mp/logs/l23b_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/l23b_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
mkdir -p /scratch/gmarraff/mp/l23b
NAMES=(L23 k8 k32)
A=$(( SLURM_ARRAY_TASK_ID / 3 ))
S=$(( SLURM_ARRAY_TASK_ID % 3 ))
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
  --vectors "$R/l23/l23_vectors.npz" --cells "$R/l23/l23_cells_${NAMES[$A]}.json" \
  --layers 23 --shard "${S}/3" --dtype fp16 \
  --out "/scratch/gmarraff/mp/l23b/gen_${NAMES[$A]}_${S}.jsonl"
echo "done=$(date)"
