#!/bin/bash
#SBATCH -J mp_roall
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-59%8
#SBATCH -o /scratch/gmarraff/mp/logs/roall_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/roall_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/roall
mkdir -p $O
I=$SLURM_ARRAY_TASK_ID
# tasks 0-23 own (24 shards, 4 strengths), 24-41 k8 (18 shards, 3 strengths), 42-59 k32 (18 shards)
if   [ "$I" -lt 24 ]; then ARM=own; S=$I;            N=24
elif [ "$I" -lt 42 ]; then ARM=k8;  S=$((I - 24));   N=18
else                       ARM=k32; S=$((I - 42));   N=18
fi
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
  --vectors "$R/l23all/ro_all_vectors.npz" --cells "$R/l23all/cells_ro_${ARM}.json" \
  --layers 23 --shard "${S}/${N}" --dtype fp16 \
  --out "$O/gen_${ARM}_${S}.jsonl"
echo "done=$(date)"
