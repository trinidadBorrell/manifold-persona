#!/bin/bash
#SBATCH -J mp_l23all
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --array=0-53%18
#SBATCH -o /scratch/gmarraff/mp/logs/l23all_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/l23all_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/l23all
mkdir -p $O
I=$SLURM_ARRAY_TASK_ID
# tasks 0-23 own (24 shards), 24-29 prompt (6), 30-41 k8 (12), 42-53 k32 (12)
if   [ "$I" -lt 24 ]; then ARM=own;    S=$I;            N=24
elif [ "$I" -lt 30 ]; then ARM=prompt; S=$((I - 24));   N=6
elif [ "$I" -lt 42 ]; then ARM=k8;     S=$((I - 30));   N=12
else                       ARM=k32;    S=$((I - 42));   N=12
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
if [ "$ARM" = prompt ]; then
  .venv/bin/python -m persona_lora.steer_pin2 \
    --vectors "$R/l23all/l23all_vectors.npz" --cells "$R/l23all/cells_prompt.json" \
    --questions "$R/pb/pb_questions.json" --layers 23 --shard "${S}/${N}" --dtype fp16 \
    --out "$O/gen_prompt_${S}.jsonl"
else
  .venv/bin/python -m persona_lora.steer_pin \
    --vectors "$R/l23all/l23all_vectors.npz" --cells "$R/l23all/cells_${ARM}.json" \
    --layers 23 --shard "${S}/${N}" --dtype fp16 \
    --out "$O/gen_${ARM}_${S}.jsonl"
fi
echo "done=$(date)"
