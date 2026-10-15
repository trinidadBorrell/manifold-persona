#!/bin/bash
#SBATCH -J mp_ro
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH --array=0-3%4
#SBATCH -o /scratch/gmarraff/mp/logs/ro_%A_%a.out
#SBATCH -e /scratch/gmarraff/mp/logs/ro_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
mkdir -p /scratch/gmarraff/mp/ro
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
cap = torch.cuda.get_device_capability()
arch = torch.cuda.get_arch_list()
ok = [a for a in arch if a.startswith(f"sm_{cap[0]}") and int(a.split("_")[1][-1]) <= cap[1]]
assert ok, f"sm_{cap[0]}{cap[1]} not in {arch}"
print(torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}", flush=True)
PY
S=$SLURM_ARRAY_TASK_ID
if [ "$S" -lt 3 ]; then
  .venv/bin/python -m persona_lora.readout_acts --gen "$R/ro/l23_merged.jsonl" \
    --shard "${S}/3" --dtype fp16 --out "/scratch/gmarraff/mp/ro/acts_l23_${S}.npz"
else
  .venv/bin/python -m persona_lora.readout_acts --gen /scratch/gmarraff/mp/pb/gen_prompt_0.jsonl /scratch/gmarraff/mp/pb/gen_prompt_1.jsonl \
    --shard "0/1" --dtype fp16 --out "/scratch/gmarraff/mp/ro/acts_pb.npz"
fi
echo "done=$(date)"
