#!/bin/bash
#SBATCH -J mp_attr
#SBATCH -p gpu-best,gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --nice=5000
#SBATCH --array=0-40%6
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/attr_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/attr_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
C=/home/mind/gmarraff/repos/manifold-persona
W=/data/parietal/store3/work/gmarraff/manifold-persona
export MP_EXTRA_ROLES=$C/attr_roles
M=Qwen/Qwen2.5-3B-Instruct
I=$SLURM_ARRAY_TASK_ID
P=$(python3 -c "import json;print(json.load(open('$C/personas_attr.json'))[$I])")
mkdir -p "$W/attr/targets" "$W/attr/acts"
T="$W/attr/targets/${P}.jsonl"
A="$W/attr/acts/${P}.npz"
echo "node=$(hostname) task=$I persona=$P start=$(date)"
cd "$C"
if [ ! -s "$T" ]; then
  .venv/bin/python -m persona_lora.gen_targets \
    --model $M --personas "$C/personas_attr.json" \
    --shard "$I" --nshards 41 \
    --limit_questions 120 --n_phrasings 5 \
    --max_new_tokens 160 --seed 0 --out "$T"
fi
if [ ! -s "$A" ]; then
  .venv/bin/python -m persona_lora.extract_acts \
    --model $M --targets "$T" --out "$A" --max_len 384
fi
echo "done=$(date)"; ls -la "$T" "$A"
