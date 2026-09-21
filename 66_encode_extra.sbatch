#!/bin/bash
#SBATCH -J mp_encx
#SBATCH -p gpu-best,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|volta|turing|ampere"
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --nice=1000
#SBATCH --array=0-3%4
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/encx_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/encx_%A_%a.err
set -euo pipefail
# Everything on store3: $HOME (95% full) never finishes importing torch.
# 8 shards, not 41: the import+load tax is ~7 min per task and unavoidable
# (TmpDisk=0, so no node-local staging).
S=/data/parietal/store3/work/gmarraff/repos/manifold-persona
W=/data/parietal/store3/work/gmarraff/manifold-persona
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MP_EXTRA_ROLES=$S/attr_roles PYTHONPATH=$S
V=$S/.venv_pascal          # cu124 build: works on pascal through ampere
M=Qwen/Qwen2.5-3B-Instruct
I=$SLURM_ARRAY_TASK_ID
mkdir -p $W/attrx/targets $W/attrx/acts
T=$W/attrx/targets/shard${I}.jsonl
A=$W/attrx/acts/shard${I}.npz
cd $S
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) shard=$I start=$(date)"
python3 -c "import json;p=json.load(open('$S/personas_extra.json'))[$I::8];print('personas:',p)"
if [ ! -s "$T" ]; then
  $V/bin/python -m persona_lora.gen_targets --model $M \
    --personas $S/personas_extra.json --shard $I --nshards 4 \
    --limit_questions 120 --n_phrasings 5 --max_new_tokens 160 \
    --dtype fp16 --seed 0 --out $T
fi
echo "generated=$(wc -l < $T) at $(date)"
if [ ! -s "$A" ]; then
  $V/bin/python -m persona_lora.extract_acts --model $M \
    --targets $T --out $A --max_len 384 --dtype fp16
fi
echo "done=$(date)"; ls -la $T $A
