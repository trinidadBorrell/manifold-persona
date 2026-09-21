#!/bin/bash
#SBATCH -J mp_tp
#SBATCH -p gpu-best,gpu,parietal
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu001,margpu004,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-1%2
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/tp_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/tp_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/data/parietal/store3/work/gmarraff/manifold-persona
MODE=(shift transport)
M=${MODE[$SLURM_ARRAY_TASK_ID]}
cd /home/mind/gmarraff/repos/manifold-persona
.venv/bin/python -m persona_lora.transplant \
  --model Qwen/Qwen2.5-0.5B-Instruct --acts "$W/acts" \
  --mode "$M" --layers 15,16,17,18,19 --fit_layer 17 --n_sub 6 \
  --only ghost,economist,musician --n_samples 4 --max_new_tokens 110 \
  --out "$W/judge_gen_tp/transplant_${M}.jsonl"
