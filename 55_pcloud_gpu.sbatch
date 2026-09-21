#!/bin/bash
#SBATCH -J mp_pcgpu
#SBATCH -p gpu-best,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu002,margpu003,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-3%4
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/pcgpu_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/pcgpu_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
W=/data/parietal/store3/work/gmarraff/manifold-persona
echo "$(date +%T) start on $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /home/mind/gmarraff/repos/manifold-persona
.venv/bin/python -m persona_lora.steer_probe \
  --model Qwen/Qwen2.5-3B-Instruct --stats "$W/fitted_stats_3b.npz" \
  --mode linear --alpha 1.0 --cells "$W/plot_cells.json" \
  --shard $SLURM_ARRAY_TASK_ID --nshards 4 \
  --layers 26,27,28,29,30 --n_sub 16 --n_questions 3 --max_new_tokens 110 \
  --out "$W/judge_gen_pcloud/shard${SLURM_ARRAY_TASK_ID}.jsonl"
echo "$(date +%T) done"
