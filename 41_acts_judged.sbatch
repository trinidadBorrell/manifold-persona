#!/bin/bash
#SBATCH -J mp_actj
#SBATCH -p gpu-best,gpu,parietal
#SBATCH --gres=gpu:1
#SBATCH --exclude=margpu001,margpu004,margpu017,margpu018,margpu019,margpu020,margpu021,margpu022,margpu023,margpu024,margpu025,margpu026,margpu027,margpu028,margpu029
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --array=0-3%4
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/actj_%A_%a.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/actj_%A_%a.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/data/parietal/store3/work/gmarraff/manifold-persona
SRC=("$W/judge_gen_conf/steer_band_a4_s1.jsonl" "$W/judge_gen_conf/pp_r1_L17_withprompt_s1.jsonl" "$W/judge_gen_n4/prompted_n4.jsonl" "$W/judge_gen_n4/cond_r1_L17_withprompt.jsonl")
TAG=(steer_a4_s1 pp_r1_L17_s1 prompted_n4 cond_r1_L17)
I=$SLURM_ARRAY_TASK_ID
cd /home/mind/gmarraff/repos/manifold-persona
.venv/bin/python -m persona_lora.extract_acts \
  --model Qwen/Qwen2.5-0.5B-Instruct --targets "${SRC[$I]}" \
  --out "$W/acts_judged/${TAG[$I]}.npz"
