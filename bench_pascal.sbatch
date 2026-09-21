#!/bin/bash
#SBATCH -J mp_bpas
#SBATCH -p gpu-best
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --nice=2000
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/bpas_%j.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/bpas_%j.err
set -euo pipefail
export OMP_NUM_THREADS=8
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C=/home/mind/gmarraff/repos/manifold-persona
W=/data/parietal/store3/work/gmarraff/manifold-persona
export MP_EXTRA_ROLES=$C/attr_roles
cd $C
echo "node=$(hostname) BENCH-PASCAL start=$(date +%s)"
t0=$(date +%s)
.venv_pascal/bin/python -m persona_lora.gen_targets --model Qwen/Qwen2.5-3B-Instruct \
  --personas $C/personas_attr.json --shard 36 --nshards 41 \
  --limit_questions 6 --n_phrasings 5 --max_new_tokens 160 --dtype fp16 \
  --out $W/bench/pas_gen.jsonl
t1=$(date +%s); echo "GEN_SECONDS_6Q=$((t1-t0))"
.venv_pascal/bin/python -m persona_lora.extract_acts --model Qwen/Qwen2.5-3B-Instruct \
  --targets $W/bench/pas_gen.jsonl --out $W/bench/pas_acts.npz --max_len 384
t2=$(date +%s); echo "EXT_SECONDS_6Q=$((t2-t1))"; echo "TOTAL=$((t2-t0))"
