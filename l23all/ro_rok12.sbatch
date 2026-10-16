#!/bin/bash
#SBATCH -J mp_rorok12
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/gmarraff/mp/logs/rorok12_%j.out
#SBATCH -e /scratch/gmarraff/mp/logs/rorok12_%j.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/rok12
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no cuda device in this allocation"
PY
.venv/bin/python -m persona_lora.readout_acts \
  --gen $O/gen_*.jsonl --layers 23 --dtype fp16 --out "$O/acts_0.npz"
echo "done=$(date)"
