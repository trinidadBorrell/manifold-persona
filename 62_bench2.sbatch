#!/bin/bash
#SBATCH -J mp_b2
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/b2_%j.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/b2_%j.err
set -euo pipefail
# Everything reads from store3 (.203). $HOME (.201, 95% full) reads at ~1 MB/s
# and never finishes importing torch.
S=/data/parietal/store3/work/gmarraff/repos/manifold-persona
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MP_EXTRA_ROLES=$S/attr_roles PYTHONPATH=$S
cd $S
echo "node=$(hostname) venv=${MP_VENV:-.venv} dtype=${MP_DTYPE:-fp32}"
$S/${MP_VENV:-.venv}/bin/python -m persona_lora.bench_load \
  --n ${MP_N:-20} --dtype ${MP_DTYPE:-fp32}
