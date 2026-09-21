#!/bin/bash
#SBATCH -J mp_red
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --nice=3000
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/red_%j.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/red_%j.err
set -euo pipefail
S=/data/parietal/store3/work/gmarraff/repos/manifold-persona
W=/data/parietal/store3/work/gmarraff/manifold-persona
export PYTHONPATH=$S OMP_NUM_THREADS=8
cd $S
$S/.venv_pascal/bin/python -m persona_lora.reduce_acts \
  --acts "$W/attrall/acts/*.npz" --out $W/attrall/reduced.npz --layers 26,28,30
ls -la $W/attrall/reduced.npz
