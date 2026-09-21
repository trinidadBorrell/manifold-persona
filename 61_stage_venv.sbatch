#!/bin/bash
#SBATCH -J mp_stage
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --nice=2000
#SBATCH -o /data/parietal/store3/work/gmarraff/manifold-persona/logs/stage_%j.out
#SBATCH -e /data/parietal/store3/work/gmarraff/manifold-persona/logs/stage_%j.err
set -euo pipefail
# $HOME is 95% full and stalls torch imports; store3 (.203) is the healthy mount.
SRC=/home/mind/gmarraff/repos/manifold-persona
DST=/data/parietal/store3/work/gmarraff/venvs
mkdir -p $DST
for v in .venv .venv_pascal; do
  t0=$(date +%s)
  rsync -a --delete "$SRC/$v/" "$DST/$v/"
  echo "$v copied in $(( $(date +%s) - t0 ))s"
done
echo "--- smoke: import torch from store3 copy ---"
for v in .venv .venv_pascal; do
  t0=$(date +%s)
  $DST/$v/bin/python -c "import torch; print('$v torch', torch.__version__)"
  echo "$v import_seconds=$(( $(date +%s) - t0 ))"
done
du -sh $DST/.venv $DST/.venv_pascal
