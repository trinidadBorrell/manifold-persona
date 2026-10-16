#!/bin/bash
#SBATCH -J mp_rosnap
#SBATCH -p gpu-best
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|rtx|hopper"
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/gmarraff/mp/logs/rosnap_%j.out
#SBATCH -e /scratch/gmarraff/mp/logs/rosnap_%j.err
set -euo pipefail
export HF_HOME=/data/parietal/store3/work/gmarraff/hf_cache
export TOKENIZERS_PARALLELISM=false
R=/data/parietal/store3/work/gmarraff/repos/manifold-persona
cd "$R"
O=/scratch/gmarraff/mp/clfaim
.venv/bin/python - <<'PY'
import glob, json
n = 0
with open("/scratch/gmarraff/mp/clfaim/snapshot.jsonl", "w") as f:
    for g in sorted(glob.glob("/scratch/gmarraff/mp/clfaim/gen_clf_*.jsonl")):
        for l in open(g):
            try:
                r = json.loads(l)
            except Exception:
                continue
            f.write(json.dumps(r) + "\n")
            n += 1
print(f"snapshot: {n} answers", flush=True)
PY
.venv/bin/python -m persona_lora.readout_acts \
  --gen $O/snapshot.jsonl --layers 23 --dtype fp16 --out "$O/acts_snapshot.npz"
echo "done=$(date)"
