#!/usr/bin/env bash
# Pod C job, one checkpoint at a time (THRESHOLDS.md G4; owner 29/10: the pod uploads to HF). Run on the Mac after
# pod.sh push and after the owner has stored a fine-grained HF token on the pod (hf auth login).
#   run_pod_c.sh [--delete-weights] [--prune] HOST FRAMES_DIR JOB RECORDS_DIR [checkpoint ...]
# FRAMES_DIR: a folder made by frames.py. It is uploaded from the Mac, copied to the pod and checked there once.
# JOB: the run name on HF: experiments/q1_emergence/261029_fixed_text_read_through/JOB/<checkpoint>/.
# Per checkpoint: extract on the pod (tmux); upload from the pod and check every file's remote sha256 and size;
# copy the receipt, manifest, records and logs (small files) to RECORDS_DIR/<checkpoint>/ and check the receipt
# on the Mac; only then does the next checkpoint load. Stops at the first failure. Default order: Instruct
# first (G3 runs there).
# Deletion needs the owner's approval, so both are off unless given:
#   --delete-weights  remove each checkpoint's model weights from the pod's HF cache after its extraction
#   --prune           remove the pod copy of a checkpoint's outputs after its HF copy passed the check
# HOST is runpodN or "ssh root@IP -p PORT".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
C=experiments/q1_emergence/261029_fixed_text_read_through
PLAN=experiments/q1_emergence/261029_stage_persona_plan
D=/root/mp
delete_weights=0 prune=0
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --delete-weights) delete_weights=1 ;;
    --prune) prune=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ $# -ge 4 ] || { echo "usage: $0 [--delete-weights] [--prune] HOST FRAMES_DIR JOB RECORDS_DIR [checkpoint ...]" >&2; exit 2; }
host=$1 frames=$2 job=$3 records=$4
shift 4
cks=("$@")
[ ${#cks[@]} -gt 0 ] || cks=(instruct base sft dpo rl_60 rl_120 rl_180 rl_240 rl_300 rl_360)
if [[ "$host" == ssh\ * ]]; then
  h=$(echo "$host" | awk '{print $2}')
  port=$(echo "$host" | awk '{for(i=1;i<NF;i++) if($i=="-p") print $(i+1)}')
  SSH=(ssh -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 "$h")
else
  SSH=(ssh -o ServerAliveInterval=60 "$host")
fi
mkdir -p "$records"
name=$(basename "$frames")
# hf auth login in a RunPod shell saves the token under the image's HF_HOME (/workspace/.cache/huggingface); the jobs
# keep the model cache in /root/.cache, so only the token path follows it
TOKEN="[ ! -e /workspace/.cache/huggingface/token ] || export HF_TOKEN_PATH=/workspace/.cache/huggingface/token"
# the pod can write to HF (prints the account name only)
"${SSH[@]}" "$TOKEN; $D/.venv/bin/python -c \"from huggingface_hub import HfApi; print('HF account on the pod:', HfApi().whoami()['name'])\""
[ -e "$frames/hf_receipt.json" ] || PYTHONPATH=src .venv/bin/python $PLAN/hf_archive.py upload --run "$frames" \
  --prefix "experiments/q1_emergence/261029_fixed_text_read_through/$job" --message "pod C $job/$name"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$(dirname "$frames")" -cf - --exclude hf_receipt.json "$name" \
  | "${SSH[@]}" "cd $D/$C && tar xf - && cd $name && sha256sum -c --quiet SHA256SUMS.txt"
for ck in "${cks[@]}"; do
  [ ! -e "$records/$ck" ] || { echo "$records/$ck exists; move it first" >&2; exit 1; }
  "${SSH[@]}" "cd $D/$C && rm -f out/$ck.done && tmux new-session -d -s c_$ck '$TOKEN; bash pod_c_one.sh $ck $name $job $delete_weights'"
  "${SSH[@]}" "while [ ! -e $D/$C/out/$ck.done ]; do sleep 30; done; tail -n 2 $D/$C/out/$ck.log $D/$C/out/$ck.upload.log; exit \$(cat $D/$C/out/$ck.done)"
  mkdir -p "$records/$ck"
  "${SSH[@]}" "cd $D/$C/out && tar cf - $ck.log $ck.upload.log $ck/hf_receipt.json $ck/manifest.json $ck/SHA256SUMS.txt $ck/fingerprint.txt $ck/*.*.json" \
    | tar xf - -C "$records"
  PYTHONPATH=src .venv/bin/python - "$records/$ck" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rec = json.loads((d / "hf_receipt.json").read_text())
names = [l.split("  ", 1)[1] for l in (d / "SHA256SUMS.txt").read_text().splitlines()]
got = {Path(f["path"]).name: f for f in rec["files"]}
bad = [n for n in names if n not in got or not got[n]["match"]]
if not rec["verified"] or bad:
    sys.exit(f"{d.name}: HF copy not verified {bad}")
print(f"{d.name}: HF revision {rec['revision']}, {len(rec['files'])} files verified")
EOF
  [ "$prune" = 1 ] && "${SSH[@]}" "rm -rf $D/$C/out/$ck"
  echo "$ck: on HF and checked; records in $records/$ck"
done
