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
HF=experiments/q1_emergence/261029_fixed_text_read_through/$job
# a receipt counts only if it is verified and binds every local hash to its full path in this job's HF folder
receipt_ok() {  # receipt_ok DIR HF_FOLDER
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
d, prefix = Path(sys.argv[1]), sys.argv[2]
rec = json.loads((d / "hf_receipt.json").read_text())
files = {f["path"]: f for f in rec["files"]}
bad = []
for line in (d / "SHA256SUMS.txt").read_text().splitlines():
    h, name = line.split("  ", 1)
    f = files.get(f"{prefix}/{name}")
    if f is None or not f["match"] or f["sha256"] != h or f["remote_sha256"] != h:
        bad.append(name)
if rec["repo"] != "brend0nc/manifold-persona-experiment-artifacts" or not rec["verified"] or bad:
    sys.exit(f"{d}: the HF receipt does not cover these files under {prefix}: {bad}")
print(f"{d.name}: HF revision {rec['revision']}, {len(files)} files verified")
EOF
}
# every job folder on HF gets its own frames copy; a receipt from another job moves to this job's records
if ! { [ -e "$frames/hf_receipt.json" ] && receipt_ok "$frames" "$HF/$name" >/dev/null 2>&1; }; then
  [ ! -e "$frames/hf_receipt.json" ] || mv "$frames/hf_receipt.json" "$records/frames_hf_receipt_other_job.json"
  PYTHONPATH=src .venv/bin/python $PLAN/hf_archive.py upload --run "$frames" --prefix "$HF" --message "pod C $job/$name"
  receipt_ok "$frames" "$HF/$name"
fi
cp "$frames/hf_receipt.json" "$records/frames_hf_receipt.json"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$(dirname "$frames")" -cf - --exclude hf_receipt.json "$name" \
  | "${SSH[@]}" "cd $D/$C && tar xf - && cd $name && sha256sum -c --quiet SHA256SUMS.txt"
for ck in "${cks[@]}"; do
  if [ -e "$records/$ck" ]; then  # a restart keeps checkpoints whose HF copy is verified
    receipt_ok "$records/$ck" "$HF/$ck" && { echo "$ck: done in an earlier run; kept"; continue; }
    echo "$records/$ck exists without a verified receipt; move it first" >&2; exit 1
  fi
  s=c_${job}_$ck  # tmux session per job and checkpoint
  "${SSH[@]}" "! tmux has-session -t $s 2>/dev/null || { echo '$s is still running on the pod' >&2; exit 1; }"
  "${SSH[@]}" "cd $D/$C && rm -f out/$ck.done && tmux new-session -d -s $s '$TOKEN; bash pod_c_one.sh $ck $name $job $delete_weights'"
  # poll in short ssh calls, so a dropped connection is retried; a job that ended without its done file (a
  # container restart kills tmux) stops the driver instead of waiting forever
  lost=0
  while :; do
    st=$("${SSH[@]}" "if [ -e $D/$C/out/$ck.done ]; then echo done; elif tmux has-session -t $s 2>/dev/null; then echo running; else echo dead; fi" || echo unreachable)
    [ "$st" = done ] && break
    [ "$st" = dead ] && { echo "$ck: the pod job ended without out/$ck.done; see out/$ck.log on the pod" >&2; exit 1; }
    if [ "$st" = unreachable ]; then
      lost=$((lost + 1))
      echo "$ck: pod unreachable ($lost of 120 tries)" >&2
      [ "$lost" -lt 120 ] || { echo "$ck: pod unreachable for an hour; the job may still run there" >&2; exit 1; }
    else
      lost=0
    fi
    sleep 30
  done
  "${SSH[@]}" "tail -n 2 $D/$C/out/$ck.log $D/$C/out/$ck.upload.log; exit \$(cat $D/$C/out/$ck.done)"
  mkdir -p "$records/$ck"
  "${SSH[@]}" "cd $D/$C/out && tar cf - $ck.log $ck.upload.log $ck/hf_receipt.json $ck/manifest.json $ck/SHA256SUMS.txt $ck/fingerprint.txt $ck/*.*.json" \
    | tar xf - -C "$records"
  receipt_ok "$records/$ck" "$HF/$ck"
  [ "$prune" = 1 ] && "${SSH[@]}" "rm -rf $D/$C/out/$ck"
  echo "$ck: on HF and checked; records in $records/$ck"
done
