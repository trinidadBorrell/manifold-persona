#!/usr/bin/env bash
# Pod side of run_pod_c.sh: extract one checkpoint, upload its folder to the private HF dataset and check the
# remote copy (hf_archive.py: remote sha256 and size of every file, receipt in out/<checkpoint>/hf_receipt.json),
# then write out/<checkpoint>.done with the exit code. The HF token is the one the owner stored with
# `hf auth login`; nothing here reads or prints it.
#   bash pod_c_one.sh CHECKPOINT FRAMES_DIR JOB DELETE_WEIGHTS(0|1)
set -uo pipefail
cd "$(dirname "$0")"
ck=$1 frames=$2 job=$3 delete_weights=$4
# no transparent huge pages for numpy: on a busy host a fresh 12 GB state array can stall for minutes in compaction
export PYTHONPATH=/root/mp/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
PY=/root/mp/.venv/bin/python
extra=()
[ "$ck" = instruct ] && extra+=(--pod_a_run /root/mp/experiments/q1_emergence/261029_instruct_persona_map/run_full)
[ "$delete_weights" = 1 ] && extra+=(--delete_weights)
mkdir -p out
$PY extract.py --frames "$frames" --checkpoint "$ck" --out "out/$ck" "${extra[@]}" >> "out/$ck.log" 2>&1 \
  && $PY ../261029_stage_persona_plan/hf_archive.py upload --run "out/$ck" \
       --prefix "experiments/q1_emergence/261029_fixed_text_read_through/$job" --message "pod C $job/$ck" \
       > "out/$ck.upload.log" 2>&1
echo $? > "out/$ck.done"
