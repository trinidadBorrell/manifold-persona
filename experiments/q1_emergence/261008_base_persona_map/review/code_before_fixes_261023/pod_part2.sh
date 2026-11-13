#!/usr/bin/env bash
# Part 2 on RunPod (THRESHOLDS.md, change of 16/10/2026). Run from this folder on the Mac.
#   ./pod_part2.sh push gpu  "ssh root@IP -p PORT"          # pod A: context extraction
#   ./pod_part2.sh push cpu  "ssh root@IP -p PORT" run_main  # pods B-D: frozen part2_analysis
# Then run the printed command on the pod (inside tmux).
set -euo pipefail
[ "$1" = push ] && shift
kind=$1; ssh_line=$2; run=${3:-run_main}
host=$(echo "$ssh_line" | awk '{print $2}'); port=$(echo "$ssh_line" | awk '{for(i=1;i<NF;i++) if($i=="-p") print $(i+1)}')
SSH=(ssh -p "$port" -o StrictHostKeyChecking=accept-new "$host")
D=/root/e3
PIN="numpy==1.26.4 scipy==1.13.1 scikit-learn==1.9.0 joblib==1.5.3"
slim() { echo "$1/replies.jsonl $1/fingerprint.txt $1/reply_mean_acts_L9_17_23.npy $1/reply_mean_acts_slim.json"; }
"${SSH[@]}" "mkdir -p $D/run_main $D/run_rep"
if [ "$kind" = gpu ]; then
  files="gen_extract.py context_extract.py part2_analysis.py part2_context.py $(slim run_main) $(slim run_rep) run_main/items.jsonl run_rep/items.jsonl p2_emb_run_main p2_emb_run_rep"
  pip="pip install -q $PIN transformers==4.47.1 tokenizers==0.21.0 accelerate"
else
  files="part2_analysis.py $(slim "$run") p2_emb_$run"
  pip="pip install -q $PIN"
fi
tar czf - $files | "${SSH[@]}" "cd $D && tar xzf -"
"${SSH[@]}" "cd $D && $pip && sha256sum run_*/reply_mean_acts_L9_17_23.npy && nproc && free -g | head -2"
grep -h '"sha256"' run_main/reply_mean_acts_slim.json run_rep/reply_mean_acts_slim.json
