#!/usr/bin/env bash
# Copy a running pod B run's resumable files to the Mac every few minutes, so a lost pod keeps its finished
# batches (the pod has no persistent volume). Run it on the Mac beside the run; it ends when the run writes
# manifest.json or stopped_d1.json. The final files still come back with pod.sh pull.
#   sync_partial.sh "ssh root@IP -p PORT" run_full [minutes]
# Resume on a new pod with the same image and pins: copy <run>.partial/ to the pod as the run folder (same
# --out name), then start gen_extract.py with the same arguments; the fingerprint must match. A cut last row
# line is dropped and a state part that is missing is rebuilt from the stored ids (gen_extract.run).
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 \"ssh root@IP -p PORT\" RUN [minutes]" >&2; exit 2; }
host=$(echo "$1" | awk '{print $2}')
port=$(echo "$1" | awk '{for(i=1;i<NF;i++) if($i=="-p") print $(i+1)}')
run=$2 every=${3:-5}
B=/root/mp/experiments/q1_emergence/261029_base_role_vs_format
dest="$(cd "$(dirname "$0")" && pwd)/$run.partial"
mkdir -p "$dest"
while :; do
  # state parts are written to *.tmp.npz and renamed, so a copied part is always whole
  rsync -a -e "ssh -p $port" --exclude='*.tmp.npz' --include='fingerprint.txt' --include='fingerprint_parts.json' \
    --include='run_id.txt' --include='grid.json' --include='rows.partial.jsonl' --include='acts_parts/' \
    --include='acts_parts/*.npz' --exclude='*' "$host:$B/$run/" "$dest/"
  echo "$(date -u +%FT%TZ) $(ls "$dest/acts_parts" 2>/dev/null | wc -l | tr -d ' ') state parts," \
       "$(wc -l < "$dest/rows.partial.jsonl" 2>/dev/null | tr -d ' ') rows on the Mac"
  if ssh -p "$port" "$host" "test -e $B/$run/manifest.json -o -e $B/$run/stopped_d1.json"; then
    echo "the run has ended; copy it back with pod.sh pull"
    break
  fi
  sleep $((every * 60))
done
