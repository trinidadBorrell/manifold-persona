#!/usr/bin/env bash
# Stage pods on RunPod (DESIGN.md, "Pod setup"). Run from anywhere in the repo, on the Mac.
#   pod.sh check                                   # local: every pushed file exists; offline tests pass
#   pod.sh push runpodN                            # copy code, install pins, fetch tokenizers, run tests on the pod
#   pod.sh pull runpodN run_canary [b]             # copy one finished pod A (or pod B) run back; check its SHA256SUMS.txt
# runpodN is the SSH entry bare-pod writes; "ssh root@IP -p PORT" works too.
# Every pod uploads its run to HF itself, and the analysis reads from HF, not from a Mac copy (owner, 29/10;
# DESIGN.md, pod setup): hf_archive.py on the pod, with a token the owner stores there himself. pull (a full
# copy to the Mac) is no longer the default. run_pod_c.sh drives pod C.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PLAN=experiments/q1_emergence/261029_stage_persona_plan
A=experiments/q1_emergence/261029_instruct_persona_map
PC=experiments/q1_emergence/261029_fixed_text_read_through
B=experiments/q1_emergence/261029_base_role_vs_format
E3=experiments/q1_emergence/261008_base_persona_map
D=/root/mp
PY="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $D/.venv/bin/python"  # 8 threads: tiny CPU models slow down on 255 cores; venv over the image packages: the image blocks system pip (PEP 668) and keeps its torch
FILES="src/manifold_persona/__init__.py src/manifold_persona/config.py src/manifold_persona/rows.py tests/test_rows.py
refs/assistant-axis/extraction_questions.jsonl refs/assistant-axis/roles/instructions
experiments/q1_emergence/261008_base_persona_map/questions_40.json $PLAN/requirements-pod.txt $PLAN/hf_archive.py
$A/gen_extract.py $A/gates.py $A/test_gen_extract.py $A/test_gates.py
$PC/checkpoints.py $PC/checkpoints.json $PC/replies.py $PC/frames.py $PC/extract.py $PC/pod_c_one.sh
$PC/test_checkpoints.py $PC/test_replies.py $PC/test_frames.py $PC/test_extract.py
$E3/descriptions_third_person.json $E3/gen_extract.py $E3/score_naming.py $E3/context_extract.py
$B/gen_extract.py $B/gates.py $B/context_extract.py $B/test_podb_gen_extract.py $B/test_podb_gates.py $B/test_podb_context_extract.py"
TESTS="tests/test_rows.py $A/test_gen_extract.py $A/test_gates.py $PC/test_checkpoints.py $PC/test_replies.py $PC/test_frames.py $PC/test_extract.py $B/test_podb_gen_extract.py $B/test_podb_gates.py $B/test_podb_context_extract.py"
TOKENIZERS="allenai/OLMo-2-1124-7B@7df9a82518afdecae4e8c026b27adccc8c1f0032 allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84"
usage() { echo "usage: $0 check | push HOST | pull HOST RUN [b]   (HOST: runpodN or \"ssh root@IP -p PORT\"; b: a pod B run)" >&2; exit 2; }
[ $# -ge 1 ] || usage
cmd=$1

if [ "$cmd" = check ]; then  # unpack exactly what push sends, then import and test from that copy only
  for f in $FILES; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
  ROOT=$(pwd) tmp=$(mktemp -d)
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - $FILES | tar xf - -C "$tmp"
  (cd "$tmp" && PYTHONPATH="$tmp/src" "$ROOT/.venv/bin/python" -c \
      "import manifold_persona.rows as r, os; assert os.path.realpath(r.__file__).startswith(os.path.realpath('$tmp')), r.__file__" \
    && PYTHONPATH="$tmp/src" "$ROOT/.venv/bin/python" -m pytest -q -p no:cacheprovider $TESTS)
  rm -rf "$tmp"
  exit 0
fi

[ $# -ge 2 ] || usage
if [[ "$2" == ssh\ * ]]; then
  host=$(echo "$2" | awk '{print $2}')
  port=$(echo "$2" | awk '{for(i=1;i<NF;i++) if($i=="-p") print $(i+1)}')
  SSH=(ssh -p "$port" -o StrictHostKeyChecking=accept-new "$host")
else
  SSH=(ssh "$2")
fi

case "$cmd" in
push)
  "${SSH[@]}" "mkdir -p $D && { [ -x $D/.venv/bin/python ] || python3 -m venv --system-site-packages $D/.venv; }"
  COPYFILE_DISABLE=1 tar --no-xattrs -czf - $FILES | "${SSH[@]}" "cd $D && tar xzf -"
  "${SSH[@]}" "cd $D && $PY -m pip install -q -r $PLAN/requirements-pod.txt && nvidia-smi --query-gpu=name,memory.total --format=csv"
  for t in $TOKENIZERS; do
    "${SSH[@]}" "$PY -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${t%@*}', revision='${t#*@}')\""
  done
  "${SSH[@]}" "cd $D && PYTHONPATH=$D/src $PY -m pytest -q -p no:cacheprovider $TESTS"
  cat <<EOF
Pushed and tested. On the pod, inside tmux:
  cd $D/$A && export PYTHONPATH=$D/src
  $PY gen_extract.py --roles canary --limit 40 --out run_smoke && $PY gates.py --run run_smoke
  $PY gen_extract.py --roles canary --out run_canary 2>&1 | tee run_canary.log
  $PY gates.py --run run_canary 2>&1 | tee run_canary/gates.log
Then on the Mac: $0 pull "$2" run_canary
Pod B, inside tmux:
  cd $D/$B && export PYTHONPATH=$D/src
  $PY gen_extract.py --grid canary --limit 40 --out run_smoke && $PY gates.py --run run_smoke && $PY context_extract.py --run run_smoke
  $PY gen_extract.py --grid canary --out run_canary 2>&1 | tee run_canary.log
  $PY gates.py --run run_canary 2>&1 | tee run_canary/gates.log
  $PY context_extract.py --run run_canary 2>&1 | tee run_canary/context.log
Then on the Mac: $0 pull "$2" run_canary b
EOF
  ;;
pull)
  [ $# -ge 3 ] || usage
  run=$3 X=$A
  [ "${4:-}" != b ] || X=$B
  [ ! -e "$X/$run" ] || { echo "$X/$run exists; move it first" >&2; exit 1; }
  stage=$(mktemp -d "$X/.$run.pulling.XXXXXX")  # a fresh folder per try; published as $X/$run only after every check
  "${SSH[@]}" "cd $D/$X && L= && { [ ! -e $run.log ] || L=$run.log; } && tar cf - --exclude=acts_parts --exclude=rows.partial.jsonl $run \$L" \
    | tar xf - -C "$stage"
  (cd "$stage/$run" && shasum -a 256 -c SHA256SUMS.txt)
  if [ -e "$stage/$run/context" ]; then (cd "$stage/$run/context" && shasum -a 256 -c SHA256SUMS.txt); fi
  [ "${4:-}" != b ] || PYTHONPATH=src .venv/bin/python "$B/gen_extract.py" --verify "$stage/$run"  # exact file sets
  mv "$stage/$run" "$X/$run"
  [ ! -e "$stage/$run.log" ] || mv "$stage/$run.log" "$X/$run.log"
  rmdir "$stage"
  echo "Copied and checked $X/$run. Read $X/$run/raw_read.md and gates.json; then upload with hf_archive.py."
  ;;
*) usage ;;
esac
