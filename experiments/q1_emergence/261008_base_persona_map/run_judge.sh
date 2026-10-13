#!/bin/bash
# Blind top-5 naming judge (E0 prompt), GPT-5.6-Sol via Codex CLI. Usage: bash run_judge.sh RUN_DIR N_BATCHES
cd /Users/brendon/persona-manifolds/manifold-persona
H=experiments/q1_emergence/261008_base_persona_map; D=$H/$1
CODEX=${CODEX:-$(command -v codex || echo /Users/brendon/.local/share/nvm/v22.22.2/bin/codex)}
RULE="Do not write or run scripts or programs that classify, score or keyword-match the replies. Use shell commands only to read the files named below (for example cat or sed -n) and to write your answer file. Do not open any other file."
mkdir -p $D/logs
job(){
  n=$1
  sed -e "s|TOOL_RULE|$RULE|" -e "s|DIR|$D|g" -e "s|batch_N|batch_$n|g" -e "s|about 150 replies|about 100 replies|" \
    $H/prompt_judge.txt > $D/logs/prompt_$n.txt
  [ -s $D/answers/batch_$n.json ] && return 0
  "$CODEX" exec -m gpt-5.6-sol -c model_reasoning_effort=high -s workspace-write --ephemeral \
    -C /Users/brendon/persona-manifolds/manifold-persona -o $D/logs/last_$n.txt \
    "$(cat $D/logs/prompt_$n.txt)" < /dev/null > $D/logs/judge_$n.log 2>&1
  echo "batch $n: $(tail -1 $D/logs/last_$n.txt 2>/dev/null)"
}
export -f job; export D H RULE CODEX
seq 1 $2 | xargs -P 4 -I{} bash -c 'job {}'
echo __JUDGE_DONE__
