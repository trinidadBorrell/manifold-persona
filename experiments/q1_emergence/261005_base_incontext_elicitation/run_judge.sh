#!/bin/bash
# GPT-5.6-Sol judges via Codex CLI, at most 8 at once.
# Usage: bash run_judge.sh role N   (literature role judge, primary)
#        bash run_judge.sh char N   (blind top-5 naming, secondary)
cd /Users/brendon/persona-manifolds/manifold-persona
D=experiments/q1_emergence/261005_base_incontext_elicitation
RULE="Do not write or run scripts or programs that classify, score or keyword-match the replies. Use shell commands only to read the files named below (for example cat or sed -n) and to write your answer file. Do not open any other file."
mkdir -p $D/logs
job(){
  n=$1
  if [ "$KIND" = role ]; then P=prompt_rolejudge.txt; OUT=ranswers; else P=prompt_judge.txt; OUT=answers; fi
  sed -e "s|TOOL_RULE|$RULE|" -e "s|DIR|$D|g" -e "s|batch_N|batch_$n|g" $D/$P > $D/logs/prompt_${KIND}_$n.txt
  [ -s $D/$OUT/batch_$n.json ] && return 0
  codex exec -m gpt-5.6-sol -c model_reasoning_effort=high -s workspace-write --ephemeral \
    -C /Users/brendon/persona-manifolds/manifold-persona -o $D/logs/last_${KIND}_$n.txt \
    "$(cat $D/logs/prompt_${KIND}_$n.txt)" < /dev/null > $D/logs/judge_${KIND}_$n.log 2>&1
  echo "$KIND batch $n: $(tail -1 $D/logs/last_${KIND}_$n.txt 2>/dev/null)"
}
KIND=$1
export -f job; export D RULE KIND
seq 1 $2 | xargs -P 8 -I{} bash -c 'job {}'
echo __JUDGE_DONE__
