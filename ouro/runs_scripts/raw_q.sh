#!/bin/bash
# raw_q.sh GPU...: the training-free untied baseline. Condition raw<m><n> (evaluate_adapter.RawRunner): the frozen model
# prefills with m loops, keeps only the last loop's prompt KV and decodes with n loops reading it, without a loop
# adapter (Ouro's last-step KV reuse applied to the prompt). Same tasks and settings as the task map, so each cell pairs
# with the adapter's S<m><n> on the same items (the main table uses raw41 / raw42). Two jobs per GPU.
# env CELLS: a subset of the conditions; env ONLY: one task (e.g. ONLY=hellaswag) to add to an earlier queue.
# env SIZE=14: Ouro-1.4B (run names O14x-raw<m><n>-<task>, RULER O14-raw<m><n>f; GSM8K and MMLU subsets as its task map).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/raw_q.txt; cd $R
if [ "${SIZE:-26}" = 14 ]; then MP=$M/model-1.4b; P=O14x; B=O14; GSM="gsm --limit 400"; MMLU="mmlu --limit 20"
else MP=$M/model-2.6b; P=X26; B=X26; GSM=gsmfull; MMLU=mmlufull; fi
for c in ${CELLS:-raw21 raw31 raw41 raw22 raw42 raw44}; do
  MA="path=$MP,condition=$c,max_length=16384"
  lm() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ] && echo "./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  set -- $GSM; lm $1 "--tasks gsm8k_cot --num_fewshot 3 $2 $3"; set -- $MMLU; lm $1 "--tasks mmlu --num_fewshot 5 $2 $3"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm winogrande "--tasks winogrande --num_fewshot 5"
  case $c in raw41|raw42) lm hellaswag "--tasks hellaswag --num_fewshot 10"
    lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local" ;; esac
  [ -z "$ONLY" ] && echo "cd /root/big-small && MODEL=$MP DATA=$M/bench BENCHES='ruler4k ruler16k' ./ouro/runs_scripts/bench3.sh $B-${c}f $c - \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
