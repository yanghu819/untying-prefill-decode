#!/bin/bash
# raw_q.sh GPU...: the training-free untied baseline on Ouro-2.6B. Condition raw<m><n> (evaluate_adapter.RawRunner): the
# frozen model prefills with m loops, keeps only the last loop's prompt KV and decodes with n loops reading it, without a
# loop adapter (Ouro's last-step KV reuse applied to the prompt). Same tasks and settings as the task map, so each
# cell pairs with the adapter's S<m><n> on the same items. Two jobs per GPU.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/raw_q.txt; cd $R
for c in raw21 raw31 raw41 raw22 raw42 raw44; do
  MA="path=$M/model-2.6b,condition=$c,max_length=16384"
  lm() { echo "./lm1.sh \$GPU X26-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm winogrande "--tasks winogrande --num_fewshot 5"
  echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler16k' ./ouro/runs_scripts/bench3.sh X26-${c}f $c - \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
