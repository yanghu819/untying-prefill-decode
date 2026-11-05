#!/bin/bash
# seed2_main.sh GPU...: the main-table benchmarks for the second training seed (X26-S-seed2) at prefill 4 / decode 1 and
# 2 that are not yet on this box (run names X26s2-S4<n>-<task>, RULER X26s2-S4<n>f), so Table 1 can report two seeds.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/seed2main_q.txt; CK=$R/X26-S-seed2/checkpoint-48m.pt; cd $R
for c in S41 S42; do
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { ls lmeval/out/X26s2-$c-$1/*/samples_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU X26s2-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/X26s2-${c}f/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler16k' ./ouro/runs_scripts/bench3.sh X26s2-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
