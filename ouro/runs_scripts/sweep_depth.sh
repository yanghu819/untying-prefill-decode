#!/bin/bash
# sweep_depth.sh GPU...: prefill-depth sweep beyond the trained range (extrapolation). The aligned loop adapter
# X26-a-universal (trained on prefill 1-4) prefills with m = 5, 6, 8, 12, 16 loops and decodes with n = 1 and 2 (run
# names X26-a<m>x<n>-<task>, RULER X26-a<m>x<n>f), on the generation benchmarks of the time study and MMLU / ARC-C.
# Together with the a<m><n> cells (m <= 4) this gives quality against prefill depth at fixed decode loops on the same
# weights. env CK / P: another adapter checkpoint and run-name prefix (the adapter trained on the wide range);
# env MS / NS: the prefill depths / decode loops; env MP: model directory (default model-2.6b; Ouro-1.4B: MP=model-1.4b
# CK=runs/O14-a-universal/checkpoint-48m.pt P=O14). Benchmarks already on this box are skipped. Queue runs/sweep_q.txt, two jobs per GPU.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/sweep_q.txt; cd $R
CK=${CK:-$R/X26-a-universal/checkpoint-48m.pt}; P=${P:-X26}; MP=$M/${MP:-model-2.6b}
for n in ${NS:-1 2}; do for m in ${MS:-5 6 8 12 16}; do
  c=a${m}x$n; MA="path=$MP,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { ls lmeval/out/$P-$c-$1/*/samples_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm math "--tasks minerva_math500 --num_fewshot 4"
  lm bbh "--tasks bbh_cot_fewshot --limit 60"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  lm mmlufull "--tasks mmlu --num_fewshot 5"; lm arc "--tasks arc_challenge --num_fewshot 25"
  ls bench/$P-${c}f/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$MP DATA=$M/bench BENCHES='ruler16k' ./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
done; done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
