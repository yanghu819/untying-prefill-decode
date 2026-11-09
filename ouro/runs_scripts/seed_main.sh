#!/bin/bash
# seed_main.sh SEED GPUS PORT: another training seed of the Ouro-2.6B loop adapter for the main table. Trains
# X26-S-seed<SEED> exactly as X26-S-universal but with --seed SEED (skipped if it exists), then queues the main-table
# benchmarks at prefill 4 / decode 1 and 2 that are not yet on this box (run names X26s<SEED>-S4<n>-<task>, RULER
# X26s<SEED>-S4<n>f) and runs them two per GPU on GPUS. ARM=a does the same for the aligned adapter (X26-a-seed<SEED>,
# cells a41/a42); with an existing checkpoint it only queues the benchmarks.
S=$1; GPUS=$2; PORT=$3; A=${ARM:-S}; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/seed${S}main${A#S}_q.txt; N=X26-$A-seed$S; P=X26s$S; cd $R
[ -f $R/$N/checkpoint-48m.pt ] || { DATA=$M/mix-data-48m ./run_mix2.sh $N $A $GPUS $PORT 48000000 --micro 1 --seed $S || exit 1; }
CK=$R/$N/checkpoint-48m.pt
for c in $(for x in ${CELLS:-41 42}; do echo $A$x; done); do   # env CELLS="11 12": the prefill-1 cells (same-weights depth gain)
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { ls lmeval/out/$P-$c-$1/*/samples_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$P-${c}f/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler16k' ./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
