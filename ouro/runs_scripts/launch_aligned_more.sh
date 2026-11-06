#!/bin/bash
# launch_aligned_more.sh WHICH GPUS PORT: the aligned loop adapter (arm a) on the other checkpoints, trained exactly as
# their arm-S adapters. WHICH = 14: Ouro-1.4B (O14-a-universal, mix-data-48m), then its task-map benchmarks and RULER at
# a41..a44 (run names O14x-<cell>-<task>, RULER O14-<cell>f). WHICH = think: Ouro-2.6B-Thinking (T26-a-universal,
# mix-think-48m); its evaluation is queued separately (the Thinking benchmarks of Figure 4). WHICH = think14:
# Ouro-1.4B-Thinking (T14-a-universal), then its eight benchmarks.
W=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/aligned_q.txt; cd $R
if [ $W = think ]; then
  [ -f $R/T26-a-universal/TRAIN_DONE ] || MODEL=$M/model-2.6b-think DATA=$M/mix-think-48m ./run_mix2.sh T26-a-universal a $GPUS $PORT 48000000 --micro 1
  exit $?
fi
if [ $W = think14 ]; then   # Ouro-1.4B-Thinking: same data choice as launch_think14.sh, then eval_think_aligned.sh 14
  MOD=$M/model-1.4b-think; DATA=$M/mix-think-48m
  cmp -s $MOD/tokenizer.json $M/model-2.6b-think/tokenizer.json 2>/dev/null || DATA=$M/mix-think14-48m
  [ -f $R/T14-a-universal/TRAIN_DONE ] || MODEL=$MOD DATA=$DATA ./run_mix2.sh T14-a-universal a $GPUS $PORT 48000000 --micro 1 || exit 1
  exec ./eval_think_aligned.sh $GPUS 14
fi
[ -f $R/O14-a-universal/TRAIN_DONE ] || MODEL=$M/model-1.4b DATA=$M/mix-data-48m ./run_mix2.sh O14-a-universal a $GPUS $PORT 48000000 --micro 1 || exit 1
CK=$R/O14-a-universal/checkpoint-48m.pt
for c in a41 a42 a43 a44; do
  MA="path=$M/model-1.4b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { echo "./lm1.sh \$GPU O14x-$c-$1 - \"$MA\" \"$2\""; }
  lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"; lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  echo "cd /root/big-small && MODEL=$M/model-1.4b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh O14-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
