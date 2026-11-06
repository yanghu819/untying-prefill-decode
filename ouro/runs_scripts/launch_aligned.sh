#!/bin/bash
# launch_aligned.sh GPUS PORT: the aligned loop adapter (arm a). Decode loop k of n reads prefill loop clip(m-n+k, 0, m-1)
# through its own map (one prompt cache per decode loop) instead of the last prefill loop, so at m = n it reads exactly
# the native model's KV. Trained as X26-S-universal (mix-data-48m, 48M tokens, global batch 24, all 16 (m, n) pairs);
# then the main-table benchmarks and RULER 4K/8K/16K at a41..a44 and the diagonal a22, a33, two jobs per GPU.
GPUS=$1; PORT=$2; M=/dev/shm/big-small-ouro; R=$M/runs; N=X26-a-universal; Q=$R/aligned_q.txt; cd $R
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 || exit 1
CK=$R/$N/checkpoint-48m.pt
for c in a41 a42 a43 a44 a22 a33; do
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { echo "./lm1.sh \$GPU X26-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
