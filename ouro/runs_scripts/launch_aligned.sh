#!/bin/bash
# launch_aligned.sh GPUS PORT: the aligned loop adapter (arm a). Decode loop k of n reads prefill loop clip(m-n+k, 0, m-1)
# through its own map (one prompt cache per decode loop) instead of the last prefill loop, so at m = n it reads exactly
# the native model's KV. Trained as X26-S-universal (mix-data-48m, 48M tokens, global batch 24, all 16 (m, n) pairs);
# then the main-table benchmarks and RULER 4K/8K/16K at a41..a44 and the diagonal a22, a33, two jobs per GPU.
# env CELLS: other cells of the map (default a41 a42 a43 a44 a22 a33).
# env SEED=2: another training seed (X26-a-seed2, run names X26as2-<cell>-...), cells a41 a42 by default.
# env TOK=12: a shorter run (X26-a-12m, run names X26a12-<cell>-...) to compare with the arm-S ablation X26-S-12m.
GPUS=$1; PORT=$2; T=${TOK:-48}; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/aligned_q.txt; cd $R
if [ -n "$SEED" ]; then N=X26-a-seed$SEED; P=X26as$SEED; CELLS=${CELLS:-a41 a42}; EXTRA="--seed $SEED"
elif [ $T = 48 ]; then N=X26-a-universal; P=X26; else N=X26-a-${T}m; P=X26a$T; fi
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N a $GPUS $PORT ${T}000000 --micro 1 $EXTRA || exit 1
CK=$(ls $R/$N/checkpoint-*.pt | tail -1)
for c in ${CELLS:-a41 a42 a43 a44 a22 a33}; do
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { echo "./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
