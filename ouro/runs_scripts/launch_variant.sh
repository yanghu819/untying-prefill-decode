#!/bin/bash
# launch_variant.sh WHICH GPUS PORT: training variants of the aligned loop adapter for the restructured study
# (same data, tokens, batch, teacher native 4/4 and seed as X26-a-universal; only the trained (m, n) cells or the maps
# differ), then their evaluation.
#   wide:   X26-a-wide, trained on prefill m in {1,2,3,4,6,8,12,16} x decode n in {1..4}, so the depth sweep beyond 4 is
#           inside the training range; evaluated by sweep_depth.sh at every m (run names X26w-a<m>x<n>-<task>).
#   fixed4: X26-a-fixed4-lora, prefill fixed at 4 (cells 4/1..4/4) and identity maps (--freeze-maps): the simple
#           "deep prefill + LoRA" control; evaluated on the main-table benchmarks at 4/1 and 4/2 (X26f4-a4<n>-<task>).
W=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
case $W in
  wide) N=X26-a-wide; P=X26w; FLAGS=""
        export OURO_CELLS=$(for m in 1 2 3 4 6 8 12 16; do for n in 1 2 3 4; do printf "%sx%s," $m $n; done; done | sed 's/,$//') ;;
  fixed4) N=X26-a-fixed4-lora; P=X26f4; FLAGS="--freeze-maps"; export OURO_CELLS=4x1,4x2,4x3,4x4 ;;
  *) echo "WHICH must be wide or fixed4"; exit 2 ;;
esac
[ -f $R/$N/TRAIN_DONE ] || [ -f $R/$N/checkpoint-48m.pt ] || \
  DATA=$M/mix-data-48m ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 $FLAGS || exit 1
unset OURO_CELLS
CK=$R/$N/checkpoint-48m.pt
if [ $W = wide ]; then exec env CK=$CK P=$P MS="1 2 3 4 5 6 8 12 16" ./sweep_depth.sh ${GPUS//,/ }; fi
Q=$R/sweep_q.txt
for c in a41 a42; do
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
