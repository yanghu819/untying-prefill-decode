#!/bin/bash
# launch_variant.sh WHICH GPUS PORT: training variants of the aligned loop adapter for the restructured study
# (same data, tokens, batch, teacher native 4/4 and seed as X26-a-universal; only the trained (m, n) cells or the maps
# differ), then their evaluation.
#   wide:   X26-a-wide, trained on prefill m in {1,2,3,4,6,8,12,16} x decode n in {1..4}, so the depth sweep beyond 4 is
#           inside the training range; evaluated by sweep_depth.sh at every m (run names X26w-a<m>x<n>-<task>).
#   fixed4: X26-a-fixed4-lora, prefill fixed at 4 (cells 4/1..4/4) and identity maps (--freeze-maps): the simple
#           "deep prefill + LoRA" control; evaluated on the main-table benchmarks at 4/1 and 4/2 (X26f4-a4<n>-<task>).
#   prefill: X26-a-pf8, the aligned adapter plus a LoRA on the prefill (writer) attention, trained on m in {1,2,3,4,6,8}
#           (a first try on the wide cells up to m=16 hit the 10-min NCCL timeout before its first step)
#           (--prefill-lora; training prompts cut to their last OURO_MAX_PREFIX=2048 tokens to bound the prefill's backward
#           memory at 16 loops; evaluation is not cut, so long-context scores test lengths it did not train on); depth
#           sweep as for wide, m up to 16 (run names X26pf8-a<m>x<n>-<task>). Tests whether the stop at
#           2-4 prefill loops comes from the frozen prefill (red team rounds 3-5).
#   w8:     X26-a-w8, its matched control: same cells (m <= 8) and prompt cut, prefill frozen (run names X26w8-...).
#   lr:     X26-a-lr<LRS>, the main adapter with the learning-rate schedule x LRS (env LRS), so Ours gets the same LR
#           check as the Distilled baseline; main-table benchmarks at 4/1 and 4/2 (X26alr<LRS>-a4<n>-<task>).
#   lora:   X26-a-lora, all 16 cells as the main adapter but identity maps (--freeze-maps): isolates the maps in the
#           aligned layout (red-team round 2); main-table benchmarks at 4/1 and 4/2 (X26lo-a4<n>-<task>). With RANK=80
#           (X26-a-lora-r80, X26lor80-...) it has as many trainable parameters as LoopBridge (review 11-13).
W=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
case $W in
  wide) N=X26-a-wide; P=X26w; FLAGS=""
        export OURO_CELLS=$(for m in 1 2 3 4 6 8 12 16; do for n in 1 2 3 4; do printf "%sx%s," $m $n; done; done | sed 's/,$//') ;;
  fixed4) N=X26-a-fixed4-lora; P=X26f4; FLAGS="--freeze-maps"; export OURO_CELLS=4x1,4x2,4x3,4x4 ;;
  lora) N=X26-a-lora; P=X26lo; FLAGS="--freeze-maps" ;;
  w8) N=X26-a-w8; P=X26w8; FLAGS=""; export OURO_MAX_PREFIX=${OURO_MAX_PREFIX:-2048}   # control for prefill: same cells
        export OURO_CELLS=$(for m in 1 2 3 4 6 8; do for n in 1 2 3 4; do printf "%sx%s," $m $n; done; done | sed 's/,$//') ;;   # and cut, frozen prefill
  lr) N=X26-a-lr${LRS:?set LRS}; P=X26alr${LRS}; FLAGS=""; export OURO_LR_SCALE=$LRS ;;   # Ours at LR x LRS (tuning check)
  prefill) N=X26-a-pf8; P=X26pf8; FLAGS="--prefill-lora"; export OURO_MAX_PREFIX=${OURO_MAX_PREFIX:-2048}
        export OURO_DIST_TIMEOUT_MIN=${OURO_DIST_TIMEOUT_MIN:-30} OURO_DDP_FIND_UNUSED=1   # without it DDP hangs (gradient
        # readiness order differs across ranks when each samples a different m; found 11-11 by 1- and 2-GPU smoke tests)
        export OURO_CELLS=$(for m in 1 2 3 4 6 8; do for n in 1 2 3 4; do printf "%sx%s," $m $n; done; done | sed 's/,$//') ;;
  *) echo "WHICH must be wide, fixed4, lora, lr, prefill or w8"; exit 2 ;;
esac
[ -n "$SEED" ] && { N=$N-s$SEED; P=${P}s$SEED; FLAGS="$FLAGS --seed $SEED"; }   # env SEED: another training seed (X26-a-lora-s2, X26los2-...)
EPRE=""; [ -n "$RANK" ] && { N=$N-r$RANK; P=${P}r$RANK; export OURO_LORA_RANK=$RANK; EPRE="OURO_LORA_RANK=$RANK "; }   # env RANK:
# LoRA rank (lora variant at RANK=80: an all-(m,n) LoRA with as many parameters as LoopBridge, X26-a-lora-r80, X26lor80-...)
[ -f $R/$N/TRAIN_DONE ] || [ -f $R/$N/checkpoint-48m.pt ] || \
  DATA=$M/mix-data-48m ./run_mix2.sh $N a $GPUS $PORT 48000000 --micro 1 $FLAGS || exit 1
unset OURO_CELLS
CK=$R/$N/checkpoint-48m.pt
if [ $W = wide ] || [ $W = prefill ] || [ $W = w8 ]; then exec env CK=$CK P=$P MS="1 2 3 4 5 6 8 12 16" ./sweep_depth.sh ${GPUS//,/ }; fi
Q=$R/sweep_q.txt
for c in a41 a42; do
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { ls lmeval/out/$P-$c-$1/*/samples_* > /dev/null 2>&1 || echo "$EPRE./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  ls bench/$P-${c}f/s*/ruler16k.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler16k' $EPRE./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
# workers must not inherit OURO_LORA_RANK (exported above for training): they also run other runs' queue lines, which
# carry their own prefix (11-13: rank-80 workers failed 24 seed-3 sweep jobs with a rank mismatch)
for g in ${GPUS//,/ }; do env -u OURO_LORA_RANK setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
