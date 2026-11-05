#!/bin/bash
# ablate_tokens.sh TOKENS_M GPUS PORT: training-data ablation of the Ouro-2.6B loop adapter. Trains X26-S-<T>m exactly as
# X26-S-universal (arm S, mix-data-48m, global batch 24, learning rate decaying to the end) but stops at T million answer
# tokens, then queues the main-table benchmarks at prefill 4 / decode 1 and 2 (run names X26t<T>-S4<n>-<task>, RULER
# X26t<T>-S4<n>f) and runs them two per GPU on the training GPUs.
# env RANK: LoRA rank ablation at the same tokens (run X26-S-<T>m-r<RANK>, names X26t<T>r<RANK>-...; default rank 16).
T=$1; GPUS=$2; PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; N=X26-S-${T}m; P=X26t$T; Q=$R/tokens_q.txt; cd $R
E=""; [ -n "$RANK" ] && { N=$N-r$RANK; P=${P}r$RANK; E="OURO_LORA_RANK=$RANK "; export OURO_LORA_RANK=$RANK; }
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N S $GPUS $PORT ${T}000000 --micro 1 || exit 1
CK=$(ls $R/$N/checkpoint-*.pt | tail -1)
for c in S41 S42; do
  MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { echo "$E./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  echo "cd /root/big-small && ${E}MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler16k' ./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
