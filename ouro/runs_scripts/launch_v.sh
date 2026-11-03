#!/bin/bash
# DEPRECATED (arm V, superseded by arm S; see DEPRECATED.md)
# Universal loop-map loop adapter (arm V) on Ouro-2.6B: one LoRA + one per-(decode loop, layer, head) KV map for all 16
# cells p, d in 1..4 (prefill costs exactly p loops, so p < d is allowed). Waits for 4 free GPUs, trains 48M KL tokens,
# then evaluates every cell on the three main tasks: RULER 16K (bench-mini), GSM8K (400), MMLU (20/subject).
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; rm -rf $R/smoke-V
while true; do
  free=""
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$n" = 0 ] && [ "$u" -lt 2000 ] && free="$free $g"
  done
  set -- $free
  [ $# -ge 4 ] && break
  sleep 60
done
G="$1,$2,$3,$4"; echo "$(date +%T) launching V on $G"
[ -f $R/X26-V-universal/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh X26-V-universal V $G 29721 48000000 --micro 1 || exit 1
CK=$R/X26-V-universal/checkpoint-48m.pt
i=0; IFS=, read -ra GS <<< "$G"
for c in V11 V21 V31 V41 V12 V22 V32 V42 V13 V23 V33 V43 V14 V24 V34 V44; do
  g=${GS[$((i % 4))]}; i=$((i + 1))
  ( MODEL=$M/model-2.6b DATA=$M/bench-mini BENCHES=ruler16k BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26-$c $c $CK $g 1
    ./lm1.sh $g X26-$c-gsm - "path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384" "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
    ./lm1.sh $g X26-$c-mmlu - "path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384" "--tasks mmlu --num_fewshot 5 --limit 20" ) &
  [ $((i % 4)) = 0 ] && wait
done
wait
