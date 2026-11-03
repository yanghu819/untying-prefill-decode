#!/bin/bash
# Single-bank universal loop-map loop adapter (arm S): like V, but the key map commutes with RoPE, so the prompt keeps ONE
# KV bank for every decode loop (prompt KV memory = 1 loop at any d). Waits for 4 free GPUs, trains 48M KL tokens, then
# evaluates all 16 cells on RULER 16K (bench-mini), GSM8K (400), MMLU (20/subject).
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; rm -rf $R/smoke-S
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
G="$1,$2,$3,$4"; echo "$(date +%T) launching S on $G"
[ -f $R/X26-S-universal/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh X26-S-universal S $G 29751 48000000 --micro 1 || exit 1
CK=$R/X26-S-universal/checkpoint-48m.pt
i=0; IFS=, read -ra GS <<< "$G"
for c in S11 S21 S31 S41 S12 S22 S32 S42 S13 S23 S33 S43 S14 S24 S34 S44; do
  g=${GS[$((i % 4))]}; i=$((i + 1))
  ( MODEL=$M/model-2.6b DATA=$M/bench-mini BENCHES=ruler16k BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26-$c $c $CK $g 1
    ./lm1.sh $g X26-$c-gsm - "path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384" "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
    ./lm1.sh $g X26-$c-mmlu - "path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384" "--tasks mmlu --num_fewshot 5 --limit 20" ) &
  [ $((i % 4)) = 0 ] && wait
done
wait
