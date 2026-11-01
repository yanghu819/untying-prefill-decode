#!/bin/bash
# nbs_cell.sh GPUS NAME PORT : Nanbeige4.2-3B universal single-bank connector (arm S, one map set for every prefill p /
# decode d in 1..2, cell sampled per global batch). Train 24M KL tokens on GPUS, then evaluate the four cells
# S11 S21 S12 S22 one per GPU of GPUS (needs >= 4 GPUs): RULER 4K/8K/16K (520 rows), full GSM8K, full MMLU.
GPUS=$1 NAME=$2 PORT=$3
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
[ -f $R/$NAME/TRAIN_DONE ] || ./run_lb.sh $NAME S $GPUS $PORT $M/nanbeige-3b-base $M/mix-nanbeige 24000000 0 || exit 1
CK=$R/$NAME/checkpoint-final.pt; IFS=, read -ra G <<< "$GPUS"; i=0
for c in S21 S11 S22 S12; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
  ( MODEL=$M/nanbeige-3b-base PP=$M/tf457: BATCH=65536 BENCHES="ruler4k ruler8k ruler16k" ./bench2.sh $NAME-$c $c $CK $g 1
    MA="path=$M/nanbeige-3b-base,condition=$c,checkpoint=$CK,max_length=16384"
    ./lm1.sh $g $NAME-$c-gsmfull $M/tf457: "$MA" "--tasks gsm8k_cot --num_fewshot 3"
    ./lm1.sh $g $NAME-$c-mmlufull $M/tf457: "$MA" "--tasks mmlu --num_fewshot 5" ) &
done
wait
