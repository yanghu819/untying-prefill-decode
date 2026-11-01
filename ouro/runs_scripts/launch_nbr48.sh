#!/bin/bash
# Nanbeige universal arm R (maps per (p, k), LoRA per p) with 48M tokens (all of mix-nanbeige-48m) = 12M per cell, 2x the first run.
# NBR48-universal (24M total) matched the specialists on RULER 4K but lost 10-20 on GSM8K: data per cell or not?
# Then RULER (520) for R21 R22 R11 R12 and full GSM8K for R21, R11.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=NBR48-universal
echo " ${GPUS//,/ }" >> /dev/shm/reserved_gpus
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || ./run_lb.sh $N R $GPUS 29871 $M/nanbeige-3b-base $M/mix-nanbeige-48m 48000000 0 || exit 1
CK=$R/$N/checkpoint-final.pt; IFS=, read -ra G <<< "$GPUS"; i=0
for c in R21 R22 R11 R12; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
  MODEL=$M/nanbeige-3b-base PP=$M/tf457: BATCH=65536 BENCHES="ruler4k ruler8k ruler16k" ./bench2.sh $N-$c $c $CK $g 1 &
  [ $((i % ${#G[@]})) = 0 ] && wait
done
wait
MA="path=$M/nanbeige-3b-base,checkpoint=$CK,max_length=16384"
for c in R21 R11; do ./lm1.sh ${G[0]} $N-$c-gsmfull $M/tf457: "$MA,condition=$c" "--tasks gsm8k_cot --num_fewshot 3" & ./lm1.sh ${G[1]} $N-$c-mmlufull $M/tf457: "$MA,condition=$c" "--tasks mmlu --num_fewshot 5"; wait; done
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
