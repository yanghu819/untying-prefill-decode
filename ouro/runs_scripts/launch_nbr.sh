#!/bin/bash
# Nanbeige universal single-bank connector, maps indexed by (p, k) AND one decode LoRA per prefill depth p (arm R),
# 24M tokens over the 4 cells. NBP (maps per (p, k), one shared LoRA) still lost ~13 to the specialists at read 2 with
# worse KL than the specialist at equal per-cell tokens -> the shared LoRA interferes. Then RULER for R11 R21 R12 R22.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=NBR-universal
echo " ${GPUS//,/ }" >> /dev/shm/reserved_gpus
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || ./run_lb.sh $N R $GPUS 29861 $M/nanbeige-3b-base $M/mix-nanbeige 24000000 0 || exit 1
CK=$R/$N/checkpoint-final.pt; IFS=, read -ra G <<< "$GPUS"; i=0
for c in R21 R22 R11 R12; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
  MODEL=$M/nanbeige-3b-base PP=$M/tf457: BATCH=65536 BENCHES="ruler4k ruler8k ruler16k" ./bench2.sh $N-$c $c $CK $g 1 &
  [ $((i % ${#G[@]})) = 0 ] && wait
done
wait
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
