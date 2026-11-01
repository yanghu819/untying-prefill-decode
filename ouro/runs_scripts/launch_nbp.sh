#!/bin/bash
# Nanbeige universal single-bank connector with maps indexed by (prefill depth p, decode loop k) (arm P), 24M tokens over
# the 4 cells. Fix for NBS-universal, whose maps (indexed by k only) had to serve both loop-1 and loop-2 KV.
# Reserves GPUS from the dispatcher; then RULER 4K/8K/16K (520 rows) for P11 P21 P12 P22, one cell per GPU.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=NBP-universal
echo " ${GPUS//,/ }" >> /dev/shm/reserved_gpus
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || ./run_lb.sh $N P $GPUS 29851 $M/nanbeige-3b-base $M/mix-nanbeige 24000000 0 || exit 1
CK=$R/$N/checkpoint-final.pt; IFS=, read -ra G <<< "$GPUS"; i=0
for c in P21 P22 P11 P12; do
  g=${G[$((i % ${#G[@]}))]}; i=$((i + 1))
  MODEL=$M/nanbeige-3b-base PP=$M/tf457: BATCH=65536 BENCHES="ruler4k ruler8k ruler16k" ./bench2.sh $N-$c $c $CK $g 1 &
  [ $((i % ${#G[@]})) = 0 ] && wait
done
wait
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
