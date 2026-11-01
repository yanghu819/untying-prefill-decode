#!/bin/bash
# Diagnostic: Nanbeige single-bank connector trained on the read 2 / write 1 cell only (24M tokens, = specialist NBX-p2d1
# budget). Matches NBX-p2d1 -> the universal run is short of data per cell; falls short -> the single-bank key map
# (complex scale) is too restrictive for Nanbeige. Reserves GPUS from the dispatcher, then RULER 4K/8K/16K (520) + GSM8K/MMLU.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=NBS21-spec; G=${GPUS%%,*}
echo "$GPUS" | tr , " " >> /dev/shm/reserved_gpus
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || ./run_lb.sh $N S $GPUS 29831 $M/nanbeige-3b-base $M/mix-nanbeige 24000000 0 --s-cells 21 || exit 1
sed -i "s/\b${GPUS##*,}\b//" /dev/shm/reserved_gpus  # release the second GPU; the first runs the evals
CK=$R/$N/checkpoint-final.pt
MODEL=$M/nanbeige-3b-base PP=$M/tf457: BATCH=65536 BENCHES="ruler4k ruler8k ruler16k" ./bench2.sh $N-S21 S21 $CK $G 1
MA="path=$M/nanbeige-3b-base,condition=S21,checkpoint=$CK,max_length=16384"
./lm1.sh $G $N-S21-gsmfull $M/tf457: "$MA" "--tasks gsm8k_cot --num_fewshot 3"
./lm1.sh $G $N-S21-mmlufull $M/tf457: "$MA" "--tasks mmlu --num_fewshot 5"
sed -i "s/\b$G\b//" /dev/shm/reserved_gpus
