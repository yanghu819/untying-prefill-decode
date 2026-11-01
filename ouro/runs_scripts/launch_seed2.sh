#!/bin/bash
# usage: launch_seed2.sh [NUM_GPUS=4]
# Training-seed robustness of the universal single-bank connector (arm S, Ouro-2.6B): same data and recipe as
# X26-S-universal, --seed 2 (LoRA init + per-micro-batch cell sampling). Then the paper's key within-connector
# comparisons on full sets: GSM8K (1,319, 3-shot CoT), MMLU (14,042, 5-shot), RULER 4K/8K/16K (520 each) for S41 S11 S42 S22.
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-S-seed2
got=""
while [ $(echo $got | wc -w) -lt $NEED ]; do
  for g in 0 1 2 3 4 5 6 7; do
    [ $(echo $got | wc -w) -ge $NEED ] && break
    echo " $got " | grep -q " $g " && continue
    grep -qw $g /dev/shm/reserved_gpus 2>/dev/null && continue
    if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then got="$got $g"; echo " $g" >> /dev/shm/reserved_gpus; fi
  done
  sleep 10
done
GPUS=$(echo $got | tr " " ",")
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N S $GPUS 29761 48000000 --micro 1 --seed 2 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
{ for c in S41 S11 S42 S22; do MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
    echo "MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26s2-${c}f $c $CK \$GPU 1"
    echo "./lm1.sh \$GPU X26s2-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
    echo "./lm1.sh \$GPU X26s2-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""; done; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
