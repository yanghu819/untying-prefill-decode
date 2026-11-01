#!/bin/bash
# usage: launch_think_seed2.sh [NUM_GPUS=4]
# Training-seed robustness on the reasoning model: Ouro-2.6B-Thinking universal single-bank connector (arm S), same data
# and recipe as T26-S-universal, --seed 2. Then GSM8K-think (4 shards) for S41 S11 S42 S22 and HELMET (NO_EMPTY) for
# S41 S11 S42 S22, paired with seed 1.
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=T26-S-seed2
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
[ -f $R/$N/TRAIN_DONE ] || MODEL=$M/model-2.6b-think DATA=$M/mix-think-48m ./run_mix2.sh $N S $GPUS 29851 48000000 --micro 1 --seed 2 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
{ for c in S41 S11 S42 S22; do
    MA="path=$M/model-2.6b-think,condition=$c,checkpoint=$CK,max_length=16384,batch_tokens=160000,max_batch=64"
    for s in 0 1 2 3; do echo "./lm1.sh \$GPU $N-$c-gsm8k_think_s$s - \"$MA\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done
    echo "NO_EMPTY=1 MODEL=$M/model-2.6b-think DATA=$M/bench-mini BENCHES=helmet BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh T26s2ne-$c-helmet $c $CK \$GPU 1"
  done; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
