#!/bin/bash
# usage: launch_fz.sh [NUM_GPUS=3]
# Control for "the map is what makes shallow writing over deep KV work": the universal single-bank loop adapter (arm S, same
# data, budget, seed and cell sampling) with the per-loop KV maps frozen at identity (--freeze-maps), i.e. LoRA-only direct
# sharing of the last prefill loop's KV. Then GSM8K / MMLU full and RULER 4K-16K (520) for S41 S11 S42 S22.
NEED=${1:-3}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-Sfz-universal
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
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N S $GPUS 29861 48000000 --micro 1 --freeze-maps || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
{ for c in S41 S11 S42 S22; do MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
    echo "MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26fz-${c}f $c $CK \$GPU 1"
    echo "./lm1.sh \$GPU X26fz-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
    echo "./lm1.sh \$GPU X26fz-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""; done; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
