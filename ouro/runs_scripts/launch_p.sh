#!/bin/bash
# usage: launch_p.sh [NUM_GPUS=4]
# Arm P: the ORIGINAL self-live recipe (the author's method) on the single-bank connector — main loss = per-loop answer
# banks (the S objective), auxiliary 0.1 x KL in the self-live view, history from one fixed-point iteration of the model's
# own compressed pass. Same data and budget as S / O (48M KL tokens). Then GSM8K / MMLU full, MATH-500, RULER 4K-16K for
# P42 P43 P44 (compare with O and S at the same cells).
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-P-universal
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
[ -f $R/$N/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh $N P $GPUS 29831 48000000 --micro 1 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
{ for c in P42 P43 P44; do MA="path=$M/model-2.6b,condition=$c,checkpoint=$CK,max_length=16384"
    echo "MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh X26-${c}f $c $CK \$GPU 1"
    echo "./lm1.sh \$GPU X26-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
    echo "./lm1.sh \$GPU X26-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""
    echo "./lm1.sh \$GPU o26x-$c-math - \"$MA\" \"--tasks minerva_math500\""; done; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
