#!/bin/bash
# Ouro-2.6B-Thinking single-bank universal connector (arm S, 48M KL tokens on the chat-template mix, teacher = native
# 4/4 Thinking). Reserves GPUS (dispatcher skips them), waits until they are idle, trains, then frees them and queues
# the reasoning evals (zero-shot chat template: MATH-500, GSM8K) for S4d / Sdd, d = 1..3.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=T26-S-universal
echo "$GPUS" | tr , " " > /dev/shm/reserved_gpus
until [ -f $M/mix-think-48m/manifest.json ] && grep -q THINK_DONE /dev/shm/dlthink.log; do sleep 60; done
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS"
[ -f $R/$N/TRAIN_DONE ] || MODEL=$M/model-2.6b-think DATA=$M/mix-think-48m ./run_mix2.sh $N S $GPUS 29781 48000000 --micro 1 || { rm -f /dev/shm/reserved_gpus; exit 1; }
rm -f /dev/shm/reserved_gpus
CK=$R/$N/checkpoint-48m.pt
for c in S41 S11 S42 S22 S43 S33; do
  MA="path=$M/model-2.6b-think,condition=$c,checkpoint=$CK,max_length=16384,batch_tokens=262144,max_batch=64"
  echo "./lm1.sh \$GPU $N-$c-math500 - \"$MA\" \"--tasks math500_think --apply_chat_template\""
  echo "./lm1.sh \$GPU $N-$c-gsm - \"$MA\" \"--tasks gsm8k_think --apply_chat_template\""
done | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
