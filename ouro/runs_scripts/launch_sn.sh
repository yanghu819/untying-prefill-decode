#!/bin/bash
# usage: launch_sn.sh [NUM_GPUS=4]
# SenseNova-Loop-9B depth-split loop adapter (ouro/sensenova_bridge.py): one 1-visit decoder + LoRA reading a 1- or
# 2-visit prefill state (p sampled 50/50), KL to native 2/2, 20M KL tokens of the re-tokenised thinking mix. Then GSM8K
# (zero-shot chat, 4 shards) for C21 (prefill 2 / decode 1) and C11 (prefill 1 / decode 1, same decoder) on the dispatcher.
# GPUs are claimed one by one as they become idle (reserved from the dispatcher the moment they are claimed).
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=SN-C-universal
until [ "$(wc -l < $M/sn-mix/train.jsonl)" -ge 60000 ]; do sleep 60; done
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
cd /root/big-small
[ -f $R/$N/checkpoint.pt ] && [ -f $R/$N/TRAIN_DONE ] || {
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$M/tf518:/root/big-small environments/ouro/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node $NEED --master_port 29941 -m ouro.sensenova_bridge --model-path $M/model-sn9b --data $M/sn-mix/train.jsonl \
    --output $R/$N --kl-tokens 20e6 --tokens-per-step 65536 > $R/$N.log 2>&1 && touch $R/$N/TRAIN_DONE; } || exit 1
cd $R
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint.pt
{ for s in 0 1 2 3; do for c in C21 C11; do
    echo "./lm1.sh \$GPU SN-$c-gsm8k_think_s$s $M/tf518: \"path=$M/model-sn9b,condition=$c,checkpoint=$CK,max_length=16384,batch_tokens=250000,max_batch=64\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done; done; } \
  | flock jobs.txt.lock sh -c "cat - jobs.txt > /tmp/jq_sn && cat /tmp/jq_sn > jobs.txt"
echo "$(date +%T) evals queued"
