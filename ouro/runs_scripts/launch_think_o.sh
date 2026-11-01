#!/bin/bash
# usage: launch_think_o.sh [NUM_GPUS=4]
# Arm O on Ouro-2.6B-Thinking: single prompt bank + one shared answer bank (self-live view), 48M KL tokens on the
# chat-template mix, teacher native 4/4 Thinking — long chains of thought are where storing the answer once matters.
# Then GSM8K-think (4 shards, 1,319) and MATH-500-think shard 0 for O42 (paired with T26-S42), and batch-1 end-to-end
# timing of O42sb (one prompt bank + one answer bank) on the same 65 GSM8K problems as think-S42sb.
NEED=${1:-4}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=T26-O-universal
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
[ -f $R/$N/TRAIN_DONE ] || MODEL=$M/model-2.6b-think DATA=$M/mix-think-48m ./run_mix2.sh $N O $GPUS 29791 48000000 --micro 1 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
MA="path=$M/model-2.6b-think,condition=O42,checkpoint=$CK,max_length=16384,batch_tokens=160000,max_batch=64"
{ echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think O42sb $CK gsm8k_think 65 $R/e2e/think-O42sb-gsm.jsonl > $R/e2e/think-O42sb-gsm.log 2>&1"
  for s in 0 1 2 3; do echo "./lm1.sh \$GPU $N-O42-gsm8k_think_s$s - \"$MA\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done
  echo "./lm1.sh \$GPU $N-O42-math500_think_s0 - \"$MA\" \"--tasks math500_think_s0 --apply_chat_template\""; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
