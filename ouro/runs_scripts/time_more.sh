#!/bin/bash
# time_more.sh GPU: 128 more GSM8K problems (ids 65-192; the first record, id 64, is the warm-up) for every
# configuration of the end-to-end comparison on Ouro-2.6B-Thinking, batch 1, one GPU with no other process. Files
# runs/e2e/think-<cond>-gsm-more.jsonl; paper_numbers.py merges them with the first 64 by item id.
G=$1; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
S=$R/T26-S-universal/checkpoint-48m.pt; D=$R/T26-D-native22/checkpoint-48m.pt
for c in "native2st -" "S42sb $S" "D2st $D" "S22sb $S" "native4st -"; do
  set -- $c
  echo "cd /root/big-small && START=64 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think $1 $2 gsm8k_think 129 $R/e2e/think-$1-gsm-more.jsonl > $R/e2e/think-$1-gsm-more.log 2>&1"
done | flock $Q.lock sh -c "cat >> $Q"
setsid nohup ./grid_worker.sh $G $Q 1 > /dev/null 2>&1 < /dev/null &
