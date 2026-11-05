#!/bin/bash
# time_small.sh GPU: end-to-end time of native Ouro-1.4B-Thinking 4/4 and 2/2 on the same 192 GSM8K problems as the
# Ouro-2.6B-Thinking timing (ids 1-64 and 65-192, the first record of each file a warm-up), batch 1, one GPU with no
# other process: the smaller model at equal answer time. Output: runs/e2e/think14-<cond>-gsm{,-more}.jsonl.
G=$1; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing14_q.txt; mkdir -p $R/e2e; cd $R
for c in native4st native2st; do
  for part in "0 65 " "64 129 -more"; do
    set -- $part
    echo "cd /root/big-small && START=$1 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-1.4b-think $c - gsm8k_think $2 $R/e2e/think14-$c-gsm$3.jsonl > $R/e2e/think14-$c-gsm$3.log 2>&1"
  done
done | flock $Q.lock sh -c "cat >> $Q"
setsid nohup ./grid_worker.sh $G $Q 1 > /dev/null 2>&1 < /dev/null &
