#!/bin/bash
# time_think_aligned.sh: end-to-end time of the aligned loop adapter on Ouro-2.6B-Thinking at 4/2 and 2/2 on the same
# GSM8K problems as the other timed Thinking conditions (ids 1-64 and 65-192, the first record of each file a warm-up),
# batch 1, exclusive GPU (appended to runs/timing_q.txt, MAX=1). Output: runs/e2e/think-a<p><d>sb-gsm{,-more}.jsonl.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
A=$R/T26-a-universal/checkpoint-48m.pt
for c in a42sb a22sb; do
  for part in "0 65 " "64 129 -more"; do
    set -- $part
    echo "cd /root/big-small && START=$1 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think $c $A gsm8k_think $2 $R/e2e/think-$c-gsm$3.jsonl > $R/e2e/think-$c-gsm$3.log 2>&1"
  done
done | flock $Q.lock sh -c "cat >> $Q"
