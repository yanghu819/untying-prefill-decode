#!/bin/bash
# time_t26d.sh GPU [RUN] [COND]: end-to-end time of a LoRA-distilled native 2/2 Ouro-2.6B-Thinking control (LoRA merged)
# on the same GSM8K problems as the other timed conditions (ids 1-64 and 65-192, the first record of each file a
# warm-up), batch 1, on a GPU with no other process (grid_worker.sh with MAX=1). RUN defaults to T26-D-native22 (COND
# D2st); the corrected control is T26-Df-native22 (COND Df2st). Output: runs/e2e/think-<COND>-gsm{,-more}.jsonl.
G=$1; RUN=${2:-T26-D-native22}; C=${3:-D2st}; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
for part in "0 65 " "64 129 -more"; do
  set -- $part
  echo "cd /root/big-small && START=$1 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think D2st $R/$RUN/checkpoint-48m.pt gsm8k_think $2 $R/e2e/think-$C-gsm$3.jsonl > $R/e2e/think-$C-gsm$3.log 2>&1"
done | flock $Q.lock sh -c "cat >> $Q"
setsid nohup ./grid_worker.sh $G $Q 1 > /dev/null 2>&1 < /dev/null &
