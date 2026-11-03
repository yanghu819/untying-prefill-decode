#!/bin/bash
# time_t26d.sh GPU: end-to-end time of the LoRA-distilled native 2/2 Ouro-2.6B-Thinking control (T26-D-native22, LoRA
# merged, condition D2st) on the same 65 GSM8K problems (first one warm-up) as the other timed conditions, batch 1, on
# a GPU with no other process (grid_worker.sh with MAX=1). Output: runs/e2e/think-D2st-gsm.jsonl.
G=$1; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think D2st $R/T26-D-native22/checkpoint-48m.pt gsm8k_think 65 $R/e2e/think-D2st-gsm.jsonl > $R/e2e/think-D2st-gsm.log 2>&1" \
  | flock $Q.lock sh -c "cat >> $Q"
setsid nohup ./grid_worker.sh $G $Q 1 > /dev/null 2>&1 < /dev/null &
