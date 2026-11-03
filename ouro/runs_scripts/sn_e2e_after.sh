#!/bin/bash
# after SenseNova loop adapter training: end-to-end batch-1 timing on GSM8K (65 problems) for native2, C11, C21, native1
until [ -f /dev/shm/big-small-ouro/runs/SN-C-universal/TRAIN_DONE ]; do sleep 120; done
CK=/dev/shm/big-small-ouro/runs/SN-C-universal/checkpoint.pt
for c in native2 C11 C21; do k=-; case $c in C*) k=$CK;; esac
echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/dev/shm/big-small-ouro/hf-lmeval PYTHONPATH=/dev/shm/big-small-ouro/tf518:/dev/shm/big-small-ouro/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py /dev/shm/big-small-ouro/model-sn9b $c $k gsm8k_think 65 /dev/shm/big-small-ouro/runs/e2e/sn-$c-gsm.jsonl > /dev/shm/big-small-ouro/runs/e2e/sn-$c-gsm.log 2>&1"
done | flock jobs.txt.lock sh -c "cat - jobs.txt > /tmp/jq_sne && cat /tmp/jq_sne > jobs.txt"
echo "$(date +%T) SN e2e queued"
