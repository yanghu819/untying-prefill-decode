#!/bin/bash
# Continue Lev's LoRA under the untied forward (state: READ prefill loops; branches: 1..4 loops reading the state's last-loop KV)
M=$1; BASE=$2; REV=$3; G=$4; STEPS=$5; OUT=$6; READ=${7:-4}; SEED=${8:-0}
cd /root/levwork/lev
export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HOME=/dev/shm/levwork/hf CUDA_VISIBLE_DEVICES=$G LEV_ROOT=/root/levwork/lev PYTHONPATH=/root/levwork/lev LEV_MODULE=kev.train LEV_READ=$READ
/root/levwork/env/bin/python /root/levwork/lev_rdws_eval.py --base $BASE --base_revision $REV --data /dev/shm/levwork/s31.jsonl \
  --init_from /dev/shm/levwork/$M --out $OUT --lora 16 --lr 1e-4 --epochs ${EPOCHS:-1} --batch 8 --accum 8 --dtype bf16 --checkpointing 1 \
  --device cuda --length_sort 1 --max_state 5120 --p_none_pair 0.25 --max_steps $STEPS --seed $SEED
echo DONE
