#!/bin/bash
# run_lb.sh NAME ARM GPUS PORT MODEL_DIR DATA_DIR STOP_TOKENS MAX_LEN [EXTRA...] : looped_bridge training (resumes)
NAME=$1 ARM=$2 GPUS=$3 PORT=$4 MODEL=$5 DATA=$6 STOP=$7 MAXLEN=$8; shift 8
M=/dev/shm/big-small-ouro; OUT=$M/runs/$NAME; mkdir -p $OUT; cd /root/big-small
N=$(echo $GPUS | tr , '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$M/tf457:/root/big-small \
  environments/ouro/.venv/bin/torchrun --nproc-per-node $N --master-port $PORT ouro/looped_bridge.py \
  --model-path $MODEL --data $DATA --output $OUT --arm $ARM --stop-tokens $STOP --max-len $MAXLEN --save-every 50 "$@" >> $OUT/train.log 2>&1
