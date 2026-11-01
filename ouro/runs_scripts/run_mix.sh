#!/bin/bash
# usage: run_mix.sh NAME ARM GPUS PORT STOP [EXTRA...]   fresh decode model on the diverse mix (mix-data), LR decays to the end at STOP
NAME=$1; ARM=$2; GPUS=$3; PORT=$4; STOP=$5; shift 5
cd /root/big-small; M=/dev/shm/big-small-ouro; O=$M/runs/$NAME; PY=environments/ouro/.venv/bin/python; mkdir -p $O
N=$(echo $GPUS | tr , "\n" | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS OURO_NO_CUDNN_SDP=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 $PY -m torch.distributed.run --nproc_per_node $N --master_addr 127.0.0.1 --master_port $PORT --module ouro.train_batched --model-path ${MODEL:-$M/model-2.6b} --data $M/mix-data --output $O --arm $ARM --micro 3 --stop-tokens $STOP --planned-tokens $STOP --global-batch 24 "$@" > $O/train.log 2>&1 || exit 1
touch $O/TRAIN_DONE
