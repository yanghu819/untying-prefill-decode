#!/bin/bash
# when a Lev training run finishes, read its LevBench development rows (READ = state loops at evaluation)
NAME=$1; G=$2; R=$3
until grep -q "^DONE" /dev/shm/levwork/train-$NAME.log 2>/dev/null; do sleep 60; done
cd /root/levwork/lev
export HF_HUB_OFFLINE=1 HF_HOME=/dev/shm/levwork/hf CUDA_VISIBLE_DEVICES=$G LEV_ROOT=/root/levwork/lev PYTHONPATH=/root/levwork/lev LEV_SCRIPT=scripts/depth_rows.py LEV_READ=$R
/root/levwork/env/bin/python /root/levwork/lev_rdws_eval.py --run /dev/shm/levwork/train/$NAME --read /dev/shm/levwork/levbench-dev.jsonl --out /dev/shm/levwork/rows/$NAME-read$R > /dev/shm/levwork/rows/$NAME-read$R.log 2>&1
echo DONE >> /dev/shm/levwork/rows/$NAME-read$R.log
