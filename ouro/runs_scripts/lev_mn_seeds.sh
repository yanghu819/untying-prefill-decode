#!/bin/bash
# Lev decision models, fewer prefill than decision loops (m < n): seeds 1-3 for every prefill count of lev_mn.sh, so
# the m < n direction has the same four-seed replication as the main comparison (controls ctl-*-s{1,2,3} exist).
# Queues one line per run (train, then read LevBench development rows) and starts one grid_worker.sh per GPU; a worker
# only takes a job when its GPU is completely free (Lev training needs most of a B300).
Q=/dev/shm/big-small-ouro/runs/lev_seeds_q.txt; cd /dev/shm/big-small-ouro/runs
R14=574fa66cb8bf5abdc979642d01cf2b79b16bfab1; R26=1ed04250da1a9936042725d302e81c8fa2ab5abd; R3=4a38e817c14b9f2c4b69b1e05cbda90395872f6b
mkdir -p /dev/shm/levwork/train /dev/shm/levwork/rows
job() {  # model base revision name m seed
  echo "/root/levwork/train_rd.sh $1 $2 $3 \$GPU 0 /dev/shm/levwork/train/$4 $5 $6 > /dev/shm/levwork/train-$4.log 2>&1 && /root/levwork/after_train.sh $4 \$GPU $5"
}
{
  for s in 1 2 3; do
    for m in 1 2 3; do
      job Lev-1.4B ByteDance/Ouro-1.4B $R14 rd$m-1.4B-s$s $m $s
      job Lev-2.6B ByteDance/Ouro-2.6B $R26 rd$m-2.6B-s$s $m $s
    done
    job Lev-3B Nanbeige/Nanbeige4.2-3B-Base $R3 rd1-3B-s$s 1 $s
  done
} | flock $Q.lock sh -c "cat >> $Q"
for g in 0 1 2 3 4 5 6 7; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
