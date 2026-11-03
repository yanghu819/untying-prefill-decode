#!/bin/bash
# Lev decision models, the other direction (prefill m < decision n): continue Lev's LoRA with the state prefilled by
# m loops (LEV_READ=m), then read LevBench development rows at every decision depth with the same m. One GPU per run.
# Prefill 4 (Lev-1.4B/2.6B) and prefill 2 (Lev-3B) are the existing rd4-* / rd2-* runs.
R14=574fa66cb8bf5abdc979642d01cf2b79b16bfab1; R26=1ed04250da1a9936042725d302e81c8fa2ab5abd; R3=4a38e817c14b9f2c4b69b1e05cbda90395872f6b
run() {  # model base revision gpu name m
  /root/levwork/train_rd.sh $1 $2 $3 $4 0 /dev/shm/levwork/train/$5 $6 0 > /dev/shm/levwork/train-$5.log 2>&1
  /root/levwork/after_train.sh $5 $4 $6
}
mkdir -p /dev/shm/levwork/train /dev/shm/levwork/rows
g=0
for m in 1 2 3; do
  (run Lev-1.4B ByteDance/Ouro-1.4B $R14 $g rd$m-1.4B $m &) ; g=$((g+1))
  (run Lev-2.6B ByteDance/Ouro-2.6B $R26 $g rd$m-2.6B $m &) ; g=$((g+1))
done
(run Lev-3B Nanbeige/Nanbeige4.2-3B-Base $R3 $g rd1-3B 1 &)
