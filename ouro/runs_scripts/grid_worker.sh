#!/bin/bash
# grid_worker.sh GPU QUEUE [MAX=1]: take the next job from QUEUE (one shell line using $GPU) whenever this GPU has fewer
# than MAX compute processes, run it, repeat until the queue is empty. Two workers with MAX=2 share one GPU between small
# evaluation jobs (Ouro-1.4B lm-eval uses a fraction of a B300). Used for the (m, n) task-map grids.
G=$1; Q=$2; MAX=${3:-1}; cd /dev/shm/big-small-ouro/runs
while true; do
  until [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | wc -l)" -lt $MAX ]; do sleep 60; done
  J=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$J" ] && exit 0
  echo "$(date +%T) gpu$G: $J" >> $Q.log
  GPU=$G bash -c "$J" > /dev/null 2>&1 < /dev/null
done
