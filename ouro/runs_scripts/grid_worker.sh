#!/bin/bash
# grid_worker.sh GPU QUEUE: take the next job from QUEUE (one shell line using $GPU) whenever this GPU has no compute
# process, run it, repeat until the queue is empty. Used for the Ouro-2.6B (m, n) task-map grids.
G=$1; Q=$2; cd /dev/shm/big-small-ouro/runs
while true; do
  until [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  J=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$J" ] && exit 0
  echo "$(date +%T) gpu$G: $J" >> $Q.log
  GPU=$G bash -c "$J" > /dev/null 2>&1 < /dev/null
done
