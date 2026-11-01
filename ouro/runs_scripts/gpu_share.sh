#!/bin/bash
# gpu_share.sh JOBFILE : companion of gpu_dispatch.sh for decode-bound eval queues. Adds a SECOND job to a GPU that runs
# exactly one compute process (never a latency / throughput / training job), uses < 90 GB and < 60% utilisation (sampled twice). Same queue, same flock, same log.
J=$1; L=$J.lock; R=/dev/shm/big-small-ouro/runs; declare -A busy
while true; do
  [ -s $J ] || { sleep 60; continue; }
  for g in 0 1 2 3 4 5 6 7; do
    grep -qw $g /dev/shm/reserved_gpus 2>/dev/null && continue
    [ -n "${busy[$g]}" ] && [ $(( $(date +%s) - ${busy[$g]} )) -lt 900 ] && continue
    pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); procs=$(echo "$pids" | grep -c .)
    # never share a GPU that is timing something (latency / throughput measurements)
    for p in $pids; do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -E "e2e_latency|bench_throughput|tput_sb|train_batched|sensenova_bridge" && procs=9; done
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    u1=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits); sleep 2
    u2=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$procs" = 1 ] && [ "$used" -lt 90000 ] && [ $(( (u1 + u2) / 2 )) -lt 60 ]; then
      # timing jobs need an idle GPU: leave them at the head for gpu_dispatch.sh
      head -1 $J | grep -q -E "e2e_latency|bench_throughput|tput_sb|train_batched|sensenova_bridge" && break
      line=$(flock $L bash -c "head -1 $J; sed -i 1d $J")
      [ -z "$line" ] && break
      busy[$g]=$(date +%s)
      echo "$(date +%F' '%T) gpu$g(shared): $line" >> $J.log
      ( cd $R; GPU=$g bash -c "$line" ) > /dev/null 2>&1 < /dev/null &
    fi
  done
  sleep 120
done
