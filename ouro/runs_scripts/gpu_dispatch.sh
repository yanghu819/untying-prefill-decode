#!/bin/bash
# gpu_dispatch.sh JOBFILE : whenever a GPU has no compute process (and < 2 GB used), pop the next line of JOBFILE
# and run it with GPU=<index> (lines are shell commands; "$GPU" is substituted). Lines are popped under flock.
# GPUs listed in /dev/shm/reserved_gpus (space separated) are left alone (kept for multi-GPU trainings).
J=$1; L=$J.lock; R=/dev/shm/big-small-ouro/runs; declare -A busy
while true; do
  [ -s $J ] || { sleep 60; continue; }
  for g in 0 1 2 3 4 5 6 7; do
    grep -qw $g /dev/shm/reserved_gpus 2>/dev/null && continue
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    procs=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$procs" = 0 ] && [ "$used" -lt 2000 ] && { [ -z "${busy[$g]}" ] || [ $(( $(date +%s) - ${busy[$g]} )) -gt 600 ]; }; then
      line=$(flock $L bash -c "head -1 $J; sed -i 1d $J")
      [ -z "$line" ] && break
      busy[$g]=$(date +%s)   # grace period: a just-launched job may take minutes to allocate GPU memory
      echo "$(date +%F' '%T) gpu$g: $line" >> $J.log
      ( cd $R; GPU=$g bash -c "$line" ) > /dev/null 2>&1 < /dev/null &
    fi
  done
  sleep 60
done
