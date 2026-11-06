#!/bin/bash
# wait_gpus.sh K COMMAND...: start COMMAND as soon as K GPUs of this box hold no memory, with the comma-separated list of
# those GPUs appended as its last argument (keeps the cards busy without reserving them in advance).
K=$1; shift
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2 < 1000 {print $1}' | head -$K | paste -sd,)
  [ "$(echo "$G" | tr , '\n' | grep -c .)" -ge "$K" ] && break
  sleep 60
done
echo "$(date +%T) starting on GPUs $G: $*"
exec "$@" "$G"
