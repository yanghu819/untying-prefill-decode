#!/bin/bash
# controls_first_chain.sh GPUS PORT ARM...: train the corrected native-LoRA controls (first answer token supervised,
# launch_x26_distill.sh ... first) one after another on the same GPUs, each once all of them are idle, and queue each
# one's 13-benchmark evaluation when it finishes (after_x26_distill.sh ... first).
G=$1; PORT=$2; shift 2; cd /dev/shm/big-small-ouro/runs
idle() { for g in ${G//,/ }; do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || return 1; done; }
for a in "$@"; do
  until idle && sleep 120 && idle; do sleep 120; done
  echo "$(date +%T) $a on $G"
  setsid nohup ./after_x26_distill.sh $a first > after-${a}48f.log 2>&1 < /dev/null &
  ./launch_x26_distill.sh $a $G $PORT first > launch-${a}48f.log 2>&1 < /dev/null
done
