#!/bin/bash
# time_think_sweep.sh GPU...: per-request time of real evaluation requests on Ouro-2.6B-Thinking (time_requests.py, the
# Thinking counterpart of time_sweep.sh), batch 1, exclusive GPUs (grid_worker.sh MAX=1). 100 requests each of BBH and
# MMLU-Pro (prompts of the T26x-a42 runs), for Original 2/2 and 4/4, Distilled 2/2 (merged) and the aligned adapter at
# prefill m in {1,2,3,4,6,8} and decode 2. Output: runs/e2e/thinkb-<cond>-<task>.jsonl.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
A=$R/T26-a-universal/checkpoint-48m.pt; D=$R/T26-Df-native22/checkpoint-48m.pt
job() {  # COND CKPT
  for t in bbh mmlupro; do
    echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/time_requests.py $M/model-2.6b-think $1 $2 $R/lmeval/out/T26x-a42-$t 100 $R/e2e/thinkb-$1-$t.jsonl > $R/e2e/thinkb-$1-$t.log 2>&1"
  done
}
{ job native2st -; job native4st -; job D2st $D; for m in 1 2 4 8 3 6; do job a${m}x2sb $A; done; } | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
