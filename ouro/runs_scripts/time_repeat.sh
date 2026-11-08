#!/bin/bash
# time_repeat.sh GPU...: a second timing pass of key configurations (Original 1/1, 2/2, 4/4 and Ours 1/1, 4/1, 4/2) on
# GSM8K and MMLU-Pro, in the reverse order of time_sweep.sh, to measure run-to-run variation of the per-request times
# (advisor: repeat key configurations, interleave the order). Same requests and runners as time_sweep.sh.
# Output: runs/e2e/rep-<cond>-<task>.jsonl.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
A=$R/X26-a-universal/checkpoint-48m.pt; P=${PROMPTS:-$R/e2e/prompts}
for c in "a4x2sb $A" "a4x1sb $A" "a1x1sb $A" "native4st -" "native2st -" "native1st -"; do
  set -- $c
  for t in mmlupro gsmfull; do
    echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/time_requests.py $M/model-2.6b $1 $2 $P/X26-a41-$t 100 $R/e2e/rep-$1-$t.jsonl > $R/e2e/rep-$1-$t.log 2>&1"
  done
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
