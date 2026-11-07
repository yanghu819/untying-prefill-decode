#!/bin/bash
# time_sweep.sh GPU...: per-request time of real evaluation requests on Ouro-2.6B (ouro/scripts/time_requests.py), batch 1,
# each job alone on its GPU (grid_worker.sh MAX=1; give it GPUs no other queue uses). 100 requests per benchmark (the same
# items for every condition) of GSM8K, MATH-500, BBH and MMLU-Pro, for the original model at 1/1, 2/2 and 4/4, the
# LoRA-distilled 1/1 and 2/2 (merged), and the aligned loop adapter at prefill m in {1,2,3,4,6,8,12,16} and decode 1, 2.
# Records prefill (time to first token), decode and total time, input and output tokens, and the replayed answer,
# which paper_numbers.py scores with the evaluation's rule. Output: runs/e2e/base-<cond>-<task>.jsonl.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
A=$R/X26-a-universal/checkpoint-48m.pt
P=${PROMPTS:-$R/lmeval/out}   # lm-eval runs whose prompts are replayed (a copy in runs/e2e/prompts on a box without them)
job() {  # COND CKPT
  for t in gsmfull math bbh mmlupro; do
    echo "cd /root/big-small && CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/time_requests.py $M/model-2.6b $1 $2 $P/X26-a41-$t 100 $R/e2e/base-$1-$t.jsonl > $R/e2e/base-$1-$t.log 2>&1"
  done
}
{
  for d in 1 2 4; do job native${d}st -; done
  job D1st $R/X26-E48f-p1d1/checkpoint-48m.pt; job D2st $R/X26-D48f-p2d2/checkpoint-48m.pt
  for m in 1 4 16 2 8 3 6 12; do for n in 1 2; do job a${m}x${n}sb $A; done; done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
