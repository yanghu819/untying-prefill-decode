#!/bin/bash
# time_batch.sh GPU...: batched serving time (batch_requests.py, continuous batching) of the timed conditions on the same
# 100 requests of GSM8K, MATH-500, BBH and MMLU-Pro as time_controls.sh, at batch sizes $BATCHES (default 32 8 1).
# Output: runs/e2e/<OUT><B>-<name>-<task>.jsonl and .summary.json (env OUT, default batch).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/btime_q.txt; mkdir -p $R/e2e; cd $R; O=${OUT:-batch}
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E=$R/X26-E48f-p1d1/checkpoint-48m.pt; D=$R/X26-D48f-p2d2/checkpoint-48m.pt
E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt; D80=$R/X26-D48fr80-p2d2/checkpoint-48m.pt; P=${PROMPTS:-$R/e2e/prompts}
job() {  # COND CKPT NAME B [ENV]
  for t in gsmfull math bbh mmlupro; do
    echo "cd /root/big-small && $5 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/batch_requests.py $M/model-2.6b $1 $2 $P/X26-a41-$t 100 $4 $R/e2e/$O$4-$3-$t.jsonl > $R/e2e/$O$4-$3-$t.log 2>&1"
  done
}
{
  for b in ${BATCHES:-32 8 1}; do
    job native1st - native1st $b; job D1st $E D1st $b; job D1st $E80 D1r80st $b "OURO_LORA_RANK=80"
    job a4x1pm $A a4x1pm $b; job a4x1pm $F f4x1pm $b
    job native2st - native2st $b; job D2st $D D2st $b; job D2st $D80 D2r80st $b "OURO_LORA_RANK=80"
    job a4x2pm $A a4x2pm $b; job a4x2pm $F f4x2pm $b
  done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
