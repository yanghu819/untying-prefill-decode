#!/bin/bash
# time_controls.sh GPU...: the controls the quality-time comparison lacked (red-team round 2), timed like time_sweep.sh
# on the same 100 requests of GSM8K, MATH-500, BBH and MMLU-Pro: the aligned adapter with its maps folded into the
# prompt banks at prefill (a<m>x<n>pm, no per-step map work), the fixed-prefill-4 LoRA control (identity maps, same
# folded path; files f4x<n>pm), the rank-80 distilled 1/1 and 2/2 (merged; files D<n>r80st) and Original 3/3.
# Output: runs/e2e/<OUT>-<name>-<task>.jsonl (env OUT, default clean).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R; O=${OUT:-clean}
A=$R/X26-a-universal/checkpoint-48m.pt; F=$R/X26-a-fixed4-lora/checkpoint-48m.pt
E80=$R/X26-E48fr80-p1d1/checkpoint-48m.pt; D80=$R/X26-D48fr80-p2d2/checkpoint-48m.pt; P=${PROMPTS:-$R/e2e/prompts}
job() {  # COND CKPT NAME [ENV]
  for t in gsmfull math bbh mmlupro; do
    echo "cd /root/big-small && $4 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/time_requests.py $M/model-2.6b $1 $2 $P/X26-a41-$t 100 $R/e2e/$O-$3-$t.jsonl > $R/e2e/$O-$3-$t.log 2>&1"
  done
}
{
  for c in a4x1pm a4x2pm a1x1pm a1x2pm a3x1pm a3x2pm; do job $c $A $c; done
  job a4x1pm $F f4x1pm; job a4x2pm $F f4x2pm
  job D1st $E80 D1r80st "OURO_LORA_RANK=80"; job D2st $D80 D2r80st "OURO_LORA_RANK=80"
  job native3st - native3st
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
